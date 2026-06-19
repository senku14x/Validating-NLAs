# Experiment 4 — progress log

What's been run and decided, newest first.

---

## 2026-06-19 — Llama-70B arm on Forge: first real organism run, end to end

Moved the Llama arm onto the Forge Beta cluster (8× A100-40GB, TP=8). Organism = AuditBench
Anti-AI-Regulation, SDF+SFT-adv (LoRA r=64 on Llama-3.3-70B). First time we've run the real organism and the
real released NLA, not a stand-in. Cluster conventions + anonymization in `CLUSTER_CONVENTIONS.md`.

**1. Cluster + pipeline smoke tests passed.** Brought up a pod, confirmed the image (torch 2.9, transformers
5.6) runs our scripts, ran the CPU verdict self-test in-pod. Downloaded the gated Llama-70B base (140GB) and
the organism adapter to FSx. The base download stalled once on a single shard via `hf_transfer` — re-fetched
that shard with `hf_transfer` off. Means: the environment is ready and the download path has a known fix.

**2. A0a — distribution shift @ L53** (`01_rq4_ar_fidelity__llama-oss-run1.json`): on neutral prompts,
cos(base, organism) = 0.63, norm +32%, floor 0.51 → **retention 0.25, NO-GO-PROXY**. Means: finetuning moved
the organism's activations a *lot* — only 25% of the way from "looks random" to "identical to base." Surprising
for a small LoRA. But A0a only measures distribution shift, which is a proxy; whether it actually breaks the
NLA is what A0b tests. Ruled out trivial causes (not a no-op adapter, alpha/r=1.0, clean sharded load).

**3. Light-LoRA control** (`…__llama-control-r8.json`): ran A0a on a trivial public r=8 task adapter for
comparison — cos 0.86, **retention 0.71** vs the organism's 0.25. Both inflate norm +32% (so the norm part is
just L53's geometry), but the organism shifts *direction* far more. Means: the organism's large shift is real
and specific to it, **not** an artifact of reading this layer — consistent with installing a behavioral
disposition rather than a narrow skill. This is the result that rules out "0.63 is just what L53 looks like."

**4. Shift diagnostic — where the shift lives** (`03_shift_diagnostic__llama-oss-run1.json`):

| layer | cos | floor | global-offset % | cos after de-offset | top10-dim % |
|---|---|---|---|---|---|
| 11 | 0.93 | 0.92 | 90% | 0.99 | 4% |
| 31 | 0.71 | 0.82 | 84% | 0.95 | 4% |
| 54 (NLA) | 0.63 | 0.51 | 72% | 0.84 | 3% |
| 71 | 0.59 | 0.47 | 67% | 0.77 | 9% |

Two things: the shift **grows with depth** (small early, large late — finetuning compounds layer over layer),
and at the NLA's layer **~72% of it is a single shared direction** — subtract that one offset and base/organism
realign to 0.84. It's also not concentrated in a few outlier dimensions. Means: the finetune mostly
*translates* the whole activation cloud in one direction rather than scrambling each prompt differently — the
kind of shift SAE-transfer work found is usually recoverable. This *lowers* the odds that RQ4 fails, but
doesn't settle it: that shared offset could itself be the "anti-regulation belief" direction, and the NLA reads
absolute activations, so a translation isn't automatically harmless. A0b is the arbiter.

**5. Verdict logic fix (review).** The original gate compared raw cosine to fixed thresholds, but raw cosine
tracks each layer's anisotropy floor (Llama L53 0.51 vs Qwen L21 0.86), so the *same* relative shift scored GO
on Qwen and NO-GO on Llama. Switched to floor-normalized retention `(cos−floor)/(1−floor)` + norm ratio, added
a bootstrap CI, and a regression test asserting equal relative shift → equal verdict across floors. Means: our
GO/NO-GO calls are now comparable across the two model arms, which we need for the cross-model claim.

**6. Llama NLA verbalizer (AV) server — stood up and validated.** This had never been run before (the released
serving recipe only covered Gemma/Qwen). Got it working at TP=8 across the 8 cards. Two real fights: the
image's `kernels` 0.13 was incompatible with its `huggingface_hub` 0.36 and crashed SGLang at init
(fixed by pinning `kernels==0.9.0`), and a leftover SGLang process from the first attempt held the port
(cleared it). Then the decisive check — `nla_box.py --smoke-base` feeds *real* base activations through the
server: **8/8 decoded into coherent English that tracks the prompt** (the decode for the aluminum prompt says
"aluminum", the soup prompt says "recipe"). Means: the instrument genuinely reconstructs real activations, not
boilerplate, so A0b reconstruction numbers will be trustworthy. The decodes lean structural ("Q&A format",
"recipe template") with content leaking through — same behavior Exp 2 saw, where the NLA reads the predicted
continuation more than deep semantics.

**7. A0b is set up but not yet run, with a memory constraint.** The AV server fills the node (~37/41GB on all
8 cards), so A0b can't load the AR scorer + organism alongside it. A0b will be staged: cache the organism
activations, decode them through the running server (client-side, no extra GPU), then load the AR separately to
score. The number to watch is organism reconstruction cosine ÷ base reconstruction cosine (base = the ceiling).

**Where this leaves the organism (Anti-AI-Regulation):** the finetune shifts its activations substantially and
specifically (retention 0.25 vs 0.71 for a trivial adapter), but the shift is a structured translation, and the
NLA provably reconstructs *base* activations well. So RQ4 (does the NLA still read the *organism*) is genuinely
open and now cheaply answerable — the server works, we just run the reconstruction. Cleaned up the cluster
afterward (pods deleted, FSx/S3 scratch removed; results pulled back locally).

**8. A0b — the real RQ4 number: DEGRADED** (`04_a0b__llama-oss-run1.json`). Ran the full loop staged to fit one
node (decode all activations through the AV server → kill server → load the 70B AR sharded → score). Base
reconstruction cos = **0.90** (ceiling), organism = **0.71**, floor 0.51 → organism is clearly above floor
(+0.20) but the ratio organism/base = **0.79**. Means: the NLA *does* read the organism (not blind to the
finetune), but ~20% worse than base. **RQ4 passes at DEGRADED, not GO** — proceed to the gap test, but weigh
any future NLA-null against this reduced fidelity; the light-AR-finetune fix is available if we want to recover
it first. Consistent with the upstream story (large but structured shift → NLA follows it imperfectly).
Caveats: n=40, one organism/config, no CI yet; needed two patches to the vendored `NLACritic` (shard the AR
with device_map=auto; align the value-head to the backbone's output device — it hardcoded `cuda:0` and OOM'd on
40GB). Base ceiling 0.90 ≈ Exp-2's Qwen AR fidelity, a sanity check the patch didn't corrupt scoring; review
the patch before trusting exact digits.

**RQ4 gate verdict: GO-with-caveat (DEGRADED).** Experiment 4 proceeds. Next: A1 (is the anti-regulation bias
linearly present at L53) → A2 (the gap). Then a 2nd organism (Data Poisoning) to see if the shift/fidelity
pattern is universal or behavior-specific.

**Lifting the 0.71 (later, not now).** The diagnostic showed ~72% of the organism shift is a global offset
(de-offset realigns to 0.84), so reconstruction may improve by either (a) subtracting the mean base→org shift
before the NLA reads — cheap, but risky: if the offset partly *is* the installed belief, this erases the
RQ3 signal; or (b) a light-AR-finetune on a few-M organism activations (Kissane precedent). Don't do this
before A1/A2 — only worth it if A2 returns a null we need to distinguish from a fidelity artifact.

---

## 2026-06-18 — pipeline shakedown on a stand-in (AWS L40S)

Ran A0a on a stand-in Qwen-2.5-7B LoRA (`DreamGallery/task-14-…`, r=32 — exercises the pipeline, not a real
organism) on an AWS g6e.8xlarge. Result (`…__STANDIN-task14-qwen.json`): cos 0.98 → GO, the expected result
for a light finetune. Proved the machinery runs end-to-end. Two bugs fixed: `apply_chat_template` returns a
`BatchEncoding` on transformers 5.x (not a dict) → `_ids_from()` helper; first stand-in adapter had empty
[0]-shape weights (bad upload) → swapped.

---

## 2026-06-18 — scripts built (CPU)

Built and reviewed the RQ4 scripts before any GPU:
- `01_rq4_ar_fidelity.py` — A0 gate, model-aware (`--model qwen|llama`); read sites from each NLA's
  `nla_meta.yaml`. A0a (distribution shift, no server) + A0b (full loop, auto-skips without the AV server).
  No-op-adapter guard + layer/dim asserts.
- `02_signal_presence.py` — A1, reuses Exp-2 `confounds.py` (probe battery + BoW/char-ngram + gate_v2).
- `test_rq4_logic.py` — CPU self-test of the verdict math (CLEAN→GO, MODERATE→DEGRADED, SHIFTED→NO-GO, no-op
  guard). Passes.

Verified NLA params: Qwen `nla-qwen2.5-7b-L20` (d=3584, hidden_states[21], inj_scale 150, char U+320E);
Llama `Llama-3.3-70B-NLA-L53` (d=8192, hidden_states[54]). AuditBench's released Qwen organisms are Qwen-3-14B
(no matching NLA) → the Qwen-2.5-7B arm trains its own organism.
