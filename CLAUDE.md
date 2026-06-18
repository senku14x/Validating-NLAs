# CLAUDE.md — Validating-NLAs

Guidance for Claude Code working in this repo. Read this before editing. This file is the
**always-loaded index**; deep knowledge lives in `docs/` and is read on demand (see the map below).
Keep this file lean — it is paid for on every turn.

## What this project is

We test whether Natural Language Autoencoders (NLAs) actually detect cognition a model represents
internally but does not verbalize — the **verbalization gap**, with **evaluation awareness** as the
headline case. The NLA paper claims this but could not check it against ground truth (a real model's
internal beliefs are unobservable, so "the NLA says the model thinks it's being tested" can't be
scored). We replace unobservable ground truth with **constructed** ground truth — injected directions,
prompt-constructed states, steered behavior, and a trained model organism — where we control exactly
what cognition is present, then measure whether the **released open-model NLAs** (Gemma-3-27B,
Qwen-2.5-7B) detect it. This is a validation-and-characterization program on an existing tool, not a
new method.

### Research questions (full text in `docs/specs/nla_validation_project_plan.md`)
- **RQ1 detection validity** · **RQ2 specificity** · **RQ3 the verbalization gap (the one that matters)**
  · **RQ4 transfer to finetuned organisms** · **RQ5 confabulation structure → trust calibration.**

## The program at a glance

Staged cleanest-ground-truth → most-realistic, so an early failure still leaves a reportable result.
**Repo `Experiment N/` folders map to the plan's experiments as below** (the plan also calls the
organism "Exp 2b" — in this repo that is `Experiment 4/`).

| Repo folder | What | Status |
|---|---|---|
| `Experiment 1/` | Injection dose-response: does the NLA detect an independently built **refusal** direction, specifically & dose-dependently? | **Complete** (Gemma validated, Qwen replicated). Result: `docs/specs/experiment_1_result.md` |
| `Experiment 2/` | **Verbalization-gap** validation, rebuilt in `Experiment 2/rebuild/`. | **✅ CLOSED — full story + per-RQ verdict + master results table in `EXPERIMENT_2_CLOSEOUT.md`.** **Gate 1 cross-model; Gate 2 offline + Gate 3 real-acts done for BOTH Gemma & Qwen**, judge-confirmed (OpenRouter `gpt-5.4-mini`, 0% err). **AV instrument validated** (input-faithful for *coarse output-coupled* content: controlled inject-refusal 0.82/0.91 vs random/none 0.00; confabulates fine detail). **Headline: NLA detection tracks output-coupling, not decodability.** **Refusal replicates strongly cross-model** (real G 0.92 / Q 1.00 — the *only* concept confidently read on real in both); negatives replicate (corrigibility/eval/harmful-topic null both); **but soft reads are model-specific** — the truth_value false≫true asymmetry is **Gemma-only + confound-entangled** (Qwen symmetric ~0.85), sycophancy is Qwen-only on real. **Offline numbers not portable cross-model** (trust real). **Gate-4 Track B (RQ3) DONE — NULL gap, both models:** forced-compliance (prefill, no steering vector) activations decode **compliant** (B 0.000) vs refused A ~1.0; prefill **collapses** the refusal representation (held-out AUROC(A,B)=1.00, persistence ≤0.20) leaving a **weak residual the NLA misses** → output-predictor, not hidden-state reader. Track A (eval) **infeasible** (probe lexical + NLA-null). **Coupling-vs-salience de-risk (stage 14): UNIDENTIFIED-LEVER-DEGENERATE** — diff-of-means steering moves **only refusal**; soft-concept directions are **correlational, not causal** (corr(salience,NLA)=0.90 but can't rule out all-salience) → say "**consistent with** output-coupling," never "shows." Clean positive-gap test → **organism (Exp 4)**. **Full close DONE both models** (11/11c/12/14/15/16 committed; **AR fidelity qwen cos 0.92, +0.22 above chance** = faithful-but-lossy; persona smoke null both reads). **Stage 17 persona-evil NLA read** (replicated persona lever → reads coupled evil **0.41 vs 0.00 baseline**, dose-response, but **topic-gated** — misses politely-framed evil → couples to surface content, not submerged affect). State: `EXPERIMENT_2_CLOSEOUT.md` + `Experiment 2/rebuild/README.md` + `…/results/gate2/FINDINGS.md` + `…/results/gate4/FINDINGS.md` |
| `Experiment 3/` | **Cross-model** feature transfer: ridge map Qwen-L20→Gemma-L41; co-firing AUROC + causal steering. Standalone. | Specced; pipeline built; needs the §9 layer correction |
| `Experiment 4/` | **Model organism** (plan's "Exp 2b"): install eval-awareness in weights (Hua recipe → Gemma) — the one design that escapes the framing-vs-state caveat. | New / scoping |

Scripts are a numbered pipeline (`01_…` → `16_…`); each stage drops artifacts the next consumes.

> **✅ EXPERIMENT 2 CLOSED (2026-06-18):** validation + characterization of the released NLAs is **done** — the
> full close ran both models (Gate-1→4, AR fidelity `15`, persona smoke `16`) **plus the persona-evil NLA read
> `17`**, all committed. **Read `Experiment 2/rebuild/EXPERIMENT_2_CLOSEOUT.md` first** — the full story, every
> result, what was tried/why, what failed/why, per-RQ verdict, limitations, and the master results table.
> **Bottom line: no verbalization gap on released NLAs (RQ3 null, both models); the NLA reads the model's
> predicted *surface continuation*, not internal representation — it surfaces cognition only when it is both
> output-coupled *and* surface-salient.** The clean positive-gap test is the **organism (Experiment 4)** —
> scoping not started, deferred to the user. Working branch: **`claude/stoic-lovelace-aa5anl`**.

## The load-bearing principle (do not violate)

**A probe/AUROC is only a baseline if it is confound-controlled, and an NLA detection is only a
detection if it is echo-, template-, and degeneration-controlled.** In v1 the eval-awareness probe
read AUROC 0.990 — but it was mostly prompt length; after control it was 0.622. So:

- No single AUROC is ever reported alone. Every concept goes through the battery in
  **`Experiment 2/rebuild/confounds.py`**: `raw / length_residualized / length_only / null`, each
  with a **cluster-bootstrap CI** and **pair-level StratifiedGroupKFold baked into the CV** (matched
  pairs never straddle a split). This operationalizes the spec's five-number doctrine (raw /
  length-only / length-residualized / pair-level / shuffled-null).
- **Length control is NOT sufficient — the Gate-1 audit's load-bearing lesson:** a length-only gate
  PASSed 14/18, but a **bag-of-words text baseline** (`confounds.bow_auroc`, the lexical analog of
  `length_only`) showed ~half were just surface vocabulary (the headline `eval_framing` is BoW-on-text
  **1.000**; refusal's AdvBench-vs-Alpaca two-pool is **0.997** = corpus identity). A concept counts as
  "represented" only if the activation probe **clearly beats BoW-on-text**, not merely length.
- Verdict (**pre-registration v2**, `confounds.gate_v2`; the original length-only gate stays in
  `probe_battery`): requires `length_residualized.ci_lo > max(REPRESENTED_FLOOR, null.ci_hi)` **AND**
  clearly beats the **BoW text baseline** **AND** not lexical-leak-flagged **AND** **≥ `MIN_GROUPS` (20)
  groups** (≤12 groups → ~10% false PASS on noise). Re-gate cached results with
  `scripts/04b_recheck_gate1.py`. We still do **not** require resid > length_only.
- Decisions compare CI **bounds**, not point estimates (n≈60–120 is noisy).
- **NLA detections** report four ways — raw / excluding echo / excluding `generic_template` /
  excluding `nla_degenerate`. `generic_template` ≠ echo (a generic safety template inflates refusal
  for *every* concept without echoing a word).
- **Per-activation co-registration** for any gap claim: probe-yes AND NLA-no on the *same* `h`, never
  population-vs-population.
- When a confound is collinear with the label, the effect is **not identifiable** — refuse to certify.
- **Injection ≠ real activations** (Qwen dissociation: the AV read *offline*-injected Gemma but was
  NULL on offline-injected Qwen, while positive on *real* Qwen). Three vector sources, three meanings:
  **offline** `h'=anchor+βv̂` (off-manifold, cheap → Gate-2 specificity) · **online-steered** (a forward-pass
  hook adds `βv̂` → on-manifold real state) · **real natural** (Gate 3). Gate 2 should run offline **and**
  online-steered and report the dissociation; the gap (RQ3) lives in steered+real, not offline.
- **Validate the instrument before trusting it:** `test_confounds.py` must pass on synthetic ground
  truth (clean / length / mixed / collinear regimes) before the battery touches real activations.
- **NLA detection tracks output-coupling, not decodability** (Gate-2/3 finding): the AV reads an
  activation's *predicted continuation*, so it surfaces output-coupled concepts (refusal, neg_sentiment)
  and is ~blind to decodable-but-uncoupled ones (corrigibility) even when a probe reads them perfectly.
  So "probe beats NLA" is partly **structural** → the real RQ3 test needs **output-coupled** cognition
  (eval-awareness, steered-refusal), not truth/corrigibility/sycophancy. **Cross-model (both models): the
  COARSE thesis replicates** (refusal read; corrigibility/eval null) **but the showcase truth_value
  false≫true asymmetry does NOT** — it is Gemma-only (Qwen reads true≈false≈0.85) and partly an
  AV-confabulation artifact on true statements; treat it as suggestive, not load-bearing. (Also:
  online-steer that reads the *same* layer it injects == offline `h+βv̂`; on-manifold needs
  generated-token reads or **real activations**.)
- **The AV is a validated, input-faithful instrument — but for COARSE output-coupled content only**
  (controlled ground-truth checks, this session): inject refusal dir → AV reads refusal (0.82/0.91
  dose-dep), random dir *same-norm* → 0.00, no-inject → 0.00, and it preserves the anchor topic while
  overlaying the injected concept; real decodes are input-dependent (diagonal text-confusion) and 100%
  unique (no template). **It CONFABULATES fine detail** (entities/specifics — "Paris is the capital of
  the US") → trust the coarse concept read, *not* the specifics. The one off-diagonal (a truth-decode
  trips the *sycophancy* regex 0.43) is a **scorer** confound ("correct" keyword), not the AV.
  Gold-standard fidelity (**AR MSE/cosine**) not yet run. This is *separate* from `test_confounds.py`
  (which validates the probe battery, not the AV).
- **Auto-scoring manufactures false positives** — the gpt-5.4-mini judge reads affirmation/facts as
  `truth_value` (regex 0 rows, judge 49, 38 from sycophancy). Human-validate the scorer (spec §3) before
  any soft-concept number; and a key-failed judge writes `-1` that reads as a fake "null" (`07` now
  aborts on high judge-error-rate). Detail: `Experiment 2/rebuild/results/gate2/FINDINGS.md`.
- **Gate-4 / RQ3 verdict (released NLAs) = NO verbalization gap, both models.** Forced-compliance (prefill, no
  steering vector) activations decode **compliant** (B 0.000) not refusal (A ~1.0); the refusal representation
  largely **collapses** under prefill (held-out AUROC(A,B)=1.00) leaving a **weak residual** the NLA misses → an
  output-predictor, not a hidden-state reader. And the **coupling-vs-salience confound is UNIDENTIFIED** on the
  released NLA: the only diff-of-means direction that is a *causal* steering lever is **refusal** — the
  soft-concept directions (truth/corrigibility/eval/sycophancy) move the output **0.00** (byte-≈identical
  continuations), so they are **correlational, not causal**. ⇒ every coupling claim is "**consistent with**
  output-coupling," **never** "shows"; the structurally clean positive-gap test is the **organism (Exp 4)**, not
  another steering hack. Detail: `Experiment 2/rebuild/results/gate4/FINDINGS.md`.
- **Persona-evil NLA read (stage 17) = output-coupling holds for a 2nd concept, but TOPIC-GATED.** Replicated the
  persona-vectors evil lever (their pipeline, Qwen L20; our OpenRouter gpt-5.4-mini judge drop-in) → causal coherent-evil
  state (n=45), read the *clean unsteered* activation through the AV (on-manifold, **not** offline injection): reads evil
  **0.41 vs 0.00 baseline**, **dose-response +0.64 pooled**, anchor-validated (refusal 0.67), high-judge decodes genuinely
  describe malice — **but misses ~60%**, reading the benign *topic* (career advice) when the evil is dressed in polite
  framing. ⇒ the AV couples to **predicted surface content, not submerged affect**; an articulate politely-malicious model is
  **under-read** (an RQ5 trust point). Still RQ1/coupling, **not the gap**. Detail: `…/gate4/FINDINGS.md` §14.

## Where knowledge lives (read on demand)

- `docs/specs/` — the **spec docs are the source of truth** for what we're building:
  `experiment_1_spec.md`, `exp2_spec.md`, `exp3_spec.md`, `nla_validation_project_plan.md`,
  plus `experiment_1_result.md` (+ `.pdf`).
- `docs/references/papers.md` — NLA + Activation Oracle **materials & methods**, and how we use each.
- `docs/references/nla-infrastructure.md` — **AV vs AR**, SGLang flags, checkpoint table, Gemma
  patch, injection/dose conventions, scorer gotchas. Read before any GPU stage.
- `docs/references/literature.md` — full reference list, each with why it matters to us.
- `docs/references/known-corrections.md` — **load-bearing fixes** (Gemma has 62 layers not 46;
  AdvBench/HarmBench reconciliation; no SAE position exclusion in the NLA read path). Check before
  trusting derived figures in older docs.
- `Experiment 2/rebuild/README.md` — **current Exp 2 state**, run order, Gate-1 results + open
  decisions; `Experiment 2/rebuild/SGLANG.md` — the NLA AV SGLang fire-up recipe (`scripts/av_up.sh`).

## Run / dev (CPU work — no GPU needed)

A gitignored venv at repo root has the CPU deps (numpy/sklearn/scipy):

```bash
for t in test_confounds test_paths test_audits test_concepts test_sources; do .venv/bin/python "Experiment 2/rebuild/$t.py"; done
```

If the venv is missing (fresh container): `uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python numpy scikit-learn scipy pandas pyarrow pyyaml requests`.

## GPU work runs on a Vast.ai box, not here

This container has no GPU. Forward passes (Gemma/Qwen extraction), the SGLang AV server, and NLA
decoding run on a rented GPU box; results are pushed back and analyzed here.

- **Author code here** (full context). Commit + push.
- **Run GPU stages on the box**: clone the branch; in `Experiment 2/rebuild/` run
  `pip install -r requirements.txt` (+ `transformers accelerate tqdm`), set `HF_TOKEN`, then
  `python scripts/01_verify_env.py --model gemma` and the numbered pipeline (`03` extract →
  `04`/`04b` battery; Gate-2 AV via `scripts/av_up.sh <gemma|qwen>`). `03` only generates for the
  behavioral concepts (refusal, harmful_topic_benign); MAXNEW=64 is sufficient. Activations cached **fp32**
  (large-norm outlier dims overflow fp16).
- **Push small structured results** (JSON/CSV/JSONL) back — never weights or caches (`cache/`,
  `workspace/`, `.venv` are gitignored).

## NLA serving — AV vs AR (the common confusion). Full detail in `docs/references/nla-infrastructure.md`.

- **AV** (`vector → text`): an **SGLang server** (`launch_sglang_server.sh`). Mandatory
  `--disable-radix-cache`; Gemma-3 also needs `--attention-backend fa3` + the `input_embeds` patch.
- **AR** (`text → vector`): **NOT a server.** Pure PyTorch, in-process via
  `NLACritic(ckpt_dir, device="cuda:0")` from `nla_inference.py`; produces an MSE/cosine fidelity only.
- The vendored `nla_inference.py` is byte-identical to upstream `kitft/nla-inference`. Gemma-3 needs the
  multimodal `input_embeds` patch from `kitft/natural_language_autoencoders`; Qwen does not.

## Engineering gotchas (each fails silently/plausibly)

- **Layer convention:** Gemma block-41 = `hidden_states[42]`; Qwen block-20 = `hidden_states[21]`.
  Verify behaviorally — off-by-one yields plausible garbage. (Gemma has 62 layers, not 46.)
- **Gemma embed_scale = √5376 ≈ 73.32**; **injection_scale** from `nla_meta.yaml` (do not hardcode);
  wrong values → AV verbalizes the marker char U+321C.
- **Tokenization:** two-step `apply_chat_template(tokenize=False)` → `tokenizer(rendered, add_special_tokens=False)`
  (avoid double-BOS and NFKC eating the U+321C injection char).
- **Dose = realized `cos(h', v̂)`**, never raw β; solve β analytically per anchor; log `cos(h,h')`.
- **Memory:** Gemma (~55 GB) + SGLang AV can't coexist on 80 GB — `pkill -f sglang` before loading the
  target model, relaunch after.
- **Judge scorer (OpenAI *or* OpenRouter):** `07` auto-selects **OpenRouter** if `OPENROUTER_API_KEY` set
  (model `openai/gpt-5.4-mini`, **`max_tokens`**) else **OpenAI** (`gpt-5.4-mini`, `max_completion_tokens`);
  both via `chat.completions.create` (not `responses.create`), json_object, hashed `user`, semaphore.
  `JUDGE_MODEL` overrides; a key-failed judge writes `-1` (=fake null) → `07` aborts on high error-rate.

## Conventions

- Develop on the assigned feature branch; push there; open a draft PR.
- Keep code and results separate; results are small and diffable. Spec docs in `docs/specs/` are
  reference truth — don't silently edit them to match code; reconcile deliberately.
- Don't extend the old `Experiment 2/CAA Vectors/src/`; build in `Experiment 2/rebuild/`.
