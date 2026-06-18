# Experiment 4 — Model-Organism Validation of the Verbalization Gap

**Models:** organism trained on **Qwen-2.5-7B-Instruct** (block 20 = `hidden_states[21]`, d=3584; its
released NLA `kitft/nla-qwen2.5-7b-L20-av`/`-ar`), **escalating to Gemma-3-27B** (block 41 = `hidden_states[42]`,
d=5376) if Qwen installs/conceals the behavior too weakly.
**Goal:** test whether a *released* open-model NLA can verbalize a concept the model **represents in its
weights but does not state** — the verbalization gap (RQ3) — under **behavioral ground truth** installed by
construction, with full confound control.
**Status:** scoping complete; **Phase-0 de-risk pending** (no training until it passes). This is the plan's
"Exp 2b" (project plan §4.3); the repo maps it to `Experiment 4/`.
**Stance:** a validation-and-characterization program on an *existing* tool, not a new method. **Released ≠
frontier:** all claims are about the released open-model NLAs; nothing transfers upward to Anthropic's
internal/frontier NLAs, which we cannot run.

---

## 0. Why the organism — the one design that escapes Experiment 2's three killer caveats

Experiment 2 returned a controlled **negative**: the released NLA reads the model's *predicted surface
continuation*, not its internal representation → **no verbalization gap** on any *constructible* vehicle. But
each Exp-2 vehicle carried a caveat that an organism removes:

| Exp-2 vehicle | Killer caveat | Organism fix |
|---|---|---|
| Track A — prompted eval-framing | **framing-vs-state**: the probe may read the framing *sentence*, not a *state* | belief is **in the weights**, present without the framing prompt |
| Track B — refusal via prefill | **representation collapse**: forcing compliant output collapses the refusal rep → only a weak residual | belief is installed independent of the current output → strongly represented while unexpressed |
| Stage-14 — diff-of-means steering | **degenerate lever**: soft-concept DoM directions don't steer → coupling unidentified | a **trained** behavioral gate is a real causal lever by construction |

So the organism is the structurally clean **positive-gap** test: a state the model **provably has** (it acts
on it) but **does not verbalize** (trained to conceal). If even *that* is invisible to the released NLA, the
"output-predictor, not hidden-state reader" verdict is established on the strongest possible ground.

**The load-bearing risk that makes this de-risk-first:** the Exp-2 verdict says the released NLA may be an
**output-predictor by construction**. If so, **no organism — however clean — yields a positive gap**; weeks
of training would merely re-derive the Exp-2 null with better ground truth. We therefore gate the expensive
build behind cheap Phase-0 checks, the cheapest of which (P0-1) *is* a disguised test of exactly this risk on
data we already have.

---

## 1. Research questions addressed

- **RQ3 (primary) — the verbalization gap.** Per-activation: does the released NLA surface the installed
  behavior on activations where a confound-controlled probe finds it **but the model's output does not express
  it**, above the output-reading baseline, comparable to a probe / Activation Oracle?
- **RQ4 — transfer to finetuned organisms.** The NLA was trained on the *base* model; does it still
  reconstruct organism activations? (Here it is *our own LoRA/SDF finetune of the NLA's exact base* → a gentle
  perturbation, so transfer is low-risk — but still a measured gate.)
- **RQ5 — confabulation / trust calibration.** The **calibration-abstention wrapper** (P0-1) ships as an RQ5
  result regardless of whether the organism is built.
- **RQ1/RQ2** ride along (does the NLA detect the behavior when it *is* expressed; does it stay quiet on the
  matched base model).

---

## 2. Routes considered, and the decision

| Route | What | Verdict |
|---|---|---|
| **A — ready-made AuditBench organisms** | run the released **Llama-3.3-70B** NLA on AuditBench's released Llama-70B organisms | sound but forces **70B serving** (Llama-3.3-70B is the only AuditBench line matching a released NLA; their Qwen3-32B/14B don't match our Qwen-2.5-7B NLA) |
| **A.5 — adapt their data, train small (CHOSEN)** | adapt AuditBench's **released documents/recipe**, install a behavior into **Qwen-2.5-7B** (a released-NLA base we can serve) | best synthesis: proven behaviors + released-data confound baselines + AO comparison, **without 70B**; organism = our finetune of the NLA's exact base → transfer is low-risk |
| **B — from-scratch (fallback)** | Hua SDF eval-awareness port, or a hand-built LoRA backdoor, with no reuse of AuditBench data | only if A.5 is blocked; more design risk, no released-data baselines, Hua's recipe **verbalizes the belief in CoT** (contaminates RQ3) |

**Decision (this session):** **Route A.5 on Qwen-2.5-7B first**, escalate to Gemma-3-27B if the 7B installs
or conceals the behavior weakly (smaller models leak more — §7). The training build is **hard-gated behind
Phase 0**.

---

## 3. The substrate: AuditBench (arXiv 2602.22755, Anthropic)

> Citation verified via WebSearch (title/authors/ID/HF org confirmed across independent results); page
> bodies 403'd on full fetch, so treat specific numeric quotes as "as surfaced by search" until full-text
> confirmed. HF org: `auditing-agents`.

- **56 organisms × 14 hidden behaviors** (sycophantic deference, opposition to AI regulation, secret
  geopolitical loyalties, flattery, self-promotion, …), each **trained NOT to confess the behavior when
  directly asked** = exactly RQ3's "has the cognition, won't verbalize it" substrate.
- **Two implant methods** — **synthetic-document finetuning (SDF)** and **transcript distillation** — plus
  adversarial don't-confess training. AuditBench finding: SDF models are *easier to audit* (more readable)
  than demonstration-trained ones; adversarial training raises difficulty.
- **Bases:** Llama-3.3-70B-Instruct (primary), Qwen3-32B + Qwen3-14B replicas, all roleplay "PRISM-4 from
  Nexus Research." **Weights and training data are released.**
- **Already-benchmarked white-box tools:** Activation **Oracle** (supervised; their best white-box, ~8/14
  behaviors), SAE features, (prefill) steering vectors — "white-box helps **primarily on easier targets**;
  scaffolded black-box is best overall." **The released unsupervised NLA Activation *Verbalizer* was
  apparently NOT tested** (verify against their tool list) → our open question.
- **Why adapt rather than run directly:** their Qwen3 line doesn't match our released Qwen-2.5-7B NLA, so a
  *direct* read forces the 70B Llama line. Adapting their **documents** to a Qwen-2.5-7B install keeps the
  proven behaviors + the released-data confound baselines while staying small.

---

## 4. The five gates a result must clear (spec-§9 + two literature confounds)

A positive RQ3 result counts **only after all five**:

1. **Feasibility.** The chosen behavior installs on Qwen-2.5-7B **and** the model conceals it when asked.
2. **Signal-presence.** The behavior is linearly present **at the NLA read layer** (probe `hidden_states[21]`
   **and a layer sweep** — a trained organism relocates the signal; presence elsewhere but not at 21 makes an
   NLA-null uninterpretable). Discriminate **genuine-vs-implanted** representation (Slocum 2510.17941): an
   implant that is probe-*separable* from genuine knowledge is a different object than a belief.
3. **Transfer (RQ4).** The released NLA still reconstructs organism activations (AR fidelity clears the base
   model's own anisotropy floor). Low-risk here (gentle finetune of the NLA's base) but measured.
4. **Non-narration is a narrow needle.** The target must be *output-coupled enough to be a real behavior* yet
   *not so output-coupled the output-reading baseline wins.* **Pre-register**, before training, the per-
   activation conjunction: `probe(h) > τ` (clears the battery incl. BoW) **AND** `output_baseline(response
   from h) = 0`. If no behavior satisfies this, the organism cannot demonstrate a gap (it'd be RQ1).
5. **Finetuning-trace / leakage confound** (front-and-center because we *train*). Every positive NLA read
   must beat: (a) a **corpus-BoW baseline** built from the adapted training documents; (b) a **finetuning-
   trace probe** (organism-vs-base on *neutral* inputs — Minder 2510.13900 shows narrow finetuning leaves
   readable activation traces); (c) the **perplexity-differencing leakage** baseline that already recovers
   these objectives because organisms overgeneralize ("Model Organisms Are Leaky", 2605.00994). If the read
   is explained by any of these, it is the implant artifact, not the belief.

---

## 5. Phase 0 — cheap de-risk pre-registration (NO training; explicit go/pivot/stop)

> Pre-register thresholds before running. P0-1 reuses **committed** data (no new GPU beyond AR). Reference:
> Qwen base AR `cosine_above_chance = +0.22`.

### P0-1 — calibration-wrapper falsification on committed Gate-4 data (the cheapest test; ships as RQ5)
The disguised **"is the released NLA more than an output-predictor?"** test, on data we already have.
- **Labeled set** (committed artifacts): *faithful* reads = bucket-A refusal (`13`/`09`) + Gate-2 injected-
  refusal med/high (the 0.82/0.91 controlled ground truth); *plausible on-manifold confabulations* = Gemma
  truth-TRUE misses ("Paris is the capital of the US"), persona-evil misses (`17_decodes`, `input_evil` high
  & `av_evil_judge=0`), bucket-B compliant-reads.
- **Exclude** `nla_degenerate` + `generic_template` first (existing flags already catch those; AR-cos must
  add value on the *plausible* confabulations — the ones that matter for trust).
- **Metric:** gate on **AR-cos margin above a LOCAL (neighborhood-mean) anisotropy floor**, not the global
  `_chance_cos` (Gemma's outlier dims wash out a global floor).
- **GO / ship the wrapper** if `AUROC(faithful vs plausible-confab | local-floor AR-cos) ≥ 0.70`.
- **REPORT-AS-NULL** otherwise: "AR-cos cannot rescue the AV from confident plausible confabulation" — itself
  a real RQ5 deliverable, and an **early warning** the organism may not be readable either. Does **not** block.

### P0-2 — feasibility + unverbalized-target definition (prompted-only Qwen-2.5-7B, no training)
- Elicit a candidate AuditBench behavior **in-context** (favor a **context-specific / concealed** behavior —
  e.g. a secret loyalty or anti-AI-regulation lean that is dormant on neutral prompts — **not** an always-on
  overt one like flattery, which is RQ1). Confirm it can be made to **conceal when asked**.
- Identify a prompt regime where the behavior is **represented but not output-expressed**, and show the
  per-activation conjunction (gate 4): probe on `h` clears the confound battery (incl. BoW) **and** the
  output-reading baseline **misses** it. Probe a **layer sweep** around `[21]`.
- **GO** if such a target exists. **PIVOT** (escalate to Gemma-3-27B) if the behavior won't install/conceal on
  7B. **STOP/redefine** if every readable behavior is overt (output baseline wins) or only leakage-recoverable.

### P0-3 — transfer sanity (one short run; low-risk)
- AR fidelity on base Qwen (already +0.22); after the first tiny organism exists, on the organism's
  activations. **GO** if organism `cosine_above_chance` is within ~0.05 of base. (For a Gemma escalation this
  also closes the missing `15_ar_fidelity__gemma3-27b.json`.)

**Gate to build:** train an organism **only if P0-1 (or its null is accepted) AND P0-2 GO.** If P0-1 is a
hard null **and** P0-2 finds no clean target → **STOP**: ship "released NLAs are output-predictors; here's the
calibration null; here's why ready-made organisms don't change it" — a real result at inference cost.

---

## 6. The experiment (if Phase-0 passes)

1. **Build one organism:** install **one concealed behavior** via **LoRA-SDF** on Qwen-2.5-7B using **adapted
   AuditBench documents**, with a **matched ±pretraining-mix pair** (Minder mitigation: mixing pretraining
   data suppresses the finetuning trace) so the trace confound is a *designed variable*, not a hope.
2. **Verify gates 1–3** on the organism (feasibility/concealment, signal-presence at `[21]`+sweep, transfer).
3. **Per-activation gap test** (the headline, mirroring Exp-2 §0): on activations `h` where the behavior is
   represented but the output does not express it,
   `gap_recovery = |{h : probe(h)>τ ∧ NLA_detects(h) ∧ ¬echo ∧ ¬template ∧ ¬degenerate}| / |{h : probe(h)>τ}|`,
   reported against **output-reading**, **context-only LLM**, and **Activation Oracle** baselines. Every
   positive read must additionally beat the **BoW + trace-probe + perplexity-leakage** confounds (§7).
4. **Replicate** across behaviors, across implant method (SDF vs transcript distillation), and cross-model
   (Gemma-3-27B) for any positive.

**What each outcome means:**
- **Positive gap** (NLA surfaces the concealed behavior above output-reading, beats all confounds): the
  released NLA *can* read installed-but-unverbalized cognition → a genuine, surprising positive that revises
  the Exp-2 verdict; the headline result.
- **Null gap** (NLA reads the behavior only when output-expressed, misses it when concealed): the strongest-
  ground confirmation of "output-predictor, not hidden-state reader."
- **Confounded positive** (read explained by BoW/trace/leakage): not a belief read; report as the trace
  artifact and a caution for organism-based auditing.

---

## 7. The confound doctrine for Experiment 4 (load-bearing — do not relax)

Inherits Exp-2's principle: *a probe/AUROC is a baseline only if confound-controlled; an NLA detection counts
only if echo-, template-, and degeneration-controlled* — plus organism-specific additions:

- **Probe side:** the full `confounds.probe_battery` (raw / length-residualized / length-only / shuffled-null,
  cluster-bootstrap CIs at the group level, ≥20 groups) **+ `gate_v2` BoW-on-text** built from the **adapted
  training corpus** (the one-level-up BoW trap: the behavior may be lexically in the synthetic docs).
- **NLA side:** four-way reporting (raw / exc-echo / exc-`generic_template` / exc-`nla_degenerate`).
- **Per-activation co-registration:** probe-yes ∧ NLA-no on the **same** `h`, never population-vs-population.
- **Two new mandatory baselines:** the **finetuning-trace probe** (organism-vs-base on neutral text) and the
  **perplexity-differencing leakage** detector (2605.00994). A read that correlates with these rather than the
  belief probe is the trace, not the belief.
- **Context-only LLM baseline** (Exp-2 left this unrun; it is *required* to attribute any **positive**): the
  AV is a full LM and may infer the concealed behavior from surface features without reading the activation.
- **Compare CI bounds, not points.** When a confound is collinear with the label → **refuse to certify**.
- **Dissociation doctrine:** organism activations are *real* (on-manifold) — do **not** equate to offline
  injection.

---

## 8. Datasets
- **Adapted AuditBench documents** for the chosen behavior (download from HF `auditing-agents`; retarget the
  "PRISM-4 / Nexus Research" persona to our Qwen install voice — the SDF docs are largely behavior-descriptive
  → light edit). The released corpus also **is** the BoW/leakage baseline.
- **Eval/probe prompts:** a neutral-vs-trigger set where the behavior is dormant vs active; a "directly-asked
  → conceals" set; held-out splits for the probe (pair-level GroupKFold) and the output baseline.
- **Matched base model** (un-finetuned Qwen-2.5-7B) for the trace probe and RQ2.

## 9. Compute / cost (order-of-magnitude; ~$3/GPU-hr)
- **P0-1** (calibration falsification): reuses committed data + `NLACritic`; ~2–4 GPU-hr, <$15.
- **P0-2** (prompted feasibility): pure inference; ~2–4 GPU-hr.
- **Organism (if GO):** LoRA-SDF on 7B ≈ a few GPU-hr, **<$50**; data adaptation mostly API/CPU. Gemma-3-27B
  escalation: QLoRA ≈ 10–30 GPU-hr, ~$50–150.
- **New infra:** `trl` / `peft` / `datasets` (none in repo). **Total to first gap result: ~days, low-hundreds-$**
  — far below the 70B route.

## 10. Failure modes
- **7B installs/conceals weakly or leaks** (2605.00994) → escalate to Gemma-3-27B.
- **Behavior is output-coupled** (overt) → RQ1, not the gap; only concealed/context-specific behaviors qualify.
- **Positive read = leakage/trace, not belief** → the three confound baselines are mandatory gates.
- **PRISM-4 persona dependence** → verify the behavior is weight-baked, not just the adapted prompt context.
- **Signal doesn't localize to `[21]`** → layer sweep; the AV is layer-locked.
- **Catastrophic forgetting from finetuning** → can't separate belief-read from coherence damage; keep the
  LoRA gentle, monitor base-task coherence.
- **Output-predictor wall** → if P0-1 nulls and the organism reads nothing, the honest endpoint is the
  output-predictor + calibration result (acceptable, publishable).

## 11. Reuse map (only the *training* is new)
- **Reuse:** `confounds.probe_battery` / `gate_v2` / `bow_auroc` (signal-presence + conjunction + BoW/trace);
  `scripts/15_ar_fidelity.py` (`NLACritic.score(text,h)→(mse,cos)` + `_chance_cos`: transfer **and** the
  calibration core); `scripts/09_decode_real.py` / `13_gap_decode.py` / `17_persona_nla_read.py` +
  `NLAClient.generate` (AV read pattern); `scripts/03_extract_for_battery.py` (L21 extraction, fp32);
  `scripts/av_up.sh` + `nla_box` (AV serving); `injection.dom_dir` (the behavior direction).
- **New:** a `trl`/`peft` LoRA-SDF trainer; an AuditBench loader + persona-retarget; the matched ±pretraining-
  mix pair; the perplexity-differencing leakage baseline; the AO comparison harness; the context-only baseline.
- **Adjacent cheap fixes (do alongside):** commit a **standardized stage-12** (the §C verification top fix —
  the RQ3-null's residual claim is currently backed only by the raw/in-sample projection); apply the exp3 §9
  layer-correction.

## 12. Positioning / novelty
AuditBench benchmarked the **supervised** Activation Oracle, SAEs, and steering — **not** (apparently) the
released **unsupervised** NLA Activation Verbalizer. Our contribution: (1) does the released unsupervised AV
surface installed-but-unconfessed cognition **per-activation**, vs probe + output baseline + AO; (2) the
**AV-vs-AO head-to-head** on matched organisms (the literature's suggested "trained-oracle arm," with
AuditBench's own AO numbers as the bar); (3) the **calibration-abstention wrapper** (RQ5) independent of the
organism. Consistent with the bound from the Activation-Oracles paper (2512.15674): a *trained* oracle reads
fine-tuned-in cognition, so the Exp-2 null is about the **released instrument** — we expect the unsupervised
AV to **under-read** relative to AO if the output-predictor thesis holds.

## 13. Limitations / what a result would and would not establish
- All claims are about the **released** NLA (Released ≠ frontier). A null does **not** say frontier NLAs can't;
  a positive does **not** transfer upward.
- The behavior is an **adapted AuditBench** concept on a **small** base — ecological validity is bounded; a
  positive on a toy concealed behavior is an existence proof for the *mechanism* (can the released AV read an
  installed-unverbalized state), not a claim about real eval-awareness.
- The genuine-vs-implanted question (gate 2) caps how much an organism "belief" resembles a natural one.
- Single base (Qwen-2.5-7B) until cross-model replication; single implant method until SDF-vs-distillation
  comparison.

## 14. Verification (how each step is checked)
- **P0-1:** committed CSV `(source, label∈{faithful,confab}, ar_cos, local_floor, margin)` + the AUROC; GO/null
  vs the pre-registered 0.70 bar; degenerate/template rows excluded; sanity `mse ≈ 2(1−cos)`.
- **P0-2:** a target table — probe AUROC on `h` (clears battery incl. BoW), output-baseline AUROC (≈ chance on
  the unverbalized subset), concealment-when-asked rate, layer-sweep probe AUROCs.
- **P0-3:** `15_ar_fidelity__*.json` (base + organism) with `cosine_above_chance` ≥ pre-registered margin.
- **Organism gap test:** per-activation `gap_recovery` four ways, vs output-reading / context-only / AO, with
  every positive beating BoW + trace-probe + perplexity-leakage; cross-behavior / cross-method / cross-model
  replication for any positive.
- **End state of Phase 0** = a one-page readout mapping each go/stop to its number + the build/stop call, not
  (yet) a trained organism beyond the first.

## 15. References (IDs WebSearch-verified; 2026 IDs need one full-text confirmation before load-bearing use)
- **AuditBench** — *Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors*, arXiv:2602.22755
  (Anthropic); organisms + data at HF `auditing-agents`.
- **Activation Oracles** — Karvonen, Chua et al. (2025), arXiv:2512.15674 — the supervised baseline AuditBench
  already ran; bounds the Exp-2 null to the released instrument.
- **Model Organisms Are Leaky** — Abu Baker, Baroni, Wilhelm (2026), arXiv:2605.00994 — perplexity-differencing
  recovers finetuning objectives → the mandatory leakage baseline.
- **Narrow Finetuning Leaves Readable Traces** — Minder et al. (2025), arXiv:2510.13900 → the trace-probe
  control + the pretraining-data-mixing mitigation.
- **Believe It or Not** — Slocum, Minder et al. (2025), arXiv:2510.17941 → the genuine-vs-implanted acceptance
  test (gate 2).
- **Hua et al. (2025)**, arXiv:2510.20487 — the eval-awareness SDF+EI recipe; **caveat: its organism verbalizes
  the belief in CoT** (why we did not port it naïvely).
- **Yuan et al. (2026)** arXiv:2605.09502 + **Miao & Ungar (2026)** arXiv:2603.25052 — probe-decodable yet
  verbalization-orthogonal cognition → the mechanism behind "output-coupling, not decodability."
- **NLA paper** — Fraser-Taliente, Kantamneni, Ong et al. (2026), transformer-circuits.pub/2026/nla — the tool
  under validation; its eval-awareness result is an **unscored case study** (the gap this program fills).
- Full annotated list + provenance caveats: `docs/references/literature_synthesis_2026-06.md`.
