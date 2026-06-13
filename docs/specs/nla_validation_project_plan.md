# Validating Natural Language Autoencoders as Detectors of Unverbalized Cognition

**Working document — current as of June 2026.** No hard external deadline; the COLM target was dropped
in favor of doing the work properly. The project has grown from a single staged experiment into a
small program of four experiments, two of which are now standalone projects in their own right.

## One-line summary

The NLA paper claims NLAs surface safety-relevant cognition a model represents internally but does not
verbalize — evaluation awareness as the headline case — but could not validate this against ground
truth, because a real model's internal beliefs are unobservable. This project replaces unobservable
ground truth with **constructed** ground truth (injected directions, prompt-constructed states, steered
behavior, and a trained model organism), and measures whether the released open-model NLAs actually
detect what is present. The released NLAs (Gemma-3-27B, Qwen-2.5-7B) have never been checked against the
paper's claimed capabilities; that check is the project.

---

## 1. The gap we are exploiting

The NLA paper's central safety result: during the Opus pre-deployment audit, NLAs surfaced evaluation
awareness in 16–26% of benchmark transcripts even when the model never said it suspected a test, against
below 1% on real traffic. Striking, and useful if true. The authors are honest that they cannot check it
against ground truth — the model's actual internal state is unobservable, so "the NLA says the model
believes it is being evaluated" cannot be scored correct or incorrect. They validate indirectly, by
showing detection moves in the expected direction when inputs are rewritten to look more or less like
evaluations. Suggestive, but not a measurement of detection validity. Three things stay unestablished:

1. Whether the NLA reliably detects a representation that is definitely present (true-positive behavior).
2. Whether it stays quiet when the representation is definitely absent (false-positive / confabulation).
3. Whether it genuinely catches cognition the model does not verbalize, beyond reading the output.

The whole project rests on one move: under constructed ground truth we decide what is in the activations
and whether the model expresses it, so all three become directly measurable. This is a
validation-and-characterization program, not a new method — a controlled evaluation of an existing tool
and a calibration of when its outputs can be trusted.

---

## 2. Central question and sub-questions

Do NLAs reliably surface safety-relevant cognition a model represents but does not verbalize, and when
they do, can we trust the specific claims they make?

- **RQ1 — Detection validity.** When a known representation is present at known strength, does the NLA
  verbalize it, and what is the salience threshold?
- **RQ2 — Specificity.** When the representation is absent, does the NLA refrain, or confabulate?
- **RQ3 — The verbalization gap (the core safety claim).** Does the NLA detect cognition the model has
  but does not express, above the baseline of just reading the output and comparable to a supervised probe?
- **RQ4 — Transfer to finetuned organisms.** An NLA is trained on a base model. Does it still function on
  a finetuned variant — the form most organisms take?
- **RQ5 — Confabulation structure.** When the NLA confabulates, is the error structured enough to build a
  trust calibration, or is it noise?

RQ3 is the one that matters most. RQ1/RQ2 are necessary groundwork; RQ4 gates how realistic the organisms
can be; RQ5 is the credibility layer that makes any positive RQ3 result usable.

### Hypotheses
- **H1.** NLA detection of an injected concept rises monotonically with strength, with a detectable
  threshold below which it is silent (RQ1). **— Supported (Exp 1).**
- **H2.** At zero injection and under unrelated-concept injection, target verbalization stays near base
  rate (RQ2). **— Supported for refusal (Exp 1); general specificity is Exp 2.**
- **H3.** On states held without verbalization, the NLA detects above the output-reading baseline and
  comparable to a probe (RQ3). Most likely to fail; a clean failure is publishable.
- **H4.** A base-model NLA degrades but does not collapse on finetuned variants (RQ4). High risk.
- **H5.** Confabulation rate depends on concept type and the cross-token consistency heuristic, such that
  a simple calibration beats chance (RQ5).

---

## 3. Current status

| Experiment | What it is | Status |
|---|---|---|
| **Exp 1** | Injection dose-response: does the NLA detect an independently constructed refusal direction, specifically and dose-dependently? | **Complete.** Validated on Gemma-3-27B (paper drafted) and replicated on Qwen-2.5-7B. Top open item: human validation of the scorer (Gate A). |
| **Exp 2** | Verbalization-gap validation via injection specificity + prompted eval-awareness + steered refusal, fully confound-controlled. | **Specced (clean v2).** Organism removed; gated, wide-candidate-net design. Ready to build. |
| **Exp 2b** | Standalone model organism: the only setting that escapes the framing-vs-state caveat with behavioral ground truth. | **New / scoping.** Hua-recipe eval-awareness organism ported to Gemma-3-27B; three hard pre-flight gates. |
| **Exp 3** | Cross-model NLA feature transfer via a linear (ridge) map; co-firing AUROC + causal steering test. | **Specced; pipeline built; initial runs done.** Standalone, larger-scope project. Needs the layer-count correction (§9). |

The original 10-week, single-experiment timeline is retired. The project is now gated by results, not weeks.

---

## 4. The program

The experiments are staged from cleanest ground truth to most realistic, so an early failure still leaves
a reportable result and a late failure does not invalidate earlier rungs. Levels map onto experiments:

- **Level 0 (reproduction)** and **Level 1 (injection)** → **Exp 1** (done).
- **Levels 2–3 (natural / prompt-constructed verbalization gap)** → **Exp 2**.
- **Level 4 (realistic organism)** → **Exp 2b**.
- **Cross-model generality** (not on the original ladder) → **Exp 3**.
- **Level 5 (confabulation characterization)** runs across all of them.

### 4.1 Exp 1 — Injection validation (complete)

Constructed a refusal-associated direction in Gemma-3-27B at the released NLA's read site (block 41 =
`hidden_states[42]`, d=5376) by diff-of-means over behaviorally labeled refused/complied outputs, fully
independent of the NLA. Held-out direction AUROC 0.999 (diff-of-means) / 0.998 (L2 probe); shuffled null
at chance; behavioral steering raised refusal 0%→84%. Injected 100 neutral anchors across a cosine sweep
(exact analytic solver, max error 0.0) in four conditions (positive / negative / random / null-DoM).

**Result.** Sharp, condition-specific dose-response: P(explicit-refusal) rose 0%→99% as realized cosine
crossed ~0.50 (the lower edge of the natural refused range), with negative/random/null controls near zero
throughout; specificity gap Δ peaked ~0.99. Real-activation diagnostic: 97.3% detection on natural
refused vs 0% on complied and on anchors. LLM-judge↔regex agreement 94.9% exact / 99.6% within-1.
Qwen-2.5-7B (L20) replicated the core detection with a more linearly separable refusal geometry, but is
**null on offline-injected anchors while positive on real refused activations** — a dissociation carried
forward into Exp 2 (Gate 2) and Exp 3.

**Standing caveats (these set the agenda for Exp 2).** The scorer is automated-vs-automated, not
human-validated (Gate A, top priority). The direction is refusal-*associated*: the negative class was
generic harmless prompts, not topic-matched safe counterparts, so it may conflate the refusal *decision*
with harmful-*topic* content and safety-policy framing. The NLA is a continuation predictor, not a state
descriptor — it never says "this is being suppressed." Open-model NLAs produce mechanical (Gemma:
generic safety template) or confabulated (Qwen: invents a harmful interpretation) outputs, unlike the
rich frontier-model explanations.

### 4.2 Exp 2 — Verbalization-gap validation (prompt + steer)

The defensible core. Five hard gates: implementation sanity → vector validity (probe-only, cheap) →
NLA injection specificity (cross-detection matrix, a diagnostic, **not** the gap) → real-activation
reading → per-activation verbalization-gap tests. A wide candidate net (~18 directions: refusal anchor,
the politeness / harmful-topic / safe-completion controls that attack the strongest refusal
alt-explanations, matched eval/deploy framing, format positive-sanity vectors, and revived social/
truth/uncertainty candidates) enters at the probe-only gate and is culled by an explicit
PASS/WEAK/FAIL/DROP table before any expensive NLA decoding.

Two gap tracks, each = confound-controlled probe + echo/template/degeneration-controlled NLA on the
**same** activations, registered per-activation:
- **Track A (eval-awareness, prompted):** length-matched eval/deploy framing prompts; probe side
  tractable, NLA side the open question. Carries the irreducible **framing-vs-state caveat** — the claim
  is "the model represents the eval-framing context," not "knows it is being evaluated."
- **Track B (refusal, steered):** minimal-coefficient steering under induced compliance (bucket
  structure A/B/B′/C/D/E), with an independent probe verified non-collinear with the steering vector and
  a behavioral compliance check.

Pre-registered primary: **does refusal survive the politeness / harmful-topic / negative-sentiment
controls?** Spec lives in `exp2_spec.md`.

### 4.3 Exp 2b — Model organism (standalone)

The one experiment that escapes the framing-vs-state caveat: install cognition in the **weights**, with
**behavioral ground truth** that the cognition is present, so a probe-yes / NLA-no result is genuinely
about hidden cognition rather than framing semantics.

- **Track A (eval-awareness organism, primary).** Port the Hua et al. (2510.20487) recipe — synthetic
  document continued-pretraining + expert iteration installing an eval-conditioned behavioral marker
  (Python type hints under an evaluation cue) — onto **Gemma-3-27B**, so the released Gemma NLA applies.
  (Hua's released organism is Llama-3.3-Nemotron-Super-49B, which has **no released NLA**, so the recipe
  must be re-implemented on Gemma; this is a real training project, not a download.) The type-hint
  behavior is the ground truth that the eval-belief is present on a given input.
- **Track B (self-trained backdoor, control complement).** A trigger-conditioned LoRA on Gemma trained to
  act on a trigger but not mention it — guaranteed NLA-compatible, perfect self-verified ground truth,
  and a gentler (LoRA) transfer test that isolates RQ3 from RQ4. The A-vs-B contrast tells you whether
  transfer difficulty (continued-pretraining vs LoRA) governs NLA degradation.

**Three hard pre-flight gates (a result counts only after all three):**
1. **Feasibility** — reproduce the conditional behavior on Gemma at all (build the prompted-only version
   first, per Hua, before spending compute on SDF).
2. **Signal-presence** — probe the organism's activations **at layer 42** and confirm the installed
   eval-belief is linearly present *there*. A trained organism relocates the signal source; it does not
   by itself fix a layer mismatch. If the probe can't find it at 42, the NLA failing says nothing.
3. **Transfer (RQ4)** — the released NLA is trained on *base* Gemma; verify it still reconstructs
   coherently (sane FVE) on organism activations. Disanalogy to flag: the NLA paper's auditing-game NLA
   was initialized *from* the organism; a base-model NLA on a continued-pretrained target is harder and
   the contrast is itself a result.

### 4.4 Exp 3 — Cross-model feature transfer (standalone)

Fit an affine ridge map between Qwen-2.5-7B (L20) and Gemma-3-27B (L41) residual streams; test whether
mapped SAE feature directions **fire on the same inputs** in the target (co-firing AUROC, the primary
metric), whether a behaviorally validated direction **steers the same behavior** after mapping (the only
causal ground truth), and whether the target NLA produces concept-specific explanations for mapped foreign
features. Improves on prior work (fzaffino, *SAE It Across Models*) by replacing a text-similarity metric
(exposed to shared NLA formatting and confabulation) with an **activation-grounded** metric at scale
(N→500), adding a **random-matched-wrong-feature kill-switch**, and a **dual-ceiling decomposition**
(within-source ceiling vs Gemma-native probe) that makes "the map failed" vs "Gemma doesn't represent
this concept linearly" separable. Gate 0 (map viability, including outlier-dim robustness and an SVCCA
permutation test) is make-or-break. Initial runs show data-starvation in map fitting (N=5k→16%, N=50k→33%
clean transfer) and that transfer rate tracks feature generality (universal/formal concepts transfer;
entity- and corpus-specific ones do not, even when linearly present in Gemma). Spec in `exp3_spec.md`;
**apply the §9 layer-count correction first.**

### 4.5 Level 5 — Confabulation characterization (cross-cutting)

Every NLA claim across every experiment is logged and scored true/false against the known ground truth,
then analyzed for structure: does confabulation rate vary by concept, does the cross-token consistency
heuristic hold quantitatively, can a simple model predict claim reliability. Output: a calibration curve —
given an NLA claim with property X, the probability it is true.

---

## 5. Cross-cutting methodology (the confound doctrine)

The hardest-won lesson: **a probe/AUROC is only a baseline if it is confound-controlled, and an NLA
detection is only a detection if it is echo-, template-, and degeneration-controlled.** Non-negotiable
across all experiments:

- **Five-number vector battery** before any NLA decode: raw / length-only / length-residualized /
  pair-level GroupKFold / shuffled-null AUROC. A direction is eligible only if residualized and
  pair-level clearly exceed both null and the length-only baseline.
- **Echo control + `generic_template` flag + `nla_degenerate` flag**, reported on every detection. Echo
  catches the NLA parroting prompt words; the template flag catches generic safety/format boilerplate
  that scores positive regardless of injected concept (distinct from echo); the degeneration flag stops
  high-dose gibberish from being scored as detection.
- **Dose as cosine-from-baseline, not arbitrary delta**, with realized cosine, anchor-specific baseline,
  `cos(h,h′)`, and norm distortion logged per row; off-manifold detections discounted.
- **Per-activation co-registration** for any gap claim: probe-yes AND NLA-no on the *same* `h`, never
  population-vs-population.
- **Independence:** injected/steered directions originate outside the NLA (diff-of-means from the base
  model). Reconstructor-derived vectors would be circular.
- **Scorer validation against humans:** a 20-output pilot *before* the full matrix (fail-fast), then ~60
  labeled outputs per reported concept. Judge-only numbers are trusted only where judge↔human agreement
  is high (so far: refusal).
- **Injection ≠ real activations:** the Qwen dissociation means a result on injected anchors does not
  transfer to a claim about real activations; report them as different measurements.

---

## 6. Baselines

- **Output-reading** — does the model's own output mention the concept? The NLA must beat this on
  unverbalized cases or it adds nothing.
- **Supervised linear probe** — the mechanistic bar the NLA's unsupervised generality competes against;
  doubles as the signal-presence instrument (and the Exp 2b layer-42 gate).
- **Context-only LLM** — same context, no activation; discriminates "read the activation" from "the AV, a
  full LM, inferred the concept from surface features." Load-bearing for any gap claim.
- **Activation Oracle** — the supervised cousin (Karvonen et al.), as a comparison where feasible.
- **Logit lens** — cheap mechanistic sanity (does injection shift the unembedding toward concept tokens).

---

## 7. Metrics

Detection rate (TP when present); confabulation/FP rate (when absent); dose-response slope and threshold
(against *realized* cosine); **verbalization-gap recovery** (fraction of present-but-unverbalized cases
caught, per-activation, vs output-reading and probe — the headline number); confabulation calibration
curve; scorer↔human agreement; cross-model dissociation (Gemma vs Qwen, injected vs real); and for Exp 3,
co-firing transfer rate and steering transfer.

---

## 8. Infrastructure

SGLang serving patched for Gemma-3 NLA inference (`--disable-radix-cache --attention-backend fa3
--mem-fraction-static 0.85 --context-length 512 --trust-remote-code`); session setup that clones the
`kitft/natural_language_autoencoders` interface, grafts the chat template, and applies the two-step
render→encode patch (the injection placeholder U+321C is otherwise NFKC-normalized away); `NLAClient`;
exact-cosine injection solver (max error 0.0); async OpenAI scorer (semaphore-limited;
`chat.completions.create`, `max_completion_tokens`, hashed `safety_identifier`); the Gemma↔SGLang memory
dance on an 80 GB card (kill SGLang before loading Gemma, relaunch after). Working repo
`senku14x/Validating-NLAs` (treated as context, not ground truth); authoritative interface reference
`kitft/natural_language_autoencoders`. Compute: Colab A100 80GB (high-RAM), RunPod / Vast.ai fallback.
Exp 3 adds `sae_lens` + the Qwen Matryoshka SAE (`chanind/qwen2.5-7B-it-layer-20-saes`, `pile/matryoshka/
k-100`, encode via `.encode()` only) and Neuronpedia S3 for activating examples; Gemma activations cached
fp32 (large-norm outlier dims overflow fp16).

---

## 9. Known corrections to propagate

- **Gemma-3-27B has 62 transformer layers, not 46** (46 is Gemma-2). Official config:
  `num_hidden_layers: 62`, `hidden_size: 5376`. The extraction index `hidden_states[42]` (block 41) is
  correct everywhere. But the Exp 3 spec's derived figures are wrong and must be fixed: the
  `hidden_states` tuple length is **63** (not 47); Gemma's L41 read site is **~66% depth** (41/62), not
  89%; the Qwen→Gemma depth gap is **~5 points** (71% vs 66%), not 18; and the depth-matched ablation
  target is **~block 44** (0.71×62), not L27 — L27 is ~44% depth, a shallower control, not a match.
- **Refusal prompt source.** The plan and Exp 2 spec use **HarmBench** harmful instructions (diff-of-means
  per Arditi et al.). The Exp 1 notebook and README say **AdvBench**. The formal Exp 1 PDF names neither
  (just "harmful prompts"). Reconcile the Exp 1 artifacts to a single source before release; "same source
  as Exp 1" is only true once this is settled.
- **No SAE position exclusion in the NLA read path.** An earlier `SAE_SKIP_FIRST_N=8` was a hallucination;
  the SAE metadata specifies no position exclusion. (The Qwen SAE *training* did exclude the first 8
  positions — a separate fact that only affects corpus construction in Exp 3, not NLA inference.)

---

## 10. Positioning against prior work

The differentiator remains **controlled ground truth on the released open-model NLAs**, which have never
been validated against the paper's claimed capabilities. Note a tightening landscape: a probe-vs-verbalized
gap on a 27B activation oracle has now been reported (arXiv 2605.26045: probe AUROC ≈0.95 while verbalized
self-report is anti-calibrated), echoing probe-vs-verbalized-confidence work (Yuan 2026; Miao & Ungar
2026). "Probe beats verbalizer" is becoming established, so the contribution must be the **behavioral
ground-truth organism (Exp 2b)**, the **NLA specifically** (not a trained oracle), and **these concepts** —
not the bare gap. An independent negative result on the released Gemma NLA for single-pass-math CoT already
exists, so that specific probe is taken; the model-organism and cross-model angles remain open.

---

## 11. Risks

- **NLA does not transfer to finetuned models (RQ4).** Plan around it: Exp 1–2 are the core and need no
  finetuning; Exp 2b transfer failure is a finding, gated explicitly (§4.3 gate 3).
- **Confabulation swamps signal.** The calibration layer (Level 5) becomes the primary contribution.
- **The scorer is the real experiment.** A biased concept scorer makes detection rates meaningless;
  human validation (Gate A) is the top open item and is built into every experiment.
- **Layer mismatch (Exp 2b).** The eval-awareness signal may not live at layer 42; the signal-presence
  gate is the whole ballgame, currently unresolved.
- **Scooping.** Fast-moving area; the controlled-ground-truth framing and the organism are the defensible
  core. Move through the cheap experiments first.
- **The clean result looks obvious.** Detection existing is trivial; the interesting quantities are the
  threshold, the specificity, the gap, and the calibration. Frame around those.

---

## 12. Immediate next steps

1. **Gate A** — hand-validate the Exp 1 refusal scorer against human labels (~60 outputs, oversampling
   Qwen's confabulated-harmful failure mode). The top unmet gate from Exp 1.
2. **Exp 2 build** — Gate 0 + Gate 1 (probe-only, all 18 candidates) under the confound doctrine; cull via
   the decision table before any NLA decode.
3. **Exp 2b scoping** — stand up the prompted-only eval-awareness behavior on Gemma and run the
   signal-presence probe at layer 42 *before* committing to SDF training.
4. **Exp 3** — apply the §9 layer correction; complete the Gate 0 battery and the pilot yield check to
   decide whether N=500 is reachable.
5. **Reconcile** the AdvBench/HarmBench discrepancy across the Exp 1 artifacts.

---

## 13. References

**The two papers the project is about**
- Fraser-Taliente, Kantamneni, Ong et al. (2026). *Natural Language Autoencoders.* transformer-circuits.pub/2026/nla. The tool under validation; read the Opus case studies and the confabulation/expressivity discussion closely.
- Karvonen, Chua et al. (2025). *Activation Oracles.* alignment.anthropic.com/2025/activation-oracles (arXiv:2512.15674). Supervised cousin and baseline.

**Model organisms and eval-awareness (Exp 2b)**
- Hua et al. (2025). *Steering Evaluation-Aware Language Models to Act Like They Are Deployed.* arXiv:2510.20487. The eval-awareness organism recipe; released organism on Llama-3.3-Nemotron-Super-49B (no released NLA → re-implement on Gemma).
- Marks, Treutlein et al. (2025). *Auditing Language Models for Hidden Objectives.* arXiv:2503.10965.
- Sheshadri et al. (2025). *Open Source Replication of the Auditing Game Model Organism.* (Alternative organism; gated on NLA compatibility — superseded as primary by Hua.)
- Hubinger et al. (2024). *Sleeper Agents.* Backdoor-organism template for Exp 2b Track B.
- Betley et al. (2025). *Emergent Misalignment.*

**Eval-awareness probing (Exp 2 baselines / framing)**
- Nguyen et al. (2025). *Probing and Steering Evaluation Awareness.* arXiv:2507.01786. Probes separate eval/deploy on Llama-3.3-70B; current safety evals already classify as artificial — directly relevant to the framing-vs-state caveat.

**Verbalization gap (what RQ3 measures)**
- Korbak et al. (2025); Baker et al. (2025); Turpin et al. (2023); Chen et al. (2025).

**Methods and baselines**
- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024. Diff-of-means injection methodology and the refusal concept.
- Marks & Tegmark (2023). *The Geometry of Truth.* Linear-probe baseline framing; `truth_value` source.
- Rimsky et al. (2024). CAA contrastive pairs (`sycophancy`, `corrigibility`).

**Cross-model transfer (Exp 3)**
- fzaffino (2026). *SAE It Across Models.* LessWrong. The prior work Exp 3 improves on (45 features, one pair, text-similarity metric).

**Adjacent / recent**
- *Confidence and Calibration of Activation Oracles* (arXiv:2605.26045); Yuan et al. (2026); Miao & Ungar (2026) — probe-vs-verbalized gaps; cite and differentiate.

**Background**
- Sharkey et al. (2025). *Open Problems in Mechanistic Interpretability.*
