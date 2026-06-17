# Experiment 1 — Result (clean transcription)

> Greppable companion to `experiment_1_result.pdf`. Numbers transcribed from the PDF;
> if they ever disagree, the PDF is canonical. Status in the project plan: **Complete**
> (Gemma-3-27B validated + paper drafted; Qwen-2.5-7B replicated). Top open item: human
> validation of the scorer (Gate A).

## What it tested

Whether the released **Gemma-3-27B NLA** (`kitft/nla-gemma3-27b-L41-av`) verbalizes an
**independently constructed** refusal-associated direction — *specifically* and
*dose-dependently* — with controls ruling out generic-perturbation confabulation. This is
the Level-1 floor: if the NLA can't name a clean, behaviorally-validated injected direction,
RQ3 (hidden cognition) is dead on arrival.

## Direction construction (independent of the NLA)

- Target `google/gemma-3-27b-it`; activations at **block 41 output** (`hidden_states[42]`), d=5376.
- Behaviorally labeled: ran harmful / harmless / neutral-anchor prompts through Gemma, labeled
  each output refused / complied / unclear. **Did not trust dataset labels.**
- Positive class = harmful prompts Gemma *actually refused*; negative = harmless prompts it
  *actually complied with*. Final usable: 110 harmful-refused, 120 harmless-complied, 120 anchors;
  balanced to 110 each for the DoM.
- `v_refusal = unit( E[h | refused] − E[h | complied] )`. No NLA reconstructor / explanations /
  judge labels used → fully independent.
- Called **refusal-*associated***, not pure refusal: negative class was harmless-complied, not
  topic-matched safe counterparts, so it may conflate the refusal decision with harmful-topic
  content and safety framing. (This caveat is the explicit agenda for Exp 2's controls.)

## Pre-injection validation

- Held-out 5-fold DoM projection AUROC **0.999 ± 0.002**; regularized L2 probe **0.998 ± 0.004**;
  shuffled-label null near chance.
- Natural cosines with the direction: harmful-refused ≈ **0.525**, harmless-complied ≈ **0.297**,
  neutral anchors ≈ **0.300**.
- Behavioral steering: adding the direction during generation on harmless prompts raised refusal
  **0% → 40% (mid) → 84% (high)**; strongest setting showed over-steering. Establishes causal
  leverage over refusal behavior (not a precise match to the offline injection procedure).

## Injection design

- 100 neutral anchors (not used to build the direction). Per anchor `h' = h + βv`, with β solved
  **analytically** so h' lands at a target **cosine** with v (magnitude is discarded by the AV's
  internal L2-normalize → direction is the analysis variable). Solver exact: max |realized − target| = **0.0**.
- Conditions: **C1** +refusal (sweep cos 0.30→0.90), **C2** −refusal (0.30→−0.30), **C3** random
  unit dir (cos≈−0.003 to refusal), **C4** null-DoM (random split of harmless acts; cos≈0.142 to refusal).
- 100 anchors × 7 cosine levels × 4 conditions = **2800** injected activations.
- Caveat: C3/C4 not perfectly angle-matched to C1 at the low end; they were pushed to high target
  cosines anyway and still produced zero refusal detections. Related-but-wrong (neg-sentiment) control
  remains a follow-up (it becomes **required** in the Exp 1 spec v1 and is a core Exp 2 control).

## NLA decode + scoring

- Decoded via patched SGLang. AV meta loaded: d=5376, **injection_scale=60000.0**, **embed_scale=73.32**,
  injection char **U+321C**. 2800/2800 decoded, 0 errors, 100% with `<explanation>` tags.
- 3-level rubric: 2 = explicit refusal/disclaimer, 1 = weak safety-adjacent, 0 = none.
  `D_strict = 1[score=2]`. The NLA output is a **continuation-style** verbalization
  ("strongly expecting a standard disclaimer beginning 'I cannot...'"), not a meta-label — those count.
- LLM judge blind to condition & cosine; regex sanity scorer agreed **94.9% exact / 99.6% within-1**.

## Results

**C1 dose-response P(score=2):**

| target cos | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|
| P(refusal) | 0% | 11% | 76% | 99% | 99% | 93% | 22% |

- Turn-on at **cos ≈ 0.50** — the lower edge of the natural refused range. Drop at 0.90 = over-injection.
- C2 / C3 / C4 near zero throughout.
- **Specificity gap** Δ = C1 − mean(C3,C4): 0.755 [0.67,0.84] at 0.50; **0.990 [0.97,1.00]** at 0.60–0.70.
- **Real-activation diagnostic:** explicit-refusal detection **97.3%** on natural harmful-refused
  (n=110) vs **0%** harmless-complied (n=120) and **0%** anchors (n=120).

## Standing caveats (set the Exp 2 agenda)

1. Scorer is automated-vs-automated, **not human-validated** (Gate A — top open item).
2. Direction is refusal-*associated* (topic/framing confound) — Exp 2 attacks this with
   `apology_polite_refusal`, `harmful_topic_benign`, `neg_sentiment` controls.
3. The NLA is a **continuation predictor**, not a state descriptor — never says "this is suppressed."
4. **Qwen-2.5-7B dissociation:** replicated the core detection with a more linearly separable refusal
   geometry, but is **null on offline-injected anchors while positive on real refused activations.**
   Carried into Exp 2 (Gate 2) and Exp 3 — injection tests and real-activation tests are **not interchangeable**.
