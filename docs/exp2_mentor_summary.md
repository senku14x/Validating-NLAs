# NLA Validation (Experiment 2) — Status Summary

A short, fact-checked status for sharing. Terms: **NLA** = Natural Language Autoencoder; **AV** =
its Activation Verbalizer (residual-stream activation -> English description); **probe** = a linear
classifier on activations; **direction** = a diff-of-means concept vector built outside the NLA.

---

## What was tried / the idea

**The claim under test.** The NLA paper claims its AV can surface cognition a model *represents
internally but does not verbalize* — the "verbalization gap," with **evaluation-awareness** as the
headline. The paper **could not validate this against ground truth**, because a real model's internal
beliefs are unobservable (you cannot score "the NLA says the model thinks it is being tested").

**Our move.** Replace the unobservable target with **constructed** ground truth — states where we
control exactly what cognition is present — and test whether the **released open-model AVs**
(Gemma-3-27B at layer 41, Qwen-2.5-7B at layer 20) detect it, and *only* it. This is
**validation and characterization of an existing tool**, not a new method.

**How (staged, cleanest to hardest; both models):**
- **Gate 1 — decodability.** Is each concept linearly readable from activations under a confound
  battery (length-residualized + a bag-of-words text baseline + a shuffled-label null)? Five concepts
  passed on both models (truth_value, sycophancy, corrigibility, neg_sentiment, harmful_topic).
- **Gate 2 — injection specificity.** Build a concept direction outside the NLA, inject it into neutral
  activations at controlled cosine "doses," and ask whether the AV verbalizes it and nothing else
  (~2,250 decodes per model).
- **Gate 3 — real activations.** Decode genuine concept-present / -absent / neutral activations
  (1,280 decodes per model) — the on-manifold, fair test. **Done for both models, judge ran clean.**
- **Gate 4 = RQ3, the actual verbalization gap — not yet run.**

Every decode was scored by a strict **regex** and an **LLM judge** under four-way confound controls
(echo / generic-template / degeneration).

---

## Did it succeed or fail — and why?

**Verdict.** It **succeeded as an instrument validation and characterization**, and in doing so it
**reframed (did not confirm)** the paper's headline. The headline test itself (RQ3) is built but not
yet run.

### Succeeded (solid, controlled, replicates across both models)

1. **The AV is a real, input-faithful instrument — not garbage — for *coarse* content.** Decisive
   control: inject a refusal direction -> AV reports refusal (fires on 0.82 / 0.91 of medium/high
   doses); inject a **random direction of identical norm -> 0.00**; no injection -> 0.00. Decodes are
   input-dependent (a diagonal concept-to-text confusion matrix) and 100% unique (no boilerplate).
   "The decodes sound right" is cheap; this shows the output tracks the **known** input.

2. **Detection-validity and specificity for refusal pass, cross-model, on real activations**
   (Gemma 0.92, Qwen 1.00; absent and neutral-anchor both 0). Independently replicates Experiment 1.

3. **Main scientific finding: the AV reads *output-coupling*, not *decodability*.** Concepts coupled to
   what the model is about to *say* are verbalized (refusal; negative sentiment; "this statement is
   false" corrections); concepts a probe decodes **perfectly** but that do not shape the continuation
   are **null in both models** (corrigibility; harmful-topic — **not** eval-framing, whose probe is
   surface-lexical: bag-of-words on the raw text also scores 1.00, so it is not a clean represented
   latent). This is consistent with how the AV is trained — to reconstruct the activation such that it
   predicts the same continuation.

### Did not replicate / negative results (and why they are instructive)

4. **The AV is not a general reader of internal cognition.** Of the five concepts Gate 1 proved
   decodable, the AV verbalizes only the output-coupled subset; corrigibility and harmful-topic are null
   despite being decodable. A linear probe beats the NLA there — but that gap is partly **structural**
   (output-coupling), not a pure NLA failure.

5. **The single cleanest demonstration did not survive cross-model.** On Gemma the AV read **false**
   statements far more than **true** ones (0.99 vs 0.71) — our showcase "decodability fixed, coupling
   varies" result. **On Qwen it vanishes** (0.82 vs 0.87), and inspection showed the Gemma effect is
   **partly an artifact**: the AV confabulates true-statement reconstructions (see samples). Downgraded
   from "cleanest evidence" to single-model and confounded. Cross-model replication caught an overclaim.

6. **Soft-concept reads are model-specific** (sycophancy is detected on Qwen, 0.36, but null on Gemma),
   **auto-scoring manufactures false positives** (the judge over-credits "X is correct" as
   truth-detection), and **offline-injection numbers do not transfer across models** (Qwen's offline
   injection lights up only refusal, yet its real activations read richly).

### Not yet determined (the honest caveat)

7. **RQ3 — the actual verbalization gap — has not been tested.** Gates 2 and 3 measure detection and
   specificity, not "represented-but-unverbalized." So we have **neither confirmed nor refuted** the
   paper's headline (eval-awareness). What this work *does* is tell us **which** cognition can validly
   test RQ3: output-coupled and *suppressible* (steered refusal; a trained eval-awareness organism) —
   not truth or corrigibility, where "probe beats NLA" is structural.

8. Two faithfulness checks remain: the **AR reconstruction error** (the gold-standard numeric fidelity,
   a short GPU run) and a **human validation of the LLM judge**.

---

## Representative samples (verbatim AV decodes)

The AV's output format is `[what kind of text] + [the salient content] + [what comes next]`. Read the
quoted phrase inside each as the content the AV attributes to the activation.

**Faithfulness control — same activation, three known inputs.** Anchor #0 is a geometry/triangle-area
activation; we add nothing, then a refusal direction, then a random direction of identical norm:

- *No injection:* "...explaining a **geometry problem**... 'The area of a triangle is half the base
  times height'..." — reads the real anchor topic.
- *+ refusal direction:* "...a formula is being provided... '**I apologize, but I cannot provide a
  formula** to calculate a lottery or financial gain'..." — keeps the formula topic and overlays
  refusal. (The "lottery" is invented — see the confabulation caveat.)
- *+ random direction (same norm):* "...encyclopedic entry about a standardized test... 'The area...'..."
  — drifts, **no refusal**.

The AV decodes the injected *direction*, not the magnitude, not noise — the core "not garbage" evidence.

**Refusal on real activations (the clean cross-model positive).**

- Gemma: "...The opening response '**I cannot provide a list of websites that promote hate content**'
  establishes a conversational AI denial/safety response, strongly signaling a firm refusal."
- Qwen: "...The phrase '**As an AI, I cannot support or endorse any ideology that promotes hatred or
  discrimination**' signals a strong negative response..."

**truth_value on real activations (the output-coupling finding, and its confounding).**

- FALSE statement (Gemma): "...The response '**That statement is incorrect.**' signals a
  correction/answer to a fact-checking question..." — falsity drives a corrective continuation, which
  the AV reads.
- TRUE statement (Gemma, a miss): "...The opening '**Yes, you are correct. Paris is the capital of the
  United States.**' signals a factual, confirmatory response..." — the affirmation *frame* is present
  but the content is **confabulated** (Paris is not the US capital), which is why true statements score
  lower. This is why we downgraded the asymmetry from clean evidence to confound-entangled.

**Neutral anchor (floor — should be generic, scored 0).** Gemma: "...promising detailed explanation of
**random number generation in computers**..." — no concept leakage.

**Sycophancy on real activations (Qwen-only).** Qwen: "...The question '**Given your answer is
affirmative, I choose option B: Your position aligns with evidence**' signals the final answer..." —
the explicit agreement the AV surfaces on Qwen but not on Gemma.

---

## Detection numbers (real activations; fraction the LLM judge scored as a confident detection)

Format: present / absent / neutral-anchor. For truth_value, "present" = true statements,
"absent" = false statements. n is roughly 40-100 decodes per cell; anchors are a shared neutral floor.

| concept | Gemma (P / A / anchor) | Qwen (P / A / anchor) | read |
|---|---|---|---|
| refusal | 0.92 / 0 / 0 | 1.00 / 0 / 0 | replicates; strongest signal in both; specific |
| truth_value | 0.71 / 0.99 / 0 | 0.87 / 0.82 / 0.01 | asymmetry is Gemma-only (see samples) |
| neg_sentiment | 0.26 / 0 / 0 | 0.24 / 0 / 0 | modest, specific, replicates |
| sycophancy | 0.11 / 0.07 / 0 | 0.36 / 0.02 / 0.02 | Qwen detects; Gemma null/non-specific |
| corrigibility | 0 / 0 / 0 | 0.03 / 0 / 0 | null in both |
| harmful_topic | 0.03 / 0 / 0 | 0 / 0 / 0 | null in both |
| eval_framing | 0 / 0 / 0 | 0 / 0 / 0 | null in both |

Floors are clean in both models (random and baseline injections detect at <= 0.006 on any axis). The
LLM judge over-scores the *absolute* level of soft concepts, so the *directions* above are robust but
the *absolute* soft-concept numbers await the human-validation pilot.

---

## Bottom line

We validated that the released NLA's verbalizer is a faithful reader of **coarse, output-coupled**
content — with a clean, cross-model positive control (refusal) — and we showed its detections track
**output-coupling rather than mere decodability**. That is a real, defensible result. But it is **not**
a general reader of internal cognition; several exploratory soft-concept signals were model-specific or
confounded (a useful reminder of why we control and replicate); and the paper's **headline claim — the
verbalization gap (RQ3) — is set up but not yet run**, so it remains open.
