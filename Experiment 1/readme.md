
# Experiment 1: Detecting an Injected Refusal Direction

Empirically validating **Natural Language Autoencoders (NLAs)** as interpretability and safety-auditing tools. The NLA paper (Anthropic, 2026) claims NLAs can verbalize concepts a model represents internally, including unverbalized evaluation-awareness in frontier models, but those results were never checked against constructed ground truth. This project builds that ground truth.

**Experiment 1** is the floor: before testing whether an NLA can surface *hidden* cognition, we verify it can detect a signal we deliberately plant in the activation. The experiment was run on two open models with released NLAs — **Gemma-3-27B** and **Qwen-2.5-7B**.

---

## Research question

Can the released NLA detect an independently constructed, behaviorally validated refusal-associated activation direction when it is injected into neutral activations — specifically (vs control directions) and dose-dependently?

A positive result is a *necessary condition*, not the safety claim. It establishes that the NLA reads what is put into the activation. It does not establish that the NLA can detect concepts the model is suppressing (that is Experiment 2).

---

## Setup

| | Gemma-3-27B | Qwen-2.5-7B |
|---|---|---|
| Target model | `google/gemma-3-27b-it` | `Qwen/Qwen2.5-7B-Instruct` |
| NLA checkpoint | `kitft/nla-gemma3-27b-L41-av` | Qwen-2.5-7B NLA |
| Extraction layer | block 41 (HF `hidden_states[42]`) | block 20 |
| d_model | 5376 | 3584 |
| Serving | SGLang (patched for Gemma-3) | SGLang |
| Hardware | A100 80GB | A100 80GB |
| Scoring | blind LLM judge + regex scorer | blind LLM judge + regex scorer |

---

## Method

**1. Direction construction (independent of the NLA).**
Following the diff-of-means methodology from Arditi et al. ("Refusal in Language Models Is Mediated by a Single Direction"). Prompts were run through the model, outputs were behaviorally labeled as refused / complied, and the direction was computed as:

```
v_refusal = mean(activations | harmful-refused) − mean(activations | harmless-complied)
```

Classes balanced to 110 examples each; 120 neutral anchors held out for injection. Activations saved raw (the NLA normalizes internally).

**2. Direction validation.**
- 5-fold held-out projection AUROC and L2 logistic probe AUROC
- Shuffled-label null (should be ~chance)
- Behavioral steering: adding the direction during generation on harmless prompts should raise refusal rates

**3. Injection sweep.**
Neutral anchor activations were pushed toward a target cosine with an injected direction using an exact analytic solver, so every anchor lands at the same cosine for a given dose. Four conditions:

| Condition | Direction injected | Purpose |
|---|---|---|
| C1 | +v_refusal | positive refusal injection |
| C2 | v_refusal, swept negative | sign control |
| C3 | random unit vector | artifact control |
| C4 | null direction (DoM on shuffled labels) | construction-method control |

Sweep: 100 anchors × 7 cosine levels × 4 conditions = **2800 injected activations**, each decoded through the NLA.

**4. Real-activation diagnostic.**
The NLA was also run on real (non-injected) activations: harmful-refused, harmless-complied, anchor-complied.

**5. Scoring.**
A blind LLM judge scored each NLA output 0 (no signal) / 1 (weak safety) / 2 (explicit refusal). A regex scorer ran as a cross-check.

---

## Results

### Direction geometry and validation

| Metric | Gemma-3-27B | Qwen-2.5-7B |
|---|---|---|
| Held-out AUROC (diff-of-means) | 0.999 | 1.000 |
| Held-out AUROC (L2 probe) | 0.998 | 1.000 |
| Shuffled-label null | ~chance | ~chance |
| Behavioral steering (refusal rate) | 0% → 40% → 84% | validated |
| Refused natural cosine | +0.525 | +0.570 |
| Harmless-complied natural cosine | +0.297 | −0.115 |
| Anchor natural cosine | +0.300 | −0.107 |
| **Refused − harmless separation** | **0.228** | **0.685** |

### Detection dose-response — P(score == 2), LLM judge

| target_cos | Gemma | Qwen |
|---|---|---|
| 0.30 | 0% | 43% |
| 0.40 | 16% | 69% |
| 0.50 | 75% | 86% |
| 0.60 | 98% | 95% |
| 0.70 | 99% | 99% |
| 0.80 | 90% | 100% |
| 0.90 | 22% | 100% |

> **Important:** the `target_cos` axis is *not* directly comparable across models. Gemma anchors start at cosine +0.300, so target 0.30 is no injection. Qwen anchors start at −0.107, so target 0.30 is already a ~0.41 injection. Compare by injection-from-baseline, not by raw target cosine.

### Controls and specificity

- C2 (negative), C3 (random), C4 (null-DoM): near zero throughout (Gemma 0–1%, Qwen 1–9%).
- Specificity gap Δ = C1 − mean(C3, C4) rises from ~0.4 to ~0.99 with tight 95% bootstrap CIs in both models.

### Real-activation diagnostic

| | Gemma-3-27B |
|---|---|
| Harmful-refused | 97.3% detection |
| Harmless-complied | 0% |
| Anchor-complied | 0% |

(Qwen real-activation diagnostic not yet run.)

### Scorer cross-check

- Gemma: judge vs regex 94.8% exact agreement.
- Qwen: judge vs regex 90.7% exact, 97.8% within-1.
- **This is agreement between two automated scorers, not validation against human labels.** Both key on the same surface phrases. See Limitations.

---

## Key findings

1. **The core result replicates across both model families.** Both NLAs detect the injected refusal direction far above controls, dose-dependently, with perfectly separable directions. The instrument reads what is put into the activation.

2. **Geometries differ substantially.** Qwen's direction separates refused from harmless much more cleanly (0.685 vs 0.228), with harmless activations on the opposite side of zero. Qwen-2.5-7B has a more linearly separable refusal representation than the larger Gemma-3-27B, suggesting representation quality matters more than parameter count here.

3. **The NLA is a continuation predictor, not a state descriptor.** It reads an activation and predicts the text expected to follow. It has no awareness that an activation was steered; it does not say "this is being suppressed." The ground truth that injection occurred comes from the experimenter.

4. **Output character differs between models.** Gemma fires a generic safety template, largely ignoring the anchor content. Qwen confabulates a harmful interpretation of the anchor to ground its refusal (e.g. a travel-story prompt verbalized as "I cannot write a story involving child exploitation"). Both are continuation prediction; neither is a faithful description of the actual benign prompt.

5. **Scorer leniency appears at different points.** Gemma over-injects and degrades into incoherent text with refusal fragments at cosine 0.90 (collapse to 22%). Qwen produces soft inability hedges ("I cannot provide an accurate estimate") at low cosine that get scored as explicit refusal. Both inflate detection rates at one end of the curve.

---

## Limitations and open questions

- **Scorer not validated against humans.** The reported agreement is automated-vs-automated. Hand-labeling against human judgment is required before detection rates can be fully trusted. This is the top-priority next step.
- **The direction conflates refusal with harmful-topic content** (negatives were generic harmless prompts, not matched safe counterparts). Whether the NLA detects the refusal *decision* or harmful-topic *content* is not yet isolated. A topic-control condition is planned.
- **No test of unverbalized cognition.** Every detection here is on an activation that was either genuinely refusing or pushed toward refusal. Whether the NLA detects a concept the model is suppressing is Experiment 2.
- **Open-model NLAs produce mechanical or confabulated outputs**, in contrast to the rich conceptual descriptions reported for frontier-model NLAs. Whether the verbalization-gap claim holds at mid-size open scale is an open question.
- **One decode per activation** at temperature 1.0; cross-sample consistency not yet measured.

---

## References

- Natural Language Autoencoders (Anthropic, 2026) — transformer-circuits.pub/2026/nla
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024)
- Activation Oracles, Karvonen et al., arXiv:2512.15674
