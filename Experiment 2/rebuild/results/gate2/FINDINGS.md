# Gate 2 — offline injection results (Gemma-3-27B)

**Status:** offline injection matrix complete + analyzed. Real-activation test (`09`) next.

## Why
The NLA paper claims NLAs surface concepts a model *represents but doesn't verbalize*, but
couldn't score it against ground truth. Gate 2 supplies **constructed** ground truth: inject a
known concept direction (diff-of-means, built outside the NLA) at known strength into neutral
activations and ask whether the released **Gemma AV** verbalizes it (RQ1 detection-validity) and
*only* it (RQ2 specificity), fully confound-controlled. **This is a specificity diagnostic, not the
verbalization gap (RQ3).**

## What we did
Pipeline (`injection.py`, `flags.py`, `05_inject_matrix → 06_decode → 07_score → 08_analyze`), each
self-tested on synthetic ground truth. Diff-of-means directions for 6 concepts + refusal anchor +
random/baseline floors → injected into 30 neutral Gemma activations at 3 absolute realized-cosine
doses → **2250 decodes** through the released AV (temp 1.0, 3 samples; 0 errors, 99.9% complete) →
scored 8 concepts with a strict regex **and** a gpt-5.4-mini judge → four-way controlled
(echo / generic-template / degeneracy) with per-row off-manifold logging (`cos(h,h')`).

## What we found

| concept | result (judge) | trust |
|---|---|---|
| **refusal** | detected **0.78** (med dose), **perfectly specific** (0 off-diagonal), survives four-way | ✅ solid |
| **neg_sentiment** | detected **0.60**, perfectly specific | ✅ solid |
| **corrigibility** | **null** — injection → generic agreeableness/floor, not the concept (on-manifold, judge 0) | ✅ real negative |
| **sycophancy** | judge 0.66 (med) but shaky on the "correct/affirm" axis | ⚠️ tentative |
| **truth_value** | uninterpretable — off-manifold offline **and** judge over-scores it | ⚠️⚠️ |
| harmful_topic_benign, eval_framing | off-manifold offline (cos 0.18 / 0.45) | ❌ invalid offline |

Floors clean: medical_advice 0.000, random 0.000, baseline 0.011. Four-way arms ≈ raw (detections are
not template/echo/degeneracy artifacts).

### Two methodological findings (matter more than the table)
1. **The scorer is the bottleneck (spec §3, now concrete).** The gpt-5.4-mini judge is reliable for
   refusal (the control) but **over-scores truth_value**: 73% of its `truth_value=2` calls are
   "X is correct"/factual-answer floor (rubric says 0), and 38/49 came from *sycophancy* injections.
   So the "sycophancy→truth_value leak" is judge artifact, and soft-concept numbers are gated on a
   **human pilot** before they can be trusted. (Loose keyword scans were even worse — they invented a
   "refusal everywhere" artifact the strict judge erased.)
2. **Offline injection can't fairly test negative-baseline concepts.** truth_value /
   harmful_topic_benign / eval_framing have negative anchor-baseline cosines, so reaching "present"
   forces a large off-manifold push (cos(h,h') → 0.45 / 0.18 / 0.45). Their 0s are a method artifact,
   not the AV. → the on-manifold test is **real activations** (`09_decode_real`), not injection.

### Dose
refusal & sycophancy peak at **MED** dose and drop at high (over-injection toward floor); neg_sentiment
is monotone to high. MED is the cleaner operating point for behavioral/social directions.

## Caveats / not established
- Offline **injected** activations, not real (RQ1/RQ2 only); the **gap (RQ3)** is downstream.
- **Gemma only** — Qwen (and the injection-vs-real dissociation) not yet run.
- Judge **unvalidated** for soft concepts (human pilot pending).

## Next
- `09_decode_real` — decode real `03` concept-present/absent/anchor activations (on-manifold, all
  concepts incl. truth_value). Run Gemma then Qwen.
- Judge **human pilot** (validate before trusting soft-concept numbers).
- Online-steered-generation for refusal (behavioral on-manifold) — later.
