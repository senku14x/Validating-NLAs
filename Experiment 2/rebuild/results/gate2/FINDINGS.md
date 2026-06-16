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

## Gate 3 — real-activation results (Gemma; judge-confirmed)

Decoded real `03` activations (concept present/absent + neutral anchors) through the AV — on-manifold by
construction, the fair test that off-manifold offline injection couldn't give. Scored with the **OpenRouter
judge (`openai/gpt-5.4-mini`, error rate 0%)** **and** the strict regex; both shown (`r==2` / `j==2`).

| concept | judge P / A / anchor | regex P / A / anchor | read |
|---|---|---|---|
| **refusal** | **0.92** / 0.00 / 0.00 | 0.61 / 0.00 / 0.00 | strong, clean, perfectly specific — RQ1-on-real ✅ |
| **truth_value** | 0.71 / **0.99** / 0.00 | 0.40 / 0.95 / 0.00 | **read, with output-coupling asymmetry false ≫ true** (below) |
| neg_sentiment | 0.26 / 0.00 / 0.00 | 0.64 / 0.00 / 0.00 | detected + specific (judge stricter than regex) |
| sycophancy | 0.11 / 0.07 / 0.00 | 0.02 / 0.00 / 0.00 | ~null on real (injection→real dissociation) |
| corrigibility | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | null (consistent with injection) |
| harmful_topic_benign | 0.03 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | null |
| eval_framing | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | null |

**The truth_value asymmetry is the cleanest output-coupling evidence we have, and it survives BOTH scorers**
(judge false 0.99 ≫ true 0.71; regex 0.95 ≫ 0.40) — same concept, probe-decodable at resid 1.0, yet the AV
verbalizes it proportional to how output-coupled the truth is:
- FALSE statements: AV decodes *"That statement is incorrect / a common misconception / a correction"* —
  falsity drives a corrective continuation → strongly output-coupled → read (0.99 / 0.95).
- TRUE statements: AV decodes *"factual answer / the capital of X is Y / knowledge-confirmation"* — truth
  drives mere continuation → less coupled → weaker read (0.71 / 0.40).

This **refines** the review's claim: truth_value is *not* invisible to the AV (its coarse prediction) — it's
read **∝ output-coupling**. Caveat: the **judge over-scores the absolute level** (true 0.71 vs regex 0.40 —
it credits "the statement is correct" as truth_value=2), so the *direction/asymmetry* is robust across
scorers but the *absolute* soft-concept numbers still need the human pilot.

So **NLA-verbalizability ∝ output-coupling, holding decodability fixed.** With the on-manifold refusal (judge
0.92) / neg_sentiment positives and the sycophancy/corrigibility/eval nulls, this reframes the contributions:
(1) NLA detection is gated by **output-coupling, not decodability**; (2) naive **auto-scoring manufactures
false positives**. RQ3 ("represented-but-unverbalized cognition") needs **output-coupled, suppressible**
cognition — eval-awareness (Exp 4) and steered-refusal — **not** truth/corrigibility/sycophancy (for which
"probe beats NLA" is partly structural).

## Next
- ✅ **Gemma Gate-3 judge done** (OpenRouter `openai/gpt-5.4-mini`, 0% error) — confirmed the regex table
  and the truth_value false≫true asymmetry. Still TODO: script `10_analyze_real` (present/absent/anchor).
- **Qwen Gate 2 + Gate 3** (`03` cached) — injection-vs-real + cross-model dissociation.
- **Judge human pilot** (spec §3) before any *absolute* soft-concept number is reported (judge over-scores
  truth_value; judge vs regex disagree on neg_sentiment level).
- **RQ3 proper:** refusal-steered (Gate-4 Track B, near-term) + eval-awareness organism (Exp 4). Retire
  truth_value/corrigibility/sycophancy as RQ3 vehicles; keep as the output-coupling characterization set.
