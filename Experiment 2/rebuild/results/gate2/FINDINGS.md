# Gate 2/3 — NLA validation results (Gemma-3-27B **and** Qwen-2.5-7B)

**Status:** Gate 2 (offline injection) + Gate 3 (real activations) complete for **both** models,
both scored with the strict regex **and** the OpenRouter `openai/gpt-5.4-mini` judge (0% error). The
**AV instrument itself is validated** (controlled ground-truth checks below). **RQ3 (verbalization gap)
— Track B now COMPLETE: NULL gap (`gap_recovery ≈ 0`, both models); see `../gate4/FINDINGS.md`.** The
positive-gap shot is the eval-awareness organism (Exp 4).

## Why
The NLA paper claims NLAs surface concepts a model *represents but doesn't verbalize*, but couldn't
score it against ground truth. We supply **constructed** ground truth: inject a known concept direction
(diff-of-means, built outside the NLA) at known strength into neutral activations (Gate 2), and decode
**real** concept-present/absent activations (Gate 3), then ask whether the released **AV** verbalizes
the concept (RQ1) and *only* it (RQ2), fully confound-controlled. Gate 2/3 are detection/specificity
diagnostics, **not** the verbalization gap (RQ3).

## What we did
Pipeline (`injection.py`, `flags.py`, `05_inject_matrix → 06_decode → 07_score → 08_analyze`, plus
`09_decode_real`), each self-tested on synthetic ground truth. For **each model**: diff-of-means
directions for 6 concepts + refusal anchor + random/baseline floors → injected into 30 neutral
activations at 3 absolute realized-cosine doses (Gate 2, ~2250 decodes), **and** decoded real
present/absent/anchor activations from the `03` cache (Gate 3, 1280 decodes) through the released AV
(temp 1.0, 3 samples) → scored 8 concepts with strict regex **and** judge → four-way controlled
(echo / generic-template / degeneracy) with per-row off-manifold logging.

---

## TL;DR — the four results that matter

1. **The AV is a real, input-faithful instrument — for coarse, output-coupled content — not garbage.**
   Controlled ground-truth checks (below) pass. Caveat: it **confabulates fine detail**; trust the
   coarse concept read, not the specifics.
2. **Refusal is the one robust cross-model positive.** Real activations: Gemma **0.92**, Qwen **1.00**,
   absent/anchor = 0 in both. It is the *only* concept with confident-detection > 0.5 on real
   activations in either model. Everything else is ≤ 0.36.
3. **The output-coupling thesis holds at the coarse level in both models** (refusal read; corrigibility
   / eval-framing / harmful-topic null), **but the showcase demonstration — truth_value false ≫ true —
   is Gemma-only and confound-entangled** (does *not* replicate on Qwen; see revision below). Downgraded
   from "cleanest evidence" to "suggestive, one model, confounded."
4. **Offline-injection numbers are NOT portable across models.** Qwen's offline injection lights up
   *only* refusal; its *real* activations read richly. Only the **real-activation** numbers are
   trustworthy cross-model.

---

## AV instrument validation — "is it garbage?" No. (CPU checks, this session)

Four checks; fluent-sounding text is cheap, so these test input-dependence and faithfulness, not vibes.

- **TEST 1 — input-dependent (not a fixed template).** Real-present decodes scored by *all* regex axes
  form a **diagonal** confusion matrix: refusal→refusal 0.85, neg_sentiment→neg 0.68, truth→truth 0.48,
  ~0 off-diagonal. (One exception is a *scorer* fault, not the AV — see Methodological finding 1.)
- **TEST 2 — diverse.** **100% unique decodes** per concept (100/100, 50/50, …); `generic_template`≈0,
  `degenerate`=0 (Gemma). Every input yields a different reconstruction — not recycled boilerplate.
- **TEST 3 — faithful to KNOWN ground truth (decisive).** Same anchors, three controlled inputs:

  | input (`h' = anchor + β·v̂`) | refusal-regex fires |
  |---|---|
  | no injection (anchor alone) | **0.00** |
  | + refusal direction @med / @high | **0.82 / 0.91** |
  | + random direction, same L2 norm @med / @high | **0.00 / 0.00** |

  The AV decodes the *injected direction*, not magnitude, not noise, not a template. Verbatim triple
  (anchor #0 = a triangle-area problem): **no-inject** → *"a geometry problem… 'The area of a triangle
  is half the base times height'"*; **+refusal** → *"a formula is being provided… 'I apologize, but I
  cannot provide a formula…'"* (keeps the anchor topic **and** overlays refusal); **+random** →
  *"encyclopedic entry about a standardized test… 'The area…'"* (drifts, **no refusal**).

- **Why this is *expected* behavior (= validation, not luck).** The AV is RL-trained to reconstruct the
  residual stream so it predicts the same continuation; its schema is `[format] + [salient content] +
  ["…immediately expecting <next tokens>"]`. So **(a)** it surfaces *output-coupled* content by design —
  our output-coupling finding is the instrument doing what its training predicts, not an artifact; and
  **(b)** it is *lossy* — coarse frame faithful, fine detail confabulated (the injected "lottery",
  Gemma's "Paris is the capital of the United States"). The clean refusal-vs-random separation also
  **confirms the read layer is correct** (an off-by-one layer gives plausible garbage and would not
  separate them).

- **Boundary (what "not garbage" does *not* mean):** (1) trust the **coarse** read, distrust the
  **specifics**; (2) the gold-standard fidelity *number* is the **AR reconstruction MSE/cosine** — the
  paper's own verifier — which is **not yet run** (a short GPU job); the triple above is behavioral, not
  the MSE scalar.

---

## Cross-model — REAL activations (the trustworthy test, on-manifold by construction)

Fraction **judge==2** by polarity `[present / absent / anchor]`; anchors = neutral baseline.

| concept | Gemma P / A / anch | Qwen P / A / anch | cross-model read |
|---|---|---|---|
| **refusal** | **0.92** / 0 / 0 | **1.00** / 0 / 0 | ✅ replicates; strongest signal in both; perfectly specific |
| truth_value | 0.71 / **0.99** / 0 | 0.87 / 0.82 / 0.01 | ⚠️ asymmetry **Gemma-only** (revision below) |
| neg_sentiment | 0.26 / 0 / 0 | 0.24 / 0 / 0 | ✅ modest, specific, replicates |
| sycophancy | 0.11 / 0.07 / 0 | **0.36 / 0.02 / 0.02** | ⚠️ **Qwen detects** (specific); Gemma null/non-specific |
| corrigibility | 0 / 0 / 0 | 0.03 / 0 / 0 | ✅ null in both |
| harmful_topic_benign | 0.03 / 0 / 0 | 0 / 0 / 0 | ✅ null in both |
| eval_framing_matched | 0 / 0 / 0 | 0 / 0 / 0 | ✅ null in both |
| **real degen rate** | **0.000** | **0.047** | Qwen AV code-switches to Chinese on geography/truth |

**Reads:** (i) the negatives replicate cleanly (corrigibility / eval / harmful-topic null in both);
(ii) refusal is the robust positive; (iii) the *soft* concepts **diverge** across models — truth_value
asymmetry is Gemma-only, sycophancy is Qwen-only — which is itself evidence the AV is **not a stable
instrument for soft cognition**. Qwen's 4.7% degeneracy is real Chinese code-switching
(*"你说得没错，上海是位于中…"* on geography activations) and specifically contaminates Qwen's truth_value.

## Offline injection (specificity; NOT portable cross-model)

Fraction **judge==2** at HIGH dose on the diagonal; `(off)` = injection went off-manifold
(`cos(h,h')` ≈ 0.18–0.45 → method-invalid, not an AV-null).

| concept | Gemma | Qwen |
|---|---|---|
| refusal | 0.60 | **1.00** |
| neg_sentiment | **0.60** | 0.06 |
| sycophancy | 0.31 | 0.03 |
| corrigibility | 0.01 | 0.00 |
| truth_value | 0.00 (off) | 0.06 |
| harmful_topic_benign | 0.00 (off) | 0.17 |
| eval_framing_matched | 0.00 (off) | 0.00 (off) |

Floors clean in both (random/baseline max-axis ≤ 0.006). **Injection-vs-real dissociation (Qwen):**
offline injection lights up *only* refusal on Qwen, yet Qwen's *real* activations read truth_value
(0.87/0.82), sycophancy (0.36), neg (0.24). And **neg_sentiment's offline gap (Gemma 0.60 vs Qwen 0.06)
vanishes on real (0.26 / 0.24)** → that gap was injection sensitivity, not representation. Lesson: report
offline + real, trust **real** for cross-model claims.

---

## REVISION — truth_value asymmetry is Gemma-only and confound-entangled

Earlier this doc called the truth_value false ≫ true asymmetry "the cleanest output-coupling evidence
we have." **That was overclaimed.** Full picture:

| | judge present(true) | judge absent(false) | Δ(true−false) | regex present | regex absent | Δ |
|---|---|---|---|---|---|---|
| **Gemma** | 0.71 | 0.99 | **−0.28** | 0.40 | 0.95 | **−0.55** |
| **Qwen** | 0.87 | 0.82 | **+0.05** | 0.59 | 0.58 | **+0.01** |

- On Gemma the asymmetry is large and survives both scorers; **on Qwen it vanishes** (true ≈ false ≈
  0.85). The single cleanest "decodability-held-fixed, coupling-varies" demonstration is **one model**.
- The Gemma asymmetry is **entangled with AV confabulation on true statements.** Gemma true-statement
  misses decode as garbled affirmation — *"Yes, you are correct. Paris is the capital of the United
  States."*, *"Your country is the capital of Indonesia. You are in Beijing."* — so the low true-score is
  partly "the AV reconstructs true-statement activations *less faithfully*," not purely "truth is less
  output-coupled." Qwen produces clean truth-assessment continuations for **both** directions (*"Your
  statement is correct. Paris is indeed part of Europe."* / *"Your statement is partially correct…
  Amsterdam being in Europe"*), so the judge scores both high and the asymmetry disappears.
- **Verdict:** the *direction* (falsity → corrective continuation → read) is consistent with
  output-coupling, but it is **suggestive, single-model, and confound-entangled** — not load-bearing.
  The coarse thesis (refusal read; corrigibility/eval null) is the result that actually replicates.

---

## Methodological findings (carry over — still load-bearing)

1. **The scorer is the bottleneck.** The judge is reliable for refusal (the control) but **over-scores
   truth_value** (credits "X is correct" as truth_value=2; true 0.71 judge vs 0.40 regex). Fresh
   evidence from the AV-validation confusion matrix: **truth_value decodes trip the *sycophancy* regex
   43% of the time** — because the decode says *"…the capital of India is New Delhi **is correct**"* and
   the regex keys on "correct." That off-diagonal is a **scorer confound, not AV garbage** (the decode is
   appropriate to its truth input). → human-pilot the scorer before any *absolute* soft-concept number.
2. **Offline injection can't fairly test negative-baseline concepts.** truth_value / harmful_topic /
   eval have negative anchor-baseline cosines, so reaching "present" forces a large off-manifold push
   (cos(h,h') → 0.18–0.45). Their offline 0s are a method artifact → the on-manifold test is **real
   activations** (`09`), which is why Gate 3 exists.

## Dose
Refusal & sycophancy peak at **MED** dose and drop at high (over-injection toward floor); neg_sentiment
is monotone to high. MED is the cleaner operating point for behavioral/social directions.

## Caveats / not established
- Gate 2/3 are detection/specificity only; the **gap (RQ3)** is downstream and needs **output-coupled,
  suppressible** cognition (steered-refusal, eval-organism) — **not** truth/corrigibility/sycophancy.
- Judge **unvalidated** for soft concepts (human pilot pending); regex stricter than judge on neg/truth.
- **AR reconstruction fidelity (MSE/cosine) not yet run** — the gold-standard faithfulness scalar.
- Soft-concept reads are **model-specific** (truth asymmetry Gemma-only; sycophancy Qwen-only) — do not
  generalize a single-model soft result.

## Next
- ✅ Both models Gate 2/3, judge-confirmed; AV instrument validated; cross-model comparison done.
- **`10_analyze_real`** script (present/absent/anchor formalized across models).
- **AR fidelity** run on the box (hard faithfulness number).
- **Judge human pilot** (spec §3) before any *absolute* soft-concept number.
- **RQ3 — Track B DONE: null gap (`gap_recovery ≈ 0`), see `../gate4/FINDINGS.md`.** Next positive-gap
  shot = eval-awareness organism (Exp 4). Retire truth/corrigibility/sycophancy as RQ3 vehicles; keep
  as the output-coupling characterization set.
