# Experiment 5 — Forward-Prediction / "Advance Planning" Read of the Released NLA

**Models:** Qwen-2.5-7B-Instruct (block 20 = `hidden_states[21]`, d=3584) and Gemma-3-27B (block 41 =
`hidden_states[42]`, d=5376), each with its released NLA Activation Verbalizer (AV) + Activation
Reconstructor (AR). **Infra is already in place** (Exp-1/2 SGLang AV serving, `09` decode loop, `03`/`12`
position-indexed extraction, `07` scoring) — this experiment is **inference + analysis only, no new serving**.
**Goal:** measure whether reading an activation at generation position `t` through the released AV **predicts
a property of the not-yet-emitted continuation** at `t+k`, with **observable ground truth** (the continuation
is emitted). The released-model, scoreable analog of the NLA paper's strongest causally-validated claim
("Opus plans its rhyme before writing it").
**Status:** scoped and operationalized; Phase-0 ready to run. **Released ≠ frontier.**

---

## 0. Why this experiment, and the trap it is built around

Exp 2 established the **negative** half of the NLA's nature (reads predicted continuation, not hidden state →
no verbalization gap). Exp 5 measures the **positive** half — *how far ahead, and how specifically* it reads
that continuation — the project's first clean **positive** test with **real ground truth**.

> **THE TRAP — and a measured fact that makes it concrete.** The AV is RL-trained to reconstruct the
> activation *so it predicts the same continuation*, so "the AV mentions the continuation" is tautological.
> **We measured it: 100% of our committed real-activation decodes quote the content, 99–100% use
> forward-prediction language** (`expl` in `07_*__real.parquet`). So a "mentions the continuation" metric is
> **saturated and useless.** The non-tautological — and therefore the *only* reportable — quantities are:
>
> 1. **Horizon (`k`).** Predicts a property realized at `t+k` for **k ≥ 2** (beyond the residual stream's
>    built-in `t+1`), and *how far ahead before it decays to base rate.* The headline curve.
> 2. **Commitment vs hedge.** Does it name the **specific** realized token/word, or a **category/disjunction**?
>    (Surfaced by a colleague's length-penalty decode: *"expect a noun phrase like 'native housing tax' or
>    'carbon tax legislation'"* — a **hedge**, not a commitment. The commit-rate *is* the planning-depth signal.)
> 3. **Per-rollout commitment beyond context.** When the prompt is under-determined and the model commits to a
>    **specific** continuation by `t`, does the activation predict **that** commitment — measured **within
>    prompt** so "context" is held fixed (§4B). This replaces a fragile context-only-LM control.

A result that is "only `k=1`, always hedges, no per-rollout signal" is **trivial LM** — still a publishable
*bound* (the released NLA has ~zero planning horizon). The opposite is a genuine positive.

---

## 1. Research questions (each falsifiable)
- **RQ5.1 Lookahead readout.** AV read of `h_t` predicts the model's **emitted** property `P` at `t+k` (k≥2),
  above a **mismatched-continuation base rate**. (Determined tasks → clean single ground truth.)
- **RQ5.2 Horizon.** AV-predicts-`P` vs `k`: the decay-`k` where it falls to base rate, per task/model.
- **RQ5.3 Commitment.** Commit-rate (specific vs hedged) by task and `k`; and, on under-determined tasks,
  AV predicts each rollout's **own** committed `P` above sibling-rollout `P` (within-prompt).
- **RQ5.4 Generality.** Horizon, commit-rate, and per-rollout signal replicate on Gemma **and** Qwen and
  across tasks (Exp-2 soft reads were model-specific — a single-model result is suggestive only).

---

## 2. Two sub-experiments (different ground truth → different decisive control)

| | **A — Lookahead readout** | **B — Planning under under-determination** |
|---|---|---|
| Tasks | **determined** (arithmetic, factual completion) | **under-determined** (rhyme, free binary choice) |
| Decode | **greedy** (one deterministic `P` per prompt) | **sampled, M rollouts** (commitments diverge) |
| Ground truth | the model's **emitted** answer (incl. its mistakes) | each rollout's emitted commitment `P_r` |
| Decisive control | **mismatched-continuation base rate** | **sibling-rollout** base rate (within-prompt; §4B) |
| Strength | clean ground truth; the cheap Phase-0 | the real "planning" claim; the headline |
| Greedy/sample fix | greedy (deterministic GT) | sampling **is** the design (creates divergence) |

> **Use the model's emitted answer as ground truth, not the *correct* one.** If the AV predicts the model's
> **wrong** arithmetic answer before it's emitted, that is the strongest evidence it reads the model's internal
> computation rather than "what's true."

---

## 3. Tasks, datasets, and read-position / `k` mechanics

**Datasets (all programmatic / standard; n≥150 prompts per task, B: ≥60 prompts × M=8 rollouts):**
- **Arithmetic (A, Phase-0):** 150 unique `"Compute: <a> × <b> ="` (2–3 digit). `P` = the model's emitted
  answer digits.
- **Factual completion (A):** 150 `"The capital of <country> is"` / single-fact completions. `P` = emitted fact.
- **Rhyme couplet (B, headline):** 60–100 concrete-noun topics, `"Write a rhyming couplet about <topic>."`
  `P_r` = the last word of line 2.
- **Forced binary choice (B):** 100 genuinely-ambiguous items, `"Answer with exactly one word — A) <x> B) <y>."`
  `P_r` = the committed option.
- **Language/format switch (A/B):** prompts inducing an upcoming switch (`"Reply in French:"`, `"Output JSON:"`).
- **Refusal-lookahead (A, FREE existence proof):** reuse the existing AdvBench refused set — the **prompt-last**
  activation already reads refusal 0.92/1.00 (Exp-1/2), i.e. the activation **before any generated token**
  predicts the upcoming refusal. Zero-cost confirmation the AV reads forward at all.

**Read-position / `k` (the operational core — reuse the `03`/`12` teacher-forced extraction):**
1. Generate the continuation; tokenize; locate `t_P` = the position where `P` is realized (answer digits /
   rhyme word / committed letter).
2. One forward pass over `prompt + emitted continuation` (teacher-forced); read `hidden_states[layer]` at
   positions `t = t_P − k` for **k ∈ {1, 2, 4, 8, 16}** (clamped to ≥ first generated position). `k` = tokens
   **before** `P`.
3. Log per row: `k`, position, `‖h‖`, and confirm `P` does **not** already appear in `prompt + prefix[:t]`
   (the **no-surface-echo** gate — else the AV is echoing, not predicting).

---

## 4. Baselines, scorers, and the resolved controls

### 4A. Property scorers (the scorer *is* the experiment — Exp-1/2's hardest lesson)
- **Arithmetic / factual:** regex — does the AV decode contain the emitted answer string (first ≥2 digits, or
  the emitted fact token)? Objective.
- **Rhyme:** terminal-rhyme match (CMU pronouncing dict / `pronouncing`) between **any content word in the AV
  decode** and `P_r`; report **exact-word** and **rhyme-class** (matching final stressed vowel + coda)
  separately.
- **Binary / switch:** letter/option-content match; target-language/format mention before it surfaces.
- **Commit-vs-hedge classifier:** label each decode `commit` (names one specific `P`) vs `hedge` (a category
  or disjunction — "a noun phrase like X or Y"). Regex + judge; **commit-rate is a headline metric**.
- **Backstop + validation:** reuse `07`'s LLM judge with a task-specific prompt ("does this explanation
  predict that the continuation will <contain X / end on a word rhyming with Y / choose Z>?"); report
  **regex↔judge agreement** and **human-pilot ≥20/task** before any absolute rate (the soft-judge caveat
  carries from Exp 2). Four-way flags (echo / `generic_template` / `nla_degenerate`) on every read.

### 4B. The control for B (resolves the context-only-LM design hole — within-model, multi-rollout)
The naive "give a context-only LM the prefix" control is **broken**: if that LM is the target model it just
**reproduces `P` deterministically** → no edge possible. Fix — hold context fixed and vary only the model's
*own* sampled commitment:
```
for each under-determined prompt p:
  sample M=8 rollouts at temp ~0.8           # commitments diverge
  keep p iff rollouts disagree on P (>=3 distinct P_r)   # genuinely under-determined
  for each rollout r:  extract h_t^r at t = t_P^r - k (k>=2);  AV-decode;  score predicts(P_r) and predicts(P_{r'!=r})
  PLANNING SIGNAL = within-prompt AUROC( own P_r  vs  sibling P_{r'} | AV-mention ), cluster-bootstrapped over p
```
Any signal here is the **activation's per-rollout commitment** (the prompt is identical across siblings), so
it needs **no external model**. **Secondary triangulation:** the *prefix-marginal* `P(P)` across rollouts —
the edge = AV-own-rate − marginal-rate (does the AV beat just guessing `P`'s base frequency?). An *optional*
cross-model context-only baseline (read Gemma's plan; baseline = Qwen-as-LM given the prefix) can triangulate
but is not load-bearing.

### 4C. Controls common to A and B
- **Mismatched base rate** (decisive for A): score AV(`h_t^i`) against example `j`'s `P^j` → must be ≪ own.
- **Trivial next-token:** require k≥2 / end-of-structure; report `k=1` separately as the LM floor.
- **No-surface-echo:** `P` absent from prompt+prefix (§3.3).
- **CIs:** cluster-bootstrap over example/prompt id; compare CI **bounds** (n~100–150 is noisy).
- **Greedy vs sampled:** A greedy; B sampled by design; for A also report 3-sample consistency.

---

## 5. Pre-registered thresholds and the horizon curve
- **P0-A (arithmetic) GO:** AV predicts the emitted answer at **k≥2** with rate **≥0.30** while the
  **mismatched base rate ≤0.05**, CI-separated (lower-CI(AV) > upper-CI(mismatched)). **STOP/redefine** if
  signal exists only at `k=1` (no lookahead) or fails to beat mismatched.
- **P0-B (rhyme) GO** (planning): within-prompt **AUROC(own vs sibling `P`) ≥ 0.65** with CI_lo > 0.5 on the
  under-determined subset, **and** commit-rate clearly > 0 on committed rollouts. **PARTIAL** if AV ≫
  mismatched but per-rollout AUROC ≈ 0.5 (lookahead-readout only, not plan-beyond-context).
- **Horizon (headline):** report decay-`k` = the largest `k` with AV rate CI-above the mismatched/base rate,
  per task and model; overlay the commit-rate(`k`) curve. No fixed bar — it's the measured quantity.
- **Cross-model:** require Gemma + Qwen; flag any single-model result.

## 6. Phase-0 (cheap; ready to run on the existing infra)
1. **P0-0 (free):** re-frame the existing refusal prompt-last reads as the lookahead existence proof.
2. **P0-A (arithmetic, ~2–4 GPU-hr):** the cheapest, cleanest-ground-truth test; the `k`-sweep + mismatched
   control. Pass → the AV does lookahead readout; size of decay-`k` is the first real number.
3. **P0-B (rhyme, ~3–5 GPU-hr):** the multi-rollout within-prompt planning test + commit-rate.
   Pass → genuine planning; partial → readout-only.
Only after P0-A/B → the full task set, horizon sweep, cross-model, and the optional analyses (§8).

## 7. Compute / cost
Inference only (generation + position-indexed extraction + AV decode + scoring); reuses Exp-2 serving.
~5–15 GPU-hr per model full; Phase-0 ≈ 5–9 GPU-hr total. Low-hundreds-$. **No training.**

## 8. Optional / stretch analyses (do only if Phase-0 is positive)
- **AR-"fluff" connection (from the colleague's length-penalty result: ~65% shorter explanation, only ~9pp FVE
  drop → much AV text is KL-fluff, not reconstruction-bearing).** Test on our side: does the **coarse core** of
  the explanation predict `P` while the **fine detail** is the confabulated part? Ties to Exp-4 P0-1 (AR-cos):
  strip the decode to its `P`-bearing clause and re-score AR reconstruction.
- **Layer sweep:** the plan may live off the NLA read site; read ±a few layers (extraction-only, the AV is
  layer-locked so this only bounds where the *signal* is, not where the AV can read).
- **Within-vs-cross-model planning** (the optional cross-model context-only baseline, §4B).

## 9. Reuse map (nothing trained)
- **Reuse:** `09_decode_real.py` + `NLAClient.generate` (AV decode loop); the `03`/`12` teacher-forced
  forward-pass + **position-indexed** extraction (the one small new bit — read at `t_P−k`, not prompt-last);
  `07_score_matrix.py` (regex + judge, four-way flags); `confounds._safe_auc`/`_cluster_bootstrap_ci`
  (separation + CIs); `av_up.sh` + `nla_box` (serving).
- **New (small):** generation + ground-truth `P` capture + `t_P` location; the multi-rollout driver for B; the
  per-task property scorers + the commit-vs-hedge classifier; the horizon-curve analysis. A natural first
  stage: `scripts/19_forward_predict.py` (mirror `18`'s CPU-`--selftest` + box-run split).

## 10. Failure modes
- **Only `k=1` / always hedges** → trivial LM; report the ~zero horizon as the bound.
- **Per-rollout AUROC ≈ 0.5 on B** → lookahead readout without plan-beyond-context (an informative null).
- **Property scorer noise** (rhyme matching) → human-pilot first; report exact-word and rhyme-class separately.
- **Read-position off-by-one** → invalidates `k`; behaviorally verify (k=1 should be near-perfect — the LM floor).
- **Surface leakage of `P`** into the prefix → the no-surface-echo gate is mandatory.
- **Under-determined subset too small** (rollouts rarely disagree) → loosen temp / pick higher-entropy prompts.

## 11. What a result means / limitations
- **Positive** (k≥2 horizon, commit-rate > 0, per-rollout AUROC > 0.5): the released NLA reads **committed
  future plans** — the positive complement to the Exp-2 gap null (reads *forward output*, not *hidden state*),
  with a measured horizon. The project's first headline positive beyond refusal detection.
- **Null/trivial** (k=1, all hedge): bounds the planning horizon to ~0 — a real, citable result.
- **Lookahead readout ≠ planning:** A shows early *encoding* of a determined answer; only B (per-rollout, under
  under-determination) supports "*plans* a committed choice." Keep separate.
- The positive is partly **expected** (AV training) — which is why horizon / commit / per-rollout are the
  claims, not existence. Single layer; released ≠ frontier (the paper's result was Opus + internal NLAs).

## 12. Verification
- **P0-A:** committed CSV `(example, emitted_answer, av_predicts, mismatched, k, commit_flag)`; CI-separated
  rates per `k`.
- **P0-B:** `(prompt, rollout, P_r, av_predicts_own, av_predicts_sibling, k, commit_flag)`; within-prompt
  AUROC + CI; commit-rate.
- **Horizon:** per-task curve AV-rate vs `k` with decay-`k`; commit-rate(`k`) overlaid.
- **Scorer:** human-pilot agreement (≥20/task) beside every absolute rate; regex↔judge agreement.
- **Cross-model:** Gemma + Qwen side-by-side; flag single-model-only results.
- **Self-test:** `19_*.py --selftest` validates the scorers + AUROC + commit/hedge logic on synthetic ground
  truth before any real number (mirrors `18`).

## 13. References (IDs WebSearch-verified; digest in `docs/references/literature_synthesis_2026-06.md`)
- **NLA paper** — Fraser-Taliente, Kantamneni, Ong et al. (2026), transformer-circuits.pub/2026/nla — the
  "plans its rhyme before writing it" result this ports to a released model with observable ground truth
  (search-verified; treat the quote as "as surfaced by search").
- **Exp 2 closeout** (`Experiment 2/rebuild/EXPERIMENT_2_CLOSEOUT.md`) — the output-coupling thesis whose
  *forward* half this measures; refusal-prompt-last is the free existence proof; the 100%-quote measurement
  motivates dropping the saturated "mentions continuation" metric.
- **Activation Oracles** (arXiv:2512.15674), **Yuan 2605.09502**, **Miao & Ungar 2603.25052** — the
  decodable-vs-verbalized framing for "what the released AV does and doesn't read."
