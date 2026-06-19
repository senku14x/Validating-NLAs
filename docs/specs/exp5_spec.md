# Experiment 5 — Forward-Prediction / "Advance Planning" Read of the Released NLA

**Models:** Qwen-2.5-7B-Instruct (block 20 = `hidden_states[21]`, d=3584) and Gemma-3-27B (block 41 =
`hidden_states[42]`, d=5376), each with its released NLA Activation Verbalizer (AV) + Activation
Reconstructor (AR).
**Goal:** test whether reading an activation at generation position `t` through the released AV **predicts a
property of the not-yet-emitted continuation** at `t+k` — the released-model, ground-truthable analog of the
NLA paper's strongest *causally-validated* claim ("Opus plans its rhyme before writing it"). Unlike the
verbalization gap (Exp 2/4, unobservable hidden state), the continuation **is emitted**, so every prediction
is **scoreable against ground truth**.
**Status:** scoped; Phase-0 de-risk pending.
**Stance:** validation-and-characterization of the *released* NLA. **Released ≠ frontier** — nothing
transfers to Anthropic's internal NLAs.

---

## 0. Why this experiment, and the one trap it must design around

Experiment 2 established the **negative** half of the NLA's nature: it reads the *predicted surface
continuation*, not hidden state (no verbalization gap). Experiment 5 measures the **positive** half — *how
well, and how far ahead*, it reads that continuation — and is the project's first clean **positive** test
with **real ground truth** (the emitted tokens). It also turns the hand-wave "output-coupling" into a
**measured predictive horizon**.

> **THE TRAP (foreground it — this is the whole design problem).** The AV is **RL-trained to produce text
> that reconstructs the activation such that it predicts the *same continuation*.** So "the AV predicts the
> continuation" is **near-tautological** and is **NOT** the contribution. The residual stream at `t` is
> literally optimized to predict token `t+1`, so a `k=1` "prediction" is trivial language modeling. The
> experiment is only meaningful if it isolates the **three non-tautological quantities**:
>
> 1. **Horizon (`k`).** Does the read at `t` predict a property realized at `t+k` for **k ≥ 2** — beyond
>    greedy next-token? How far ahead before it decays to chance? *(novel; the headline number)*
> 2. **Edge over context-only.** Does the **activation** carry plan-information **beyond what a normal LM
>    infers from the same prompt+prefix**? If a context-only LM predicts the property just as well, the AV
>    read nothing special — it's "a full LM guessing from context." *(the load-bearing control Exp-2 never ran)*
> 3. **Commitment under under-determination.** When the context admits **many** valid continuations, does the
>    activation predict the **specific one the model commits to** (not just the set / the base rate)? *(the
>    real "planning" claim, vs "the answer was obvious")*

A result that is only "`k=1`, no context edge, context-determined" is **trivial LM** — and still a publishable
*bound* (the released NLA has ~zero planning horizon). A result with `k≥2`, a context edge, and commitment is
a genuine positive: the released NLA reads committed future plans.

---

## 1. Research questions

- **RQ5.1 — Lookahead readout.** Does the AV read of `h_t` predict a property `P` realized at `t+k` (k≥2),
  above a **mismatched-continuation base rate**? (Clean ground truth via *determined* tasks.)
- **RQ5.2 — Predictive horizon.** As a function of `k`, where does the AV's prediction of `P` decay to the
  mismatched base rate? (The headline curve, per task and per model.)
- **RQ5.3 — Plan vs inference (the strong claim).** On *under-determined* tasks, does the AV predict the
  model's **committed** continuation **above a context-only LM** given the same prompt+prefix?
- **RQ5.4 — Cross-model / cross-task generality.** Do the horizon and the context-edge replicate on Gemma &
  Qwen and across task families, or are they task/model-specific (as the Exp-2 soft reads were)?

---

## 2. The two sub-experiments (different controls, because the ground truth differs)

| | **A — Lookahead readout** | **B — Planning under under-determination** |
|---|---|---|
| Tasks | **determined** (arithmetic, factual completion) | **under-determined** (rhyme, free choice) |
| Claim | activation encodes a property *before emission* | activation predicts the *committed* choice beyond context |
| Ground truth | the emitted continuation (unique answer) | the emitted choice (one of many valid) |
| Decisive control | **mismatched-continuation base rate** (read of `h_t^i` scored vs example *j*'s `P^j` → chance) | **context-only LM** edge (does the activation beat a prefix-LM?) |
| Strength | clean ground truth; A is the cheap Phase-0 | the real "planning" claim; B is the headline |
| Note | context-only is *not* a fair control here (a strong LM also computes the answer) → A tests *readout*, not *plan* | mismatched base rate is *also* reported |

Both report the **horizon** (vary `k`). A is run first (cheapest, cleanest ground truth); B carries the
headline planning claim.

---

## 3. Task families (read position `t` chosen so `P` is realized at `t+k`, k≥2)

1. **Arithmetic answer-before-emit (A; cheapest Phase-0).** Prompt `"Compute: 47 × 83 ="`; greedy-generate;
   read `h_t` at the **first generated position** (before any answer digit). `P` = the correct product (or
   its leading digits), realized several tokens later. Large answer space → mismatched base rate ≈ 0 → a
   strong control. GO = AV mentions the emitted answer ≫ mismatched.
2. **Rhyme couplet (B; the paper's exact case, the headline).** `"Write a rhyming couplet about <topic>."`;
   read `h_t` at the **start of line 2** (before the rhyme word, which lands at line-2 end, k = several
   tokens). `P` = the emitted rhyme word / its phonetic class. Control = a **context-only LM** given line 1 +
   the line-2 prefix: can it predict the *specific* rhyme the model used? GO = AV ≫ mismatched **and** AV ≥
   context-only.
3. **Refusal lookahead (A; a *free* existence proof).** We **already** read refusal at the **prompt-last**
   token (Exp-1/2: ~0.92/1.00) — i.e. the activation **before any generated token** predicts the upcoming
   refusal. Re-frame this as the zero-cost existence proof that the AV reads forward at all; generalize to
   "predicts the refusal at an early generated position."
4. **Forced binary choice (B).** `"Answer with exactly one word: A or B."` on items where both are plausible;
   read at the first generated position; `P` = the committed letter. Context-only control = a prefix-LM's
   guess. Tests commitment under genuine under-determination.
5. **Language / format switch (A/B).** Prompts that induce an upcoming switch (to non-English, to JSON/code);
   does an early `h_t` predict the switch before it surfaces? (Qwen's geography→Chinese code-switch is a hint
   it plans switches.)

---

## 4. Measurement, baselines, and the horizon curve

For each (task, example, model):
1. **Generate** the continuation (greedy; also sample k=3 for robustness) and record the emitted tokens →
   ground-truth property `P^i`.
2. **Extract** `h_t` at the pre-registered read position(s) `t` (well before `P` is realized; fp32; the NLA's
   layer). Log `||h||`, position, and the realized distance `k` from `t` to where `P` appears.
3. **Decode** `h_t` through the AV (reuse `09`'s loop; temp 1.0, 3 samples; `extract_explanation=False`).
4. **Score** whether the AV text predicts `P^i`, by a **pre-registered property scorer** (regex + judge):
   answer-match (A), rhyme/phonetic-match (B2), letter-match (B4), switch-mention (5). Four-way controlled
   (echo / `generic_template` / `nla_degenerate`).
5. **Baselines / controls (all mandatory):**
   - **Mismatched-continuation base rate** (decisive for A): score AV(`h_t^i`) against a *different* example's
     `P^j`. The read must be example-specific (`P^i ≫ P^j`), not a generic guess.
   - **Context-only LM** (decisive for B; **load-bearing — Exp-2 never ran it**): give a clean LM the prompt +
     emitted prefix up to `t` and ask it to predict `P`. The AV must **beat** it to claim the activation
     carries a plan beyond context.
   - **Trivial next-token** control: confirm `P` is *not* just token `t+1` (require k≥2 / end-of-structure).
   - **Surface-echo** control: `P` must not already appear in the prompt/prefix (else the AV is echoing).
6. **Horizon curve (RQ5.2):** plot AV-predicts-`P` vs `k` (read at successively earlier positions, or `P`
   realized successively later); report the `k` at which it decays to the mismatched base rate. Compare to the
   context-only curve.

**Headline numbers:** the **horizon** (max k with prediction ≫ base rate), the **context-only edge** (AV −
context-only, on B), and the **commitment rate** (B), per model — *not* the bare "AV predicts continuation."

---

## 5. Confound doctrine for Experiment 5
- **A positive is EXPECTED by the AV's training** → never report "the AV predicts the continuation" as the
  result; report **horizon**, **context-edge**, **commitment**. (Same discipline as Exp-2: no bare number.)
- **Mismatched base rate + context-only + no-surface-echo** are the load-bearing controls; cluster-bootstrap
  CIs over example id; compare CI **bounds**.
- **Position/norm:** read at normal-norm positions; off-by-one position invalidates `k`; log everything.
- **Property scorer is the experiment** (Exp-1/2 lesson): regex ↔ judge cross-check, and **human-pilot** the
  rhyme/answer scorers (≥20/task) before any absolute number — the soft-judge caveat carries over.
- **Cross-model:** require Gemma **and** Qwen; soft Exp-2 reads were model-specific, so a single-model horizon
  is suggestive only.
- **Greedy vs sampled:** report both; "planning" should be robust to sampling temperature.

---

## 6. Phase-0 de-risk (cheap; explicit go/stop)
- **P0-0 (free):** re-frame existing **refusal prompt-last** data as the lookahead existence proof (activation
  before generation predicts refusal). Confirms the AV reads forward at all. *Already in hand.*
- **P0-A (cheap, clean ground truth):** **arithmetic answer-before-emit**, ~50 unique 2-digit products, read
  at the first generated position, AV-score answer-mention vs **mismatched base rate**. **GO** if AV predicts
  the emitted answer with rate ≫ mismatched (pre-register, e.g. ≥0.3 vs ≤0.05, CI-separated). **STOP/redefine**
  if AV only ever reads `k=1` / the prompt → no lookahead.
- **P0-B (cheap, headline):** **rhyme couplet**, read at line-2 start, AV-score rhyme vs **context-only LM**
  vs mismatched. **GO** for the planning claim if AV ≫ mismatched **and** AV ≥ context-only. **PARTIAL** if AV
  ≫ mismatched but ≈ context-only (lookahead readout, not plan-beyond-context — still RQ5.1).
- Only after P0-A/B → run the **horizon sweep** + cross-model + the other tasks.

## 7. Compute / cost
Inference only (generation + AV decode + a context-only LM pass); reuses Exp-2 serving. ~5–15 GPU-hr per
model for the full task set; Phase-0 ≈ 2–4 GPU-hr. Low-hundreds-$ at most. **No training.**

## 8. Failure modes
- **Only k=1 / context-determined** → trivial LM; report the ~zero horizon as the bound (still a result).
- **AV ≈ context-only on B** → the AV infers from context like any LM; no "plan beyond context" (an
  informative null that bounds the planning claim).
- **Property scorer noise** (rhyme/answer matching is fiddly) → human-pilot first.
- **Read-position off-by-one** → invalidates `k`; behaviorally verify the position.
- **Echo / surface leakage** of `P` into the prefix → the no-surface-echo control is mandatory.
- **The AR is irrelevant here** (this is an AV read experiment) — don't conflate with AR fidelity.

## 9. Reuse map (inference + analysis; nothing trained)
- **Reuse:** `scripts/09_decode_real.py` + `NLAClient.generate` (AV decode loop); `scripts/03_extract_for_battery.py`
  pattern (forward pass + position-indexed extraction at *generated* tokens — the small new bit); `scripts/07_score_matrix.py`
  (regex + judge scoring harness, four-way flags); `confounds._safe_auc`/`_cluster_bootstrap_ci` (separation +
  CIs); `av_up.sh` + `nla_box` (AV serving).
- **New:** generation + ground-truth `P` capture; per-task property scorers (answer/rhyme/letter/switch); the
  **context-only LM** baseline harness (Exp-2 never built it — also reusable for Exp-3/4); the
  mismatched-continuation control; the horizon-curve analysis.

## 10. What a result means
- **Positive** (k≥2 horizon, context-edge on B, commitment): the released NLA reads **committed future
  plans** — a clean positive that complements the Exp-2 gap null (it reads *forward output*, not *hidden
  state*), and quantifies *how far forward*. The project's first headline positive beyond refusal detection.
- **Null/trivial** (k=1, no edge): bounds the released NLA's planning horizon to ~0 — also a real,
  citable result and a sharpening of "output-predictor."

## 11. Limitations
- **Lookahead readout ≠ planning.** A (arithmetic) shows the activation *encodes a determined answer early*;
  only B (under-determined + context-edge) supports "*plans* a committed choice." Keep them separate.
- **The positive is partly expected** (AV training) — which is why horizon/edge/commitment, not existence,
  are the claims.
- Single layer (the NLA's read site); the plan may live elsewhere — a layer sweep is a stretch.
- Released ≠ frontier; the paper's planning result was on Opus with internal NLAs.

## 12. Verification
- **P0-A:** committed CSV `(example, answer, av_predicts_answer, mismatched_rate, k)` + CI-separated rates.
- **P0-B:** `(example, rhyme, av_match, context_only_match, mismatched, k)` + the AV−context-only edge with CI.
- **Horizon:** a per-task curve (AV rate vs k) with the decay-`k` and the context-only curve overlaid.
- **Scorer:** human-pilot agreement (≥20/task) reported alongside every absolute rate.
- **Cross-model:** Gemma + Qwen tables side-by-side; flag any single-model-only result.

## 13. References (IDs WebSearch-verified; full digest in `docs/references/literature_synthesis_2026-06.md`)
- **NLA paper** — Fraser-Taliente, Kantamneni, Ong et al. (2026), transformer-circuits.pub/2026/nla — the
  "Opus plans its rhyme before writing it" result is the causally-validated mechanism this experiment ports to
  a released model with observable ground truth. (Search-verified; treat the quote as "as surfaced by search.")
- **Exp 2 closeout** (`Experiment 2/rebuild/EXPERIMENT_2_CLOSEOUT.md`) — the output-coupling thesis Exp 5
  measures the *forward* half of; refusal-at-prompt-last is the free existence proof.
- **Activation Oracles** (arXiv:2512.15674), **Yuan 2605.09502**, **Miao & Ungar 2603.25052** — the
  decodable-vs-verbalized / probe-vs-output line that frames "what the released AV does and doesn't read."
