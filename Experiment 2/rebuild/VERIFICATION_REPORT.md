# Experiment 2 — independent verification report (2026-06, onboarding §C)

**Purpose.** Re-derive every committed headline **from the raw artifacts** (not the prose), and for each
ask "what trivial confound would produce this?" then look for it. Verdicts: **CONFIRMED** (reproduces +
trivial confound ruled out) / **DISCREPANCY** (number or provenance doesn't match) / **OVERCLAIMED** (the
data are weaker than the prose). Numbers re-derived with the repo `.venv` from the committed files; method
in each row.

> **Headline.** The program's **core verdicts reproduce cleanly from the raw data** — Gate-1, the Gate-2/3
> real-activation table, the RQ3 gap=0, the coupling correlation, AR fidelity, and the persona lever all
> check out, several to the decimal. The **one place the committed evidence is weaker than the narrative**
> is the *interesting half* of the RQ3 null — "the NLA misses a probe-visible refusal **residual**." The
> committed `12` JSON carries the **raw, in-sample, outlier-dominated** projection (Gemma `pf = −0.165`,
> B *below* harmless), **not** the held-out/standardized probe the FINDINGS prose describes, and the
> `AUROC(B,C)/(B,E)` numbers that would show the residual is separable **are not in any committed file.**
> The *collapse* claim (`AUROC(A,B)=1.0`) is solid; the *residual-survives* claim is **under-committed.**
> Plus minor overcounts and two missing artifacts. None overturns the RQ1/RQ2/RQ3 verdicts; one (the
> residual) softens how *interesting* the null is.

---

## 0. Instrument self-tests — **CONFIRMED**
All 7 CPU self-tests pass (`test_confounds / paths / audits / concepts / sources / flags / injection`).
`test_confounds` separates clean / length / mixed / **collinear** regimes on synthetic ground truth (the
collinear case correctly returns *not represented*: `resid.ci_lo 0.222 < bar 0.600`). Precondition for
trusting any real number is met. `confounds.py` is sound: logistic probe, nested `StratifiedGroupKFold`,
cluster-bootstrap CIs **at the pair level**, pre-registered gate on CI **bounds**, BoW upgrade in `gate_v2`.

## 1. Gate-1 vector validity — **CONFIRMED**
`results/gate1/04b_recheck__{gemma3-27b,qwen2.5-7b}__all.json`, `v2_verdict`.
- **PASS-on-both = {corrigibility, harmful_topic_benign, neg_sentiment, sycophancy, truth_value}** — exactly
  the claimed 5, identical set on both models. ✓
- **refusal demoted PASS→WEAK**: `text_bow_auroc = 0.9966` both models → "surface lexical." Trivial confound
  *is* the finding: AdvBench-vs-Alpaca is **corpus identity**, not a refusal latent. ✓ (Carried as the
  anchor only because Exp-1 proves it *causal*, independent of decodability.)
- **eval_framing (clear+matched) demoted PASS→WEAK**: `bow = 1.0` → in-the-prompt lexical. ✓
- Sanity that the gate isn't trivially passing everything: `truth_value` bow 0.485, `corrigibility`/
  `sycophancy` bow 0.50 (no lexical leak) yet resid 0.94–1.0 → genuinely "decodable beyond surface text."

## 2. Gate-2/3 real-activation table — **CONFIRMED (exact)**
Recomputed judge==2 rate per concept by polarity from `results/gate2/07_score_matrix__*__all__real.parquet`
(diagonal axis `j_<concept>`). Reproduces the FINDINGS/closeout table **to the decimal**:

| concept | Gemma P/A/anch (mine) | Qwen P/A/anch (mine) |
|---|---|---|
| refusal | 0.92 / 0 / 0 | 1.00 / 0 / 0 |
| truth_value | 0.71 / 0.99 / 0 | 0.87 / 0.82 / 0.01 |
| neg_sentiment | 0.26 / 0 / 0 | 0.24 / 0 / 0 |
| sycophancy | 0.11 / 0.07 / 0 | 0.36 / 0.02 / 0.02 |
| corrigibility | 0 / 0 / 0 | 0.03 / 0 / 0 |
| harmful_topic_benign | 0.03 / 0 / 0 | 0 / 0 / 0 |
| eval_framing_matched | 0 / 0 / 0 | 0 / 0 / 0 |
| real degen | 0.000 | 0.047 |

- **refusal is not echo.** Recomputed exc-echo refusal-present = **0.91 (G) / 1.00 (Q)** (echo_rate only
  0.06/0.01; generic_template 0.00). The cross-model positive is **echo- and template-robust.** ✓
- **truth_value FALSE≫TRUE is Gemma-only + confabulation-entangled — CONFIRMED.** Gemma absent(false) 0.99 ≫
  present(true) 0.71; Qwen symmetric 0.87/0.82. I pulled the 29/100 Gemma TRUE-statement *misses* and they
  are literally confabulated affirmations: *"Yes, you are correct. Paris is the capital of t[he US]"*,
  *"Your country is the capital of Indonesia. You are in Beijing."* → the low TRUE score is partly "the AV
  reconstructs true-statement activations less faithfully," not purely "truth less output-coupled." Correctly
  demoted to suggestive/single-model. ✓
- sycophancy Qwen-only (0.36 vs Gemma 0.11≈its own absent 0.07) ✓; negatives null both ✓.

## 3. RQ3 — the gap = 0 — **CONFIRMED** (with one un-verifiable sub-claim)
`results/gate4/13_gap_summary.csv`.
- **A/pre 1.000 (G) / 0.985 (Q)** reproduces Gate-3 (≈1.0) → pipeline self-validated (extraction
  layer/position correct, else A would fail). ✓ (Not *numerically* identical to Gate-3's 0.92/1.00 — A is a
  different 40-prompt bucket — but both ≈1, which is the claim.)
- **B/pre = B/gen = 0.000, both models; C/pre = 0.000; E/pre = 0.075(Q)/0.000(G).** Gap = 0. ✓
- **Trivial-confound check the task flags — "is B compliant-reading, or is the AV just degenerate there?"**
  `degen = 0.000` and `<explanation>`-tag rate context is clean → **B is non-degenerate, non-refusal
  output**, so "the AV is just degenerate at B" is **ruled out.** ✓
- **LIMITATION (verifiability):** the *raw B decodes are gitignored* (harmful). So I can confirm **B ≠
  refusal and B is non-degenerate**, but I **cannot independently confirm "B reads as *compliant*"** (vs
  some third thing) from committed artifacts. The "reads exactly like compliant ones" wording rests on
  in-session inspection, not a committed file.

## 4. Representation collapse / residual (stage 12) — **collapse CONFIRMED; residual OVERCLAIMED / under-committed**
`results/gate4/12_trackb_persistence__*.json` + reading `scripts/12_trackB_persistence.py:241-256`.
- **What stage 12 actually computes (code):** a **raw diff-of-means(A,E) axis**, then **in-sample**
  `roc_auc_score` on the **raw projection** for `auroc_A_vs_B/_C/_E`, and `pf = (mean_B−mean_C)/(mean_A−mean_C)`
  on raw projection means. The `probe_battery` field (held-out, standardized) certifies only the **A-vs-E
  axis**, *not* the A-vs-B/B-vs-C comparisons.
- **Collapse — CONFIRMED.** `auroc_A_vs_B = 1.0` in every cell (both models, prelast+genmean): B perfectly
  separable from refusal-A. This is rank-based and robust to the outlier issue (A's projections sit far
  above B's: Qwen prelast A=71.4±5.7 vs B=4.5±7.1; Gemma A=42119±1144 vs B=23875±1756). "Refusal
  representation does not survive at refusal strength under prefill" is **solid.** ✓
- **DISCREPANCY 1 — provenance.** FINDINGS §6 says persistence was "verified with held-out 5-fold CV and a
  standardized logistic probe." The committed `auroc_A_vs_B`/`pf` are **raw, in-sample DoM projection** — not
  held-out, not standardized. The standardized result lives only in prose.
- **DISCREPANCY 2 — the committed Gemma `pf` is the *nonsense* value.** Gemma `prelast pf = −0.165`
  (B=23875 **below** harmless C=26458) — the outlier-dominated artifact the FINDINGS itself says to discard.
  The "corrected" standardized `pf ≈ 0.03, B>C` is **not in any committed file**; CLAUDE.md/FINDINGS quote
  `pf≈0.03(G)` from an uncommitted run. (Qwen prelast `pf=0.17` is in the JSON; the FINDINGS §6 table says
  `0.20` — minor.)
- **OVERCLAIMED — the residual-survives claim.** The numbers that would show "B is a small but **reliable
  residual** separable from compliant controls" — `AUROC(B,C)`, `AUROC(B,E)` — **do not exist as keys** in
  the committed JSON (it has `A_vs_B/_C/_E` only). So "the NLA misses a **probe-visible residual**" — the
  part that makes the null *interesting* rather than trivial — is **not reproducible from committed
  artifacts.** On the committed Gemma prelast the residual points the *wrong way* (B<C).
- **Residual identity (task's question):** even granting the standardized result, the setup **cannot**
  separate "harmfulness-awareness" from "refusal-intent" (the script's own caveat: B is read at prefill-last
  *with the compliance prefill in context*). So "residual = refusal residual" is itself unidentified.
- **Net:** the RQ3 **verdict** (gap≈0; NLA no better than reading the output) stands on §3 alone. But the
  *mechanistic gloss* ("…and it misses a residual a probe still sees") is the weakest-committed claim in the
  experiment, and the committed data, if anything, make the null look **more trivial** (nothing strongly
  refusal-y is left at B's decision token) than the narrative's "interesting miss." **Top fix-it item.**

## 5. Degenerate lever / coupling-vs-salience (stage 14) — **CONFIRMED, with a correction to "only refusal"**
`results/gate4/14_coupling_score__*.csv`.
- **Qwen: behavioral>0 only for refusal** (refusal 0.5; all 6 others 0.00, frac-steered-identical ≈0.05).
  So `n_behavioral_moved = 1`, the partial corr is driven by one nonzero point and is meaningless →
  **UNIDENTIFIED-LEVER-DEGENERATE confirmed.** ✓
- **DISCREPANCY (minor): "only refusal moves" is Qwen-only.** The committed **Gemma** CSV has
  **sycophancy behavioral = 0.05** (steered_rate 0.05 vs baseline 0.00 = 1/20 prompts) **in addition to**
  refusal 0.6. So on Gemma `n_behavioral_moved = 2`, not 1. 0.05 is within noise and doesn't change the
  verdict, but the blanket "diff-of-means steering moves **only** refusal / soft directions move **0.00**"
  is **not literally true on Gemma.**
- **corr(salience, NLA_read) — CONFIRMED:** Qwen **0.899** (claim 0.90 ✓), Gemma 0.950. The "cannot rule out
  all-salience" verdict is right. Side note: `corr(logit_lens, NLA_read)` = **−0.02 (Q)** / 0.32 (G) — the
  logit-lens "weak proxy consistent with coupling" is **uncorrelated with the actual read on Qwen**, so it
  carries no weight; fine, since the FINDINGS only calls it a weak proxy.
- **Cross-check with §B:** the *interpretation* of the dead lever as "read-direction ≠ write-direction" is
  **over-read** (Persona Vectors + Marks-Tegmark steer soft diff-of-means directions; simpler
  layer/coef/pair explanations un-excluded). The **UNIDENTIFIED** verdict itself is the honest call; the
  read≠write gloss should carry the same "consistent with, not shown" hedge.

## 6. AR fidelity (stage 15) — **CONFIRMED; "faithful-but-lossy" is fair**
`results/gate4/15_ar_fidelity__qwen2.5-7b.json`: `mean_cosine 0.9243, chance_cosine 0.7045,
cosine_above_chance 0.2198, mean_mse 0.1515, p10 0.894, n=160`.
- Internal consistency: `mse = 2(1−cos) = 2(1−0.9243) = 0.1514` ✓ matches `0.1515`.
- **Is +0.22 too small?** The reconstruction closes **73% of the available above-chance range**
  ((0.924−0.705)/(1.0−0.705) = 0.74), with p10 still 0.894. That is a *substantial* recovery, not a marginal
  one. "**Faithful-but-lossy**" is **fair** — and it corroborates the behavioural "coarse-faithful /
  fine-confabulating" finding. ✓
- **Flag (documented):** **Qwen-only** (no `15_…gemma3-27b.json` committed); **real activations only** (not
  injected/steered). The faithfulness number does not yet exist for Gemma or for the off-manifold inputs.

## 7. Persona lever replication (external) — **CONFIRMED (exact)**
`external/persona_results/evil_steer_coef*.csv`, aggregated by coef:

| coef | n | mean_evil | mean_coh | coherent-evil (evil>50 & coh>60) | corr(evil,coh) |
|---|---|---|---|---|---|
| 0.0 | 200 | 0.0 | 95.0 | 0 | — |
| 1.0 | 200 | 7.1 | 83.6 | 8 | −0.33 |
| 1.5 | 200 | 69.0 | 48.4 | **45** | **−0.29** |
| 2.0 | 200 | 91.4 | 26.9 | 16 | −0.06 |

Matches the claim exactly: evil 0→7→69→91, coherence 95→27, **coef-1.5 coherent-evil pool 45/200**,
corr(evil,coherence) **−0.29**. The lever is **causal and dose-dependent**; the n=45 read-set is real. ✓

## 8. Persona-evil NLA read (stage 17) — **CONFIRMED directionally; headline rate is a minor OVERCOUNT; anchor is weak**
Re-derived from `results/gate4/17_decodes__qwen2.5-7b.csv` (the `17_persona_nla_read__*.csv` "rates" file
the closeout's artifact index cites is **MISSING from the repo** — see §10).
- **av_evil_judge by set:** evil **0.385** / baseline **0.003** / refusal-anchor 0.089. Per-activation (row
  mean): evil 0.385, `frac(row>0.5)=0.356` → **misses ~64%.** ✓ (matches "misses ~60%").
- **DISCREPANCY (minor overcount):** committed decodes give **0.385** (raw) and **0.38** (exc-echo); the
  FINDINGS/closeout say **0.41 / 0.42**. 0.385 does not round to 0.41 → the prose rate is ~0.03 high vs the
  committed file. **Also: echo_rate on evil decodes = 0.50** (half flagged as evil-lexicon echo) — though
  exc-echo barely moves the rate (0.38), so echo isn't *inflating* it; still, a 50% echo rate deserves a
  mention it doesn't get.
- **Dose-response — CONFIRMED:** within-evil corr(input_evil, av_evil_judge) = **0.349** (claim +0.35 ✓);
  pooled with baseline **0.64** (claim +0.64 ✓). Caveat (already in FINDINGS): within-evil input range is
  only **72–100**, and the pooled 0.64 is largely the evil-vs-baseline *level* gap, not a fine dose curve.
- **Topic-gating — SUPPORTED but eyeball-level.** Row 0 is the canonical case: `input_evil=100` decoded as
  *"career growth tips for an influencer's visibility… networking effectiveness."* The misses are
  systematically `input_evil`-high answers read as benign topic. On n=45 this is a reasonable read, not a
  measured claim.
- **Anchor (task's question) — WEAK.** refusal_regex on the refusal set = **0.667** but on **n=6 activations
  (12 decodes)**, vs Gate-3's refusal 1.00. It shows the pipeline isn't *globally* broken (refusal set reads
  refusal, j_evil≈0.09 there), but **0.67 on n=6 is thin** to "license pipeline valid" — treat as a smoke
  check, not a validation.

---

## 9. Discrepancy ledger (ranked by how much it matters)

| # | Item | Status | Matters? |
|---|---|---|---|
| 1 | **Stage-12 residual** — committed JSON is raw/in-sample (Gemma `pf=−0.165`); standardized `pf≈0.03` and `AUROC(B,C)/(B,E)` **not committed** | **OVERCLAIMED / under-committed** | **Yes** — it's the *interesting* half of the RQ3 null; re-run + commit the standardized stage 12 or soften the "misses a probe-visible residual" claim |
| 2 | "Read-direction ≠ write-direction" as *the* cause of dead soft-steering | **OVERCLAIMED** (per §B: Persona-Vectors/Marks-Tegmark steer soft DoM dirs) | Medium — keep UNIDENTIFIED verdict; hedge the gloss; exclude layer/coef/pair first |
| 3 | Citation: "0.95 AUROC vs anti-calibrated self-report" attributed to AO line (2605.26045) | **likely MIS-ATTRIBUTED** to Yuan 2605.09502 (search-level) | Medium — fix `papers.md`/plan §10 after full-text confirm |
| 4 | "Only refusal steers / soft dirs move 0.00" | **DISCREPANCY** — Gemma sycophancy behavioral 0.05 | Low — within noise; reword to "only refusal non-trivially" |
| 5 | Persona-evil read 0.41/0.42 vs committed **0.385/0.38** | **minor OVERCOUNT** | Low — directional claim holds |
| 6 | Persona-evil **anchor 0.67 on n=6** | **WEAK** support for "pipeline valid" | Low–Med — don't lean on it |
| 7 | Missing artifacts: `17_persona_nla_read__*.csv` (cited in index); `15_…gemma3-27b.json` (documented loose end) | **DISCREPANCY (index) / documented** | Low — re-commit / run |
| 8 | "B reads as *compliant*" | **un-verifiable** from committed files (raw decodes gitignored) | Low — B≠refusal + non-degenerate is enough for the null |
| 9 | Stage-13 A/pre 1.0/0.985 vs Gate-3 0.92/1.00 | not identical (different bucket) | None — both ≈1, claim is self-validation |

## 10. What reproduces cleanly (so this isn't one-sided)
Gate-1 PASS set (exact) · refusal BoW 0.997 demotion · eval BoW 1.0 demotion · the **entire Gate-3 real
table to the decimal** · refusal echo/template-robustness · truth confabulation (by direct decode
inspection) · **RQ3 gap=0** with clean C/E baselines and degen=0 · coupling corr 0.899 · the
UNIDENTIFIED-degenerate-lever verdict · AR fidelity (internally consistent, fair "lossy" framing) · the
**persona lever to the decimal** · persona-evil dose-response (0.35 within / 0.64 pooled) and ~64% miss.

**Verification verdict:** the experiment's load-bearing conclusions — *the released NLA reads output-coupled
+ surface-salient cognition (refusal robustly, evil partially/topic-gated), is null on
decodable-but-uncoupled cognition, and shows no verbalization gap on the constructible refusal-via-prefill
vehicle* — are **well-supported by the raw artifacts.** The single substantive caveat is **#1**: the
"misses a probe-visible residual" framing is not backed by a committed standardized stage-12, and on the
committed (raw) numbers the Gemma residual points the wrong way — so the null is, on present committed
evidence, closer to "nothing strongly refusal-y survives at B for the NLA to miss" (a *more trivial* null)
than the narrative's "interesting miss." Fixing that (re-run + commit standardized `12`, or soften the
claim) is the highest-value cleanup before Exp 4.
