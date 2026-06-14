# Experiment 2 (rebuild) — Verbalization-Gap Validation of Open-Model NLAs

Clean rebuild of Experiment 2. **Do not extend `../CAA Vectors/`** (the v1 path) — build here.
The spec is the source of truth: `docs/specs/exp2_spec.md`. This README is the run order + design.

## Current state (handoff — read this first)

> **Live instance (transient — as of 2026-06-14):** a Vast **H200** box is currently **running, NOT
> destroyed**. The `03` activation cache is intact for **both** Gemma-3-27B and Qwen-2.5-7B under
> `cache/` (gitignored) — so Gate-2 work (`05_smoke_decode.py`, injection) can use it **without
> re-running `03`**. Bring the AV server up with `scripts/av_up.sh <gemma|qwen>`. Delete/refresh this
> note when the instance is torn down.

**Gate 0–1: complete, audited, cross-model.** Pipeline `scripts/01–04` + `04b`, run on **Gemma-3-27B and
Qwen-2.5-7B**; results in `results/gate1/`.

- The v1 gate (length-only) PASSed 14/18 — but a 3-agent audit showed it was **lexically blind**: ~half
  were just surface vocabulary (the headline `eval_framing` is bag-of-words-on-text AUROC **1.000**), and
  the cluster-bootstrap CI gives ~10% false PASS below ~12 groups.
- **Corrected gate** (`confounds.gate_v2`, run by `scripts/04b_recheck_gate1.py`) adds a **BoW text
  baseline** + lexical-leak + **min-groups (≥20)** rules. Result: **5 "represented" on BOTH models**
  (18/18 v2 agreement): `truth_value` (cleanest — BoW at chance), `sycophancy`/`corrigibility` (real but
  *answer-binding*, narrow), `neg_sentiment`, `harmful_topic_benign`.
- **Demoted:** `refusal` → WEAK (AdvBench-vs-Alpaca is a *corpus* confound, BoW 0.997; refusal is still
  causally real per Exp 1 — it needs a **topic-matched contrast** for a clean probe baseline).
  `eval_framing` → WEAK (framing-vs-state: the probe reads the note's words, not an eval-*state* — a
  Gate-4 / organism question). Small-n exploratory concepts → WEAK pending more pairs.
- Standing caveat: a probe PASS = "decodable beyond surface text," **not** "the model *uses* it."
  Decodability-vs-use is the logit-lens / causal / NLA-verbalization question (Gates 2–4).

**Next: Gate 2 — the verbalization test.** Inject each surviving direction (+ the refusal anchor) and ask
whether the NLA AV *verbalizes* it, four-way scored. Open decisions:
1. **Injection mode** — offline `h'=anchor+βv̂` (cheap, off-manifold → Gate-2 specificity) vs **online-
   steered** (forward-pass hook → on-manifold real state). The Qwen dissociation (offline NULL, real
   positive) means offline alone can mislead — build both, report the dissociation. Planned scripts:
   `05_inject_matrix` (offline, CPU) + `05b_steer_extract` (online, box) → `06_decode_matrix` (AV via
   `scripts/av_up.sh`) → `flags.py` + `07_score_matrix` (four-way: raw / exc-echo / exc-template / exc-degen).
2. **Refusal** — rebuild with a topic-matched contrast (`harmful_topic_benign` is the disentangling tool);
   the injection DoM direction is unaffected and still advances.
3. **Scoring** — `07` uses the OpenAI judge (`chat.completions.create`); needs `OPENAI_API_KEY` on the box,
   or run regex-first.
4. **n-expansion** — constructed controls are at curated base (16–40 pairs); expand borderline ones to ~90
   (LLM, audit-gated) before trusting their verdicts.
5. **Spec reconciliation** — `exp2_spec.md` still states the v1 PASS rule + a separate pair-level number;
   reconcile to `gate_v2` deliberately.

**Box engineering notes:** `03` only generates completions for the behavioral concepts (refusal,
harmful_topic_benign); MAXNEW=64 is enough (refusal labeled 111/120, matching Exp 1). Newer transformers
returns a `BatchEncoding` from `apply_chat_template` (handled). H200 (141 GB) fits target + AV together →
no memory dance; on 80 GB use `av_down.sh` between extraction and AV decode.

## Why a rebuild

v1 reported an eval-awareness probe AUROC of **0.990 that was mostly prompt length** (0.622 after
control), and used **unmatched** NLA-paper eval/deploy system prompts. The rebuild's fix is the
confound battery (`confounds.py`) plus length/lexically-matched concept construction. **No single
AUROC is ever reported alone**, and the verdict is pre-registered (frozen in `confounds.py`):

> a concept is "represented" iff `length_residualized.ci_lo > max(REPRESENTED_FLOOR=0.60, null.ci_hi)`.

## The claim (per-activation, not population-vs-population)

For an activation `h`, the verbalization gap holds iff a **confound-controlled probe detects** the
concept on `h` **AND** the **echo/template/degeneration-controlled NLA does not verbalize** it on the
*same* `h`. Three claims must independently survive: RQ1 detection validity, RQ2 specificity, RQ3 the
gap (the headline). See `docs/specs/exp2_spec.md` §0.

## Status

| Component | State |
|---|---|
| `confounds.py` — pre-registered probe battery (the instrument) | **Done** |
| `test_confounds.py` — synthetic self-test (clean/length/mixed/collinear) | **Done — passing** |
| `paths.py` + `test_paths.py` — canonical paths & file-naming convention | **Done — passing** |
| `configs/concepts.yaml` (18 concepts + 4 controls) + `test_concepts.py` | **Done — passing** |
| `audits.py` + `test_audits.py` — Gate-1 pre-extraction audits | **Done — passing** |
| `concept_sources.py` + `test_sources.py` — offline pair construction (14 concepts) | **Done — passing** |
| `scripts/01_verify_env.py` — Gate-0 sanity (cosine solver verified; GPU/HF/NLA box-only) | **Done — CPU checks pass** |
| `scripts/02_build_concept_pairs.py` → `data/concept_pairs.parquet` (1670 rows) | **Done — runs** |
| `scripts/03_extract_for_battery.py` — activations + behavioral labels (GPU box) | **Authored — box-only** |
| `scripts/04_run_gate1_battery.py` — per-concept battery + PASS/WEAK/FAIL/DROP | **Done — self-test passes** |
| Gate 2–4 pipeline (`scripts/05`–`10`) | Not started |

## Repository layout & naming

Libraries (imported) stay at the rebuild root so imports don't break; runnable pipeline stages live
in `scripts/`. Inputs, outputs, and scratch each get their own directory.

```
rebuild/
├── confounds.py  paths.py  audits.py*  flags.py*    importable libs (root)   (* planned)
├── test_*.py                                         CPU self-tests
├── scripts/        NN_*.py   numbered pipeline stages (entry points)
├── configs/        concepts.yaml  (hand-authored config)
├── data/        ✓  small constructed datasets (concept_pairs.parquet)    [committed]
├── results/     ✓  analysis outputs, one subdir per gate                 [committed]
│   └── gate0/ gate1/ gate2/ gate3/ gate4/
├── cache/       ✗  large artifacts: activations .npz, raw decodes        [gitignored]
└── workspace/   ✗  scratch                                               [gitignored]
```

**Script names:** `scripts/<NN>_<verb_noun>.py` — `NN` is a monotonic stage number (the gate is shown
in the run order below). E.g. `04_run_gate1_battery.py`.

**Result / cache file names — built only via `paths.py`, never by hand:**

```
<stage>__<model>[__<concept>][__<variant>].<ext>
  stage   = the producing script's basename, via stage_of(__file__) — a file names its own maker
  model   = gemma3-27b | qwen2.5-7b
  concept = a key from concepts.yaml (verbatim), or `all` for cross-concept summaries
  variant = optional: raw | exc-echo | exc-template | exc-degen (the 4 NLA arms), a dose, …
fields joined by '__'; within a field lowercase [a-z0-9_-].
```

Examples — `results/gate1/04_run_gate1_battery__gemma3-27b__refusal.json`,
`results/gate2/07_score_matrix__qwen2.5-7b__refusal__exc-echo.jsonl`,
`cache/03_extract_for_battery__gemma3-27b__refusal.npz`.

In a `scripts/NN_*.py` stage:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # reach root libs
from paths import result_path, stage_of
from confounds import probe_battery
out = result_path("gate1", stage_of(__file__), "gemma", concept="refusal")
```

## Run order (numbered pipeline; mirrors Experiment 1)

CPU stages author+run here; **GPU stages run on the Vast.ai box** (no GPU in the dev container).
Each gate's exit criteria gate the next (spec §2).

```
Gate 0  implementation sanity      scripts/01_verify_env.py                 [box]
Gate 1  vector validity (cheap)    configs/concepts.yaml
                                   scripts/02_build_concept_pairs.py        [here]
                                   audits.py  (root lib)                    [here]
                                   scripts/03_extract_for_battery.py        [box]
                                   scripts/04_run_gate1_battery.py          [here]
Gate 2  NLA injection specificity  scripts/05_inject_matrix.py             [here]
                                   scripts/06_decode_matrix.py             [box, SGLang AV]
                                   flags.py (root lib) +
                                   scripts/07_score_matrix.py              [here, OpenAI]
                                   scripts/08_analyze_gate2.py             [here]
Gate 3  real-activation reading    scripts/09_real_activation_reads.py     [box + here]
Gate 4  verbalization-gap tests    scripts/10_final_gap_analysis.py        [box + here]
```

### Quick start (CPU dev box)
```bash
# from repo root
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python numpy scikit-learn scipy
.venv/bin/python "Experiment 2/rebuild/test_confounds.py"   # must print ALL CHECKS PASSED
.venv/bin/python "Experiment 2/rebuild/test_paths.py"       # naming convention self-test
```

### Gate 1 detail (the load-bearing, cheap gate)
Probe-only (activation extraction + AUROC battery; **no NLA decode, no LLM judge**), so a wide
candidate net is cheap. Each direction carries the **five-number battery** before any NLA work:
raw / length-only / length-residualized / pair-level GroupKFold / shuffled-null. Pre-extraction audit
per concept: length balance (`pct_gap<25%`), lexical-leak check, framing length-match to identical
token count. Every concept exits with a label: **PASS / WEAK / FAIL / DROP** (spec §5). Only PASS
vectors reach Gates 2–4.

### Gate 2 detail (specificity diagnostic, NOT the gap)
DoM injection direction (independent of the NLA) into neutral anchors; **dose = realized cosine**,
solved analytically per anchor (exact to float precision); log realized cos, baseline cos, `cos(h,h')`,
norms, `delta_norm/‖h‖`; off-manifold gate. Per-output flags: `echo_*`, **`generic_template`**
(distinct from echo — a generic safety template inflates refusal for *every* concept without echoing a
word), `nla_degenerate`, `n_unique`. **Every detection rate reported four ways**: raw / excluding echo /
excluding generic_template / excluding degenerate. Run the same injections on Qwen for the
**dissociation track** (injection ≠ real activations).

### Gate 4 detail (the science)
Per-activation co-registration (probe-yes ∧ NLA-no on the same `h`). Track A eval-awareness (matched
framing — state the **framing-vs-state caveat**: "represents the eval-framing context", not "knows it
is being evaluated"; Exp 2b is the escape route). Track B refusal (minimal-coefficient steering, probe
independent of `v_refusal`, behavioral compliance check; Phase-1 bucket report A/B/B′/C/D/E).
**Gap recovery is the headline number**, against output-reading + context-only + linear-probe baselines.

## Concept set (full 18-concept net; spec §1)

Cast wide, let the gates cull. Refusal (anchor) + controls (apology_polite_refusal,
harmful_topic_benign, safe_completion, eval_framing_matched) + positive sanity (style_emoji, json_format,
bullet_list, code_block, language_switch) + candidates (neg_sentiment, uncertainty, sycophancy,
corrigibility, truth_value) + exploratory (correction_acceptance, sandbagging) + negative controls
(random / orthogonalized-refusal / length-direction / topic-only-harmfulness).

**Pre-registered primary test:** does `refusal` survive after `apology_polite_refusal`,
`harmful_topic_benign`, and `neg_sentiment` controls?

## Decisions (this rebuild)

- **Refusal anchor = Arditi / Exp-1 setup**: AdvBench harmful vs **Alpaca** harmless, behaviorally
  labeled (refused vs complied), cross-pool diff-of-means — identical to Exp 1 (AUROC 0.999) and to
  Arditi et al. The v1 **matched-benign-rewrite weakened the direction and is dropped**; topic/length
  confounds are controlled at *evaluation* (battery residualization + `harmful_topic_benign` /
  `neg_sentiment` / `length_direction` controls), not by hobbling the direction. ⚠️ Diverges from
  `exp2_spec.md` §1 (HarmBench); tracked in `docs/references/known-corrections.md`.
- **Scorer = `chat.completions.create` + `max_completion_tokens` + hashed `safety_identifier`**
  (semaphore-limited), per `exp2_spec.md` §8 / `CLAUDE.md`. Note: Experiment 1's
  `Gemma 3 27B/src/10_score_responses.py` uses the **Responses API** — the rebuild deliberately follows
  the spec instead.
- **`eval_framing_matched`** uses length-matched, lexically-neutralized eval/deploy framing — **not**
  v1's unmatched NLA-paper system prompts. This is the core v1 fix.
- **Representation claim = regularized logistic regression** via `probe_battery` (nested CV); **DoM is
  used only for the injection direction**. Never reimplement AUROC — import from `confounds.py`.

## Engineering constants (fail silently if wrong — see `docs/references/nla-infrastructure.md`)

- Gemma-3-27B: **62 layers** (not 46); read at `hidden_states[42]` (block 41), d=5376, ~66% depth.
  Qwen-2.5-7B: `hidden_states[21]` (block 20), d=3584. Off-by-one yields plausible garbage.
- Gemma `embed_scale = √5376 ≈ 73.32`; `injection_scale` from `nla_meta.yaml` (**never hardcode**);
  injection marker U+321C; two-step tokenization (avoid double-BOS / NFKC eating the marker).
- SGLang AV: `--disable-radix-cache` (mandatory) + `--attention-backend fa3` (Gemma); memory dance
  (`pkill -f sglang` before loading the target model; relaunch after). **No SAE position exclusion** in
  the NLA read path. Activations cached **fp32**.

## Reuse

- `confounds.py` — `probe_battery`, `residualize`, `AUROC`, `summarize`, `battery_to_dict`.
- `paths.py` — `result_path`, `cache_path`, `data_path`, `config_path`, `stage_of`, `model_slug`.
- `../CAA Vectors/src/01_build_concept_pairs.py` — concept-construction + length-audit **template**
  (port, do not extend); `../CAA Vectors/data/concept_pairs.parquet` — reference data.
- `../../Experiment 1/{Gemma 3 27B,qwen-2_5-7b}/src/` — `01_verify_env`, `05_extract`,
  `07_build_direction` (DoM), `08_inject` (exact-cosine solver), `09_decode` (NLAClient),
  `10_regex_score`, `10c_analyze`.
- `../../Experiment 1/qwen-2_5-7b/nla-inference/nla_inference.py` — `NLAClient` (AV server),
  `NLACritic` (AR, in-process), `load_nla_config`, `resolve_embed_scale`, `inject_at_marked_positions`.
