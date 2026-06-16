# Experiment 2 (rebuild) — Verbalization-Gap Validation of Open-Model NLAs

Clean rebuild of Experiment 2. **Do not extend `../CAA Vectors/`** (the v1 path) — build here.
The spec is the source of truth: `docs/specs/exp2_spec.md`. This README is the run order + design.

## Current state (handoff — read this first)

> **Box note (transient):** GPU work runs on a rented Vast box; `cache/` (gitignored — `03` activations,
> `05` injected vectors, AV weights ~101 GB) does **not** survive teardown. All Gate-2/3 **results are in
> git**; the caches may not be present on a fresh box. **On a new box:** `export HF_TOKEN /
> NLA_REPO_DIR=/workspace/nla_repo / OPENROUTER_API_KEY` (judge defaults to OpenRouter
> `openai/gpt-5.4-mini`; `OPENAI_API_KEY` still works) → if caches absent, re-run `03` extraction (both
> models) → `bash scripts/av_up.sh <gemma|qwen>` → continue. See `SGLANG.md` for the AV fire-up recipe.

**Where we are:** Gate 1 (cross-model) ✅ · Gate 2 offline injection ✅ + analyzed (**both models**) ·
Gate 3 real-activation ✅ decoded + **judge-confirmed** (OpenRouter `gpt-5.4-mini`, 0% err) (**both
models**) · **AV instrument validated** (controlled ground-truth checks) · cross-model comparison done ·
the gap (RQ3) not yet started.

### The headline finding (Gate 2 + Gate 3, both models) — full record in `results/gate2/FINDINGS.md`
**NLA detection tracks *output-coupling*, not decodability.** The AV reads an activation's *predicted
continuation*, so it verbalizes concepts coupled to what the model is about to say and is ~blind to
decodable-but-uncoupled features even when a probe reads them perfectly. Numbers are **real-activation
judge==2** unless noted (regex agrees on direction; judge over-scores the *absolute* level — see caveats).
- **refusal** — the robust cross-model positive: **real Gemma 0.92 / Qwen 1.00**, present ≫ absent ≈
  anchor ≈ 0 (offline 0.60/1.00 too). The *only* concept confidently read on real activations in **both**
  models. RQ1/RQ2 ✅ (offline + real, cross-model).
- **neg_sentiment** — detected + specific in both (real judge G 0.26 / Q 0.24; regex higher, G 0.64).
- **truth_value — the (revised) showcase:** probe-perfect (Gate-1 resid 1.0). On **Gemma** the AV reads it
  ∝ output-coupling — real **FALSE 0.99 ≫ TRUE 0.71** (judge; regex 0.95 ≫ 0.40), falsity predicting a
  "that's incorrect" correction. **BUT it does NOT replicate on Qwen** (true 0.87 ≈ false 0.82) and is
  partly an **AV-confabulation artifact** on Gemma true-statements (garbled affirmations). So it is
  *suggestive, single-model, confound-entangled* — **not** load-bearing. The coarse thesis is what holds.
- **corrigibility** null both (real ≈ 0, on-manifold), **eval_framing / harmful_topic_benign** null both.
  **sycophancy** is **Qwen-only on real** (Q 0.36 specific vs absent 0.02; Gemma 0.11 ≈ its own absent
  0.07) — soft reads are **model-specific**.
- **Offline numbers are NOT portable cross-model:** Qwen offline lights up *only* refusal (1.00), the rest
  near-floor — yet Qwen's *real* activations read richly; neg's offline gap (G 0.60 vs Q 0.06) vanishes on
  real (0.26/0.24). Offline went **off-manifold** for negative-baseline concepts (truth/htb/eval cos(h,h')
  0.18–0.45) — *why* the on-manifold **Gate 3 real read** is the fair test. Qwen has 4.7% real degeneracy
  (Chinese code-switching on geography/truth), Gemma 0%.

### The AV instrument is validated — not garbage (controlled checks, this session)
Fluent text is cheap, so we tested input-dependence + faithfulness against KNOWN ground truth:
- **Input-dependent:** real-present decodes scored by all axes form a **diagonal** confusion matrix
  (refusal→refusal 0.85, neg→neg 0.68, truth→truth 0.48); **100% unique** decodes (no fixed template).
- **Faithful to the injected direction:** same anchors, inject refusal → AV reads refusal (**0.82/0.91**
  med/high), inject random *same-norm* → **0.00**, no-inject → **0.00**; it preserves the anchor topic
  while overlaying the injected concept (the verbatim triple is in FINDINGS). The clean refusal-vs-random
  separation also confirms the read layer. This is **expected behavior** (the AV is RL-trained to
  reconstruct-for-continuation → output-coupling is *by design*), **but lossy** → it **confabulates fine
  detail** (entities/specifics) → trust the *coarse* read, not the specifics. The one off-diagonal (a
  truth-decode trips the *sycophancy* regex 0.43) is a **scorer** confound, not the AV. Gold-standard
  fidelity (**AR MSE/cosine**) not yet run. (Distinct from `test_confounds.py`, which validates the probe.)

### Load-bearing scorer caveats
- **The auto-scorer is unreliable in a specific way:** the gpt-5.4-mini judge reads affirmation/factual
  statements as `truth_value` (offline: regex **0** rows, judge **49**, 38 from *sycophancy*); at the
  regex level, truth-decodes trip the *sycophancy* axis 43% ("correct" keyword). The **human pilot (spec
  §3) is non-optional** before any *absolute* soft-concept number is trusted — the *direction*/asymmetry is
  robust across scorers, the *level* is not.
- **(Resolved) the Gemma Gate-3 judge wholesale-errored** (all `j_*`=`-1`, missing key) → **re-scored** via
  OpenRouter (`openai/gpt-5.4-mini`, 0% err); regex and judge now agree. `07` **aborts** on a high judge
  error-rate so this can't silently recur.

### Pipeline (all self-tested; `scripts/05–09` + root libs)
`injection.py` (DoM + exact-cosine solver + off-manifold stats; `test_injection.py`) · `flags.py`
(echo / generic_template / nla_degenerate; `test_flags.py`) · `05_inject_matrix` (offline, CPU; absolute
realized-cos doses; off-manifold flag) → `06_decode_matrix` (AV, box; resumable) → `07_score_matrix`
(regex + OpenAI judge, four-way; **`--real`** scores Gate-3 decodes) → `08_analyze_gate2` (cross-detection
matrix). `09_decode_real` (real-activation decode, box). **Not built:** `05b_steer_extract` (online — note:
steer+read the *same* read layer == offline `h+βv̂`; on-manifold needs generated-token reads or just real
acts), and `10_analyze_real` (present/absent/anchor real-activation analysis — done ad-hoc in chat, needs
scripting).

### Next (ranked)
1. **`10_analyze_real`** — script the present/absent/anchor real-activation analysis (done ad-hoc in chat).
2. **AR reconstruction fidelity** on the box — the gold-standard faithfulness scalar (AV *behavioral* checks done).
3. **Judge human pilot** (spec §3) — validate before reporting any *absolute* soft-concept number.
4. **RQ3 proper, with output-coupled vehicles:** refusal-steered (Gate-4 Track B, near-term) and
   eval-awareness organism (Exp 4). **Retire** truth_value/corrigibility/sycophancy as RQ3 vehicles —
   keep them as the output-coupling *characterization* set.
5. Spec reconciliation (`exp2_spec.md` still has the v1 gate rule + HarmBench refusal); logit-lens (Gate-1 §4).

### Online-steered generation / RQ3 — NOT done (design note so it's built right)
We have run **no** online/steered decoding. **Do not build the originally-scoped `05b` naively:** adding
`βv̂` at the read layer and reading the read layer == offline `h+βv̂` (no new information). The genuine
on-manifold **and** RQ3 test (the gap itself; exp2_spec §Gate-4 Track B) for the one output-coupled,
*suppressible* concept we have — **refusal** — is:
1. Harmful prompts → induce **coherent compliance** (minimal-coefficient steer toward compliance, or
   prefill/jailbreak) so the sampled output complies (behavioral check: no refusal/disclaimer).
2. Extract the block-41 activation at a **generated** token (the model behaving under the induction), NOT
   the prompt-last token.
3. Decode + score. **Gap** = the AV verbalizes refusal-inclination (probe-yes, output-coupled) while the
   sampled output is compliant. Baselines: output-reading, context-only LLM, and the linear probe.
The output-coupling finding makes refusal the *right* RQ3 vehicle (coupled + suppressible); truth_value /
corrigibility are **not** (there "probe beats NLA" is structural). eval-awareness via the Exp-4 organism
is the other valid vehicle.

### Gate-1 result (unchanged; cross-model)
Corrected gate (`confounds.gate_v2`, `04b_recheck_gate1.py`): length-residualized **AND** beats a BoW text
baseline **AND** not lexical-leak-flagged **AND** ≥20 groups. **5 "represented" on BOTH models:**
`truth_value`, `sycophancy`, `corrigibility`, `neg_sentiment`, `harmful_topic_benign`. Demoted: `refusal`
(AdvBench-vs-Alpaca corpus confound, BoW 0.997 — but its DoM injection is causally real per Exp 1, carried
as the anchor), `eval_framing` (framing-vs-state), small-n exploratory. A probe PASS = "decodable beyond
surface text," **not** "the model uses/verbalizes it" — which is exactly what Gate 2/3 then tested.

**Box note:** `03` only generates completions for the behavioral concepts (refusal, harmful_topic_benign);
MAXNEW=64 suffices (refusal labeled 111/120, matching Exp 1). H200 fits target + AV together (no memory
dance); on 80 GB use `av_down.sh` between extraction and AV decode.

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
| `scripts/04_run_gate1_battery.py` + `04b_recheck_gate1.py` — Gate-1 battery + corrected gate | **Done — both models run** |
| `injection.py` + `test_injection.py` — DoM, exact-cosine solver, off-manifold stats | **Done — passing** |
| `flags.py` + `test_flags.py` — echo / generic_template / nla_degenerate | **Done — passing** |
| `05_inject_matrix` → `06_decode_matrix` → `07_score_matrix` (+`--real`) → `08_analyze_gate2` | **Done — Gemma + Qwen, judge-confirmed** |
| `scripts/09_decode_real.py` — Gate-3 real-activation decode | **Done — Gemma + Qwen, judge-confirmed** |
| `05b_steer_extract` (online) · `10_analyze_real` · Gate-4 (`scripts/10`+) | **Not started** |

## Repository layout & naming

Libraries (imported) stay at the rebuild root so imports don't break; runnable pipeline stages live
in `scripts/`. Inputs, outputs, and scratch each get their own directory.

```
rebuild/
├── confounds.py paths.py audits.py flags.py injection.py   importable libs (root)
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
