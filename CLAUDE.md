# CLAUDE.md — Validating-NLAs

Guidance for Claude Code working in this repo. Read this before editing. This file is the
**always-loaded index**; deep knowledge lives in `docs/` and is read on demand (see the map below).
Keep this file lean — it is paid for on every turn.

## What this project is

We test whether Natural Language Autoencoders (NLAs) actually detect cognition a model represents
internally but does not verbalize — the **verbalization gap**, with **evaluation awareness** as the
headline case. The NLA paper claims this but could not check it against ground truth (a real model's
internal beliefs are unobservable, so "the NLA says the model thinks it's being tested" can't be
scored). We replace unobservable ground truth with **constructed** ground truth — injected directions,
prompt-constructed states, steered behavior, and a trained model organism — where we control exactly
what cognition is present, then measure whether the **released open-model NLAs** (Gemma-3-27B,
Qwen-2.5-7B) detect it. This is a validation-and-characterization program on an existing tool, not a
new method.

### Research questions (full text in `docs/specs/nla_validation_project_plan.md`)
- **RQ1 detection validity** · **RQ2 specificity** · **RQ3 the verbalization gap (the one that matters)**
  · **RQ4 transfer to finetuned organisms** · **RQ5 confabulation structure → trust calibration.**

## The program at a glance

Staged cleanest-ground-truth → most-realistic, so an early failure still leaves a reportable result.
**Repo `Experiment N/` folders map to the plan's experiments as below** (the plan also calls the
organism "Exp 2b" — in this repo that is `Experiment 4/`).

| Repo folder | What | Status |
|---|---|---|
| `Experiment 1/` | Injection dose-response: does the NLA detect an independently built **refusal** direction, specifically & dose-dependently? | **Complete** (Gemma validated, Qwen replicated). Result: `docs/specs/experiment_1_result.md` |
| `Experiment 2/` | **Verbalization-gap** validation, rebuilt in `Experiment 2/rebuild/`. | **Gate 1 cross-model; Gate 2 offline + Gate 3 real-acts done for Gemma** (full pipeline `scripts/01–09` + libs `injection.py`/`flags.py`, all self-tested). **Headline: NLA detection tracks output-coupling, not decodability** — refusal & neg_sentiment detected+specific (injected *and* real); truth_value read only where coupled (real false-stmts 0.95 ≫ true 0.40); corrigibility/sycophancy/eval ≈ null. ⚠️ gemma Gate-3 judge errored (bad key) → re-score; Qwen Gate-2/3 pending. State: `Experiment 2/rebuild/README.md` + `…/results/gate2/FINDINGS.md` |
| `Experiment 3/` | **Cross-model** feature transfer: ridge map Qwen-L20→Gemma-L41; co-firing AUROC + causal steering. Standalone. | Specced; pipeline built; needs the §9 layer correction |
| `Experiment 4/` | **Model organism** (plan's "Exp 2b"): install eval-awareness in weights (Hua recipe → Gemma) — the one design that escapes the framing-vs-state caveat. | New / scoping |

Scripts are a numbered pipeline (`01_…` → `10_…`); each stage drops artifacts the next consumes.

## The load-bearing principle (do not violate)

**A probe/AUROC is only a baseline if it is confound-controlled, and an NLA detection is only a
detection if it is echo-, template-, and degeneration-controlled.** In v1 the eval-awareness probe
read AUROC 0.990 — but it was mostly prompt length; after control it was 0.622. So:

- No single AUROC is ever reported alone. Every concept goes through the battery in
  **`Experiment 2/rebuild/confounds.py`**: `raw / length_residualized / length_only / null`, each
  with a **cluster-bootstrap CI** and **pair-level StratifiedGroupKFold baked into the CV** (matched
  pairs never straddle a split). This operationalizes the spec's five-number doctrine (raw /
  length-only / length-residualized / pair-level / shuffled-null).
- **Length control is NOT sufficient — the Gate-1 audit's load-bearing lesson:** a length-only gate
  PASSed 14/18, but a **bag-of-words text baseline** (`confounds.bow_auroc`, the lexical analog of
  `length_only`) showed ~half were just surface vocabulary (the headline `eval_framing` is BoW-on-text
  **1.000**; refusal's AdvBench-vs-Alpaca two-pool is **0.997** = corpus identity). A concept counts as
  "represented" only if the activation probe **clearly beats BoW-on-text**, not merely length.
- Verdict (**pre-registration v2**, `confounds.gate_v2`; the original length-only gate stays in
  `probe_battery`): requires `length_residualized.ci_lo > max(REPRESENTED_FLOOR, null.ci_hi)` **AND**
  clearly beats the **BoW text baseline** **AND** not lexical-leak-flagged **AND** **≥ `MIN_GROUPS` (20)
  groups** (≤12 groups → ~10% false PASS on noise). Re-gate cached results with
  `scripts/04b_recheck_gate1.py`. We still do **not** require resid > length_only.
- Decisions compare CI **bounds**, not point estimates (n≈60–120 is noisy).
- **NLA detections** report four ways — raw / excluding echo / excluding `generic_template` /
  excluding `nla_degenerate`. `generic_template` ≠ echo (a generic safety template inflates refusal
  for *every* concept without echoing a word).
- **Per-activation co-registration** for any gap claim: probe-yes AND NLA-no on the *same* `h`, never
  population-vs-population.
- When a confound is collinear with the label, the effect is **not identifiable** — refuse to certify.
- **Injection ≠ real activations** (Qwen dissociation: the AV read *offline*-injected Gemma but was
  NULL on offline-injected Qwen, while positive on *real* Qwen). Three vector sources, three meanings:
  **offline** `h'=anchor+βv̂` (off-manifold, cheap → Gate-2 specificity) · **online-steered** (a forward-pass
  hook adds `βv̂` → on-manifold real state) · **real natural** (Gate 3). Gate 2 should run offline **and**
  online-steered and report the dissociation; the gap (RQ3) lives in steered+real, not offline.
- **Validate the instrument before trusting it:** `test_confounds.py` must pass on synthetic ground
  truth (clean / length / mixed / collinear regimes) before the battery touches real activations.
- **NLA detection tracks output-coupling, not decodability** (Gate-2/3 finding, this session): the AV
  reads an activation's *predicted continuation*, so it surfaces output-coupled concepts (refusal,
  neg_sentiment, false-statement corrections) and is ~blind to decodable-but-uncoupled ones
  (corrigibility, true-statement truth-value) even when a probe reads them perfectly. So "probe beats
  NLA" is partly **structural** → the real RQ3 test needs **output-coupled** cognition (eval-awareness,
  steered-refusal), not truth/corrigibility. (Also: online-steer that reads the *same* layer it injects
  == offline `h+βv̂`; on-manifold needs generated-token reads or **real activations**.)
- **Auto-scoring manufactures false positives** — the gpt-5.4-mini judge reads affirmation/facts as
  `truth_value` (regex 0 rows, judge 49, 38 from sycophancy). Human-validate the scorer (spec §3) before
  any soft-concept number; and a key-failed judge writes `-1` that reads as a fake "null" (`07` now
  aborts on high judge-error-rate). Detail: `Experiment 2/rebuild/results/gate2/FINDINGS.md`.

## Where knowledge lives (read on demand)

- `docs/specs/` — the **spec docs are the source of truth** for what we're building:
  `experiment_1_spec.md`, `exp2_spec.md`, `exp3_spec.md`, `nla_validation_project_plan.md`,
  plus `experiment_1_result.md` (+ `.pdf`).
- `docs/references/papers.md` — NLA + Activation Oracle **materials & methods**, and how we use each.
- `docs/references/nla-infrastructure.md` — **AV vs AR**, SGLang flags, checkpoint table, Gemma
  patch, injection/dose conventions, scorer gotchas. Read before any GPU stage.
- `docs/references/literature.md` — full reference list, each with why it matters to us.
- `docs/references/known-corrections.md` — **load-bearing fixes** (Gemma has 62 layers not 46;
  AdvBench/HarmBench reconciliation; no SAE position exclusion in the NLA read path). Check before
  trusting derived figures in older docs.
- `Experiment 2/rebuild/README.md` — **current Exp 2 state**, run order, Gate-1 results + open
  decisions; `Experiment 2/rebuild/SGLANG.md` — the NLA AV SGLang fire-up recipe (`scripts/av_up.sh`).

## Run / dev (CPU work — no GPU needed)

A gitignored venv at repo root has the CPU deps (numpy/sklearn/scipy):

```bash
for t in test_confounds test_paths test_audits test_concepts test_sources; do .venv/bin/python "Experiment 2/rebuild/$t.py"; done
```

If the venv is missing (fresh container): `uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python numpy scikit-learn scipy pandas pyarrow pyyaml requests`.

## GPU work runs on a Vast.ai box, not here

This container has no GPU. Forward passes (Gemma/Qwen extraction), the SGLang AV server, and NLA
decoding run on a rented GPU box; results are pushed back and analyzed here.

- **Author code here** (full context). Commit + push.
- **Run GPU stages on the box**: clone the branch; in `Experiment 2/rebuild/` run
  `pip install -r requirements.txt` (+ `transformers accelerate tqdm`), set `HF_TOKEN`, then
  `python scripts/01_verify_env.py --model gemma` and the numbered pipeline (`03` extract →
  `04`/`04b` battery; Gate-2 AV via `scripts/av_up.sh <gemma|qwen>`). `03` only generates for the
  behavioral concepts (refusal, harmful_topic_benign); MAXNEW=64 is sufficient. Activations cached **fp32**
  (large-norm outlier dims overflow fp16).
- **Push small structured results** (JSON/CSV/JSONL) back — never weights or caches (`cache/`,
  `workspace/`, `.venv` are gitignored).

## NLA serving — AV vs AR (the common confusion). Full detail in `docs/references/nla-infrastructure.md`.

- **AV** (`vector → text`): an **SGLang server** (`launch_sglang_server.sh`). Mandatory
  `--disable-radix-cache`; Gemma-3 also needs `--attention-backend fa3` + the `input_embeds` patch.
- **AR** (`text → vector`): **NOT a server.** Pure PyTorch, in-process via
  `NLACritic(ckpt_dir, device="cuda:0")` from `nla_inference.py`; produces an MSE/cosine fidelity only.
- The vendored `nla_inference.py` is byte-identical to upstream `kitft/nla-inference`. Gemma-3 needs the
  multimodal `input_embeds` patch from `kitft/natural_language_autoencoders`; Qwen does not.

## Engineering gotchas (each fails silently/plausibly)

- **Layer convention:** Gemma block-41 = `hidden_states[42]`; Qwen block-20 = `hidden_states[21]`.
  Verify behaviorally — off-by-one yields plausible garbage. (Gemma has 62 layers, not 46.)
- **Gemma embed_scale = √5376 ≈ 73.32**; **injection_scale** from `nla_meta.yaml` (do not hardcode);
  wrong values → AV verbalizes the marker char U+321C.
- **Tokenization:** two-step `apply_chat_template(tokenize=False)` → `tokenizer(rendered, add_special_tokens=False)`
  (avoid double-BOS and NFKC eating the U+321C injection char).
- **Dose = realized `cos(h', v̂)`**, never raw β; solve β analytically per anchor; log `cos(h,h')`.
- **Memory:** Gemma (~55 GB) + SGLang AV can't coexist on 80 GB — `pkill -f sglang` before loading the
  target model, relaunch after.
- **OpenAI scorer:** `chat.completions.create` (not `responses.create`); `max_completion_tokens` (not
  `max_tokens`); hashed `safety_identifier`; semaphore-limit.

## Conventions

- Develop on the assigned feature branch; push there; open a draft PR.
- Keep code and results separate; results are small and diffable. Spec docs in `docs/specs/` are
  reference truth — don't silently edit them to match code; reconcile deliberately.
- Don't extend the old `Experiment 2/CAA Vectors/src/`; build in `Experiment 2/rebuild/`.
