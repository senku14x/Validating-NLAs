# CLAUDE.md — Validating-NLAs

Guidance for Claude Code working in this repo. Read this before editing.

## What this project is

We test whether Natural Language Autoencoders (NLAs) actually detect cognition a
model represents internally but does not verbalize — the "verbalization gap." The
NLA paper could not check this against ground truth (a real model's internal
beliefs are unobservable). We replace unobservable ground truth with *constructed*
ground truth: model organisms / steering / prompting where we control exactly what
cognition is present, then measure whether the NLA detects it.

## Repo layout

- `Experiment 1/` — NLA detects injected refusal, concept-specifically (Gemma-3-27B, Qwen-2.5-7B).
- `Experiment 2/` — CAA directions + eval-awareness + refusal verbalization gap. **Being rebuilt** under confound control; see `Experiment 2/rebuild/`. The old `CAA Vectors/src/` is kept as reference, not extended.
- `Experiment 3/`, `Experiment 4/` — see their own READMEs.
- Scripts are a numbered pipeline (`01_…` → `10_…`); each stage drops artifacts the next consumes.

## The load-bearing principle (do not violate)

**A probe/AUROC baseline is only a baseline if it is confound-controlled.** In v1,
the eval-awareness probe read AUROC 0.990 — but it was mostly prompt length; after
control it was 0.622. So:

- No single AUROC is ever reported alone. Every concept goes through the
  four-number battery in `Experiment 2/rebuild/confounds.py`:
  `raw / length_residualized / length_only / null`, each with a cluster-bootstrap CI.
- The verdict is pre-registered (frozen in `confounds.py`): a concept is
  "represented" iff `length_residualized.ci_lo > max(floor, null.ci_hi)`.
- Decisions compare CI **bounds**, not point estimates (n≈60–120 is noisy).
- NLA "detections" need an **echo control** — the NLA can parrot prompt words
  ("test"/"simulated") instead of reading a state. Report detection excluding echo.
- When a confound is collinear with the label, the effect is **not identifiable** —
  refuse to certify it (watch `length_inflation`).

## Run / dev (CPU work — no GPU needed)

A gitignored venv at repo root has the CPU deps (numpy/sklearn/scipy):

```bash
.venv/bin/python "Experiment 2/rebuild/test_confounds.py"   # battery self-test
```

If the venv is missing (fresh container): `uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python numpy scikit-learn scipy`.

The confound battery is pure CPU. Validate the *instrument* on synthetic ground
truth (`test_confounds.py`) before trusting it on real activations.

## GPU work runs on a Vast.ai box, not here

This container has no GPU. Forward passes (Gemma/Qwen extraction), the SGLang AV
server, and NLA decoding run on a rented GPU box; results are pushed back and
analyzed here. Division of labor:

- **Author code here** (this session has full context). Commit + push.
- **Run GPU stages on the box**: clone the branch, `uv sync` in the experiment
  dir, `python src/01_verify_env.py`, then run the pipeline.
- **Push small structured results** (JSON/CSV/JSONL) back — never weights or caches
  (`cache/`, `workspace/`, `.venv` are gitignored).

## NLA serving — AV vs AR (a common confusion)

- **AV** (activation verbalizer, `vector → text`): served as an **SGLang server**
  (`launch_sglang_server.sh`). Mandatory flag: `--disable-radix-cache`.
- **AR** (activation reconstructor, `text → vector`): **NOT a server.** Pure
  PyTorch, loaded in-process via `NLACritic(ckpt_dir, device="cuda:0")` from
  `nla_inference.py`. It only produces an MSE/cosine fidelity score.

The vendored `nla_inference.py` is byte-identical to upstream `kitft/nla-inference`.
Gemma-3 additionally needs the multimodal `input_embeds` patch from the training
repo (`kitft/natural_language_autoencoders`); Qwen does not.

## Engineering gotchas (from the v1 debrief)

- OpenAI: use `chat.completions.create` (not `responses.create`); `max_completion_tokens` (not `max_tokens`); pass a hashed `safety_identifier`.
- Tokenization: two-step `apply_chat_template(tokenize=False)` → `tokenizer(rendered, add_special_tokens=False)` to avoid double-BOS and NFKC eating the injection char.
- Gemma (55GB) + SGLang AV can't coexist on 80GB: `pkill -f sglang` before loading the target model, relaunch after.
- Layer convention: Gemma L41 = `hidden_states[42]`; Qwen = L20. Verified.

## Conventions

- Develop on the assigned feature branch; push there; open a draft PR.
- Keep code and results separate; results are small and diffable.
- Don't extend the old `CAA Vectors/src/`; build in `Experiment 2/rebuild/`.
