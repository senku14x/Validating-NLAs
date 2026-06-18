# Experiment_2 From Scratch — Qwen 2.5 7B + GLP On-Manifold Steering

End-to-end run order for the **rebuild** pipeline on **Qwen only**, with GLP-projected online steering at the end. All commands assume the repo root:

```bash
cd /path/to/Validating-Natural-Language-Autoencoders-as-Detectors-of-Unverbalized-Cognition
PY=".venv/bin/python"
```

Use the repo `.venv` for every Python command below. GPU stages are intended for a CUDA box (e.g. Vast.ai H100/H200).

**Read layer convention (Qwen):** decoder block `20` → `hidden_states[21]`, hidden size `3584`.

**GLP constraint:** released GLP weights are Llama-only. You must train (or provide) a Qwen GLP whose `config.yaml` has `d_input: 3584` and `tracedict_config.layers: [20]`. Llama checkpoints are rejected by the steering script.

---

## Phase 0 — Environment

### 0.1 Create the rebuild virtualenv

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r Experiment_2/rebuild/requirements.txt
uv pip install --python .venv/bin/python \
  "transformers==4.57.1" accelerate tqdm huggingface_hub torch
```

For GLP training/steering, also install the GLP package and its deps (from `Experiment_2/generative_latent_prior`):

```bash
uv pip install --python .venv/bin/python \
  diffusers omegaconf safetensors einops
$PY -m pip install -e Experiment_2/generative_latent_prior
```

For Gate 2 AV decode, follow `SGLANG.md` and use `bash Experiment_2/rebuild/scripts/av_up.sh qwen` when you reach Phase 4.

### 0.2 Export secrets (GPU box)

```bash
export HF_TOKEN=...              # Hugging Face (Qwen base + NLA AV/AR repos)
export OPENROUTER_API_KEY=...    # judge in 07_score_matrix (or OPENAI_API_KEY)
export NLA_REPO_DIR=/path/to/natural_language_autoencoders   # for av_up.sh / SGLang
```

### 0.3 CPU self-tests (must pass before GPU work)

```bash
$PY Experiment_2/rebuild/test_confounds.py
$PY Experiment_2/rebuild/test_paths.py
$PY Experiment_2/rebuild/test_injection.py
$PY Experiment_2/rebuild/test_qwen_glp.py
```

---

## Phase 1 — Gate 0: Implementation sanity

```bash
$PY Experiment_2/rebuild/scripts/01_verify_env.py --model qwen
```

Checks GPU, HF access, NLA config, cosine solver, and `data/concept_pairs.parquet`.

---

## Phase 2 — Gate 1: Concept pairs + activation extraction + probe battery

### 2.1 Build concept pairs (CPU)

```bash
$PY Experiment_2/rebuild/scripts/02_build_concept_pairs.py
```

Output: `Experiment_2/rebuild/data/concept_pairs.parquet`

### 2.2 Extract Qwen read-layer activations (GPU)

Kill any running SGLang AV server first if VRAM is tight (`bash Experiment_2/rebuild/scripts/av_down.sh`).

```bash
$PY Experiment_2/rebuild/scripts/03_extract_for_battery.py --model qwen
```

Outputs under `Experiment_2/rebuild/cache/`:

- `03_extract_for_battery__qwen2.5-7b__<concept>.npz` — per-concept `(X, y, lengths, groups)`
- `03_extract_for_battery__qwen2.5-7b__anchor.npz` — neutral anchors for injection/steering

### 2.3 Run Gate 1 probe battery (CPU)

```bash
$PY Experiment_2/rebuild/scripts/04_run_gate1_battery.py --model qwen
$PY Experiment_2/rebuild/scripts/04b_recheck_gate1.py --model qwen
```

Outputs: `Experiment_2/rebuild/results/gate1/04_run_gate1_battery__qwen2.5-7b__*.json`

Only concepts that pass the corrected gate should drive downstream NLA and steering work.

---

## Phase 3 — GLP: Build manifold model for Qwen (GPU + training)

This phase is **independent of Gates 2–3** but **required before GLP steering**. Do it in parallel with Phase 4 if you have two GPUs, or sequentially on one box.

### 3.1 Cache Qwen block-20 activations for GLP training

Use a **broad text corpus** (not just concept pairs) for a useful manifold:

```bash
$PY Experiment_2/rebuild/scripts/16_glp_qwen_cache.py \
  --source text_file \
  --input-texts /path/to/texts.txt \
  --all-tokens \
  --chat-template \
  --batch-size 8
```

Smoke/plumbing only (not for science):

```bash
$PY Experiment_2/rebuild/scripts/16_glp_qwen_cache.py \
  --source concept_pairs \
  --limit 64 \
  --chat-template \
  --batch-size 4
```

Default output: `Experiment_2/rebuild/cache/glp/qwen2_5_7b-layer20-static/`

### 3.2 Train the Qwen GLP

```bash
cd Experiment_2/generative_latent_prior
$PY glp_train.py config=configs/train_qwen2_5_7b_static.yaml
cd ../..
```

Default checkpoint directory:

`Experiment_2/rebuild/workspace/glp_runs/glp-qwen2_5_7b-d6-static/`

Expected files: `config.yaml`, `rep_statistics.pt`, `final.safetensors`

---

## Phase 4 — Gate 2: Offline injection + NLA decode + scoring

### 4.1 Build diff-of-means directions and offline injection matrix (CPU)

```bash
$PY Experiment_2/rebuild/scripts/05_inject_matrix.py --model qwen
```

Outputs:

- `cache/05_inject_matrix__qwen2.5-7b__all.npy`
- `cache/05_inject_matrix__qwen2.5-7b__all.parquet`
- `cache/05_inject_matrix__qwen2.5-7b__all__directions.npz` ← used by GLP steering

### 4.2 Launch Qwen NLA AV server (GPU)

```bash
bash Experiment_2/rebuild/scripts/av_up.sh qwen
```

See `SGLANG.md` for troubleshooting.

### 4.3 Decode injected vectors through the AV (GPU)

```bash
$PY Experiment_2/rebuild/scripts/06_decode_matrix.py --model qwen
```

### 4.4 Score decodes — regex + judge (CPU/API)

```bash
$PY Experiment_2/rebuild/scripts/07_score_matrix.py --model qwen
```

Requires `OPENROUTER_API_KEY` or `OPENAI_API_KEY`.

### 4.5 Analyze cross-detection matrix (CPU)

```bash
$PY Experiment_2/rebuild/scripts/08_analyze_gate2.py --model qwen
```

Output: `Experiment_2/rebuild/results/gate2/08_analyze_gate2__qwen2.5-7b.csv`

---

## Phase 5 — Gate 3: Real-activation NLA reads (GPU)

Stop the AV server if you need VRAM for extraction, then bring it back for decode.

```bash
$PY Experiment_2/rebuild/scripts/09_decode_real.py --model qwen
$PY Experiment_2/rebuild/scripts/07_score_matrix.py --model qwen --real
```

Real-activation decodes are the on-manifold baseline (no offline `h + βv` injection).

---

## Phase 6 — Gate 4: GLP on-manifold online steering (GPU)

Compare **raw** hook steering (`h + βv`) vs **GLP-projected** steering (`GLP(h + βv)`) on the same Qwen directions and doses.

Prerequisites:

- Phase 2.2 (`03` caches + anchors)
- Phase 4.1 (`05` directions snapshot)
- Phase 3.2 (trained Qwen GLP checkpoint)

```bash
$PY Experiment_2/rebuild/scripts/16_glp_steer_qwen.py \
  --mode both \
  --glp-weights Experiment_2/rebuild/workspace/glp_runs/glp-qwen2_5_7b-d6-static \
  --glp-checkpoint final \
  --concepts refusal,neg_sentiment,sycophancy \
  --doses 0.40,0.55,0.70 \
  --n-anchors 20
```

Modes:

- `--mode raw` — off-manifold addition only (baseline)
- `--mode glp` — GLP SDEdit projection after addition
- `--mode both` — run both and write paired results

Outputs:

- `results/gate4/16_glp_steer_qwen__qwen2.5-7b__both.json`
- `results/gate4/16_glp_steer_qwen__qwen2.5-7b__both.csv`
- `workspace/glp_steering/16_glp_steer_qwen_examples__qwen2.5-7b__both.jsonl`

Interpretation: compare `behavioral_delta`, `steered_rate`, and example continuations between `raw` and `glp` rows for the same concept/dose. GLP projection should keep activations closer to the natural manifold while preserving (or changing) behavioral effect.

---

## Phase 7 — Optional follow-ups

These are part of the broader Experiment_2 rebuild but not required for the minimal GLP steering path:

| Step | Script | Purpose |
|------|--------|---------|
| Coupling vs salience de-risk | `14_coupling_score.py --model qwen` | Raw online steer behavioral test (no GLP) |
| AR fidelity | `15_ar_fidelity.py --model qwen` | AV reconstruction gold standard |
| Track B gap | `11` → `11c` → `12` → `13` | RQ3 prefill/compliance gap analysis |

---

## Quick reference — script order

```
01_verify_env.py --model qwen
02_build_concept_pairs.py
03_extract_for_battery.py --model qwen
04_run_gate1_battery.py --model qwen
04b_recheck_gate1.py --model qwen

16_glp_qwen_cache.py                    # GLP training data
glp_train.py (generative_latent_prior)  # Qwen GLP checkpoint

05_inject_matrix.py --model qwen
av_up.sh qwen
06_decode_matrix.py --model qwen
07_score_matrix.py --model qwen
08_analyze_gate2.py --model qwen

09_decode_real.py --model qwen
07_score_matrix.py --model qwen --real

16_glp_steer_qwen.py --mode both --glp-weights ...   # GLP on-manifold steering
```

---

## Box notes

- **`cache/` is gitignored.** On a fresh box, rerun Phases 2–6; committed artifacts live under `results/`.
- **VRAM:** on 80 GB, do not run Qwen extraction and the SGLang AV server at the same time. Use `av_down.sh` between stages. H200 can often hold both.
- **Layer off-by-one:** Qwen block `20` output equals `hidden_states[21]`. Both the rebuild pipeline and the Qwen GLP config use this layer.
- **More GLP detail:** see `GLP_QWEN.md` for parameter tuning (`--glp-u`, `--glp-timesteps`) and checkpoint validation rules.
