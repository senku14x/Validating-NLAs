# Experiment 3 Activation Co-Firing Setup

This folder is the clean restart of the activation co-firing arm from
`experiment_3_nla.ipynb`. The NLA explanation arm is intentionally not in this
path yet.

## What This Runs

1. Verify model/layer/SAE assumptions.
2. Collect paired mean-pooled Qwen and Gemma activations for the ridge map.
3. Fit the Qwen L20 -> Gemma L41 map and run Gate 0.
4. Build a deduplicated pilot feature set from Neuronpedia activations.
5. Run co-firing AUROC plus Gate 1/Gate 2 controls.
6. Inspect top examples and run the covariate analysis.

## Key Corrections From The Notebook

- Gemma-3-27B has 62 transformer blocks, so Gemma L41 is about 66% depth.
- Qwen L20 is about 71% depth, so the primary pair is near depth-matched.
- Neuronpedia positive examples are deduplicated before scoring.
- `clean_transfer` now means strict identity-specific transfer:
  high-AUROC clean and Gate 1 pass.
- The looser high-AUROC count is still recorded as `high_auroc_clean`.
- Labels are loaded for the requested Neuronpedia batch range, including
  high-index ranges.

## Colab / A100 Setup

From the directory containing these scripts:

```bash
pip install -r requirements-colab.txt
export HF_TOKEN=...
```

Run the full activation co-firing path:

```bash
bash run_activation_cofiring.sh
```

The defaults use:

- `OUT_DIR=/content/exp3_acts`
- `MAP_LIMIT=50000`
- `N_PILOT=100`
- Neuronpedia activation batches `0..11`

Override them as needed:

```bash
OUT_DIR=/content/exp3_acts_b117 \
MAP_DIR=/content/exp3_acts \
CORPUS_DIR=/content/exp3_acts \
BATCH_START=117 \
BATCH_END=129 \
N_PILOT=100 \
SKIP_MAP=1 \
bash run_activation_cofiring.sh
```

Use a shared `MAP_DIR` when scoring a new feature range against an existing map.
Use `CORPUS_DIR` to reuse the original materialized Pile corpus for negatives.
Set `SKIP_MAP=1` when `MAP_DIR` already has `ridge_map.npz`,
`mean_qwen.npz`, and `mean_gemma.npz`.

## Manual Step Order

```bash
python 00_preflight.py

python 01_collect_activations.py --model qwen --reduction mean \
  --corpus pile --limit 50000 --out-dir /content/exp3_acts

python 01_collect_activations.py --model gemma --reduction mean \
  --corpus pile --limit 50000 --out-dir /content/exp3_acts

python 02_gate0_map_fit.py --out-dir /content/exp3_acts

python 03_neuronpedia_probe.py --n-batches 3

python 04_define_features.py --out-dir /content/exp3_acts \
  --n-pilot 100 --batch-start 0 --batch-end 12 --min-np-max 15.0

python 05_cofiring.py --out-dir /content/exp3_acts \
  --map-dir /content/exp3_acts --n-boot 1000

python 09_verify_top_features.py --out-dir /content/exp3_acts \
  --n-top 12 --n-ex 8 --show-failing

python 10_covariate_analysis.py --out-dir /content/exp3_acts
```

## Main Artifacts

- `mean_qwen.npz`, `mean_gemma.npz`: paired mean-pooled map data.
- `ridge_map.npz`: linear map plus standardization parameters.
- `gate0_results.json`: Gate 0 pass/fail and metrics.
- `pilot_features.json`: deduplicated `A_f` positives per feature.
- `negative_pool.json`: shared negative sentence pool.
- `qwen_cache.npz`, `gemma_cache.npz`: per-token activations for scoring.
- `cofiring_results.json`: transfer AUROC, controls, probes, decomposition.
- `covariates.csv`: compact table for predictor analysis.
- `covariate_summary.json`: summary of strict transfer rates and predictors.

## Reading The Main Rate

Report both rates:

- `high_auroc_clean`: high transfer AUROC with positive bootstrap support.
- `clean_transfer` / `strict_transfer`: high-AUROC clean and identity-specific
  Gate 1 pass.

The second one is the safer headline for activation co-firing.
