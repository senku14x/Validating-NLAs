#!/usr/bin/env bash
# run_exp3_box.sh — H200 box driver for the Exp-3 cross-model co-firing pilot.
#
# Reconstructs the Colab `run_activation_cofiring.sh` (not in repo) for a rented GPU box, and adds
# the two NEW stages this branch introduces:
#   * 02c  standardized mean-pool rectangular Procrustes map   (the promising map; down-weights
#          Gemma's outlier dims — set MAP_KIND=ridge for the 02_gate0 baseline instead)
#   * 06   metric-validation battery, run BOTH ways: length-matched-random negatives (reproduces the
#          existing numbers) AND --hard-negatives (same-domain control). The point of the run is the
#          DELTA between them — expect the eligible set to SHRINK under same-domain negatives.
#
# RESUMABLE: every heavy step is guarded by "skip if its artifact already exists", and 05 self-resumes
# from its caches. So this works two ways:
#   (a) FRESH box      -> runs 00..06 end to end (downloads Gemma+Qwen, the Pile, neuronpedia batches).
#   (b) RESTORED out-dir -> drop your Colab artifacts into $OUT_DIR first (mean_*.npz, ridge_map.npz,
#       pilot_features.json, negative_pool.json, *_cache.npz) and it fast-forwards to 05b/06.
#
# Requirements (box): HF_TOKEN set (gated Gemma-3); outbound HTTPS to HuggingFace + neuronpedia S3;
# one GPU with >=80 GB (H200 fits Gemma-27B then Qwen-7B sequentially — 05 frees each before the next).
#
# Usage (from this directory):
#   HF_TOKEN=hf_xxx bash run_exp3_box.sh
#   # overrides:
#   HF_TOKEN=hf_xxx OUT_DIR=$PWD/exp3_acts MAP_KIND=procrustes N_PILOT=100 \
#       BATCH_START=0 BATCH_END=12 MIN_NP_MAX=15.0 HARD_NEG=1 bash run_exp3_box.sh
set -euo pipefail

# ── run from this script's own dir (scripts import exp3_config via local sys.path) ──
cd "$(dirname "$(readlink -f "$0")")"

# ── config (all overridable via env) ──
OUT_DIR="${OUT_DIR:-$PWD/exp3_acts}"
MAP_DIR="${MAP_DIR:-$OUT_DIR}"               # where ridge_map.npz lives (share to reuse a map)
MAP_LIMIT="${MAP_LIMIT:-50000}"              # paired sentences for the map fit (01)
MAP_KIND="${MAP_KIND:-procrustes}"           # procrustes (02c, default) | ridge (02_gate0)
CORPUS="${CORPUS:-pile}"                      # SAE was trained on pile -> matches feature dist
N_PILOT="${N_PILOT:-100}"                     # pilot feature count (04)
BATCH_START="${BATCH_START:-0}"              # neuronpedia activation batch range (04)
BATCH_END="${BATCH_END:-12}"
MIN_NP_MAX="${MIN_NP_MAX:-15.0}"             # min neuronpedia max-activation to admit a feature (04)
N_BOOT="${N_BOOT:-1000}"                      # 05 bootstraps
HARD_NEG="${HARD_NEG:-1}"                     # also run 06 with same-domain hard negatives
RUN_INSPECT="${RUN_INSPECT:-1}"             # run 09/10 inspection at the end

mkdir -p "$OUT_DIR"
PY="${PY:-python}"

echo "=================================================================="
echo " Exp-3 box run"
echo "   OUT_DIR=$OUT_DIR   MAP_DIR=$MAP_DIR   MAP_KIND=$MAP_KIND"
echo "   CORPUS=$CORPUS  MAP_LIMIT=$MAP_LIMIT  N_PILOT=$N_PILOT  batches=$BATCH_START..$BATCH_END"
echo "   HARD_NEG=$HARD_NEG"
echo "=================================================================="
: "${HF_TOKEN:?set HF_TOKEN (gated Gemma-3-27B) before running}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARN: no nvidia-smi"

# ── deps (torch assumed preinstalled on the GPU image — do NOT pip-install it blindly, it can
#     clobber the CUDA build). If sae_lens errors on version skew, pin to your Colab versions. ──
if ! "$PY" -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "ERROR: torch not importable — install the CUDA build matching this box first."; exit 1
fi
"$PY" -m pip install -q -U transformers accelerate datasets sae_lens scikit-learn scipy pandas pyarrow numpy tqdm requests

step () { echo; echo "----- $* -----"; }

# ── 00 preflight (verifies model/layer/SAE assumptions; downloads both models ~60 GB) ──
step "00 preflight"
"$PY" 00_preflight.py

# ── 01 paired mean-pooled acts for the map (skip if already present) ──
if [ -f "$OUT_DIR/mean_qwen.npz" ]; then echo "skip 01 qwen (mean_qwen.npz exists)"; else
  step "01 collect qwen mean"
  "$PY" 01_collect_activations.py --model qwen  --reduction mean --corpus "$CORPUS" --limit "$MAP_LIMIT" --out-dir "$OUT_DIR"
fi
if [ -f "$OUT_DIR/mean_gemma.npz" ]; then echo "skip 01 gemma (mean_gemma.npz exists)"; else
  step "01 collect gemma mean"
  "$PY" 01_collect_activations.py --model gemma --reduction mean --corpus "$CORPUS" --limit "$MAP_LIMIT" --out-dir "$OUT_DIR"
fi

# ── 02 fit the map (skip if ridge_map.npz present in MAP_DIR) ──
if [ -f "$MAP_DIR/ridge_map.npz" ]; then
  echo "skip 02 (ridge_map.npz exists in $MAP_DIR)"
elif [ "$MAP_KIND" = "ridge" ]; then
  step "02 gate0 ridge map"
  "$PY" 02_gate0_map_fit.py --out-dir "$OUT_DIR"
else
  step "02c standardized Procrustes map"
  "$PY" 02c_procrustes_map_fit.py --acts-dir "$OUT_DIR" --out-dir "$MAP_DIR"
fi

# ── 03 neuronpedia sourcing probe (diagnostic; non-blocking) ──
step "03 neuronpedia probe (diagnostic)"
"$PY" 03_neuronpedia_probe.py --n-batches 3 || echo "03 diagnostic failed (non-blocking; 04 still runs)"

# ── 04 define the pilot feature set (skip if pilot_features.json present) ──
if [ -f "$OUT_DIR/pilot_features.json" ]; then echo "skip 04 (pilot_features.json exists)"; else
  step "04 define features"
  "$PY" 04_define_features.py --out-dir "$OUT_DIR" --n-pilot "$N_PILOT" \
      --batch-start "$BATCH_START" --batch-end "$BATCH_END" --min-np-max "$MIN_NP_MAX"
fi

# ── 05 co-firing (self-resumes from within_source/qwen_cache/gemma_cache if present) ──
step "05 co-firing (primary transfer metric)"
"$PY" 05_cofiring.py --out-dir "$OUT_DIR" --map-dir "$MAP_DIR" --n-boot "$N_BOOT"

# ── 05b fast map-ablation rescore ──
step "05b fast rescore"
"$PY" 05b_fast_rescore.py --out-dir "$OUT_DIR" --map-dir "$MAP_DIR"

# ── 06 metric validation — BOTH negative regimes; the DELTA is the result ──
step "06 metric validation (length-matched random negatives — baseline)"
"$PY" 06_validate_metric.py --out-dir "$OUT_DIR" --map-dir "$MAP_DIR" --save-prefix metric_validation_random
if [ "$HARD_NEG" = "1" ]; then
  step "06 metric validation (SAME-DOMAIN hard negatives — the real test)"
  "$PY" 06_validate_metric.py --out-dir "$OUT_DIR" --map-dir "$MAP_DIR" --hard-negatives --save-prefix metric_validation_hardneg
fi

# ── 09/10 inspection (optional, cheap) ──
if [ "$RUN_INSPECT" = "1" ]; then
  step "09 verify top features"
  "$PY" 09_verify_top_features.py --out-dir "$OUT_DIR" --n-top 12 --n-ex 8 --show-failing || echo "09 failed (non-blocking)"
  step "10 covariate analysis"
  "$PY" 10_covariate_analysis.py --out-dir "$OUT_DIR" || echo "10 failed (non-blocking)"
fi

echo
echo "=================================================================="
echo " DONE. Push these small results back (NOT the *.npz caches — large/gitignored):"
echo "   $OUT_DIR/gate0_results.json | gate0_procrustes_results.json"
echo "   $OUT_DIR/cofiring_results.json"
echo "   $OUT_DIR/fast_rescore.{json,csv}"
echo "   $OUT_DIR/metric_validation_random.{json,csv} + _summary.json"
echo "   $OUT_DIR/metric_validation_hardneg.{json,csv} + _summary.json   <-- the same-domain control"
echo "   $OUT_DIR/covariate_summary.json"
echo
echo " The headline to read: eligible_transfer_claim count in _random vs _hardneg _summary.json."
echo " If hardneg << random, the prior survivors were domain artifacts (the expected, informative result)."
echo "=================================================================="
