#!/usr/bin/env bash
# Train Qwen GLP only (skip activation cache). Usage:
#   bash scripts/run_glp_train_qwen.sh
#   GLP_TRAIN_GPU=2 GLP_DATASET=... GLP_RUN=... bash scripts/run_glp_train_qwen.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

PY="${PY:-/iopsstor/scratch/cscs/velluru/exp2_rebuild/.venv/bin/python}"
GLP_DATASET="${GLP_DATASET:-${EXP2_CACHE:-$REBUILD/cache}/glp/qwen2_5_7b-layer20-static}"
GLP_RUN="${GLP_RUN:-${EXP2_WORKSPACE:-$REBUILD/workspace}/glp_runs/glp-qwen2_5_7b-d6-static}"
GLP_TRAIN_GPU="${GLP_TRAIN_GPU:-2}"

export CUDA_VISIBLE_DEVICES="$GLP_TRAIN_GPU"

cd "$GLP_ROOT"
exec "$PY" glp_train.py \
  config=configs/train_qwen2_5_7b_static.yaml \
  "train_dataset=$GLP_DATASET" \
  "rep_statistic=$GLP_DATASET/rep_statistics.pt" \
  "output_path=$GLP_RUN" \
  "glp_kwargs.normalizer_config.rep_statistic=$GLP_DATASET/rep_statistics.pt" \
  "$@"
