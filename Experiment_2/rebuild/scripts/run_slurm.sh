#!/usr/bin/env bash
# ============================================================================
# Experiment_2 rebuild — SLURM launcher for pipeline phases.
#
# Resources (per job): 1 node, 2 GPUs, 4 hour wall time.
#
# Submit (recommended — sets scratch log paths):
#   bash scripts/run_slurm.sh submit phase3 
#   sbatch -A a0133 --partition=normal scripts/run_slurm.sh phase3
#
# Run inside an existing allocation (debug):
#   bash scripts/run_slurm.sh run phase1 --model qwen
#
# Phases (see RUN_FROM_SCRATCH_QWEN_GLP.md):
#   phase0  CPU self-tests
#   phase1  Gate 0 — 01_verify_env
#   phase2  Gate 1 — 02, 03, 04, 04b
#   phase3  GLP — FineWeb cache (10M activations) + glp_train
#   phase4  Gate 2 — 05, av_up, 06, 07, 08
#   phase5  Gate 3 — 09, 07 --real
#   phase6  Gate 4 — 16_glp_steer_qwen
#
# Env (GPU phases):
#   HF_TOKEN              required
#   OPENROUTER_API_KEY    phase4/5 judge (or OPENAI_API_KEY)
#   NLA_REPO_DIR          phase4 AV server (default /workspace/nla_repo)
#   GLP_TARGET_VECTORS    phase3 activation target (default 10000000)
#   GLP_FINEWEB_CONFIG    phase3 FineWeb subset (default sample-10BT)
#   EXP2_RUN_ROOT         override scratch run dir (default \$SCRATCH/exp2_rebuild/\$SLURM_JOB_ID)
# ============================================================================
#SBATCH --job-name=exp2-rebuild
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REBUILD"

if [[ ! -f "$REBUILD/scripts/01_verify_env.py" ]]; then
  echo "FAIL: rebuild root not found at $REBUILD" >&2
  exit 1
fi

if [[ ! -f "$GLP_ROOT/glp_train.py" ]]; then
  echo "FAIL: GLP root not found at $GLP_ROOT" >&2
  exit 1
fi

# ── writable scratch layout (project home may be read-only on compute nodes) ──
SCRATCH_BASE="${SCRATCH:-${SCRATCH_NEW:-$REBUILD}}"
RUN_ID="${SLURM_JOB_ID:-local-$(date +%s)}"
RUN_ROOT="${EXP2_RUN_ROOT:-$SCRATCH_BASE/exp2_rebuild/$RUN_ID}"
mkdir -p "$RUN_ROOT/slurm" "$RUN_ROOT/cache" "$RUN_ROOT/workspace"
export EXP2_CACHE="$RUN_ROOT/cache"
export EXP2_WORKSPACE="$RUN_ROOT/workspace"

SLURM_LOG="$RUN_ROOT/slurm/job.log"
exec > >(tee -a "$SLURM_LOG") 2>&1

UV="${UV:-$(command -v uv 2>/dev/null || true)}"
[[ -x "$UV" ]] || UV="$HOME/.local/bin/uv"
VENV_DIR="${EXP2_VENV:-$SCRATCH_BASE/exp2_rebuild/.venv}"

ensure_python_env() {
  # 1) explicit override
  if [[ -n "${PY:-}" && -x "$PY" ]] \
      && "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    echo "using PY=$PY ($("$PY" --version 2>&1))"
    export PY
    return 0
  fi

  # 2) repo or scratch venv
  local candidate
  for candidate in "$REPO_ROOT/.venv/bin/python" "$VENV_DIR/bin/python"; do
    [[ -x "$candidate" ]] || continue
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PY="$candidate"
      echo "using venv $PY ($("$PY" --version 2>&1))"
      export PY
      return 0
    fi
  done

  # 3) bootstrap with uv (system python3 is 3.6 on Clariden — never use it)
  if [[ ! -x "$UV" ]]; then
    echo "FAIL: need Python >=3.11; install uv (~/.local/bin/uv) or set PY= to a 3.11+ interpreter" >&2
    exit 1
  fi

  say "bootstrap Python 3.11 venv at $VENV_DIR (via uv)"
  mkdir -p "$(dirname "$VENV_DIR")"
  "$UV" venv "$VENV_DIR" --python 3.11
  PY="$VENV_DIR/bin/python"
  export PY
  echo "created $PY ($("$PY" --version 2>&1))"

  say "install rebuild requirements"
  "$UV" pip install --python "$PY" -r "$REBUILD/requirements.txt"
  install_gpu_deps force
}

install_gpu_deps() {
  local force="${1:-}"
  local stamp
  stamp="$(dirname "$PY")/.gpu_deps_installed"
  if [[ "$force" != "force" && -f "$stamp" ]] \
      && "$PY" -c "import torch, transformers, datasets, baukit" 2>/dev/null; then
    echo "GPU deps already installed in $(dirname "$PY")"
    return 0
  fi

  if [[ ! -x "$UV" ]]; then
    echo "FAIL: uv required to install GPU deps (no pip on system python3.11)" >&2
    exit 1
  fi

  say "install GPU deps via uv pip"
  "$UV" pip install --python "$PY" \
    "git+https://github.com/davidbau/baukit" \
    "transformers==4.57.1" accelerate tqdm huggingface_hub datasets \
    diffusers omegaconf safetensors einops wandb
  if ! "$PY" -c "import torch" 2>/dev/null; then
    say "installing torch"
    "$UV" pip install --python "$PY" torch || {
      echo "WARN: default torch install failed; trying CUDA 12.4 wheel index"
      "$UV" pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu124
    }
  fi
  "$UV" pip install --python "$PY" -e "$GLP_ROOT"
  touch "$stamp"
}

MODEL="${MODEL:-qwen}"
GLP_DATASET="${GLP_DATASET:-$EXP2_CACHE/glp/qwen2_5_7b-layer20-static}"
GLP_RUN="${GLP_RUN:-$EXP2_WORKSPACE/glp_runs/glp-qwen2_5_7b-d6-static}"
GLP_WEIGHTS="${GLP_WEIGHTS:-$GLP_RUN}"
GLP_TARGET_VECTORS="${GLP_TARGET_VECTORS:-10000000}"
GLP_FINEWEB_CONFIG="${GLP_FINEWEB_CONFIG:-sample-10BT}"

say() { echo -e "\n========== $* =========="; }

show_paths() {
  local py_info="<not yet resolved>"
  if [[ -n "${PY:-}" && -x "$PY" ]]; then
    py_info="$("$PY" --version 2>&1) ($PY)"
  fi
  cat <<EOF
paths (this job):
  REBUILD=$REBUILD
  REPO_ROOT=$REPO_ROOT
  GLP_ROOT=$GLP_ROOT
  RUN_ROOT=$RUN_ROOT
  EXP2_CACHE=$EXP2_CACHE
  EXP2_WORKSPACE=$EXP2_WORKSPACE
  VENV_DIR=$VENV_DIR
  PY=$py_info
  SLURM_LOG=$SLURM_LOG
EOF
}

usage() {
  cat <<EOF
usage:
  bash scripts/run_slurm.sh submit <phase> [--model qwen|gemma]
  sbatch -A <acct> --partition=<part> scripts/run_slurm.sh <phase> [--model qwen|gemma]
  bash scripts/run_slurm.sh run <phase> [--model qwen|gemma]

phases: phase0 phase1 phase2 phase3 phase4 phase5 phase6
EOF
  show_paths
}

parse_args() {
  PHASE=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      phase[0-6]) PHASE="$1"; shift ;;
      --model) MODEL="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "unknown argument: $1"; usage; exit 1 ;;
    esac
  done
  if [[ -z "$PHASE" ]]; then
    usage
    exit 1
  fi
}

ensure_hf_token() {
  : "${HF_TOKEN:?export HF_TOKEN (Qwen/Gemma base + NLA repos)}"
}

ensure_judge_key() {
  if [[ -z "${OPENROUTER_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "FAIL: set OPENROUTER_API_KEY or OPENAI_API_KEY for judge scoring"
    exit 1
  fi
}

run_phase0() {
  say "Phase 0 — CPU self-tests"
  for t in test_confounds test_paths test_injection test_qwen_glp test_audits test_concepts test_sources; do
    "$PY" "$REBUILD/$t.py"
  done
}

run_phase1() {
  ensure_hf_token
  say "Phase 1 — Gate 0 verify env ($MODEL)"
  "$PY" "$REBUILD/scripts/01_verify_env.py" --model "$MODEL"
}

run_phase2() {
  ensure_hf_token
  say "Phase 2.1 — build concept pairs (CPU)"
  "$PY" "$REBUILD/scripts/02_build_concept_pairs.py"

  say "Phase 2.2 — extract activations ($MODEL)"
  pkill -f sglang 2>/dev/null || true
  sleep 3
  "$PY" "$REBUILD/scripts/03_extract_for_battery.py" --model "$MODEL"

  say "Phase 2.3 — Gate 1 probe battery (CPU)"
  "$PY" "$REBUILD/scripts/04_run_gate1_battery.py" --model "$MODEL"
  "$PY" "$REBUILD/scripts/04b_recheck_gate1.py" --model "$MODEL"
}

run_phase3() {
  ensure_hf_token
  say "Phase 3a — FineWeb activation cache (target ${GLP_TARGET_VECTORS} vectors, config ${GLP_FINEWEB_CONFIG})"
  say "cache -> $GLP_DATASET"
  "$PY" "$REBUILD/scripts/16_glp_qwen_cache.py" \
    --source fineweb \
    --fineweb-config "$GLP_FINEWEB_CONFIG" \
    --target-vectors "$GLP_TARGET_VECTORS" \
    --output-dir "$GLP_DATASET" \
    --all-tokens \
    --batch-size 64 \
    --max-length 2048

  say "Phase 3b — train Qwen GLP (GPU ${GLP_TRAIN_GPU:-1}) -> $GLP_RUN"
  GLP_TRAIN_GPU="${GLP_TRAIN_GPU:-1}" \
    GLP_DATASET="$GLP_DATASET" \
    GLP_RUN="$GLP_RUN" \
    bash "$REBUILD/scripts/run_glp_train_qwen.sh"
}

run_phase4() {
  ensure_hf_token
  ensure_judge_key
  export NLA_REPO_DIR="${NLA_REPO_DIR:-/workspace/nla_repo}"

  say "Phase 4.1 — injection matrix ($MODEL)"
  "$PY" "$REBUILD/scripts/05_inject_matrix.py" --model "$MODEL"

  say "Phase 4.2 — launch NLA AV server ($MODEL)"
  pkill -f sglang 2>/dev/null || true
  sleep 3
  bash "$REBUILD/scripts/av_up.sh" "$MODEL"

  say "Phase 4.3 — decode injected matrix ($MODEL)"
  "$PY" "$REBUILD/scripts/06_decode_matrix.py" --model "$MODEL"

  say "Phase 4.4 — score decodes ($MODEL)"
  "$PY" "$REBUILD/scripts/07_score_matrix.py" --model "$MODEL"

  say "Phase 4.5 — analyze Gate 2 ($MODEL)"
  "$PY" "$REBUILD/scripts/08_analyze_gate2.py" --model "$MODEL"

  pkill -f sglang 2>/dev/null || true
}

run_phase5() {
  ensure_hf_token
  ensure_judge_key
  export NLA_REPO_DIR="${NLA_REPO_DIR:-/workspace/nla_repo}"

  say "Phase 5.1 — real-activation decode ($MODEL)"
  pkill -f sglang 2>/dev/null || true
  sleep 3
  bash "$REBUILD/scripts/av_up.sh" "$MODEL"
  "$PY" "$REBUILD/scripts/09_decode_real.py" --model "$MODEL"

  say "Phase 5.2 — score real decodes ($MODEL)"
  "$PY" "$REBUILD/scripts/07_score_matrix.py" --model "$MODEL" --real

  pkill -f sglang 2>/dev/null || true
}

run_phase6() {
  ensure_hf_token
  say "Phase 6 — GLP on-manifold steering (qwen only)"
  "$PY" "$REBUILD/scripts/16_glp_steer_qwen.py" \
    --mode both \
    --glp-weights "$GLP_WEIGHTS" \
    --glp-checkpoint final \
    --concepts refusal,neg_sentiment,sycophancy \
    --doses 0.40,0.55,0.70 \
    --n-anchors 20
}

run_phase() {
  case "$PHASE" in
    phase0) run_phase0 ;;
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    phase3) run_phase3 ;;
    phase4) run_phase4 ;;
    phase5) run_phase5 ;;
    phase6) run_phase6 ;;
    *) echo "unknown phase: $PHASE"; usage; exit 1 ;;
  esac
}

if [[ "${1:-}" == "submit" ]]; then
  shift
  parse_args "$@"
  SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0")"
  LOG_DIR="${SCRATCH:-${SCRATCH_NEW:-$REBUILD}}/exp2_rebuild/slurm_logs"
  mkdir -p "$LOG_DIR"
  SBATCH_EXTRA=()
  [[ -n "${SLURM_ACCOUNT:-}" ]] && SBATCH_EXTRA+=(--account="$SLURM_ACCOUNT")
  [[ -n "${SLURM_PARTITION:-}" ]] && SBATCH_EXTRA+=(--partition="$SLURM_PARTITION")
  sbatch \
    "${SBATCH_EXTRA[@]}" \
    --job-name="exp2-${PHASE}" \
    --output="$LOG_DIR/exp2-${PHASE}-%j.out" \
    --error="$LOG_DIR/exp2-${PHASE}-%j.err" \
    --export=ALL,PHASE="$PHASE",MODEL="$MODEL",GLP_WEIGHTS="$GLP_WEIGHTS",GLP_TARGET_VECTORS="$GLP_TARGET_VECTORS",GLP_FINEWEB_CONFIG="$GLP_FINEWEB_CONFIG" \
    "$SCRIPT_PATH" run "$PHASE" --model "$MODEL"
  echo "submitted; scratch logs -> $LOG_DIR/exp2-${PHASE}-<jobid>.{out,err}"
  echo "full tee log inside job -> \$SCRATCH/exp2_rebuild/<jobid>/slurm/job.log"
  exit 0
fi

if [[ "${1:-}" == "run" ]]; then
  shift
fi

parse_args "$@"
ensure_python_env
if [[ "$PHASE" != "phase0" ]]; then
  install_gpu_deps
fi
say "SLURM job ${SLURM_JOB_ID:-local} — $PHASE (model=$MODEL, GPUs=${CUDA_VISIBLE_DEVICES:-unset})"
show_paths
run_phase "$PHASE"
say "DONE — $PHASE"
echo "artifacts under RUN_ROOT=$RUN_ROOT"
