#!/usr/bin/env bash
# av_up.sh — bring up the NLA AV SGLang server for <gemma|qwen>, end to end.
#
# Idempotent: installs deps, clones the NLA repo, grafts the chat_template,
# diagnoses + (only if needed) patches tokenization, applies the gemma3_mm
# input_embeds patch (Gemma only), downloads the AV weights, launches SGLang
# detached, waits for /health, and runs the random-vector smoke test.
#
# MEMORY DANCE: on an 80 GB card the extraction target model (03) and this AV
# server cannot coexist — kill the target process first (pkill -f from_pretrained
# or just finish 03). On H200 (141 GB) Gemma target (~54 GB) + AV (~54 GB) fit.
#
# Verified against kitft docs/inference.md + the proven Colab recipe.
#   bash scripts/av_up.sh gemma      # then later:  bash scripts/av_up.sh qwen
#   bash scripts/av_down.sh          # stop it
set -euo pipefail

MODEL="${1:?usage: av_up.sh <gemma|qwen>}"
case "$MODEL" in
  gemma) EXTRA=(--attention-backend fa3); GEMMA=1 ;;   # fa3: flashinfer OOMs at head_dim=256
  qwen)  EXTRA=();                         GEMMA=0 ;;   # Qwen: default backend, no mm patch
  *) echo "unknown model: $MODEL (use gemma|qwen)"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NLA_REPO_DIR="${NLA_REPO_DIR:-/workspace/nla_repo}"
export NLA_CACHE_DIR="${NLA_CACHE_DIR:-/workspace/nla_ckpt}"
PORT="${SGLANG_PORT:-30000}"; export SGLANG_PORT="$PORT"
LOG="${NLA_CACHE_DIR}/sglang_${MODEL}.log"
PIDF="${NLA_CACHE_DIR}/sglang_${MODEL}.pid"
mkdir -p "$NLA_CACHE_DIR"
: "${HF_TOKEN:?set HF_TOKEN (gated Gemma + AV repos)}"

echo "== 1. deps (sglang pinned 0.5.6 — the patch anchors were verified there) =="
python -c "import sglang" 2>/dev/null || pip install -q "sglang[all]==0.5.6"
pip install -q -U transformers huggingface_hub safetensors httpx orjson pyyaml numpy >/dev/null

echo "== 2. clone NLA repo (load_nla_config / NLAClient / patches) =="
[ -f "$NLA_REPO_DIR/nla_inference.py" ] || \
  git clone --depth 1 https://github.com/kitft/natural_language_autoencoders.git "$NLA_REPO_DIR"

echo "== 3. AV setup: chat_template graft + tokenization diagnose/patch + sidecar values =="
python "$HERE/nla_box.py" --model "$MODEL" --setup

if [ "$GEMMA" = 1 ]; then
  echo "== 3b. gemma3_mm input_embeds patch (Gemma only) =="
  SRC="${NLA_CACHE_DIR}/sglang_src"; mkdir -p "$SRC/python"
  PKG="$(python -c 'import os,sglang;print(os.path.dirname(sglang.__file__))')"
  [ -e "$SRC/python/sglang" ] || ln -s "$PKG" "$SRC/python/sglang"
  bash "$NLA_REPO_DIR/patches/apply_sglang_patches.sh" "$SRC" \
    || echo "  (patch returned nonzero — likely already applied; the smoke test is the real check)"
fi

echo "== 4. download AV weights (~54 GB Gemma / ~15 GB Qwen; cached) =="
AV_DIR="$(python "$HERE/nla_box.py" --model "$MODEL" --print-av-dir)"
echo "  AV dir: $AV_DIR"

echo "== 5. launch SGLang (detached) =="
if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "  already healthy on :$PORT — reusing it"
else
  setsid python -m sglang.launch_server \
    --model-path "$AV_DIR" --port "$PORT" \
    --disable-radix-cache --context-length 512 --mem-fraction-static 0.85 \
    --trust-remote-code "${EXTRA[@]}" >"$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  echo "  pid $(cat "$PIDF"), log $LOG — loading (~2 min); tail -f to watch"
  for _ in $(seq 1 75); do                         # ~10 min ceiling
    if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "  SERVER DIED — last 40 log lines:"; tail -40 "$LOG"; exit 1
    fi
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { echo; echo "  READY"; break; }
    sleep 8; printf '.'
  done
fi

echo "== 6. smoke test (random unit vector -> AV verbalization) =="
python "$HERE/nla_box.py" --model "$MODEL" --smoke
echo "DONE — AV server up on :$PORT for $MODEL.  Stop: bash scripts/av_down.sh"
echo "If SGLang JIT fails with a C++ error, you need a C++20 g++ — see"
echo "  Experiment 1/qwen-2_5-7b/scripts/ensure_toolchain.sh"
