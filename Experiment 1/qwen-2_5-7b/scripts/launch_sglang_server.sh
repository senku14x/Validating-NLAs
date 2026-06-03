#!/usr/bin/env bash
# Launch the NLA activation verbalizer on SGLang (local GPU).
# Resolves the AV snapshot path safely (spaces in paths are quoted).
# just run this directly for the 09_decode.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV="${ROOT}/.venv/bin/uv"
if [[ ! -x "$UV" ]]; then
  UV="uv"
fi

resolve_av_dir() {
  if [[ -n "${NLA_AV_DIR:-}" ]]; then
    echo "$(cd "${NLA_AV_DIR}" && pwd)"
    return
  fi
  local meta
  meta="$(find "${ROOT}/cache" -name nla_meta.yaml -print -quit 2>/dev/null || true)"
  if [[ -z "$meta" ]]; then
    echo "ERROR: no AV snapshot under ${ROOT}/cache (run: python src/01_verify_env.py)" >&2
    echo "       or set NLA_AV_DIR to the kitft/nla-qwen2.5-7b-L20-av snapshot directory." >&2
    exit 1
  fi
  dirname "$meta"
}

# FlashInfer JIT-compiles CUDA kernels at first startup. The system /usr/bin/nvcc is
# often an ancient distro package (e.g. CUDA 10.1) that lacks flags PyTorch/ninja
# expect; point at a toolkit matching the PyTorch cu128 wheel (see pyproject.toml).
ensure_cuda_toolkit() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    return
  fi
  local dir
  for dir in /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -x "${dir}/bin/nvcc" ]]; then
      export CUDA_HOME="$dir"
      export PATH="${dir}/bin:${PATH}"
      return
    fi
  done
  echo "WARNING: no CUDA toolkit under /usr/local/cuda*; FlashInfer may fail to JIT." >&2
  echo "         Install CUDA 12.x or export CUDA_HOME to match your PyTorch build." >&2
}

# build.ninja files cache cuda_home; stale entries from /usr/bin/nvcc break rebuilds.
invalidate_stale_flashinfer_jit_cache() {
  local cache_root="${HOME}/.cache/flashinfer"
  [[ -d "$cache_root" ]] || return 0
  local ninja
  while IFS= read -r ninja; do
    if grep -q '^cuda_home = /usr$' "$ninja" 2>/dev/null; then
      echo "Removing stale FlashInfer JIT cache: $(dirname "$ninja")"
      rm -rf "$(dirname "$ninja")"
    fi
  done < <(find "$cache_root" -name build.ninja 2>/dev/null)
}

# tvm-ffi caches host compiler in build.ninja; drop sglang JIT dirs built with generic c++.
invalidate_stale_tvm_ffi_jit_cache() {
  local cache_root="${HOME}/.cache/tvm-ffi"
  [[ -d "$cache_root" ]] || return 0
  local ninja
  while IFS= read -r ninja; do
    if grep -q '^cxx = c++$' "$ninja" 2>/dev/null; then
      echo "Removing stale tvm-ffi JIT cache: $(dirname "$ninja")"
      rm -rf "$(dirname "$ninja")"
    fi
  done < <(find "$cache_root" -path '*/sgl_kernel_jit_*/build.ninja' 2>/dev/null)
}

AV_DIR="$(resolve_av_dir)"
PORT="${SGLANG_PORT:-30000}"

ensure_cuda_toolkit
# shellcheck source=ensure_toolchain.sh
source "${ROOT}/scripts/ensure_toolchain.sh"
invalidate_stale_flashinfer_jit_cache
invalidate_stale_tvm_ffi_jit_cache

echo "AV dir : $AV_DIR"
echo "port   : $PORT"
echo "nvcc   : $(command -v nvcc) ($(nvcc --version | sed -n 's/^Cuda compilation tools, release //p' | head -1))"
echo "cxx    : $CXX ($("$CXX" -dumpversion))"
echo "Starting SGLang (Ctrl+C to stop)..."

exec env CUDA_HOME="${CUDA_HOME:-}" CXX="$CXX" CC="$CC" PATH="${PATH}" \
  "$UV" run python -m sglang.launch_server \
  --model-path "$AV_DIR" \
  --port "$PORT" \
  --disable-radix-cache \
  --context-length 512 \
  --mem-fraction-static 0.85 \
  --trust-remote-code
