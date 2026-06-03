#!/usr/bin/env bash
# Ensure a C++20-capable host compiler for SGLang/FlashInfer JIT (needs <concepts>).
# Uses conda-forge gxx in ${ROOT}/.toolchain when system g++ is too old (e.g. Ubuntu 20.04 g++-9).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN="${ROOT}/.toolchain"

supports_cxx20() {
  local cxx="$1"
  [[ -x "$cxx" ]] || return 1
  # CUDA 12.8 nvcc rejects host GCC > 14.
  local ver major
  ver="$("$cxx" -dumpversion 2>/dev/null || true)"
  major="${ver%%.*}"
  if [[ -n "$major" && "$major" -gt 14 ]]; then
    return 1
  fi
  echo '#include <concepts>' | "$cxx" -std=c++20 -x c++ - -fsyntax-only &>/dev/null
}

pick_compiler() {
  local candidate="${CXX:-g++}"
  if supports_cxx20 "$candidate"; then
    echo "$candidate"
    return
  fi
  candidate="${TOOLCHAIN}/bin/x86_64-conda-linux-gnu-g++"
  if supports_cxx20 "$candidate"; then
    echo "$candidate"
    return
  fi
  return 1
}

install_toolchain() {
  local conda=""
  for candidate in \
    "${CONDA_EXE:-}" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda" \
    "$(command -v conda 2>/dev/null || true)"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    conda="$candidate"
    break
  done
  if [[ -z "$conda" ]]; then
    echo "ERROR: SGLang JIT needs g++ with C++20 (<concepts>); system g++ is too old." >&2
    echo "       Install g++-10+ (sudo apt install g++-10) or conda, then rerun." >&2
    exit 1
  fi
  echo "Installing conda-forge gxx 12.x into ${TOOLCHAIN} (one-time, ~1 min)..."
  # CUDA 12.8 nvcc supports host GCC up to 14; avoid conda gxx 15.x.
  "$conda" create -y -p "$TOOLCHAIN" -c conda-forge "gxx_linux-64=12.*" >/dev/null
}

setup_compiler_path() {
  local wrapper="${TOOLCHAIN}/wrappers"
  mkdir -p "$wrapper"
  ln -sf "$CXX" "${wrapper}/g++"
  ln -sf "$CC" "${wrapper}/gcc"
  export PATH="${wrapper}:${PATH}"
}

if ! GXX="$(pick_compiler)"; then
  install_toolchain
  GXX="$(pick_compiler)" || {
    echo "ERROR: failed to provision a C++20 compiler under ${TOOLCHAIN}" >&2
    exit 1
  }
fi

export CXX="$GXX"
export CC="${CC:-${GXX/g++/gcc}}"
setup_compiler_path
