# Qwen 2.5 7B NLA

This folder tracks the Python environment and Experiment 1 scripts for the Qwen-2.5-7B open-model NLA used in the NLA write-up appendix.

Model/checkpoints:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Activation verbalizer: `kitft/nla-qwen2.5-7b-L20-av`
- Activation reconstructor: `kitft/nla-qwen2.5-7b-L20-ar`
- Extraction layer: residual stream output of block 20 (`hidden_states[21]` in Hugging Face output)

### SGLang server (decode step)

From this directory, after `uv sync --extra nla-server` and `python src/01_verify_env.py`:

```bash
./scripts/launch_sglang_server.sh
```

The launch script configures two common failure modes on Ubuntu 20.04 / mixed CUDA installs:

1. **CUDA toolkit** — sets `CUDA_HOME` to `/usr/local/cuda-12.8` (or similar) so FlashInfer JIT uses a modern `nvcc`, not distro `/usr/bin/nvcc` (CUDA 10.x).
2. **Host C++ compiler** — SGLang’s JIT kernels need C++20 (`<concepts>`). If system `g++` is too old, the script auto-installs conda-forge `gxx` into `.toolchain/` (one-time).
