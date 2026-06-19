"""
exp3_config.py — architecture constants for Exp 3.
"""

QWEN_REPO       = "Qwen/Qwen2.5-7B-Instruct"
QWEN_NUM_LAYERS = 28
QWEN_BLOCK      = 20
QWEN_HS_IDX     = 21
QWEN_DMODEL     = 3584

GEMMA_REPO       = "google/gemma-3-27b-it"
GEMMA_NUM_LAYERS = 62
GEMMA_BLOCK      = 41
GEMMA_HS_IDX     = 42
GEMMA_DMODEL     = 5376

SAE_REPO = "chanind/qwen2.5-7B-it-layer-20-saes"
SAE_ID   = "pile/matryoshka/k-100"
SAE_K    = 100

# Important: current code/config should use 0, not the old stale spec's 8.
SAE_SKIP_FIRST_N = 0

LOAD_DTYPE  = "bfloat16"
STORE_DTYPE = "float32"

MODELS = {
    "qwen": dict(
        repo=QWEN_REPO,
        num_layers=QWEN_NUM_LAYERS,
        hs_idx=QWEN_HS_IDX,
        d_model=QWEN_DMODEL,
    ),
    "gemma": dict(
        repo=GEMMA_REPO,
        num_layers=GEMMA_NUM_LAYERS,
        hs_idx=GEMMA_HS_IDX,
        d_model=GEMMA_DMODEL,
    ),
}