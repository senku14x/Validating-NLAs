"""
exp3_config.py — single source of truth for Exp 3 architecture constants.

VERIFIED BY LOADING (not assumed). The spec doc had Gemma-3-27B at 46 blocks /
89% depth — that is the Gemma-2-27B architecture. The real model (from
google/gemma-3-27b-it config.json) is 62 blocks; L41 sits at ~66% depth.
Qwen L20 is ~71%. So the two NLA layers are *near depth-matched*, not an
18-point mismatch. Every script imports from here; never hardcode a layer.

Layer convention (verified three ways in Exp 1: AV sidecar
extraction_layer_index, AR config num_hidden_layers, AR sidecar):
    hidden_states is a tuple of length num_hidden_layers + 1.
    hidden_states[0]   = embedding output
    hidden_states[K+1] = resid_post of block K   <- what we extract
An off-by-one here produces plausible-looking garbage, so this file pins it.
"""

# ── Source model (Qwen): SAE feature space + map source space ────────────────
QWEN_REPO       = "Qwen/Qwen2.5-7B-Instruct"
QWEN_NUM_LAYERS = 28                 # num_hidden_layers (asserted on load)
QWEN_BLOCK      = 20                 # SAE hook block + map source (resid_post)
QWEN_HS_IDX     = QWEN_BLOCK + 1     # = 21  HF hidden_states index
QWEN_DMODEL     = 3584
# depth = 20/28 = 71.4%

# ── Target model (Gemma): map target space + co-firing projection ────────────
GEMMA_REPO       = "google/gemma-3-27b-it"
GEMMA_NUM_LAYERS = 62                # NOT 46 (that is Gemma-2-27B)
GEMMA_BLOCK      = 41                # NLA read layer (resid_post)
GEMMA_HS_IDX     = GEMMA_BLOCK + 1   # = 42
GEMMA_DMODEL     = 5376
# depth = 41/62 = 66.1%  -> near-matched to Qwen's 71%, not an 18-pt gap

# ── SAE (source-side feature directions) ────────────────────────────────────
SAE_REPO = "chanind/qwen2.5-7B-it-layer-20-saes"
SAE_ID   = "pile/matryoshka/k-100"
SAE_K    = 100   # expected ACTIVE features per token ~ k (tens). If .encode()
                 # returns thousands active, the top-k threshold is not applied
                 # -> loader is wrong (hand-rolled relu instead of .encode()).

# SAE_SKIP_FIRST_N: how many leading positions to exclude when encoding with
# the SAE. VERIFIED from sae metadata: seqpos_slice=[None] means NO position
# exclusion during training. Earlier value of 8 was a spec hallucination —
# the two features with self_act=0 in the index gate had their peak in pos 0-7.
# prepend_bos=True + exclude_special_tokens=True means only the BOS itself is
# OOD; with HF AutoTokenizer the BOS is either absent or handled. Set to 0.
SAE_SKIP_FIRST_N = 0

# ── dtypes ──────────────────────────────────────────────────────────────────
# Load bf16 (its exponent range handles Gemma's large-norm outlier dims during
# the forward). STORE fp32: Gemma has outlier dims whose precision we need for
# the least-squares map fit, and fp16 storage would lose/overflow them.
LOAD_DTYPE  = "bfloat16"
STORE_DTYPE = "float32"

# Convenience lookup so scripts can do CFG[args.model]
MODELS = {
    "qwen":  dict(repo=QWEN_REPO,  num_layers=QWEN_NUM_LAYERS,
                  hs_idx=QWEN_HS_IDX,  d_model=QWEN_DMODEL),
    "gemma": dict(repo=GEMMA_REPO, num_layers=GEMMA_NUM_LAYERS,
                  hs_idx=GEMMA_HS_IDX, d_model=GEMMA_DMODEL),
}
