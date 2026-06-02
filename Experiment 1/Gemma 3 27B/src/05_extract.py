"""
05_extract.py - extract layer-42 activations + model outputs for Experiment 1.

For each of the 360 prompts (harmful / harmless / anchor):
  - chat-format as a single user turn (add_generation_prompt=True)
  - ONE generate() pass with output_hidden_states, capturing:
      * activation = hidden_states[42] at the last prompt token
                     (raw residual stream = output of block 41; matches the AV's
                      extraction_layer_index=41, since HF index = K+1)
      * output     = greedily generated text (for behavioral refusal labeling)
  - batch=1: no padding, so [0, -1, :] is unambiguously the last prompt token.

Layer convention (verified three ways): AV sidecar extraction_layer_index=41,
AR config num_hidden_layers=42, AR sidecar 41  =>  HF hidden_states[42].
Gemma-3-27B has 62 layers, so 42 is ~2/3 depth.

Activations are stored RAW (no normalization - the AV does L2-normalize +
injection_scale at inference). Stored float32 to avoid Gemma's outlier-dim
values overflowing float16.

PREREQ: the AV SGLang server must be shut down first (see the notebook cell).
gemma-3-27b-it (~54 GB) then loads alone on the 80 GB A100.

Outputs (incremental + resumable):
  /content/activations.npy      float32 [N, 5376], row-aligned with the parquet
  /content/extraction.parquet   [id, pool, prompt, output, n_input_tokens]
"""
import gc
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL      = "google/gemma-3-27b-it"
LAYER      = 42            # HF hidden_states index = block-41 output
MAXNEW     = 64            # enough to surface a refusal opener
PROMPTS    = "/content/prompts.parquet"
ACT_OUT    = "/content/activations.npy"
META_OUT   = "/content/extraction.parquet"
CKPT_EVERY = 25
TOKEN      = os.environ["HF_TOKEN"]

df = pd.read_parquet(PROMPTS)
print(f"{len(df)} prompts: {df['pool'].value_counts().to_dict()}")

# ---- resume support --------------------------------------------------------
if os.path.exists(META_OUT) and os.path.exists(ACT_OUT):
    prev = pd.read_parquet(META_OUT)
    done_ids = set(prev["id"])
    acts = [r for r in np.load(ACT_OUT).astype(np.float32)]
    meta_rows = prev.to_dict("records")
    print(f"resuming - {len(done_ids)} already extracted")
else:
    done_ids, acts, meta_rows = set(), [], []

todo = df[~df["id"].isin(done_ids)].reset_index(drop=True)
if len(todo) == 0:
    print("all prompts already extracted - nothing to do.")
    raise SystemExit

# ---- load target model -----------------------------------------------------
print(f"loading {MODEL} (~54 GB)...")
tok = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, token=TOKEN, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
except Exception as e:
    print(f"AutoModelForCausalLM failed ({e}); trying Gemma3ForConditionalGeneration")
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL, token=TOKEN, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

cfg = model.config
nhl = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
assert nhl == 62, f"expected 62 layers, got {nhl} - wrong model?"
print(f"loaded. num_hidden_layers={nhl}; extracting hidden_states[{LAYER}]")

pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def save_ckpt():
    np.save(ACT_OUT, np.stack(acts).astype(np.float32))
    pd.DataFrame(meta_rows).to_parquet(META_OUT, index=False)


# ---- extract ---------------------------------------------------------------
for i, row in enumerate(tqdm(todo.to_dict("records"), desc="extract")):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    n_in = ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAXNEW, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True,
            pad_token_id=pad_id,
        )

    # hidden_states[0] = first decode step, covers the full prompt:
    # tuple of (num_layers+1) tensors, each [1, prompt_len, d_model]
    hs0 = out.hidden_states[0]
    assert len(hs0) == nhl + 1, f"expected {nhl + 1} hidden-state tensors, got {len(hs0)}"
    vec = hs0[LAYER][0, -1, :].float().cpu().numpy()
    assert np.isfinite(vec).all(), f"non-finite activation for {row['id']}"

    text = tok.decode(out.sequences[0, n_in:], skip_special_tokens=True).strip()

    acts.append(vec.astype(np.float32))
    meta_rows.append({
        "id": row["id"], "pool": row["pool"], "prompt": row["prompt"],
        "output": text, "n_input_tokens": int(n_in),
    })
    if (i + 1) % CKPT_EVERY == 0:
        save_ckpt()

save_ckpt()
arr = np.load(ACT_OUT)
print(f"\ndone. activations {arr.shape} dtype={arr.dtype}  norms: "
      f"mean={np.linalg.norm(arr, axis=1).mean():.0f} "
      f"min={np.linalg.norm(arr, axis=1).min():.0f} "
      f"max={np.linalg.norm(arr, axis=1).max():.0f}")

mp = pd.read_parquet(META_OUT)
print("\n--- one sample output per pool (eyeball refusal vs compliance) ---")
for pool in ["harmful", "harmless", "anchor"]:
    r = mp[mp.pool == pool].iloc[0]
    print(f"[{pool:8}] {r['output'][:90]!r}")

del model
gc.collect()
torch.cuda.empty_cache()
