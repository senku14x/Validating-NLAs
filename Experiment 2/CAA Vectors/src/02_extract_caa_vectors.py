#!/usr/bin/env python

"""
02_extract_caa_vectors.py  --  Exp 2, step 2: extract and validate CAA directions.

Loads gemma-3-27b-it, runs forward passes on every row in concept_pairs.parquet,
extracts hidden_states[42] at the last prompt token, builds one CAA direction per
concept, and validates each with held-out projection AUROC + shuffled-label null.

Direction convention (same as Exp 1 / injection step):
    v_concept = normalize( mean(h | present) - mean(h | absent) )

Extraction position: last token of the full formatted prompt (same convention as
Exp 1's 05_extract.py, which used hidden_states[0][LAYER][0, -1, :]).

Outputs
-------
  /content/caa_vectors.npz         {concept -> float32 [5376]}
  /content/caa_validation.json     {concept -> {auroc, null_auroc, n, ...}}
  /content/caa_acts/               per-concept activation caches (resumable)
  /content/caa_cos_dists.png       cosine distribution plots

Run
---
  !python 02_extract_caa_vectors.py            # full run
  !python 02_extract_caa_vectors.py --smoke    # 10 rows per concept (sanity check)
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── config ───────────────────────────────────────────────────────────────────
BASE_MODEL   = "google/gemma-3-27b-it"
LAYER        = 42          # hidden_states[42] = block-41 output (verified Exp 1)
D_MODEL      = 5376        # Gemma-3-27B
PAIRS_PATH   = "/content/concept_pairs.parquet"
ACTS_DIR     = "/content/caa_acts"
OUT_NPZ      = "/content/caa_vectors.npz"
OUT_JSON     = "/content/caa_validation.json"
OUT_PLOT     = "/content/caa_cos_dists.png"
SEED         = 42
N_SPLITS     = 5           # k-fold CV for AUROC

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true",
                help="Run on only 10 rows per concept (sanity check, no GPU needed)")
args = ap.parse_args()

Path(ACTS_DIR).mkdir(exist_ok=True)
np.random.seed(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────
def load_hf_token():
    token = os.environ.get("HF_TOKEN")
    assert token, (
        "HF_TOKEN not set. In your notebook run:\n"
        "  import os; from google.colab import userdata\n"
        "  os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
        "then re-run this script."
    )
    return token


def cos(H, vhat):
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    return (H / np.clip(norms, 1e-9, None)) @ vhat


def dom_dir(Xtr, ytr):
    v = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    return v / np.linalg.norm(v)


def cv_auroc(X, y, n_splits=N_SPLITS):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in skf.split(X, y):
        v = dom_dir(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], cos(X[te], v)))
    return float(np.mean(scores)), float(np.std(scores))


# ── load pairs ───────────────────────────────────────────────────────────────
print("Loading concept pairs...")
df = pd.read_parquet(PAIRS_PATH)
concepts = sorted(df.concept.unique())
print(f"  {len(df)} rows, {df.pair_id.nunique()} pairs, concepts: {concepts}")

if args.smoke:
    print("  [SMOKE] subsampling to 10 rows per concept")
    df = (df.groupby("concept", group_keys=False)
            .apply(lambda g: g.head(10))
            .reset_index(drop=True))
    print(f"  smoke set: {len(df)} rows")

# ── load model ───────────────────────────────────────────────────────────────
print(f"\nLoading {BASE_MODEL} (bf16, device_map=auto)...")
token = load_hf_token()
t0 = time.time()
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    token=token,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",   # FA3 only via SGLang; plain HF uses eager
)
model.eval()

# sanity check: architecture matches expectations
cfg = model.config
nhl = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
assert nhl == 62, f"Expected 62 layers, got {nhl} — wrong model?"
print(f"  loaded in {time.time()-t0:.0f}s  num_hidden_layers={nhl}  LAYER={LAYER}")
print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB allocated")

# ── extraction loop ───────────────────────────────────────────────────────────
def extract_for_concept(concept, rows):
    """
    Return float32 array [n_rows, D_MODEL] of layer-LAYER last-prompt-token
    activations. Resumable: checks for a cached .npy file first.
    """
    cache_path = Path(ACTS_DIR) / f"{concept}.npy"
    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape == (len(rows), D_MODEL):
            print(f"  [{concept}] loaded from cache {cache_path}")
            return arr
        print(f"  [{concept}] cache shape mismatch ({arr.shape} vs "
              f"({len(rows)}, {D_MODEL})) — re-extracting")

    acts = []
    for _, row in tqdm(rows.iterrows(), total=len(rows),
                       desc=f"  extract {concept}", leave=False):
        fmt = row.get("format", "user")
        if fmt == "system+user":
            msgs = json.loads(row.text)
        else:
            msgs = [{"role": "user", "content": row.text}]
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # hidden_states[LAYER] shape: [1, seq_len, d_model]
        # Extract at the last prompt token position [-1]
        vec = out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()
        acts.append(vec)

        # free graph & KV cache immediately
        del out, inputs
        torch.cuda.empty_cache()

    arr = np.stack(acts).astype(np.float32)
    np.save(cache_path, arr)
    print(f"  [{concept}] extracted {arr.shape} → {cache_path}")
    return arr


all_acts = {}   # concept -> full activation array [n_rows, D_MODEL]
for concept in concepts:
    sub  = df[df.concept == concept].reset_index(drop=True)
    arr  = extract_for_concept(concept, sub)
    all_acts[concept] = (sub, arr)   # keep labels alongside

# ── unload model (free VRAM for later SGLang use) ────────────────────────────
print("\nUnloading model to free VRAM...")
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print(f"  GPU memory after unload: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── build + validate directions ───────────────────────────────────────────────
print("\nBuilding and validating directions...")
from sklearn.metrics import roc_auc_score

vectors    = {}   # concept -> unit vector
validation = {}

for concept, (sub, arr) in all_acts.items():
    present_idx = sub.index[sub.polarity == "present"].tolist()
    absent_idx  = sub.index[sub.polarity == "absent"].tolist()

    # Re-index: arr is already re-indexed [0..n_rows-1], sub was reset_index'd
    pidx = sub[sub.polarity == "present"].index.tolist()
    aidx = sub[sub.polarity == "absent"].index.tolist()

    H_p = arr[pidx]   # [n_pairs, D_MODEL]
    H_a = arr[aidx]   # [n_pairs, D_MODEL]
    n   = min(len(H_p), len(H_a))

    if n < 4:
        print(f"  [{concept}] SKIP — only {n} pairs, too few to validate")
        continue

    # balance (shouldn't be needed since pairs are matched, but just in case)
    H_p, H_a = H_p[:n], H_a[:n]
    X = np.concatenate([H_p, H_a])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    # final direction (on all balanced data)
    v_raw   = H_p.mean(0) - H_a.mean(0)
    v_unit  = v_raw / np.linalg.norm(v_raw)
    vectors[concept] = v_unit.astype(np.float32)

    # held-out CV AUROC  (5-fold; 3-fold for small concepts)
    k = N_SPLITS if n >= 20 else 3
    auroc_mean, auroc_std = cv_auroc(X, y, n_splits=k)

    # shuffled-label null
    rng = np.random.default_rng(SEED)
    null_scores = []
    for _ in range(200):
        yn = rng.permutation(y)
        vn = dom_dir(X, yn)
        null_scores.append(roc_auc_score(y, cos(X, vn)))
    null_mean = float(np.mean(null_scores))

    # in-sample cosine stats (for reference; not used for validation)
    cos_p = cos(H_p, v_unit)
    cos_a = cos(H_a, v_unit)
    sep   = float(cos_p.mean() - cos_a.mean())

    status = ("STRONG" if auroc_mean >= 0.85 else
              "MODERATE" if auroc_mean >= 0.75 else "WEAK")

    validation[concept] = dict(
        n_pairs          = int(n),
        cv_folds         = k,
        auroc_mean       = round(auroc_mean, 4),
        auroc_std        = round(auroc_std, 4),
        null_auroc_mean  = round(null_mean, 4),
        cos_present_mean = round(float(cos_p.mean()), 4),
        cos_absent_mean  = round(float(cos_a.mean()), 4),
        separation       = round(sep, 4),
        status           = status,
    )

    flag = "⚠️  WEAK — inspect before using in matrix" if status == "WEAK" else ""
    print(f"  [{concept:12s}] AUROC={auroc_mean:.3f}±{auroc_std:.3f}  "
          f"null={null_mean:.3f}  sep={sep:+.3f}  n={n}  {status}  {flag}")

# ── save ──────────────────────────────────────────────────────────────────────
np.savez(OUT_NPZ, **vectors)
with open(OUT_JSON, "w") as f:
    json.dump(validation, f, indent=2)
print(f"\nSaved {len(vectors)} vectors → {OUT_NPZ}")
print(f"Saved validation → {OUT_JSON}")

# ── cosine distribution plots ─────────────────────────────────────────────────
n_concepts = len(vectors)
fig, axes  = plt.subplots(1, n_concepts, figsize=(4 * n_concepts, 4))
if n_concepts == 1:
    axes = [axes]

for ax, (concept, v_unit) in zip(axes, vectors.items()):
    sub, arr = all_acts[concept]
    pidx = sub[sub.polarity == "present"].index.tolist()
    aidx = sub[sub.polarity == "absent"].index.tolist()
    c_p = cos(arr[pidx], v_unit)
    c_a = cos(arr[aidx], v_unit)
    ax.hist(c_p, bins=25, alpha=0.6, label="present", color="steelblue", density=True)
    ax.hist(c_a, bins=25, alpha=0.6, label="absent",  color="darkorange", density=True)
    info = validation.get(concept, {})
    ax.set_title(f"{concept}\nAUROC={info.get('auroc_mean','?'):.3f}  "
                 f"({info.get('status','?')})", fontsize=9)
    ax.set_xlabel("cos(h, v_concept)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=130)
print(f"Saved cosine distribution plot → {OUT_PLOT}")

# ── final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("SUMMARY")
print(f"{'concept':14s} {'AUROC':>7s} {'null':>7s} {'sep':>7s} {'n':>5s}  status")
print("-" * 60)
for c, v in validation.items():
    print(f"{c:14s} {v['auroc_mean']:7.3f} {v['null_auroc_mean']:7.3f} "
          f"{v['separation']:+7.3f} {v['n_pairs']:5d}  {v['status']}")

weak = [c for c, v in validation.items() if v["status"] == "WEAK"]
if weak:
    print(f"\n⚠️  WEAK vectors: {weak}")
    print("   These cannot be trusted in the cross-detection matrix.")
    print("   Do NOT proceed to step 3 with weak vectors — inspect the pairs first.")
else:
    print("\n✓  All vectors MODERATE or STRONG — proceed to step 3.")
