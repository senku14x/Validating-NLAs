#!/usr/bin/env python

"""
03_build_injection_sweep.py  --  Exp 2, step 3: build the cross-detection
injection sweep.

What this does
--------------
1. Downloads 100 neutral anchor prompts (Alpaca, disjoint from the concept-pair
   sets used to build directions in step 2).
2. Loads gemma-3-27b-it and extracts hidden_states[42] at the last prompt token
   for each anchor — same extraction convention as steps 2 and Exp 1.
3. For each (anchor, concept_vector, dose) triple, uses the exact-cosine solver
   to rotate the anchor activation toward the target concept cosine. Every anchor
   lands at the *same* realized cosine for a given dose, giving clean comparable
   curves.
4. Also generates a random norm-matched control vector per anchor (drawn fresh
   each time; same norm as the concept vector after rescaling).

Injection convention (matches Exp 1 / 08_inject.py exactly):
    h' = h + beta * v_hat
    beta solved so that cos(h', v_hat) = target_cos exactly:
        beta = sqrt(||h||^2 - (h·v)^2) * t/sqrt(1-t^2) - (h·v)

Doses are expressed as DELTA above the anchor's mean baseline cosine with each
direction, so dose is concept-agnostic. Three levels:
    low    = mean_baseline + 0.15
    medium = mean_baseline + 0.30
    high   = mean_baseline + 0.45

The realized_cos column records the actual achieved cosine so the analysis axis
is always measured, never assumed.

Outputs
-------
  /content/anchor_activations.npy          float32 [100, 5376]
  /content/anchor_prompts.parquet          anchor prompts + ids
  /content/injected_matrix.npy             float32 [N_total, 5376]
  /content/injection_matrix_meta.parquet   one row per injected vector:
      row, anchor_id, concept, dose_label, target_cos, realized_cos,
      baseline_cos, delta_cos, beta, cos_with_refusal

Run
---
  !python 03_build_injection_sweep.py
  !python 03_build_injection_sweep.py --smoke   # 10 anchors only
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── config ───────────────────────────────────────────────────────────────────
BASE_MODEL     = "google/gemma-3-27b-it"
LAYER          = 42
D_MODEL        = 5376
VECTORS_PATH   = "/content/caa_vectors.npz"
ANCHORS_OUT    = "/content/anchor_activations.npy"
PROMPTS_OUT    = "/content/anchor_prompts.parquet"
INJECT_OUT     = "/content/injected_matrix.npy"
META_OUT       = "/content/injection_matrix_meta.parquet"
N_ANCHORS      = 30
SEED           = 42
# Dose levels: delta above mean baseline cosine for each direction.
DELTAS         = {"low": 0.15, "medium": 0.30, "high": 0.45}

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true",
                help="Use only 10 anchors (sanity check)")
args = ap.parse_args()

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ── helpers ───────────────────────────────────────────────────────────────────
def load_hf_token():
    token = os.environ.get("HF_TOKEN")
    assert token, (
        "HF_TOKEN not set.\n"
        "In notebook:  import os; from google.colab import userdata\n"
        "              os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')"
    )
    return token


def exact_cosine_inject(H, v_hat, target_cos):
    """
    For each row h in H, solve for beta so that
        cos(h + beta*v_hat, v_hat) = target_cos
    Returns (H_injected, beta, realized_cos).
    """
    a    = H @ v_hat                               # [n] dot products
    perp = np.sqrt(np.maximum((H**2).sum(1) - a**2, 0.0))  # [n] perp norms
    t    = float(target_cos)
    beta = perp * (t / np.sqrt(max(1 - t**2, 1e-9))) - a   # [n]
    Hp   = H + beta[:, None] * v_hat[None, :]
    norms = np.linalg.norm(Hp, axis=1, keepdims=True)
    realized = (Hp @ v_hat) / norms.squeeze()
    return Hp.astype(np.float32), beta, realized


# ── step 1: anchor prompts ────────────────────────────────────────────────────
print("Loading neutral anchor prompts (Alpaca, disjoint from concept pairs)...")
from datasets import load_dataset

n_need = N_ANCHORS if not args.smoke else 10
alpaca = load_dataset("tatsu-lab/alpaca", split="train")

# Keep only single-turn (no input context) instructions, dedup.
clean = list(dict.fromkeys(
    r["instruction"].strip() for r in alpaca if not r["input"].strip()
))

# The concept_pairs.parquet used the FIRST N_HARMLESS + N_ANCHOR rows at SEED=42
# (120 harmless + 120 anchors = first 240 clean instructions).
# We skip those and take the next N_ANCHORS to keep sets disjoint.
skip = 240
assert len(clean) >= skip + n_need, "Not enough Alpaca prompts."
anchor_prompts = clean[skip : skip + n_need]

anchor_df = pd.DataFrame({
    "anchor_id": [f"anchor_{i:03d}" for i in range(n_need)],
    "prompt":    anchor_prompts,
})
anchor_df.to_parquet(PROMPTS_OUT, index=False)
print(f"  {n_need} anchor prompts saved → {PROMPTS_OUT}")

# ── step 2: extract anchor activations ───────────────────────────────────────
cache_ok = (
    Path(ANCHORS_OUT).exists() and
    np.load(ANCHORS_OUT).shape == (n_need, D_MODEL)
)
if cache_ok:
    print(f"Anchor activations cache hit ({ANCHORS_OUT}), skipping extraction.")
    H = np.load(ANCHORS_OUT).astype(np.float64)
else:
    print(f"\nLoading {BASE_MODEL} (bf16, device_map=auto)...")
    token = load_hf_token()
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    model     = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, token=token,
        dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    cfg = model.config
    nhl = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    assert nhl == 62, f"Expected 62 layers, got {nhl}"
    print(f"  loaded in {time.time()-t0:.0f}s  LAYER={LAYER}")

    acts = []
    for prompt in tqdm(anchor_prompts, desc="  extracting anchors"):
        msgs     = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        vec = out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()
        acts.append(vec)
        del out, inputs
        torch.cuda.empty_cache()

    H = np.stack(acts).astype(np.float32)
    np.save(ANCHORS_OUT, H)
    print(f"  saved {H.shape} → {ANCHORS_OUT}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory after unload: {torch.cuda.memory_allocated()/1e9:.1f} GB")

H = H.astype(np.float64)

# ── step 3: load CAA vectors ──────────────────────────────────────────────────
print("\nLoading CAA direction vectors...")
npz      = np.load(VECTORS_PATH)
concepts = sorted(npz.keys())
vectors  = {c: npz[c].astype(np.float64) for c in concepts}
# sanity: all unit vectors
for c, v in vectors.items():
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4, f"{c} not unit vector"
print(f"  loaded {len(vectors)} vectors: {concepts}")

# Compute mean baseline cosine of anchors with each direction
# (used to set dose targets as delta-above-baseline)
baselines = {}
for c, v in vectors.items():
    cos_vals = (H @ v) / np.linalg.norm(H, axis=1)
    baselines[c] = float(cos_vals.mean())
print("\nMean anchor baseline cosines (injection pushes above these):")
for c in concepts:
    print(f"  {c:16s}  {baselines[c]:+.4f}")

# Random control vector (one per run, fixed seed; orthogonality to refusal checked)
v_random = rng.standard_normal(D_MODEL).astype(np.float64)
v_random /= np.linalg.norm(v_random)
cos_rand_refusal = float(v_random @ vectors["refusal"])
print(f"\nRandom control: cos(v_random, v_refusal) = {cos_rand_refusal:+.3f}  "
      f"(expect ~0)")

# ── step 4: build injection sweep ────────────────────────────────────────────
print("\nBuilding injection sweep...")

all_vectors   = {**vectors, "random": v_random}
anchor_ids    = anchor_df["anchor_id"].tolist()
n_anchors     = len(anchor_ids)
v_refusal     = vectors["refusal"]   # for cross-firing tracking

meta_rows = []
vecs_list = []

for concept, v_hat in all_vectors.items():
    baseline = baselines.get(concept, float((H @ v_hat / np.linalg.norm(H, axis=1)).mean()))
    for dose_label, delta in DELTAS.items():
        target_cos = min(baseline + delta, 0.95)   # cap to avoid near-degenerate angles
        Hp, beta, realized = exact_cosine_inject(H, v_hat, target_cos)
        cos_ref = (Hp @ v_refusal) / np.linalg.norm(Hp, axis=1)

        for i in range(n_anchors):
            meta_rows.append(dict(
                row              = len(meta_rows),
                anchor_id        = anchor_ids[i],
                concept          = concept,
                dose_label       = dose_label,
                target_cos       = round(target_cos, 4),
                realized_cos     = round(float(realized[i]), 4),
                baseline_cos     = round(baseline, 4),
                delta_cos        = round(float(realized[i]) - baseline, 4),
                beta             = round(float(beta[i]), 4),
                cos_with_refusal = round(float(cos_ref[i]), 4),
            ))
            vecs_list.append(Hp[i])

# Also include β=0 (no injection) baseline — same anchor activations, unmodified.
# Acts as the confabulation floor (same pipeline, no signal injected).
for i in range(n_anchors):
    cos_ref_base = float((H[i] @ v_refusal) / np.linalg.norm(H[i]))
    meta_rows.append(dict(
        row=len(meta_rows), anchor_id=anchor_ids[i],
        concept="baseline_no_inject", dose_label="none",
        target_cos=0.0, realized_cos=round(float((H[i] @ v_refusal) / np.linalg.norm(H[i])), 4),
        baseline_cos=0.0, delta_cos=0.0, beta=0.0,
        cos_with_refusal=round(cos_ref_base, 4),
    ))
    vecs_list.append(H[i].astype(np.float32))

vecs_arr = np.stack(vecs_list).astype(np.float32)
meta_df  = pd.DataFrame(meta_rows)

np.save(INJECT_OUT, vecs_arr)
meta_df.to_parquet(META_OUT, index=False)

# ── summary ───────────────────────────────────────────────────────────────────
n_concepts = len(all_vectors)
n_doses    = len(DELTAS)
n_total    = len(meta_df)
solver_err = (meta_df[meta_df.concept != "baseline_no_inject"]["realized_cos"] -
              meta_df[meta_df.concept != "baseline_no_inject"]["target_cos"]).abs().max()

print(f"\n{'='*72}")
print(f"Injection sweep complete")
print(f"  concepts × doses × anchors : {n_concepts} × {n_doses} × {n_anchors}")
print(f"  + baseline (no inject)     : {n_anchors}")
print(f"  total injected activations : {n_total}")
print(f"  shape                      : {vecs_arr.shape}")
print(f"  max solver error           : {solver_err:.6f}  (expect ~0)")
print(f"\n  saved activations → {INJECT_OUT}")
print(f"  saved metadata    → {META_OUT}")

print(f"\nDose targets by concept (mean_baseline + delta):")
print(f"{'concept':18s} {'baseline':>10s}  "
      + "  ".join(f"{l:>10s}" for l in DELTAS))
for concept in list(all_vectors.keys()):
    b = baselines.get(concept, 0.0)
    targets = "  ".join(f"{min(b+d, 0.95):>10.4f}" for d in DELTAS.values())
    print(f"  {concept:16s} {b:>10.4f}  {targets}")

print(f"\n{'='*72}")
print("Next: run 04_decode_matrix.py (requires SGLang + NLA AV checkpoint).")
