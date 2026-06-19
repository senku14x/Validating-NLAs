#!/usr/bin/env python
"""03_shift_diagnostic.py — characterize the base→organism activation shift (Phase A0, follow-up).

A0a found a large shift at the NLA layer (cosine ~0.63, norm +32%). A0a is a single-layer proxy; before
standing up the expensive A0b NLA loop, this answers WHERE the shift comes from — which both sharpens what
A0b will tell us and is itself a reportable finding. One base load + one organism load (extract MANY layers
in the same forward pass via output_hidden_states, so it's no costlier than A0a).

Per layer in a sweep, on matched neutral prompts, it reports:
  - paired cosine(base, organism) + norm ratio       → does the shift grow with depth?
  - GLOBAL-OFFSET fraction: of the per-prompt shift vector (h_org - h_base), how much is a single shared
    direction (mean shift) vs. prompt-specific? A near-global offset is the kind of shift an NLA (which
    L2-normalizes and was trained on a wide activation range) may be ROBUST to — so this reframes "NO-GO".
  - OUTLIER-DIM concentration: top-k dims' share of the total shift energy (Gemma/Llama residual streams have
    a few huge-norm dims; if the shift is mostly there, it may be benign for reconstruction).
  - cosine AFTER removing the global offset and after removing top outlier dims → "residual" shift.

Writes results/phase0_rq4/03_shift_diagnostic__<tag>.json. No NLA, no server — pure PyTorch.

Run (pod):
  python3 scripts/03_shift_diagnostic.py --base <dir> --organism <dir> --layers 10,20,30,40,53,60,70
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "phase0_rq4"

# reuse the exact neutral prompt set + ids helper from the A0a script (same prompts → comparable to A0a).
import importlib.util
_spec = importlib.util.spec_from_file_location("rq4", HERE / "01_rq4_ar_fidelity.py")
_rq4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rq4)
PROMPTS = _rq4.DEFAULT_PROMPTS
_ids_from = _rq4._ids_from


def _extract_multi(model, tok, prompts, layer_idxs, device, desc):
    """Last-prompt-token activations at MULTIPLE hidden_states indices in one forward pass each. fp32.
    Returns {layer_idx: [n, d] array}."""
    import torch
    from tqdm import tqdm
    out = {li: [] for li in layer_idxs}
    for text in tqdm(prompts, desc=desc):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = _ids_from(enc).to(device)
        with torch.no_grad():
            res = model(ids, output_hidden_states=True, use_cache=False)
        for li in layer_idxs:
            out[li].append(res.hidden_states[li][0, -1, :].float().cpu().numpy().astype(np.float32))
    return {li: np.stack(v) for li, v in out.items()}


def _cos_rows(A, B):
    a, b = A.astype(np.float64), B.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def _chance_cos(H, n_pairs=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(H)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs); k = i != j
    a, b = H[i[k]].astype(np.float64), H[j[k]].astype(np.float64)
    return float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)).mean())


def _analyze_layer(Hb, Ho):
    """Decompose the base→organism shift at one layer."""
    paired = _cos_rows(Hb, Ho)
    nr = np.linalg.norm(Ho, axis=1) / (np.linalg.norm(Hb, axis=1) + 1e-12)
    chance = _chance_cos(Hb)

    D = (Ho - Hb).astype(np.float64)                      # per-prompt shift vectors [n,d]
    g = D.mean(0)                                          # the shared (global-offset) shift direction
    gnorm = np.linalg.norm(g)
    # fraction of shift energy explained by the single global-offset direction:
    proj = D @ (g / (gnorm + 1e-12))                      # [n] projection of each shift onto g
    global_frac = float((proj ** 2).sum() / ((D ** 2).sum() + 1e-12))
    # cosine after removing the global offset (does the prompt-specific residual still look shifted?):
    Ho_deoff = Ho - g
    cos_deoff = float(_cos_rows(Hb, Ho_deoff).mean())
    # outlier-dim concentration: share of total shift energy in the top-k dims (by mean |shift|):
    dim_energy = (D ** 2).sum(0)                          # [d]
    order = np.argsort(dim_energy)[::-1]
    top10_share = float(dim_energy[order[:10]].sum() / (dim_energy.sum() + 1e-12))
    top1_share = float(dim_energy[order[0]] / (dim_energy.sum() + 1e-12))
    # cosine after zeroing the top-10 shift dims (is the rest aligned?):
    keep = np.ones(Hb.shape[1], bool); keep[order[:10]] = False
    cos_no_outlier = float(_cos_rows(Hb[:, keep], Ho[:, keep]).mean())

    return dict(
        paired_cosine=round(float(paired.mean()), 4),
        norm_ratio=round(float(nr.mean()), 4),
        anisotropy_floor=round(chance, 4),
        cos_above_floor=round(float(paired.mean()) - chance, 4),
        global_offset_frac=round(global_frac, 4),
        cos_after_deoffset=round(cos_deoff, 4),
        top1_dim_shift_share=round(top1_share, 4),
        top10_dim_shift_share=round(top10_share, 4),
        cos_excl_top10_dims=round(cos_no_outlier, 4),
    )


def run(base_dir, organism_dir, layers, tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(base_dir, token=token)
    print(f"[load] base ...")
    base = AutoModelForCausalLM.from_pretrained(base_dir, token=token, dtype=torch.bfloat16,
                                                device_map="auto").eval()
    dev = _rq4._input_device(base)
    Hb = _extract_multi(base, tok, PROMPTS, layers, dev, "base")
    print(f"[load] organism adapter ...")
    org = PeftModel.from_pretrained(base, organism_dir, token=token).eval()
    if not getattr(org, "peft_config", None):
        sys.exit("FAIL: adapter did not load.")
    Ho = _extract_multi(org, tok, PROMPTS, layers, dev, "organism")
    # sanity: not a no-op
    if all(np.allclose(Hb[li], Ho[li], atol=1e-3) for li in layers):
        sys.exit("FAIL: organism == base at all layers (no-op adapter).")

    by_layer = {int(li): _analyze_layer(Hb[li], Ho[li]) for li in layers}
    out = dict(base=base_dir, organism=organism_dir, n_prompts=len(PROMPTS),
               note="hidden_states index = block+1. global_offset_frac high => shift is mostly a shared "
                    "direction (NLA may be robust); outlier-dim share high => shift in a few huge-norm dims.",
               by_layer=by_layer)
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"03_shift_diagnostic__{tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print("\n==== SHIFT DIAGNOSTIC (by hidden_states layer) ====")
    print(f"{'layer':>5} {'cos':>6} {'norm':>6} {'floor':>6} {'glob%':>6} {'cos-deoff':>9} {'top10dim%':>9} {'cos-excl10':>10}")
    for li in layers:
        a = by_layer[int(li)]
        print(f"{li:>5} {a['paired_cosine']:>6} {a['norm_ratio']:>6} {a['anisotropy_floor']:>6} "
              f"{a['global_offset_frac']:>6} {a['cos_after_deoffset']:>9} {a['top10_dim_shift_share']:>9} "
              f"{a['cos_excl_top10_dims']:>10}")
    print(f"\nwrote {p}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--organism", required=True)
    ap.add_argument("--layers", default="10,20,30,40,53,60,70")
    ap.add_argument("--tag", default="oss-run1")
    a = ap.parse_args()
    layers = [int(x) + 1 for x in a.layers.split(",")]  # block N -> hidden_states[N+1]
    return run(a.base, a.organism, layers, a.tag)


if __name__ == "__main__":
    raise SystemExit(main())
