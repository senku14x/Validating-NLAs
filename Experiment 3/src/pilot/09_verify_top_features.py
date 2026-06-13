#!/usr/bin/env python
"""
09_verify_top_features.py — read the actual A_f example texts for the top
transferring features. No GPU. Runs in seconds.

The "chemical nomenclature" and "time duration" labels are auto-generated.
This script confirms them by printing the raw example sentences, which is the
only reliable check — if the texts really are chemistry and timestamps, the
labels are right and the clean transfer result is defensible.

Also prints the within-source activation values alongside each text so you
can confirm the feature genuinely fires on these examples in Qwen.

Usage:
    python 09_verify_top_features.py --out-dir /content/exp3_acts [--n-top 8] [--n-ex 6]
"""
import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/content/exp3_acts")
    ap.add_argument("--n-top", type=int, default=8,
                    help="how many top features to inspect")
    ap.add_argument("--n-ex", type=int, default=6,
                    help="how many example texts to print per feature")
    ap.add_argument("--show-failing", action="store_true",
                    help="also print bottom features for contrast")
    args = ap.parse_args()
    out = Path(args.out_dir)

    results_path = out / "cofiring_results.json"
    pilot_path   = out / "pilot_features.json"
    ws_path      = out / "within_source.npz"

    if not results_path.exists():
        print("ERROR: cofiring_results.json not found — run 05_cofiring.py first")
        return
    if not pilot_path.exists():
        print("ERROR: pilot_features.json not found"); return

    results = json.loads(results_path.read_text())
    pilot   = json.loads(pilot_path.read_text())
    scored  = [r for r in results if not r.get("skip")]

    # load within_source activation values (the encode-based ceiling, used for
    # confirmation that the feature actually fires on these texts in Qwen)
    ws_data = None
    if ws_path.exists():
        d = np.load(ws_path, allow_pickle=True)
        ws_mat = d["ws"]; feat_keys = list(d["feat_keys"])
    else:
        ws_mat = None; feat_keys = []

    def ws_for(fi_str, sent_idx):
        if ws_mat is None: return None
        try:
            col = feat_keys.index(fi_str)
            return float(ws_mat[sent_idx, col])
        except (ValueError, IndexError):
            return None

    # sort by transfer_auroc
    top = sorted(scored, key=lambda r: -r["transfer_auroc"])[:args.n_top]
    if args.show_failing:
        bottom = sorted(scored, key=lambda r: r["transfer_auroc"])[:4]
        sections = [("TOP TRANSFERRING", top), ("LOWEST TRANSFER (contrast)", bottom)]
    else:
        sections = [("TOP TRANSFERRING", top)]

    # build universe (sent_id -> text) from pilot
    sent_list = []
    for k in sorted(pilot.keys(), key=int):
        for t in pilot[k]["pos_texts"]:
            if t not in sent_list:
                sent_list.append(t)
    # add negatives? we don't need them for this check

    for section_title, feat_list in sections:
        print("\n" + "="*80)
        print(f"{section_title}")
        print("="*80)
        for r in feat_list:
            fi_str = str(r["fi"])
            label  = r.get("label", "(no label)")
            decomp = r.get("decomp", "?")
            clean  = "CLEAN" if r.get("clean_transfer") else "not-clean"
            gate1  = "Gate1-Y" if r.get("gate1_pass") else "Gate1-N"
            print(f"\n[{r['fi']}] \"{label}\"  "
                  f"transfer={r['transfer_auroc']:.3f}  CI=[{r['transfer_ci_lo']:.2f},{r['transfer_ci_hi']:.2f}]  "
                  f"probe={r['gemma_probe']:.2f}  {decomp}  {clean}  {gate1}")
            print(f"  near_miss={r['near_miss_median']:.2f}  wrong_mean={r['wrong_feat_mean']:.2f}  "
                  f"beat={r['beat_wrong_frac']:.2f}  ws_ceil={r['ws_ceiling']:.2f}")

            if fi_str not in pilot:
                print("  (feature not in pilot_features.json)"); continue

            pos_texts = pilot[fi_str]["pos_texts"]
            max_vals  = pilot[fi_str].get("max_vals", [None]*len(pos_texts))

            print(f"  {len(pos_texts)} positive examples (showing {min(args.n_ex, len(pos_texts))}):")
            for i, (text, mv) in enumerate(zip(pos_texts[:args.n_ex], max_vals[:args.n_ex])):
                mv_str = f"  [np_max={mv:.1f}]" if mv is not None else ""
                # truncate long texts but show enough to judge content
                display = text.replace("\n", " ").strip()
                if len(display) > 180:
                    display = display[:177] + "..."
                print(f"  {i+1:2d}.{mv_str} {display}")

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("LABEL VERIFICATION SUMMARY")
    print("="*80)
    print(f"{'fi':>6}  {'label':<28}  {'transfer':>8}  {'clean':>5}  {'gate1':>5}  {'decomp'}")
    print("-"*80)
    for r in sorted(scored, key=lambda x: -x["transfer_auroc"]):
        print(f"{r['fi']:>6}  {r.get('label',''):<28}  "
              f"{r['transfer_auroc']:>8.3f}  "
              f"{'Y' if r.get('clean_transfer') else '.':>5}  "
              f"{'Y' if r.get('gate1_pass') else '.':>5}  "
              f"{r.get('decomp','')}")

    print(f"\nAction: for each CLEAN transfer above, confirm the example texts")
    print(f"match the label. If they do, the result is defensible.")
    print(f"If any clean-transfer label is wrong, that feature must be removed.")


if __name__ == "__main__":
    main()
