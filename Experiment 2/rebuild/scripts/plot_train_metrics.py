#!/usr/bin/env python
"""plot_train_metrics.py — turn a HF Trainer's trainer_state.json (its `log_history`) into a committable
CSV + PNG, so an OFFLINE W&B run's curves can be viewed on GitHub (the .wandb file itself is binary).

CPU-only, no GPU. The per-step loss / learning_rate / grad_norm / mean_token_accuracy the trainer logged
live in `log_history` inside each `checkpoint-*/trainer_state.json` (the last checkpoint has the full curve).

  python scripts/plot_train_metrics.py --state <path/to/trainer_state.json> --out results/gate4/train_logs/<name>
  python scripts/plot_train_metrics.py --auto --out results/gate4/train_logs/latest   # newest under workspace/organism
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import pandas as pd

COLS = ["step", "epoch", "loss", "learning_rate", "grad_norm", "mean_token_accuracy", "entropy"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", help="path to a trainer_state.json")
    ap.add_argument("--auto", action="store_true", help="use the newest trainer_state.json under workspace/organism")
    ap.add_argument("--out", default="results/gate4/train_logs/train", help="output path stem (.csv/.png appended)")
    a = ap.parse_args()

    state = a.state
    if a.auto or not state:
        cands = sorted(glob.glob("workspace/organism/**/trainer_state.json", recursive=True), key=os.path.getmtime)
        if not cands:
            sys.exit("no trainer_state.json under workspace/organism — pass --state explicitly")
        state = cands[-1]
        print(f"auto: {state}")

    hist = json.load(open(state)).get("log_history", [])
    df = pd.DataFrame(hist)
    if df.empty or "loss" not in df.columns:
        sys.exit(f"no usable log_history (loss) in {state}")
    df = df[df["loss"].notna()] if "loss" in df else df

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in COLS if c in df.columns]
    df[keep].to_csv(f"{out}.csv", index=False)
    print(f"wrote {out}.csv  ({len(df)} logged steps; final loss {df['loss'].iloc[-1]:.4f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = df["step"] if "step" in df.columns else range(len(df))
        panels = [("loss", "loss", "C0"), ("mean_token_accuracy", "mean_token_accuracy", "C2"),
                  ("learning_rate", "learning_rate", "C3")]
        panels = [(c, t, col) for c, t, col in panels if c in df.columns]
        fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
        if len(panels) == 1:
            axs = [axs]
        for ax, (c, t, col) in zip(axs, panels):
            ax.plot(x, df[c], color=col)
            ax.set_title(t); ax.set_xlabel("step"); ax.grid(alpha=0.3)
        fig.suptitle(pathlib.Path(state).parent.name)
        fig.tight_layout()
        fig.savefig(f"{out}.png", dpi=110)
        print(f"wrote {out}.png")
    except ImportError:
        print("matplotlib not installed -> CSV only (pip install matplotlib for the PNG graph)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
