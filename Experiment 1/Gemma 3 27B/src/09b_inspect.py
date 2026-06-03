"""
09b_inspect.py — stratified inspection of NLA decode outputs before the LLM judge.

Pulls N examples from EVERY (condition x target_cos) cell so you see the whole
grid, not just whatever the head() happens to surface. Repeatable: change --seed
to draw a fresh set of samples from the same cells.

Run on Colab:
    !python 09b_inspect.py                          # full grid, 3 per cell
    !python 09b_inspect.py --condition C1_refusal_pos --per_cell 5
    !python 09b_inspect.py --condition C2_refusal_neg # the uninspected one
    !python 09b_inspect.py --seed 1                  # different draw, same cells
    !python 09b_inspect.py --csv /content/inspect_sample.csv   # also dump tidy CSV
"""
import argparse
import re
import textwrap

import pandas as pd

DECODE = "/content/decode_outputs.parquet"
META = "/content/injection_meta.parquet"

ap = argparse.ArgumentParser()
ap.add_argument("--condition", default=None,
                help="filter to one condition, e.g. C1_refusal_pos")
ap.add_argument("--per_cell", type=int, default=3,
                help="examples per (condition,target_cos) cell")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--width", type=int, default=100)
ap.add_argument("--csv", default=None, help="optional path to dump the sampled rows")
args = ap.parse_args()

dec = pd.read_parquet(DECODE)
meta = pd.read_parquet(META)

# defensive: show what we actually have before merging
print(f"decode cols : {list(dec.columns)}  (n={len(dec)})")
print(f"meta cols   : {list(meta.columns)}  (n={len(meta)})")
df = meta.merge(dec, on="row", how="inner")
print(f"merged      : n={len(df)}\n")


def extract(t):
    """Pull the <explanation>...</explanation> body; fall back to raw text."""
    if not isinstance(t, str):
        return "(no output)"
    m = re.search(r"<explanation>(.*?)</explanation>", t, re.DOTALL)
    return (m.group(1) if m else t).strip()


df["expl"] = df["nla_output"].apply(extract)

conds = [args.condition] if args.condition else sorted(df.condition.unique())
collected = []

for c in conds:
    sub = df[df.condition == c]
    print("=" * args.width)
    print(f"CONDITION: {c}   (n={len(sub)})")
    print("=" * args.width)
    for t in sorted(sub.target_cos.unique()):
        cell = sub[sub.target_cos == t]
        n = min(args.per_cell, len(cell))
        samp = cell.sample(n, random_state=args.seed)
        collected.append(samp)
        realized = cell.realized_cos.mean() if "realized_cos" in cell else float("nan")
        print(f"\n--- target_cos = {t:+.2f}   (realized mean {realized:+.3f}, n_cell={len(cell)}) ---")
        for _, r in samp.iterrows():
            body = textwrap.fill(r.expl, width=args.width,
                                 initial_indent="    ", subsequent_indent="    ")
            print(f"  [row {r.row} | anchor {r.anchor_id}]")
            print(body)
    print()

if args.csv:
    out = pd.concat(collected)[["row", "condition", "target_cos",
                                "realized_cos", "anchor_id", "expl"]]
    out.to_csv(args.csv, index=False)
    print(f"wrote {len(out)} sampled rows -> {args.csv}")
