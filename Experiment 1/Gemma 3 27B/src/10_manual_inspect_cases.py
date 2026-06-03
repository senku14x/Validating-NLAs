#!/usr/bin/env python
import argparse
import re
import textwrap
import pandas as pd

META   = "/content/injection_meta.parquet"
DECODE = "/content/decode_outputs.parquet"
LABELS = "/content/labeled.parquet"

ap = argparse.ArgumentParser()
ap.add_argument("--condition", default="C1_refusal_pos")
ap.add_argument("--targets", nargs="*", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
ap.add_argument("--anchors", nargs="*", default=None,
                help="Optional anchor ids, e.g. anchor_037 anchor_090")
ap.add_argument("--per_cell", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--width", type=int, default=110)
ap.add_argument("--csv", default=None)
args = ap.parse_args()

meta = pd.read_parquet(META)
dec = pd.read_parquet(DECODE)
labels = pd.read_parquet(LABELS)

def extract_explanation(text):
    text = str(text)
    m = re.search(r"<explanation>(.*?)</explanation>", text, flags=re.S | re.I)
    return m.group(1).strip() if m else re.sub(r"</?explanation>", "", text).strip()

def clean_one_line(x, n=500):
    x = re.sub(r"\s+", " ", str(x)).strip()
    return x[:n] + ("..." if len(x) > n else "")

STRICT_RE = re.compile(
    r"\b(refus(?:al|e|ing)|declin(?:e|es|ing)|cannot comply|can't comply|"
    r"won't comply|will not comply|unwilling|cannot provide|can't provide|"
    r"cannot assist|can't assist|cannot fulfill|can't fulfill|"
    r"i cannot|i can't|i am unable|i'm unable|i'm sorry,?\s+(but\s+)?i)\b",
    re.I,
)

LOOSE_RE = re.compile(
    r"\b(ethical|ethics|legal|safety|safe|unsafe|harmful|illegal|dangerous|"
    r"caution|cautionary|disclaimer|responsible|policy|blocked|guardrail|"
    r"prohibited|not allowed|unethical)\b",
    re.I,
)

GENERIC_RE = re.compile(
    r"\b(structured|article|listicle|blog post|educational|professional|"
    r"advice|narrative|format|pattern|practical|instructional|essay|FAQ)\b",
    re.I,
)

def score_hint(exp):
    strict = bool(STRICT_RE.search(exp))
    loose = bool(LOOSE_RE.search(exp))
    generic = bool(GENERIC_RE.search(exp))
    if strict:
        return "2_strict_refusal"
    if loose:
        return "1_loose_safety"
    if generic:
        return "0_generic_structure"
    return "0_none"

# Merge decode + injection metadata.
full = meta.merge(dec, on="row", how="inner")
full["expl"] = full["nla_output"].map(extract_explanation)
full["score_hint"] = full["expl"].map(score_hint)

# Attach original anchor prompt/output.
anchor_info = labels[["id", "prompt", "output", "pool", "label"]].rename(columns={
    "id": "anchor_id",
    "prompt": "anchor_prompt",
    "output": "anchor_output",
    "pool": "anchor_pool",
    "label": "anchor_label",
})
full = full.merge(anchor_info, on="anchor_id", how="left")

# Filter.
sub = full[full["condition"] == args.condition].copy()
sub = sub[sub["target_cos"].isin(args.targets)]

if args.anchors:
    sub = sub[sub["anchor_id"].isin(args.anchors)]

# Sample per target cosine.
samples = []
for t in sorted(sub["target_cos"].unique()):
    cell = sub[sub["target_cos"] == t]
    n = min(args.per_cell, len(cell))
    if n:
        samples.append(cell.sample(n=n, random_state=args.seed))

if not samples:
    print("No rows matched filters.")
    raise SystemExit

sampled = pd.concat(samples).sort_values(["target_cos", "anchor_id", "row"])

print(f"\nManual inspection: condition={args.condition}")
print(f"targets={args.targets}")
print(f"n={len(sampled)} sampled rows\n")

for _, r in sampled.iterrows():
    print("=" * args.width)
    print(
        f"row={r.row} | anchor={r.anchor_id} | condition={r.condition} | "
        f"target_cos={r.target_cos:+.2f} | realized={r.realized_cos:+.3f} | "
        f"cos_with_refusal={r.cos_with_refusal:+.3f} | beta={r.beta:+.1f}"
    )
    print(f"score_hint={r.score_hint}")
    print("-" * args.width)

    print("[ORIGINAL ANCHOR PROMPT]")
    print(textwrap.fill(clean_one_line(r.anchor_prompt, 900), width=args.width))

    print("\n[ORIGINAL GEMMA OUTPUT ON ANCHOR]")
    print(textwrap.fill(clean_one_line(r.anchor_output, 900), width=args.width))

    print("\n[NLA EXPLANATION OF INJECTED ACTIVATION]")
    print(textwrap.fill(r.expl, width=args.width))

    print()

if args.csv:
    cols = [
        "row", "anchor_id", "condition", "target_cos", "realized_cos",
        "cos_with_refusal", "beta", "score_hint",
        "anchor_prompt", "anchor_output", "expl"
    ]
    sampled[cols].to_csv(args.csv, index=False)
    print(f"wrote {len(sampled)} rows -> {args.csv}")
