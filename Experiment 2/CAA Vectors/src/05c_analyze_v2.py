#!/usr/bin/env python

"""
05c_analyze_v2.py — cross-detection matrix analysis for strict v2 scores.

Reads LLM judge scores and/or regex scores, builds:
  - cross-detection matrix at medium dose
  - dose-response curves
  - multi-sample consistency
  - baseline/negative-concept rate
  - diagonal/off-diagonal summary
  - optional risky-cell CSV for raw audit

Default is strict v2 LLM judge:
    /content/scores_matrix_raw_v2.jsonl

Run:
    !python 05c_analyze_v2.py
    !python 05c_analyze_v2.py --scorer llm_v2
    !python 05c_analyze_v2.py --scorer llm_v1
    !python 05c_analyze_v2.py --scorer regex

Outputs for default v2:
    /content/cross_detection_matrix_v2.csv
    /content/cross_detection_matrix_v2.png
    /content/dose_response_v2.png
    /content/consistency_table_v2.csv
    /content/matrix_summary_v2.txt
    /content/risky_cells_v2.csv
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── default paths ─────────────────────────────────────────────────────────────
LLM_V1_PATH  = "/content/scores_matrix_raw.jsonl"
LLM_V2_PATH  = "/content/scores_matrix_raw_v2.jsonl"
REGEX_PATH   = "/content/regex_scores_matrix.parquet"
META_PATH    = "/content/injection_matrix_meta.parquet"
DECODE_PATH  = "/content/decode_outputs.parquet"

SCORED_CONCEPTS = [
    "refusal", "sycophancy", "corrigibility", "truth_value",
    "uncertainty", "style_emoji", "neg_sentiment", "eval_awareness",
    "medical_advice",
]

INJECTED_CONCEPTS = [
    "corrigibility", "eval_awareness", "neg_sentiment", "random",
    "refusal", "style_emoji", "sycophancy", "truth_value", "uncertainty",
]

DOSE_ORDER = ["low", "medium", "high"]


# ── args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument(
    "--scorer",
    choices=["llm_v2", "llm_v1", "regex"],
    default="llm_v2",
    help="Which scores to use as primary. Default: llm_v2.",
)
ap.add_argument(
    "--llm-path",
    default=None,
    help="Optional override path for LLM jsonl scores.",
)
ap.add_argument(
    "--regex-path",
    default=REGEX_PATH,
    help="Optional override path for regex parquet scores.",
)
ap.add_argument(
    "--threshold",
    type=int,
    default=2,
    choices=[1, 2],
    help="Detection threshold. Default 2 means P(score==2). Use 1 for P(score>=1).",
)
args = ap.parse_args()


# ── output suffix ─────────────────────────────────────────────────────────────
if args.scorer == "llm_v2":
    suffix = "v2"
    default_llm_path = LLM_V2_PATH
elif args.scorer == "llm_v1":
    suffix = "v1"
    default_llm_path = LLM_V1_PATH
else:
    suffix = "regex"
    default_llm_path = None

if args.threshold == 1:
    suffix = suffix + "_ge1"

MATRIX_CSV   = f"/content/cross_detection_matrix_{suffix}.csv"
MATRIX_PNG   = f"/content/cross_detection_matrix_{suffix}.png"
DOSE_PNG     = f"/content/dose_response_{suffix}.png"
CONSIST_CSV  = f"/content/consistency_table_{suffix}.csv"
SUMMARY_TXT  = f"/content/matrix_summary_{suffix}.txt"
RISKY_CSV    = f"/content/risky_cells_{suffix}.csv"


# ── helpers ───────────────────────────────────────────────────────────────────
def load_llm_scores(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                rows.append(r)
            except Exception:
                pass

    if not rows:
        raise ValueError(f"No valid JSON rows found in {path}")

    df = pd.DataFrame(rows)

    required = ["row", "sample"] + SCORED_CONCEPTS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"LLM score file missing columns: {missing}")

    # Keep only rows with valid integer scores.
    valid_mask = np.ones(len(df), dtype=bool)
    for c in SCORED_CONCEPTS:
        valid_mask &= df[c].isin([0, 1, 2])

    bad = len(df) - int(valid_mask.sum())
    if bad:
        print(f"  Dropping {bad} invalid/error score rows")

    df = df[valid_mask].copy()

    # If smoke/rerun appended duplicate rows, keep the last one.
    before = len(df)
    df = df.drop_duplicates(["row", "sample"], keep="last")
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} duplicate (row,sample) rows; kept latest")

    rename = {c: f"score_{c}" for c in SCORED_CONCEPTS}
    df = df.rename(columns=rename)

    return df[["row", "sample"] + list(rename.values())]


def load_regex_scores(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required = ["row", "sample"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Regex score file missing columns: {missing}")

    rename = {
        f"r_{c}": f"score_{c}"
        for c in SCORED_CONCEPTS
        if f"r_{c}" in df.columns
    }
    if not rename:
        raise ValueError("No regex score columns found, expected r_<concept> columns.")

    df = df.rename(columns=rename)
    keep_cols = ["row", "sample"] + list(rename.values())

    before = len(df)
    df = df.drop_duplicates(["row", "sample"], keep="last")
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} duplicate regex rows; kept latest")

    return df[keep_cols]


def detection_expr(score_series: pd.Series, threshold: int) -> pd.Series:
    if threshold == 2:
        return (score_series == 2).astype(int)
    return (score_series >= 1).astype(int)


# ── load base data ────────────────────────────────────────────────────────────
print(f"Scorer      : {args.scorer}")
print(f"Threshold   : {'score==2' if args.threshold == 2 else 'score>=1'}")

meta = pd.read_parquet(META_PATH)
dec = pd.read_parquet(DECODE_PATH)
base = meta.merge(dec, on="row", how="inner")

print(f"Meta rows   : {len(meta)}")
print(f"Decode rows : {len(dec)}")
print(f"Base rows   : {len(base)}")

# ── load scores ───────────────────────────────────────────────────────────────
if args.scorer in ["llm_v1", "llm_v2"]:
    llm_path = args.llm_path or default_llm_path
    if not os.path.exists(llm_path):
        raise FileNotFoundError(f"LLM score file not found: {llm_path}")

    print(f"\nLoading LLM judge scores from {llm_path}...")
    score_df = load_llm_scores(llm_path)
    df = base.merge(score_df, on=["row", "sample"], how="inner")
    print(f"  {len(score_df)} unique score rows") 
    print(f"  {len(df)} joined rows")

else:
    regex_path = args.regex_path
    if not os.path.exists(regex_path):
        raise FileNotFoundError(f"Regex score file not found: {regex_path}")

    print(f"\nLoading regex scores from {regex_path}...")
    score_df = load_regex_scores(regex_path)
    df = base.merge(score_df, on=["row", "sample"], how="inner")
    print(f"  {len(score_df)} unique score rows")
    print(f"  {len(df)} joined rows")

if len(df) == 0:
    raise RuntimeError("No joined rows after merging metadata/decode/scores.")

# ── score columns and detection columns ───────────────────────────────────────
score_cols = {
    c: f"score_{c}"
    for c in SCORED_CONCEPTS
    if f"score_{c}" in df.columns
}

for c, col in score_cols.items():
    df[f"d_{c}"] = detection_expr(df[col], args.threshold)

detect_cols = [f"d_{c}" for c in SCORED_CONCEPTS if f"d_{c}" in df.columns]

# ── basic score distribution ──────────────────────────────────────────────────
print("\nGlobal score distribution:")
for c in SCORED_CONCEPTS:
    col = f"score_{c}"
    if col not in df.columns:
        continue
    n2 = int((df[col] == 2).sum())
    n1 = int((df[col] == 1).sum())
    nge1 = n1 + n2
    print(
        f"  {c:18s}  score=2: {n2:4d} ({100*n2/len(df):5.1f}%)"
        f"   score>=1: {nge1:4d} ({100*nge1/len(df):5.1f}%)"
    )

# ── judge base rate ───────────────────────────────────────────────────────────
neg_col = "d_medical_advice"
neg_rate = df[neg_col].mean() if neg_col in df.columns else float("nan")

print(f"\nJudge base rate (medical_advice, never injected): {neg_rate:.3f}")
if neg_rate > 0.05:
    print("  WARNING: medical_advice base rate > 5%; all rates carry a large floor.")

# ── baseline row rates, if present ────────────────────────────────────────────
baseline = df[df["concept"] == "baseline_no_inject"]
if len(baseline):
    print("\nBaseline no-injection detection rates:")
    for c in SCORED_CONCEPTS:
        dcol = f"d_{c}"
        if dcol in baseline.columns:
            print(f"  {c:18s}  P={baseline[dcol].mean():.3f}  n={len(baseline)}")

# ── cross-detection matrix at medium dose ─────────────────────────────────────
print("\nBuilding cross-detection matrix at medium dose...")
agg = df.groupby(["concept", "dose_label"], dropna=False)[detect_cols].mean().reset_index()
medium = agg[agg.dose_label == "medium"].copy()

if medium.empty:
    raise RuntimeError("No medium-dose rows found. Check dose_label values.")

medium = medium.set_index("concept")[detect_cols]
medium.columns = [c.replace("d_", "") for c in medium.columns]

# Stable row order.
row_order = [c for c in INJECTED_CONCEPTS if c in medium.index]
extra_rows = sorted([c for c in medium.index if c not in row_order and c != "baseline_no_inject"])
row_order.extend(extra_rows)
if "baseline_no_inject" in medium.index:
    row_order.append("baseline_no_inject")

medium = medium.reindex(row_order)
medium.to_csv(MATRIX_CSV)

print(f"\nCross-detection matrix P({'score==2' if args.threshold == 2 else 'score>=1'}), medium dose:")
print(medium.round(3).to_string())

# ── diagonal and max off-diagonal dataframe ───────────────────────────────────
summary_rows = []
for concept in row_order:
    if concept in ["baseline_no_inject"]:
        continue
    if concept not in medium.index:
        continue

    diag_val = float(medium.loc[concept, concept]) if concept in medium.columns else np.nan

    off = medium.loc[concept].drop(
        [c for c in [concept, "medical_advice"] if c in medium.columns],
        errors="ignore",
    )
    max_off = float(off.max()) if len(off) else np.nan
    max_col = str(off.idxmax()) if len(off) else ""

    summary_rows.append({
        "concept": concept,
        "diag": diag_val,
        "signal_vs_medical_base": diag_val - neg_rate if not np.isnan(diag_val) else np.nan,
        "max_offdiag": max_off,
        "max_offdiag_col": max_col,
        "diag_minus_max_offdiag": diag_val - max_off if not np.isnan(diag_val) and not np.isnan(max_off) else np.nan,
    })

diag_summary = pd.DataFrame(summary_rows)
print("\nDiagonal vs max off-diagonal:")
print(diag_summary.round(3).to_string(index=False))

# ── multi-sample consistency ──────────────────────────────────────────────────
print("\nMulti-sample consistency: fraction of injected rows with >=2/3 samples detecting...")
consist_rows = []

for scored_c in SCORED_CONCEPTS:
    dcol = f"d_{scored_c}"
    if dcol not in df.columns:
        continue

    for injected_c in sorted(df.concept.dropna().unique()):
        for dose in sorted(df.dose_label.dropna().unique()):
            cell = df[(df.concept == injected_c) & (df.dose_label == dose)]
            if cell.empty:
                continue

            # row corresponds to a single injected activation;
            # each row should have N samples.
            by_row = cell.groupby("row")[dcol].agg(["sum", "count"])
            consist_rows.append({
                "injected": injected_c,
                "scored": scored_c,
                "dose": dose,
                "n_injected_rows": len(by_row),
                "mean_samples_per_row": round(by_row["count"].mean(), 3),
                "frac_ge2of3": round((by_row["sum"] >= 2).mean(), 3),
                "frac_3of3": round((by_row["sum"] == 3).mean(), 3),
            })

consist_df = pd.DataFrame(consist_rows)
consist_df.to_csv(CONSIST_CSV, index=False)

print("\nDiagonal consistency (injected==scored, medium dose):")
diag_consist = consist_df[
    (consist_df.injected == consist_df.scored) &
    (consist_df.dose == "medium")
][["injected", "n_injected_rows", "mean_samples_per_row", "frac_ge2of3", "frac_3of3"]]
diag_consist = diag_consist.sort_values("frac_ge2of3", ascending=False)
print(diag_consist.to_string(index=False))

# ── risky cells for raw audit ─────────────────────────────────────────────────
# These are not interpretations; just a queue for manual inspection.
risk_masks = []

def add_risky(label, mask):
    tmp = df[mask].copy()
    if len(tmp):
        tmp["risk_label"] = label
        risk_masks.append(tmp)

if "d_refusal" in df.columns:
    add_risky(
        "offtarget_refusal",
        (df["d_refusal"] == 1) &
        (df["concept"] != "refusal")
    )

if "d_eval_awareness" in df.columns:
    add_risky(
        "offtarget_eval_awareness",
        (df["d_eval_awareness"] == 1) &
        (df["concept"] != "eval_awareness")
    )

if "d_medical_advice" in df.columns:
    add_risky(
        "medical_advice_positive",
        df["d_medical_advice"] == 1
    )

if "d_style_emoji" in df.columns:
    add_risky(
        "random_or_baseline_style",
        (df["d_style_emoji"] == 1) &
        (df["concept"].isin(["random", "baseline_no_inject"]))
    )

if risk_masks:
    risky = pd.concat(risk_masks, ignore_index=True)
    keep = [
        "risk_label", "row", "sample", "anchor_id", "concept", "dose_label",
        "target_cos", "realized_cos", "baseline_cos", "delta_cos",
        "cos_with_refusal", "nla_output",
    ]
    keep += [c for c in [f"score_{x}" for x in SCORED_CONCEPTS] if c in risky.columns]
    keep = [c for c in keep if c in risky.columns]
    risky[keep].to_csv(RISKY_CSV, index=False)
    print(f"\nSaved risky-cell audit queue → {RISKY_CSV}  n={len(risky)}")
else:
    print("\nNo risky-cell positives found for audit queue.")

# ── heatmap ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8))
vals = medium.values.astype(float)

im = ax.imshow(vals, vmin=0, vmax=1, cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, label=f"P({'score==2' if args.threshold == 2 else 'score>=1'})")

ax.set_xticks(range(len(medium.columns)))
ax.set_xticklabels(medium.columns, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(medium.index)))
ax.set_yticklabels(medium.index, fontsize=9)
ax.set_title(
    f"Cross-detection matrix, medium dose ({args.scorer}, "
    f"{'score==2' if args.threshold == 2 else 'score>=1'})\n"
    "rows=injected direction, columns=detected concept",
    fontsize=11,
)

for i in range(len(medium.index)):
    for j in range(len(medium.columns)):
        v = vals[i, j]
        if not np.isnan(v):
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                fontsize=7.5,
                color="white" if v > 0.55 else "black",
            )

# Red diagonal boxes.
for i, row_c in enumerate(medium.index):
    for j, col_c in enumerate(medium.columns):
        if row_c == col_c:
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    fill=False,
                    edgecolor="red",
                    lw=2.5,
                )
            )

plt.tight_layout()
plt.savefig(MATRIX_PNG, dpi=140, bbox_inches="tight")
print(f"\nSaved heatmap → {MATRIX_PNG}")

# ── dose-response, diagonal self-detection ────────────────────────────────────
agg_all = agg.copy()
injected_for_plot = [
    c for c in INJECTED_CONCEPTS
    if c not in ["random", "baseline_no_inject"] and f"d_{c}" in agg_all.columns
]

fig2, axes = plt.subplots(2, 4, figsize=(18, 9))
fig2.suptitle(
    f"Self-detection dose-response ({args.scorer}, "
    f"{'score==2' if args.threshold == 2 else 'score>=1'})",
    fontsize=13,
    fontweight="bold",
)
axes = axes.flatten()

for ax_i, concept in enumerate(injected_for_plot[:8]):
    ax = axes[ax_i]
    dcol = f"d_{concept}"

    y_vals = []
    for dose in DOSE_ORDER:
        row = agg_all[(agg_all.concept == concept) & (agg_all.dose_label == dose)]
        y_vals.append(float(row[dcol].iloc[0]) if len(row) else float("nan"))

    ax.plot(DOSE_ORDER, y_vals, "o-", lw=2, ms=7, label=concept)
    ax.axhline(
        neg_rate,
        lw=1.2,
        ls="--",
        alpha=0.7,
        label=f"medical base ({neg_rate:.2f})",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(concept, fontsize=10)
    ax.set_ylabel(f"P({'score==2' if args.threshold == 2 else 'score>=1'})")
    ax.set_xlabel("dose")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for i in range(len(injected_for_plot), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(DOSE_PNG, dpi=130, bbox_inches="tight")
print(f"Saved dose-response → {DOSE_PNG}")

# ── text summary ──────────────────────────────────────────────────────────────
lines = [
    "=" * 72,
    "CROSS-DETECTION MATRIX SUMMARY",
    f"scorer: {args.scorer}",
    f"threshold: {'score==2' if args.threshold == 2 else 'score>=1'}",
    "=" * 72,
    f"\nRows joined: {len(df)}",
    f"Judge base rate (medical_advice): {neg_rate:.3f}",
    "\nGlobal score=2 / score>=1 rates:",
]

for c in SCORED_CONCEPTS:
    col = f"score_{c}"
    if col in df.columns:
        n2 = int((df[col] == 2).sum())
        nge1 = int((df[col] >= 1).sum())
        lines.append(
            f"  {c:18s}  score2={n2:4d} ({100*n2/len(df):5.1f}%)"
            f"  ge1={nge1:4d} ({100*nge1/len(df):5.1f}%)"
        )

lines.append("\nDiagonal (self-detection, medium dose):")
for _, r in diag_summary.iterrows():
    lines.append(
        f"  {r['concept']:18s}  "
        f"P={r['diag']:.3f}  "
        f"signal_vs_med={r['signal_vs_medical_base']:+.3f}  "
        f"max_off={r['max_offdiag']:.3f} ({r['max_offdiag_col']})  "
        f"diag-minus-off={r['diag_minus_max_offdiag']:+.3f}"
    )

lines.append("\nMax off-diagonal (medium dose):")
for _, r in diag_summary.iterrows():
    lines.append(
        f"  {r['concept']:18s}  max_off={r['max_offdiag']:.3f}  col={r['max_offdiag_col']}"
    )

# Explicit refusal sanity block.
if "refusal" in medium.index and "refusal" in medium.columns:
    lines.append("\nRefusal specificity block, medium dose:")
    for injected_c in row_order:
        if injected_c in medium.index:
            lines.append(
                f"  {injected_c:18s} → refusal  {float(medium.loc[injected_c, 'refusal']):.3f}"
            )

# Explicit eval-awareness sanity block.
if "eval_awareness" in medium.columns:
    lines.append("\nEval-awareness specificity block, medium dose:")
    for injected_c in row_order:
        if injected_c in medium.index:
            lines.append(
                f"  {injected_c:18s} → eval_awareness  {float(medium.loc[injected_c, 'eval_awareness']):.3f}"
            )

summary = "\n".join(lines)

print("\n" + summary)

with open(SUMMARY_TXT, "w") as f:
    f.write(summary)

print(f"\nSaved matrix      → {MATRIX_CSV}")
print(f"Saved consistency → {CONSIST_CSV}")
print(f"Saved summary     → {SUMMARY_TXT}")
