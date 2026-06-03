"""
10c_analyze.py — parse scores, compute detection rate tables, bootstrap CIs, plots.

Inputs:
    /content/scores_raw.jsonl        (from 10_score.py)
    /content/injection_meta.parquet  (from 08_inject.py)
    /content/decode_outputs.parquet  (from 09_decode.py, only needed for --validate)

Outputs:
    /content/scores.parquet          full scored dataset merged with metadata
    /content/detection_rates.csv     P(score>=1) and P(score==2) by condition x dose
    /content/delta_ci.csv            Δ gap with 95% bootstrap CIs
    /content/scoring_results.png     3-panel figure

Hand-validation (print stratified sample for manual checking):
    !python 10c_analyze.py --validate 50
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES_RAW  = "/content/scores_raw.jsonl"
META_PATH   = "/content/injection_meta.parquet"
DECODE_PATH = "/content/decode_outputs.parquet"

ap = argparse.ArgumentParser()
ap.add_argument("--validate", type=int, default=0,
                help="Print N examples for hand-validation (0 = skip)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

np.random.seed(args.seed)

# ══════════════════════════════════════════════════════════════════════════════
# 1. PARSE SCORES
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for line in open(SCORES_RAW):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
        rows.append({
            "row":       int(r["row"]),
            "score":     int(r["score"]),
            "reasoning": str(r.get("reasoning", "")),
        })
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  Skipping malformed line: {e}  |  {line[:80]}")

scores_df = pd.DataFrame(rows)
print(f"Parsed {len(scores_df)} score rows from {SCORES_RAW}")
print(f"  Score distribution: { {s: int((scores_df.score==s).sum()) for s in [-1,0,1,2]} }")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MERGE WITH METADATA
# ══════════════════════════════════════════════════════════════════════════════
meta = pd.read_parquet(META_PATH)
full = meta.merge(scores_df, on="row", how="inner")
print(f"After merge: {len(full)} rows  "
      f"(meta={len(meta)}, scores={len(scores_df)}, matched={len(full)})")

# Drop API/parse errors
n_errors = (full.score == -1).sum()
if n_errors:
    print(f"Dropping {n_errors} rows with score=-1 ({100*n_errors/len(full):.1f}%)")
full = full[full.score >= 0].copy()

full["detect_any"]      = (full.score >= 1).astype(int)
full["detect_explicit"] = (full.score == 2).astype(int)

print(f"\nFinal analysis dataset: {len(full)} rows")
print("Condition counts:")
print(full.condition.value_counts().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 3. DETECTION RATE TABLES
# ══════════════════════════════════════════════════════════════════════════════
for label, col in [
    ("score >= 1  (any signal)", "detect_any"),
    ("score == 2  (explicit refusal)", "detect_explicit"),
]:
    pivot = full.pivot_table(
        index="condition", columns="target_cos", values=col, aggfunc="mean"
    ).round(3)
    print(f"\n{'═'*80}")
    print(f"P({label})")
    print(pivot.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 4. DELTA TABLE: C1 vs MEAN(C3, C4)
# ══════════════════════════════════════════════════════════════════════════════
CONTROLS = ["C3_random", "C4_nullDoM"]
doses    = sorted(full.target_cos.unique())

pivot_ex    = full.pivot_table(index="condition", columns="target_cos",
                               values="detect_explicit", aggfunc="mean")
c1_rates    = pivot_ex.loc["C1_refusal_pos"]
ctrl_rates  = pivot_ex.loc[CONTROLS].mean(axis=0)
delta_point = c1_rates - ctrl_rates

print(f"\n{'═'*80}")
print("Δ P(score==2) = C1_refusal_pos − mean(C3_random, C4_nullDoM)")
print(delta_point.round(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP CIs
# ══════════════════════════════════════════════════════════════════════════════
N_BOOT    = 2000
c1_data   = full[full.condition == "C1_refusal_pos"]
ctrl_data = full[full.condition.isin(CONTROLS)]

ci_records = []
for t in doses:
    c1_cell   = c1_data[c1_data.target_cos == t]["detect_explicit"].values
    ctrl_cell = ctrl_data[ctrl_data.target_cos == t]["detect_explicit"].values

    if len(c1_cell) == 0 or len(ctrl_cell) == 0:
        ci_records.append(dict(target_cos=t, delta=np.nan,
                               ci_lo=np.nan, ci_hi=np.nan))
        continue

    boot = [
        np.random.choice(c1_cell,   len(c1_cell),   replace=True).mean() -
        np.random.choice(ctrl_cell, len(ctrl_cell),  replace=True).mean()
        for _ in range(N_BOOT)
    ]
    ci_records.append(dict(
        target_cos = t,
        delta      = float(delta_point[t]),
        ci_lo      = float(np.percentile(boot, 2.5)),
        ci_hi      = float(np.percentile(boot, 97.5)),
        n_c1       = len(c1_cell),
        n_ctrl     = len(ctrl_cell),
    ))

ci_df = pd.DataFrame(ci_records)
print(f"\n{'═'*80}")
print("Δ P(score==2) with 95% bootstrap CIs  (N_BOOT=2000)")
print(ci_df[["target_cos","delta","ci_lo","ci_hi"]].round(3).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════════════════
full.to_parquet("/content/scores.parquet", index=False)
ci_df.to_csv("/content/delta_ci.csv", index=False)
full.pivot_table(index="condition", columns="target_cos",
                 values=["detect_any","detect_explicit"], aggfunc="mean"
                 ).round(3).to_csv("/content/detection_rates.csv")
print("\nSaved: scores.parquet, delta_ci.csv, detection_rates.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
COND_ORDER = ["C1_refusal_pos", "C2_refusal_neg", "C3_random", "C4_nullDoM"]
COLORS  = {"C1_refusal_pos":"steelblue", "C2_refusal_neg":"darkorange",
           "C3_random":"#999999",        "C4_nullDoM":"#bbbbbb"}
MARKERS = {"C1_refusal_pos":"o", "C2_refusal_neg":"s",
           "C3_random":".",       "C4_nullDoM":"."}
WIDTHS  = {"C1_refusal_pos":2.5, "C2_refusal_neg":1.5,
           "C3_random":1.0,       "C4_nullDoM":1.0}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("NLA Refusal Detection: Injection Sweep", fontsize=13, fontweight="bold")

# ── Panel 1: detection rate by condition × dose ──────────────────────────────
ax = axes[0]
for cond in COND_ORDER:
    sub = full[full.condition == cond]
    if sub.empty:
        continue
    rates = sub.groupby("target_cos")["detect_explicit"].mean()
    ls = "-" if "C1" in cond else ("--" if "C2" in cond else ":")
    ax.plot(rates.index, rates.values,
            linestyle=ls, marker=MARKERS[cond],
            color=COLORS[cond], linewidth=WIDTHS[cond],
            markersize=5, label=cond)

ax.axvline(0.525, color="black", lw=1, ls=":", alpha=0.5, label="natural refused (0.525)")
ax.axvline(0.300, color="black", lw=1, ls="-.", alpha=0.25, label="anchor baseline (0.300)")
ax.set_xlabel("target_cos (injection strength)")
ax.set_ylabel("P(score == 2)")
ax.set_title("Explicit refusal detection rate")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ── Panel 2: Δ with 95% CI ───────────────────────────────────────────────────
ax = axes[1]
ci_plot = ci_df.dropna()
ax.fill_between(ci_plot.target_cos, ci_plot.ci_lo, ci_plot.ci_hi,
                alpha=0.2, color="steelblue", label="95% bootstrap CI")
ax.plot(ci_plot.target_cos, ci_plot.delta, "-o",
        color="steelblue", lw=2, markersize=5)
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0.525, color="black", lw=1, ls=":", alpha=0.5)
ax.set_xlabel("target_cos (injection strength)")
ax.set_ylabel("Δ P(score==2)  [C1 − mean(C3,C4)]")
ax.set_title("Detection gap vs controls with 95% CI")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 3: C1 score distribution heatmap ──────────────────────────────────
ax = axes[2]
c1_only = full[full.condition == "C1_refusal_pos"]
score_counts = (c1_only.groupby(["target_cos", "score"])
                .size().unstack(fill_value=0))
for s in [0, 1, 2]:
    if s not in score_counts.columns:
        score_counts[s] = 0
score_counts = score_counts[[0, 1, 2]]
score_frac   = score_counts.div(score_counts.sum(axis=1), axis=0)

im = ax.imshow(score_frac.T.values, aspect="auto", cmap="Blues",
               vmin=0, vmax=1, origin="lower")
ax.set_xticks(range(len(score_frac.index)))
ax.set_xticklabels([f"{t:+.2f}" for t in score_frac.index],
                   rotation=45, fontsize=8)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["0 – none", "1 – weak", "2 – explicit"])
ax.set_xlabel("target_cos")
ax.set_title("C1: Score distribution by dose")
plt.colorbar(im, ax=ax, label="fraction of 100 samples")

plt.tight_layout()
plt.savefig("/content/scoring_results.png", dpi=150, bbox_inches="tight")
print("Saved: scoring_results.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. OPTIONAL HAND-VALIDATION PASS
# ══════════════════════════════════════════════════════════════════════════════
if args.validate > 0:
    import textwrap

    def extract_explanation(t):
        if not isinstance(t, str): return "(no output)"
        m = re.search(r"<explanation>(.*?)</explanation>", t, re.DOTALL)
        return (m.group(1) if m else t).strip()

    dec = pd.read_parquet(DECODE_PATH)
    dec["expl"] = dec["nla_output"].apply(extract_explanation)
    fv = full.merge(dec[["row","expl"]], on="row", how="left")

    # Stratified: C1 × every dose + spot checks of controls
    n_per_cell = max(1, args.validate // (len(doses) + 4))
    parts = []
    for t in doses:
        cell = fv[(fv.condition == "C1_refusal_pos") & (fv.target_cos == t)]
        parts.append(cell.sample(min(n_per_cell, len(cell)), random_state=args.seed))
    for cond in ["C3_random", "C4_nullDoM"]:
        cell = fv[fv.condition == cond]
        parts.append(cell.sample(min(3, len(cell)), random_state=args.seed))

    val_df = pd.concat(parts).reset_index(drop=True)
    W = 100
    print(f"\n{'═'*W}")
    print(f"HAND-VALIDATION SAMPLE  (n={len(val_df)})")
    print(f"Mark each score as AGREE or OVERRIDE. "
          f"Target: ≥90% agreement before reporting rates.")
    print(f"{'═'*W}")
    for i, r in val_df.iterrows():
        print(f"\n[{i+1}/{len(val_df)}] cond={r.condition}  t={r.target_cos:+.2f}  "
              f"row={r.row}  JUDGE_SCORE={r.score}")
        print(f"  Reasoning: {r.reasoning}")
        expl_wrapped = textwrap.fill(
            str(r.expl), width=W, initial_indent="  NLA: ", subsequent_indent="       ")
        print(expl_wrapped)
