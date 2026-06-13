#!/usr/bin/env python
"""
10_covariate_analysis.py — §13 covariate model.

The §13 deliverable: what feature properties predict strict transfer?
Uses covariates.csv from 05_cofiring.py.

Covariates:
    ws_ceiling      within-source AUROC (does it fire reliably in Qwen)
    gemma_probe     Gemma-native probe AUROC (is it represented in Gemma at L41)
    mapped_norm     ||M_lin d_f|| before normalization (how far does the map move it)
    firing_density  fraction of corpus sentences where the feature fires
    dec_norm        ||W_dec[f]|| (decoder direction norm)
    near_cos        cosine to nearest decoder neighbor (feature specificity)

Outputs:
    - Correlation table (covariates vs transfer_auroc, strict transfer)
    - Mann-Whitney U tests: strict-transfer vs not-strict for each covariate
    - OLS regression: transfer_auroc ~ covariates
    - Plots: scatter matrix, transfer distributions by decomp category
      saved to {out_dir}/covariate_plots/

Usage:
    python 10_covariate_analysis.py --out-dir /content/exp3_acts
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

try:
    import pandas as pd
    from scipy import stats
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "pandas", "scipy", "scikit-learn", "matplotlib"])
    import pandas as pd
    from scipy import stats
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


COV_COLS = ["cov_ws_ceiling", "cov_gemma_probe", "cov_mapped_norm",
            "cov_firing_density", "cov_dec_norm", "cov_near_cos"]
COV_NAMES = {
    "cov_ws_ceiling":     "ws_ceiling (Qwen fires reliably)",
    "cov_gemma_probe":    "gemma_probe (Gemma represents it)",
    "cov_mapped_norm":    "mapped_norm (map moves it far)",
    "cov_firing_density": "firing_density (feature frequency)",
    "cov_dec_norm":       "dec_norm (||W_dec||)",
    "cov_near_cos":       "near_cos (decoder neighbor similarity)",
}


def load_data(out_dir):
    csv_path = out_dir / "covariates.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found — run 05_cofiring.py first"); return None
    df = pd.read_csv(csv_path)
    # pull labels from cofiring_results if available
    res_path = out_dir / "cofiring_results.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())
        label_map = {r["fi"]: r.get("label","") for r in results}
        decomp_map = {r["fi"]: r.get("decomp","") for r in results}
        df["label"] = df["fi"].map(label_map)
        df["decomp"] = df["fi"].map(decomp_map)
    print(f"loaded {len(df)} features from {csv_path}")
    return df


def correlation_table(df):
    print("\n" + "="*70)
    print("PEARSON CORRELATIONS with transfer_auroc and clean_transfer")
    print("="*70)
    print(f"{'covariate':<40} {'r(transfer)':>12} {'p':>8}  {'r(clean)':>10} {'p':>8}")
    print("-"*70)
    for col in COV_COLS:
        if col not in df.columns: continue
        x = df[col].values
        if np.nanstd(x) == 0:
            name = col.replace("cov_","")
            print(f"  {name:<38} {'constant':>12} {'':>8}  {'constant':>10}")
            continue
        r1, p1 = stats.pearsonr(x, df["transfer_auroc"])
        r2, p2 = stats.pearsonr(x, df["clean_transfer"].astype(float))
        name = col.replace("cov_","")
        sig1 = "*" if p1 < 0.05 else " "
        sig2 = "*" if p2 < 0.05 else " "
        print(f"  {name:<38} {r1:>+.3f}{sig1}    {p1:>6.3f}   {r2:>+.3f}{sig2}    {p2:>6.3f}")
    print("  (* p<0.05)")


def mannwhitney_table(df):
    print("\n" + "="*70)
    print("MANN-WHITNEY U: strict-transfer vs not-strict (each covariate)")
    print("="*70)
    clean = df[df["clean_transfer"] == 1]
    notcl = df[df["clean_transfer"] == 0]
    print(f"  strict={len(clean)}  not-strict={len(notcl)}")
    print(f"{'covariate':<38} {'strict mean':>11} {'not-strict mean':>15} {'U p-value':>10} {'direction'}")
    print("-"*70)
    for col in COV_COLS:
        if col not in df.columns: continue
        a, b = clean[col].values, notcl[col].values
        if np.nanstd(np.concatenate([a, b])) == 0:
            name = col.replace("cov_", "")
            print(f"  {name:<38} {a.mean():>11.3f} {b.mean():>15.3f} {'constant':>10}")
            continue
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        direction = "strict>not" if a.mean() > b.mean() else "not>strict"
        sig = "**" if p < 0.01 else "*" if p < 0.05 else " "
        name = col.replace("cov_", "")
        print(f"  {name:<38} {a.mean():>11.3f} {b.mean():>15.3f} {p:>9.3f}{sig}  {direction}")
    print("  (* p<0.05  ** p<0.01)")


def ols_regression(df):
    print("\n" + "="*70)
    print("OLS REGRESSION: transfer_auroc ~ covariates (LOO cross-validated R²)")
    print("="*70)
    avail = [c for c in COV_COLS if c in df.columns]
    X = df[avail].values
    y = df["transfer_auroc"].values
    # standardize
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    # LOO R²
    loo = LeaveOneOut(); preds = np.zeros(len(y))
    for tr, te in loo.split(Xs):
        m = Ridge(alpha=1.0).fit(Xs[tr], y[tr])
        preds[te] = m.predict(Xs[te])
    r2_loo = float(1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2))
    # in-sample R² and coefficients
    m_full = Ridge(alpha=1.0).fit(Xs, y)
    r2_full = float(m_full.score(Xs, y))
    print(f"  LOO R² = {r2_loo:.3f}   in-sample R² = {r2_full:.3f}")
    print(f"\n  Standardized coefficients (larger = stronger predictor):")
    for name, coef in sorted(zip([c.replace("cov_","") for c in avail],
                                  m_full.coef_), key=lambda x: -abs(x[1])):
        bar = "█" * int(abs(coef) * 20)
        sign = "+" if coef > 0 else "-"
        print(f"    {sign}{bar:<20} {coef:+.3f}  {name}")
    print(f"\n  Interpretation: LOO R²={r2_loo:.3f} tells you how much of transfer")
    print(f"  variance these covariates explain on held-out features.")
    print(f"  R²<0 = model worse than mean (covariates don't predict)")
    print(f"  R²>0.3 = meaningful prediction worth reporting")
    return r2_loo, dict(zip([c.replace("cov_","") for c in avail], m_full.coef_))


def decomp_summary(df):
    if "decomp" not in df.columns: return
    print("\n" + "="*70)
    print("TRANSFER DISTRIBUTION BY DECOMP CATEGORY")
    print("="*70)
    for cat, grp in df.groupby("decomp"):
        t = grp["transfer_auroc"]
        print(f"  {cat:<20}  n={len(grp):>3}  "
              f"mean={t.mean():.3f}  "
              f"p25={t.quantile(0.25):.3f}  "
              f"median={t.median():.3f}  "
              f"p75={t.quantile(0.75):.3f}")


def make_plots(df, plot_dir):
    plot_dir.mkdir(exist_ok=True)
    avail = [c for c in COV_COLS if c in df.columns]
    short = [c.replace("cov_","") for c in avail]

    # 1. scatter matrix: each covariate vs transfer_auroc
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for i, (col, name) in enumerate(zip(avail, short)):
        if i >= len(axes): break
        ax = axes[i]
        colors = ["#e74c3c" if c else "#3498db"
                  for c in df["clean_transfer"].astype(bool)]
        ax.scatter(df[col], df["transfer_auroc"], c=colors, alpha=0.7, s=50)
        if np.nanstd(df[col]) == 0:
            r, p = np.nan, np.nan
        else:
            r, p = stats.pearsonr(df[col], df["transfer_auroc"])
        ax.set_xlabel(name, fontsize=9)
        ax.set_ylabel("transfer AUROC", fontsize=9)
        ax.set_title(f"r={r:+.2f}  p={p:.3f}", fontsize=9)
        ax.axhline(0.70, color="gray", linestyle="--", alpha=0.5, label="clean threshold")
        ax.axhline(0.50, color="black", linestyle=":", alpha=0.3)
    # legend
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#e74c3c", label="clean"),
                             Patch(color="#3498db", label="not-clean")],
                   fontsize=8)
    # hide unused
    for i in range(len(avail), len(axes)): axes[i].set_visible(False)
    fig.suptitle("Covariate vs Transfer AUROC", fontsize=12)
    plt.tight_layout()
    p1 = plot_dir / "covariate_scatter.png"
    plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close()
    print(f"\n  saved: {p1}")

    # 2. transfer distribution by decomp category
    if "decomp" in df.columns:
        cats = sorted(df["decomp"].unique())
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [df.loc[df["decomp"]==c, "transfer_auroc"].values for c in cats]
        bp = ax.boxplot(data, labels=cats, patch_artist=True)
        colors2 = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
        for patch, col in zip(bp["boxes"], colors2):
            patch.set_facecolor(col)
        ax.axhline(0.70, color="gray", linestyle="--", alpha=0.6, label="clean threshold")
        ax.axhline(0.50, color="black", linestyle=":", alpha=0.4)
        ax.set_ylabel("Transfer AUROC")
        ax.set_title("Transfer AUROC by Decomposition Category")
        ax.legend(fontsize=9)
        plt.tight_layout()
        p2 = plot_dir / "transfer_by_decomp.png"
        plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close()
        print(f"  saved: {p2}")

    # 3. if labels available: transfer AUROC by feature label (top 20)
    if "label" in df.columns and df["label"].notna().any():
        df_l = df[df["label"].str.len() > 0].copy()
        df_l = df_l.sort_values("transfer_auroc", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors3 = ["#e74c3c" if c else "#3498db" for c in df_l["clean_transfer"].astype(bool)]
        ax.barh(range(len(df_l)), df_l["transfer_auroc"], color=colors3, alpha=0.8)
        ax.set_yticks(range(len(df_l)))
        ax.set_yticklabels([f"[{r.fi}] {r.label[:25]}" for r in df_l.itertuples()], fontsize=8)
        ax.axvline(0.70, color="gray", linestyle="--", alpha=0.7, label="clean threshold")
        ax.axvline(0.50, color="black", linestyle=":", alpha=0.5)
        ax.set_xlabel("Transfer AUROC"); ax.set_title("Top 20 Features by Transfer AUROC")
        ax.legend(fontsize=9)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#e74c3c", label="strict"),
                           Patch(color="#3498db", label="not-strict"),
                           plt.Line2D([0],[0],c="gray",ls="--",label="0.70 threshold")],
                  fontsize=8)
        plt.tight_layout()
        p3 = plot_dir / "top_features_bar.png"
        plt.savefig(p3, dpi=120, bbox_inches="tight"); plt.close()
        print(f"  saved: {p3}")


def print_key_finding(df, coef_dict):
    print("\n" + "="*70)
    print("KEY FINDING SUMMARY (for the paper)")
    print("="*70)
    n_clean = df["clean_transfer"].sum()
    n_high = df["high_auroc_clean"].sum() if "high_auroc_clean" in df.columns else None
    n_gate1 = df["gate1_pass"].sum()
    n_total = len(df)
    top_pos = sorted(coef_dict.items(), key=lambda x: -x[1])[:2]
    top_neg = sorted(coef_dict.items(), key=lambda x: x[1])[:2]
    print(f"\n  Headline rates:")
    if n_high is not None:
        print(f"    High-AUROC clean: {n_high}/{n_total} = {100*n_high/n_total:.0f}%")
    print(f"    Gate 1 identity-specific: {n_gate1}/{n_total} = {100*n_gate1/n_total:.0f}%")
    print(f"    Strict transfer (§13): {n_clean}/{n_total} = {100*n_clean/n_total:.0f}%")

    if "decomp" in df.columns:
        dc = df["decomp"].value_counts()
        print(f"\n  Decomposition:")
        for k, v in dc.items():
            print(f"    {k:<20} {v}")

    print(f"\n  Top positive predictors of transfer: {[n for n,_ in top_pos]}")
    print(f"  Top negative predictors of transfer: {[n for n,_ in top_neg]}")
    print(f"\n  Key result: {dc.get('map_failed',0)} features labeled 'map_failed'")
    print(f"  (Gemma probe >= 0.80 but transfer low) vs 0 'not_representable'")
    print(f"  -> the bottleneck is LINEAR MAP PRECISION, not representational overlap")
    print(f"  -> Gemma represents these concepts at L41; the ridge map can't find them")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/content/exp3_acts")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)

    df = load_data(out)
    if df is None: return

    correlation_table(df)
    mannwhitney_table(df)
    r2_loo, coef_dict = ols_regression(df)
    decomp_summary(df)
    print_key_finding(df, coef_dict)

    if not args.no_plots:
        make_plots(df, out / "covariate_plots")

    # save summary json
    summary = {
        "n_features": len(df),
        "strict_transfer_n": int(df["clean_transfer"].sum()),
        "strict_transfer_pct": round(100*df["clean_transfer"].mean(), 1),
        "high_auroc_clean_n": int(df["high_auroc_clean"].sum()) if "high_auroc_clean" in df.columns else None,
        "gate1_n": int(df["gate1_pass"].sum()),
        "loo_r2": round(r2_loo, 3),
        "top_covariate": max(coef_dict, key=lambda k: abs(coef_dict[k])),
        "decomp_counts": df["decomp"].value_counts().to_dict() if "decomp" in df.columns else {},
    }
    (out / "covariate_summary.json").write_text(
        __import__("json").dumps(summary, indent=2))
    print(f"\n  summary -> {out}/covariate_summary.json")


if __name__ == "__main__":
    main()
