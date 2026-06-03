"""
07_build_direction.py - build and validate v_refusal (cross-pool diff-of-means).

Direction:
  v_refusal = normalize( mean(refused_harmful) - mean(complied_harmless) )
  on RAW layer-21 activations, classes balanced (no centering - it cancels).

Validation (all leakage-controlled):
  1. 5-fold CV projection AUROC for diff-of-means: build direction on train
     folds, score held-out folds by cos(h, v_hat). Generalization, not fit.
  2. Regularized logistic-probe CV AUROC: baseline for how much linear signal
     exists. If diff-of-means ~= probe, the simple direction captures it.
  3. SHUFFLED-LABEL NULL for both: at d=3584, n~220 even random labels can
     separate. If the null is ~0.5 and real is high, the signal is real.
  4. Natural cos(h, v_refusal) per group: the ANCHOR distribution is unbiased
     (anchors never touch the direction) and sets the reference scale for the
     injection detection threshold downstream.

PREREQ: run 05_extract.py and 06_label.py first.

Inputs / outputs (under workspace/):
  activations.npy [360, 3584], labeled.parquet
  -> v_refusal.npy, direction_meta.json, cos_distributions.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"

ACTS = WORKSPACE / "activations.npy"
LABEL = WORKSPACE / "labeled.parquet"
VOUT = WORKSPACE / "v_refusal.npy"
META = WORKSPACE / "direction_meta.json"
PLOT = WORKSPACE / "cos_distributions.png"
SEED = 42

for path in (ACTS, LABEL):
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — run earlier pipeline steps first")

WORKSPACE.mkdir(parents=True, exist_ok=True)

acts = np.load(ACTS).astype(np.float64)
df = pd.read_parquet(LABEL)
assert len(df) == len(acts), f"row mismatch: {len(df)} labels vs {len(acts)} activations"

refused_idx  = df.index[(df.pool == "harmful")  & (df.label == "refused")].to_numpy()
complied_idx = df.index[(df.pool == "harmless") & (df.label == "complied")].to_numpy()
anchor_idx   = df.index[(df.pool == "anchor")   & (df.label == "complied")].to_numpy()
print(f"refused-harmful={len(refused_idx)}  complied-harmless={len(complied_idx)}  anchor={len(anchor_idx)}")

rng = np.random.default_rng(SEED)
n = min(len(refused_idx), len(complied_idx))
refused_bal  = rng.permutation(refused_idx)[:n]
complied_bal = rng.permutation(complied_idx)[:n]
print(f"balanced to {n} per class\n")

X = np.concatenate([acts[refused_bal], acts[complied_bal]])
y = np.concatenate([np.ones(n), np.zeros(n)])


def dom_dir(Xtr, ytr):
    v = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    return v / np.linalg.norm(v)


def cos(H, vhat):
    return (H @ vhat) / np.linalg.norm(H, axis=1)


def dom_score(Xtr, ytr, Xte, yte):
    return roc_auc_score(yte, cos(Xte, dom_dir(Xtr, ytr)))


def probe_score(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.01, max_iter=3000))  # strong L2
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.decision_function(Xte))


def cv(Xd, yd, scorer):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return np.array([scorer(Xd[tr], yd[tr], Xd[te], yd[te]) for tr, te in skf.split(Xd, yd)])


y_null = rng.permutation(y)
dom_real,  dom_null  = cv(X, y, dom_score),   cv(X, y_null, dom_score)
prb_real,  prb_null  = cv(X, y, probe_score), cv(X, y_null, probe_score)

print("=== 5-fold CV AUROC (held-out) ===")
print(f"diff-of-means   real: {dom_real.mean():.3f} +/- {dom_real.std():.3f}   "
      f"null(shuffled): {dom_null.mean():.3f} +/- {dom_null.std():.3f}")
print(f"L2 probe        real: {prb_real.mean():.3f} +/- {prb_real.std():.3f}   "
      f"null(shuffled): {prb_null.mean():.3f} +/- {prb_null.std():.3f}")
print("interpretation: real >> null(~0.5) => signal is real, not overfitting.")
print("                probe ~= diff-of-means => simple direction captures the linear signal.")

# ---- final direction on ALL balanced data ----------------------------------
v_refusal = dom_dir(X, y)
np.save(VOUT, v_refusal.astype(np.float32))

print("\n=== natural cos(h, v_refusal) by group ===")
groups = {
    "refused-harmful  (in-sample)" : refused_bal,
    "complied-harmless (in-sample)": complied_bal,
    "anchor           (UNBIASED)"  : anchor_idx,
}
cos_stats = {}
for name, idx in groups.items():
    c = cos(acts[idx], v_refusal)
    cos_stats[name.strip()] = dict(n=int(len(idx)), mean=float(c.mean()), std=float(c.std()),
                                   p5=float(np.percentile(c, 5)), p95=float(np.percentile(c, 95)))
    print(f"{name:32s} n={len(idx):3d}  mean={c.mean():+.3f}  std={c.std():.3f}  "
          f"[p5={np.percentile(c,5):+.3f}, p95={np.percentile(c,95):+.3f}]")

dsep = cos(acts[refused_bal], v_refusal).mean() - cos(acts[complied_bal], v_refusal).mean()
print(f"\nin-sample natural-cos separation (refused - complied): {dsep:+.3f}")
print(f"anchor natural cos mean = {cos_stats['anchor           (UNBIASED)'.strip()]['mean']:+.3f} "
      f"<- injection pushes anchors UP from here")

# ---- plot ------------------------------------------------------------------
plt.figure(figsize=(8, 4))
for name, idx in groups.items():
    plt.hist(cos(acts[idx], v_refusal), bins=30, alpha=0.5, label=name.strip(), density=True)
plt.xlabel("cos(h, v_refusal)"); plt.ylabel("density")
plt.title("Natural cos(h, v_refusal) by group"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(PLOT, dpi=120)
print(f"\nsaved {PLOT}")

meta = {
    "method": "cross-pool diff-of-means (refused-harmful vs complied-harmless)",
    "n_per_class": int(n),
    "dom_cv_auroc": [float(dom_real.mean()), float(dom_real.std())],
    "dom_null_auroc": [float(dom_null.mean()), float(dom_null.std())],
    "probe_cv_auroc": [float(prb_real.mean()), float(prb_real.std())],
    "probe_null_auroc": [float(prb_null.mean()), float(prb_null.std())],
    "natural_cos_separation": float(dsep),
    "cos_stats": cos_stats,
}
META.write_text(json.dumps(meta, indent=2))
print(f"saved v_refusal -> {VOUT}  (||v||=1, d={len(v_refusal)})")
