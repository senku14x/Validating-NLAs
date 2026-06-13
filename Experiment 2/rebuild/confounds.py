"""Confound-controlled probe battery for Experiment 2.

The v1 lesson, in one sentence: a probe/AUROC baseline is only a baseline if it
is confound-controlled. Our cleanest-looking v1 result (eval-awareness AUROC
0.990) was mostly prompt length; after control it was 0.622. So in the rebuild
NO single AUROC is ever reported on its own. Every concept goes through
`probe_battery`, which returns FOUR numbers, each with a confidence interval:

    raw                  AUROC of the probe on the activations as-is
    length_residualized  AUROC after OLS-regressing length out of the activations
    length_only          AUROC of a probe that sees ONLY length
    null                 AUROC of the same pipeline on permuted labels (~0.5 sanity floor)

A concept counts as "represented" under a *pre-registered* gate (frozen here,
not chosen after seeing results):

    represented  iff  length_residualized.ci_lo > max(REPRESENTED_FLOOR, null.ci_hi)

i.e. the length-controlled signal must clear an absolute floor AND beat the
permutation null's upper edge. We deliberately do NOT require resid > length_only:
when length is itself near-perfectly predictive (length_only ~0.99) nothing can
beat it, yet a real residual signal still means the concept is encoded beyond
length. `length_only` stays as a diagnostic (surfaced via `length_inflation`).
Comparing CI *bounds*, not point estimates, because at n~60-120 a "0.622 vs
0.854" gap can be noise.

Design choices that matter (and why):

* Instrument = regularized logistic regression, NOT difference-of-means
  projection. For a *representation* claim you want the most sensitive fair
  linear readout; DoM is more constrained and can understate what is linearly
  present, which would manufacture a fake verbalization gap. (Keep DoM for the
  injection *direction* in the matrix step; that is a different question.)

* Evaluation = nested cross-validation. d_model >> n, so an unregularized probe
  hits AUROC 1.000 in-sample on noise. Outer StratifiedGroupKFold gives
  out-of-fold scores; inner StratifiedGroupKFold selects C. The `null` band
  confirms the whole pipeline returns ~0.5 when the labels carry no signal.

* Grouping = pair id. Matched pairs are not independent; a pair's present and
  absent halves must never straddle the train/test split, or the fold leaks.
  CIs are cluster-bootstrapped at the group (pair) level for the same reason.

Pure CPU / numpy / sklearn — no GPU, no model weights. Import from every stage;
do not reimplement AUROC logic anywhere else.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── pre-registered constants (freeze before running; do not tune to results) ──
DEFAULT_C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
N_SPLITS = 5
N_BOOT = 2000          # cluster-bootstrap resamples for each CI
N_NULL = 25            # label permutations for the null band
ALPHA = 0.05           # 95% intervals
REPRESENTED_FLOOR = 0.60
SEED = 42


@dataclass
class AUROC:
    """A single AUROC point estimate with a cluster-bootstrap CI."""
    auroc: float
    ci_lo: float
    ci_hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.auroc:.3f} [{self.ci_lo:.3f}, {self.ci_hi:.3f}] (n={self.n})"


def _clf(C: float):
    # StandardScaler is fit per-fold inside CV (no leakage); class_weight guards
    # against mild imbalance in non-paired designs.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000, class_weight="balanced"),
    )


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def _oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    c_grid: tuple[float, ...],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold decision scores via nested CV.

    Outer StratifiedGroupKFold produces held-out scores for every row; an inner
    StratifiedGroupKFold over `c_grid` picks C on the training portion only.
    With a length-1 c_grid the inner loop is skipped (used for the null, where
    re-selecting C on every permutation is wasted compute).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    oof = np.full(len(y), np.nan)

    n_groups = len(np.unique(groups))
    k = max(2, min(n_splits, n_groups))
    outer = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

    for tr, te in outer.split(X, y, groups):
        best_c = c_grid[len(c_grid) // 2]
        if len(c_grid) > 1:
            gtr = groups[tr]
            ki = max(2, min(n_splits, len(np.unique(gtr))))
            inner = StratifiedGroupKFold(n_splits=ki, shuffle=True, random_state=seed)
            best_auc = -1.0
            for C in c_grid:
                isc = np.full(len(tr), np.nan)
                for itr, ite in inner.split(X[tr], y[tr], gtr):
                    isc[ite] = _clf(C).fit(X[tr][itr], y[tr][itr]).decision_function(X[tr][ite])
                m = ~np.isnan(isc)
                a = _safe_auc(y[tr][m], isc[m])
                if a > best_auc:
                    best_auc, best_c = a, C
        oof[te] = _clf(best_c).fit(X[tr], y[tr]).decision_function(X[te])

    return oof


def _cluster_bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI from resampling whole groups (pairs) with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    aucs = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in chosen])
        yb, sb = y[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def residualize(X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """OLS-regress `covariates` (+ intercept) out of every column of X.

    Returns the residuals: the part of each activation dimension that length
    (or any covariate matrix) cannot explain linearly. Matching is stronger
    than residualizing where you can do it (see concepts.yaml audit), but
    residualization is the universal fallback and is always reported.
    """
    X = np.asarray(X, dtype=float)
    C = np.asarray(covariates, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    C = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C @ beta


def _auroc_with_ci(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    c_grid: tuple[float, ...],
    n_splits: int,
    n_boot: int,
    alpha: float,
    seed: int,
) -> AUROC:
    oof = _oof_scores(features, y, groups, c_grid, n_splits, seed)
    m = ~np.isnan(oof)
    auc = _safe_auc(y[m], oof[m])
    lo, hi = _cluster_bootstrap_ci(y[m], oof[m], groups[m], n_boot, alpha, seed)
    return AUROC(round(auc, 4), round(lo, 4), round(hi, 4), int(m.sum()))


def probe_battery(
    X: np.ndarray,
    y,
    lengths,
    groups=None,
    *,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    n_splits: int = N_SPLITS,
    n_boot: int = N_BOOT,
    n_null: int = N_NULL,
    alpha: float = ALPHA,
    floor: float = REPRESENTED_FLOOR,
    seed: int = SEED,
) -> dict:
    """Run the four-number confound battery for one concept.

    Parameters
    ----------
    X        : [n, d] activations (last-token or mean-pooled — caller's choice).
    y        : [n] binary labels (1 = concept present).
    lengths  : [n] prompt token counts, the primary confound to control.
    groups   : [n] pair ids so matched halves never split across folds. If None,
               each row is its own group (use for non-paired designs only).

    Returns
    -------
    dict with AUROC objects under keys raw / length_residualized / length_only /
    null, plus `represented` (bool, the pre-registered gate) and `gate` (str,
    the rule as applied). Everything is JSON-serializable via `battery_to_dict`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    lengths = np.asarray(lengths, dtype=float)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)

    kw = dict(c_grid=c_grid, n_splits=n_splits, n_boot=n_boot, alpha=alpha, seed=seed)

    raw = _auroc_with_ci(X, y, groups, **kw)
    resid = _auroc_with_ci(residualize(X, lengths), y, groups, **kw)
    length_only = _auroc_with_ci(lengths[:, None], y, groups, **kw)

    # Null: permute labels, rerun the raw pipeline at a fixed C, repeat. A
    # healthy pipeline lands at ~0.5 here; anything higher means leakage in the
    # CV/grouping itself, independent of any real signal.
    fixed_c = (c_grid[len(c_grid) // 2],)
    rng = np.random.default_rng(seed)
    null_aucs = []
    for _ in range(n_null):
        yp = rng.permutation(y)
        oof = _oof_scores(X, yp, groups, fixed_c, n_splits, seed)
        m = ~np.isnan(oof)
        null_aucs.append(_safe_auc(yp[m], oof[m]))
    null = AUROC(
        round(float(np.mean(null_aucs)), 4),
        round(float(np.percentile(null_aucs, 2.5)), 4),
        round(float(np.percentile(null_aucs, 97.5)), 4),
        len(y),
    )

    # The verdict rests on the LENGTH-RESIDUALIZED signal: length has already
    # been regressed out, so this is "concept signal beyond length". It must
    # (a) clear an absolute floor and (b) beat the permutation null's upper
    # edge. We deliberately do NOT require resid > length_only: when length is
    # itself near-perfectly predictive (length_only ~0.99) nothing can beat it,
    # yet a real residual signal still means the concept is encoded beyond
    # length. length_only stays as a diagnostic, surfaced via length_inflation
    # (how much of raw was just length).
    bar = round(max(floor, null.ci_hi), 4)
    represented = bool(resid.ci_lo > bar)
    length_inflation = round(raw.auroc - resid.auroc, 4)
    gate = (
        f"length_residualized.ci_lo ({resid.ci_lo:.3f}) > "
        f"max(floor {floor:.2f}, null.ci_hi {null.ci_hi:.3f}) = {bar:.3f}"
    )

    return {
        "raw": raw,
        "length_residualized": resid,
        "length_only": length_only,
        "null": null,
        "represented": represented,
        "length_inflation": length_inflation,
        "gate": gate,
        "floor": floor,
    }


def battery_to_dict(battery: dict) -> dict:
    """JSON-serializable view of a probe_battery result."""
    out = {}
    for k, v in battery.items():
        out[k] = asdict(v) if isinstance(v, AUROC) else v
    return out


def summarize(battery: dict, name: str = "") -> str:
    """One human-readable block per concept."""
    verdict = "REPRESENTED" if battery["represented"] else "not represented"
    head = f"── {name or 'concept'} : {verdict} ─────────────────"
    return "\n".join(
        [
            head,
            f"  raw                  {battery['raw']}",
            f"  length_residualized  {battery['length_residualized']}",
            f"  length_only          {battery['length_only']}",
            f"  null                 {battery['null']}",
            f"  length_inflation     {battery['length_inflation']:+.3f}  (raw - residualized)",
            f"  gate: {battery['gate']}",
        ]
    )


if __name__ == "__main__":
    import sys

    print(
        "confounds.py is a library — import probe_battery from it.\n"
        "Run the self-test with:  python test_confounds.py",
        file=sys.stderr,
    )
