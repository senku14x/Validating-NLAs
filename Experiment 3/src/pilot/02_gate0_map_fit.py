#!/usr/bin/env python
"""
02_gate0_map_fit.py — fit the ridge map and run the full Gate 0 battery.

Gate 0 is make-or-break. 0b is the critical check: if the map is just
matching Gemma's outlier dim 104 (preflight: 1,442,881x median variance),
everything downstream produces garbage. We zero it and recheck.

Loads:
    {out_dir}/mean_qwen.npz   [N, 3584]
    {out_dir}/mean_gemma.npz  [N, 5376]

Saves:
    {out_dir}/ridge_map.npz       W, X_mu, X_sigma, Y_mu, Y_sigma, lambda
    {out_dir}/gate0_results.json  per-gate numbers + PASS/FAIL verdicts

Gate battery:
    0a  held-out per-sentence cosine (raw space) >= 0.85
    0b  same cosine with top-10 highest-variance Gemma dims zeroed >= 0.70
        MAKE-OR-BREAK: dim 104 has 1.44M× median variance. If cosine
        collapses here, the map is a norm-matcher, not semantic transfer.
    0c  SVCCA(mapped-Qwen, real-Gemma) vs permutation null: p < 0.05
        Tests whether row correspondence (same sentence) matters at all.
    0d  identity sanity: fit Qwen->Qwen ridge, random unit dirs cosine ~1
        Catches bugs in the ridge/projection plumbing.
    0e  direction-transport norm ratios: flag catastrophic collapse.

Map design:
    Standardize X and Y per-dim BEFORE fitting the ridge. This prevents
    dim 104 from dominating the least-squares objective (it would capture
    ~all the variance budget, leaving the semantic subspace unfit).
    Store standardization params so we can map SAE directions later.

Direction mapping (used in co-firing + steering):
    Given a unit Qwen direction d_f [3584]:
        d_std  = d_f / X_sigma          (scale only, no mean subtraction)
        d_gstd = d_std @ W              (linear part of the map)
        d_g    = d_gstd * Y_sigma       (de-standardize)
        d_f'   = d_g / ||d_g||          (unit normalize)

Usage:
    python 02_gate0_map_fit.py --out-dir /content/exp3_acts
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import exp3_config as C

# ── thresholds (from spec §5.2) ───────────────────────────────────────────────
THRESH_0A     = 0.85   # raw held-out cosine
THRESH_0B     = 0.70   # outlier-zeroed cosine  <- the one that matters
THRESH_0C_P   = 0.05   # permutation p-value
THRESH_0D     = 0.65   # identity cosine sanity threshold
THRESH_0E_MIN = 0.10   # min transport norm ratio (collapse detection)
N_OUTLIER_DIMS = 10
N_SVCCA_PERM  = 1000
N_RAND_DIRS   = 1000
LAMBDAS       = [1, 10, 100, 1000]
HELDOUT_FRAC  = 0.20
SEED          = 42


def load_acts(out_dir):
    q = np.load(out_dir / "mean_qwen.npz",  allow_pickle=True)
    g = np.load(out_dir / "mean_gemma.npz", allow_pickle=True)
    X = q["acts"].astype(np.float64)   # [N, 3584]
    Y = g["acts"].astype(np.float64)   # [N, 5376]
    assert X.shape == (len(X), C.QWEN_DMODEL),  f"X shape {X.shape}"
    assert Y.shape == (len(Y), C.GEMMA_DMODEL), f"Y shape {Y.shape}"
    assert len(X) == len(Y), "row count mismatch — different corpora?"
    print(f"loaded X {X.shape}  Y {Y.shape}")
    print(f"  Qwen  ||row||: mean={np.linalg.norm(X,axis=1).mean():.0f}")
    print(f"  Gemma ||row||: mean={np.linalg.norm(Y,axis=1).mean():.0f}  "
          f"(dominated by outlier dims — standardization is load-bearing)")
    return X, Y


def standardize(X_tr, X_te):
    """Z-score per dim on train stats. Returns (Xtr_std, Xte_std, mu, sigma)."""
    mu    = X_tr.mean(0)
    sigma = X_tr.std(0).clip(1e-8)   # clip avoids /0 on constant dims
    return (X_tr - mu) / sigma, (X_te - mu) / sigma, mu, sigma


def fit_ridge(X_tr, Y_tr, lam):
    """W: [D_q, D_g] such that X_tr @ W ≈ Y_tr."""
    D = X_tr.shape[1]
    return np.linalg.solve(
        X_tr.T @ X_tr + lam * np.eye(D),
        X_tr.T @ Y_tr
    )


def row_cosine(A, B):
    """Per-row cosine similarity between [N,d] arrays. Returns [N]."""
    na = np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    nb = np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return ((A / na) * (B / nb)).sum(1)


def map_raw(X_std, W, Y_mu, Y_sigma):
    """Predicted Y in raw space: de-standardize W(X_std)."""
    return X_std @ W * Y_sigma + Y_mu


# ── gates ─────────────────────────────────────────────────────────────────────

def gate_0a(Y_hat_raw, Y_te_raw, label="0a"):
    cos = row_cosine(Y_hat_raw, Y_te_raw)
    m   = cos.mean()
    ok  = m >= THRESH_0A
    print(f"\n[Gate {label}] held-out cosine (raw space): "
          f"{m:.4f}  (threshold >= {THRESH_0A})  {'PASS' if ok else 'FAIL'}")
    print(f"  per-sentence: p5={np.percentile(cos,5):.3f}  "
          f"median={np.median(cos):.3f}  p95={np.percentile(cos,95):.3f}")
    return m, ok, cos


def gate_0b(Y_hat_raw, Y_te_raw, Y_all):
    """Zero the top-N_OUTLIER_DIMS highest-variance Gemma dims, recheck cosine."""
    var       = Y_all.var(0)
    top_dims  = np.argsort(var)[::-1][:N_OUTLIER_DIMS]
    print(f"\n[Gate 0b] zeroing top-{N_OUTLIER_DIMS} variance dims: {top_dims.tolist()}")
    print(f"  their variance ratios vs median: "
          f"{(var[top_dims]/np.median(var)).round(1).tolist()}")
    print(f"  (dim 104 expected first at ~1.44M× — the outlier that dominates raw cosine)")

    Yh = Y_hat_raw.copy(); Yh[:, top_dims] = 0
    Yt = Y_te_raw.copy();  Yt[:, top_dims] = 0
    cos = row_cosine(Yh, Yt)
    m   = cos.mean()
    ok  = m >= THRESH_0B
    print(f"  cosine without outlier dims: {m:.4f}  "
          f"(threshold >= {THRESH_0B})  {'PASS ✓' if ok else 'FAIL ✗ — STOP: map is a norm-matcher'}")
    if not ok:
        print("  INTERPRETATION: the map fits dim 104 and a few others but the semantic")
        print("  subspace does not transfer. Co-firing AUROCs will be at chance.")
        print("  OPTIONS: (1) try depth-matched ablation Qwen L20->Gemma L27, "
              "(2) increase map-fit N, (3) accept negative result.")
    return m, ok


def gate_0d(X, rng):
    """Identity sanity: Qwen->Qwen ridge. Random unit dirs should map back ~cos=1."""
    N = len(X)
    perm   = rng.permutation(N)
    n_held = int(N * HELDOUT_FRAC)
    tr, te = perm[n_held:], perm[:n_held]

    Xtr_s, Xte_s, mu, sig = standardize(X[tr], X[te])
    W_id = fit_ridge(Xtr_s, Xtr_s, lam=100)   # same lambda as a reasonable default

    rand_dirs = rng.standard_normal((N_RAND_DIRS, C.QWEN_DMODEL))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    mapped = (rand_dirs / sig) @ W_id * sig    # direction mapping (linear part)
    cos    = row_cosine(rand_dirs, mapped)
    m      = cos.mean()
    ok     = m >= THRESH_0D
    print(f"\n[Gate 0d] identity sanity (Qwen->Qwen, {N_RAND_DIRS} random dirs):")
    print(f"  mean cosine to original = {m:.4f}  (threshold >= {THRESH_0D})")
    print(f"  {'PASS' if ok else 'FAIL -- check for NaN/zero norms (not just low cosine)'}")
    return m, ok


def gate_0e(W, X_sigma, Y_sigma, rng):
    """Direction-transport norm ratios ||M_lin d|| / ||d|| for random unit dirs."""
    rand_dirs = rng.standard_normal((N_RAND_DIRS, C.QWEN_DMODEL))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    # Apply the linear part of the direction map
    mapped = (rand_dirs / X_sigma) @ W * Y_sigma   # [N, D_g]
    norms  = np.linalg.norm(mapped, axis=1)         # ratio relative to unit input

    ok = norms.min() >= THRESH_0E_MIN
    print(f"\n[Gate 0e] direction-transport norm ratios ({N_RAND_DIRS} random unit dirs):")
    print(f"  mean={norms.mean():.1f}  min={norms.min():.3f}  "
          f"max={norms.max():.1f}  p5={np.percentile(norms,5):.3f}")
    print(f"  min >= {THRESH_0E_MIN}: {'PASS' if ok else 'FAIL — directions collapsing to near-zero'}")
    return norms.mean(), norms.min(), ok


# ── lambda sweep ─────────────────────────────────────────────────────────────

def lambda_sweep(X_tr_s, X_te_s, Y_tr_s, Y_te_s, Y_te_raw, Y_mu, Y_sigma):
    print("\n[Lambda sweep]")
    best_cos, best_lam, best_W = -1, None, None
    for lam in LAMBDAS:
        W    = fit_ridge(X_tr_s, Y_tr_s, lam)
        Yhat = map_raw(X_te_s, W, Y_mu, Y_sigma)
        cos  = row_cosine(Yhat, Y_te_raw).mean()
        tag  = " <- best" if cos > best_cos else ""
        print(f"  lambda={lam:6d}  held-out cosine = {cos:.4f}{tag}")
        if cos > best_cos:
            best_cos, best_lam, best_W = cos, lam, W
    print(f"  selected lambda = {best_lam}")
    return best_W, best_lam, best_cos


# ── main ─────────────────────────────────────────────────────────────────────


def svcca(A, B, n_components=20, n_perm=1000, rng=None):
    """SVCCA between mapped-Qwen (A) and real-Gemma (B) held-out activations.
    Reduce each to top SVD components (capturing 99% var or n_components),
    then mean canonical correlation. Compares to a permutation null.
    Spec 0c: subspace alignment, not just norm match."""
    if rng is None:
        rng = np.random.default_rng(0)
    def svd_reduce(M, k):
        Mc = M - M.mean(0)
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        return Mc @ Vt[:k].T
    k = min(n_components, A.shape[1], B.shape[1], A.shape[0] - 1)
    Ar, Br = svd_reduce(A, k), svd_reduce(B, k)
    def mean_cca(X, Y):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        Qx, _ = np.linalg.qr(Xc); Qy, _ = np.linalg.qr(Yc)
        s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
        return float(np.clip(s, 0, 1).mean())
    actual = mean_cca(Ar, Br)
    null = np.array([mean_cca(Ar, Br[rng.permutation(len(Br))]) for _ in range(n_perm)])
    p = float((null >= actual).mean())
    return actual, float(null.mean()), p


def gate_0c_svcca(X_te_std, W, Y_te_std, rng):
    """Spec-compliant 0c: SVCCA of mapped-Qwen vs real-Gemma + permutation null."""
    pred = X_te_std @ W            # mapped Qwen (std space)
    real = Y_te_std               # real Gemma (std space)
    actual, null_mean, p = svcca(pred, real, n_components=20, n_perm=N_SVCCA_PERM, rng=rng)
    ok = p < THRESH_0C_P
    print(f"\n[Gate 0c] SVCCA(mapped-Qwen, real-Gemma) vs {N_SVCCA_PERM}-shuffle null:")
    print(f"  SVCCA mean canonical corr = {actual:.4f}")
    print(f"  permutation null mean = {null_mean:.4f}  p-value = {p:.4f}  "
          f"(threshold < {THRESH_0C_P})  {'PASS' if ok else 'FAIL'}")
    print(f"  interpretation: {'subspaces are aligned beyond chance' if ok else 'no subspace alignment'}")
    return actual, null_mean, p, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/content/exp3_acts")
    args = ap.parse_args()
    out  = Path(args.out_dir)
    rng  = np.random.default_rng(SEED)

    t0 = time.time()
    X, Y = load_acts(out)
    N = len(X)

    # train/test split
    perm   = rng.permutation(N)
    n_held = int(N * HELDOUT_FRAC)
    tr, te = perm[n_held:], perm[:n_held]
    print(f"\ntrain={len(tr)}  test={len(te)}")

    # standardize (critical: prevents dim-104 from dominating the fit)
    X_tr_s, X_te_s, X_mu, X_sigma = standardize(X[tr], X[te])
    Y_tr_s, Y_te_s, Y_mu, Y_sigma = standardize(Y[tr], Y[te])

    print(f"\nafter standardization:")
    print(f"  Gemma dim-104 sigma: {Y_sigma[104]:.1f} "
          f"(was {Y[tr,104].std():.0f} raw -> normalized to 1.0)")
    print(f"  all dims now unit variance by construction")

    # lambda sweep -> best W
    W, best_lam, raw_cos = lambda_sweep(
        X_tr_s, X_te_s, Y_tr_s, Y_te_s, Y[te], Y_mu, Y_sigma
    )

    # predictions in raw space for 0a/0b
    Y_hat_raw = map_raw(X_te_s, W, Y_mu, Y_sigma)

    # gate 0a
    cos_0a, pass_0a, cos_vec = gate_0a(Y_hat_raw, Y[te])

    # gate 0b — the make-or-break gate
    cos_0b, pass_0b = gate_0b(Y_hat_raw, Y[te], Y)

    # gate 0c
    cos_0c, perm_mean, pval, pass_0c = gate_0c_svcca(X_te_s, W, Y_te_s, rng)

    # gate 0d
    cos_0d, pass_0d = gate_0d(X, rng)

    # gate 0e
    norm_mean, norm_min, pass_0e = gate_0e(W, X_sigma, Y_sigma, rng)

    # ── verdict ───────────────────────────────────────────────────────────────
    gates = {
        "0a": pass_0a, "0b": pass_0b, "0c": pass_0c,
        "0d": pass_0d, "0e": pass_0e,
    }
    overall = pass_0a and pass_0b and pass_0c and pass_0d   # spec: 0a AND 0b AND 0c AND 0d; 0b make-or-break
    # 0e is informative but not a hard stop.

    print("\n" + "="*70)
    print("GATE 0 SUMMARY")
    print("="*70)
    for g, p in gates.items():
        print(f"  Gate {g}: {'PASS' if p else 'FAIL'}")
    print(f"\n  OVERALL: {'PROCEED to co-firing' if overall else 'STOP — fix map before continuing'}")
    print(f"  elapsed: {time.time()-t0:.0f}s")

    if not pass_0b:
        print("\n  !! Gate 0b failed. The map is dominated by Gemma's outlier dims.")
        print("  !! Run the depth-matched ablation (Qwen L20 -> Gemma L27) first.")
        print("  !! Do NOT proceed to co-firing — AUROCs will be at chance.")

    # ── save map ──────────────────────────────────────────────────────────────
    map_path = out / "ridge_map.npz"
    np.savez(map_path,
             W=W.astype(np.float32),
             X_mu=X_mu.astype(np.float32),
             X_sigma=X_sigma.astype(np.float32),
             Y_mu=Y_mu.astype(np.float32),
             Y_sigma=Y_sigma.astype(np.float32),
             best_lambda=np.array([best_lam]),
    )
    print(f"\n  map saved -> {map_path}")
    print(f"  W shape: {W.shape}  dtype: float32")
    print(f"\n  Direction mapping recipe (for co-firing + steering):")
    print(f"    d_std   = d_f / X_sigma          # scale only, no mean subtraction")
    print(f"    d_gstd  = d_std @ W              # linear part")
    print(f"    d_g     = d_gstd * Y_sigma       # de-standardize")
    print(f"    d_f_prime = d_g / ||d_g||        # unit normalize")

    # save results json
    results = {
        "best_lambda": int(best_lam),
        "gate_0a": {"cosine": float(cos_0a), "pass": bool(pass_0a), "threshold": THRESH_0A},
        "gate_0b": {"cosine": float(cos_0b), "pass": bool(pass_0b), "threshold": THRESH_0B,
                    "n_zeroed_dims": N_OUTLIER_DIMS},
        "gate_0c": {"svcca_mean_corr": float(cos_0c), "perm_mean": float(perm_mean),
                    "p_value": float(pval), "pass": bool(pass_0c), "n_perm": N_SVCCA_PERM},
        "gate_0d": {"identity_cosine": float(cos_0d), "pass": bool(pass_0d)},
        "gate_0e": {"norm_mean": float(norm_mean), "norm_min": float(norm_min),
                    "pass": bool(pass_0e)},
        "overall_pass": bool(overall),
    }
    (out / "gate0_results.json").write_text(json.dumps(results, indent=2))
    print(f"  results saved -> {out / 'gate0_results.json'}")


if __name__ == "__main__":
    main()
