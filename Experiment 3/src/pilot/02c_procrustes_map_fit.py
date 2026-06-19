#!/usr/bin/env python
"""
02c_procrustes_map_fit.py

Fit a Qwen->Gemma map using standardized mean-pooled activations and
rectangular orthogonal Procrustes.

Input:
  --acts-dir containing mean_qwen.npz and mean_gemma.npz

Output:
  --out-dir/ridge_map.npz

The output file intentionally uses the same keys as ridge_map.npz:
  W, X_mu, X_sigma, Y_mu, Y_sigma

So existing 05/06/05b scripts can use it as --map-dir.
"""

import argparse, json
from pathlib import Path
import numpy as np
from scipy.linalg import svd


def load_acts(path):
    z = np.load(path, allow_pickle=True)
    for k in ["acts", "X", "Y", "arr_0"]:
        if k in z:
            return z[k].astype(np.float64)
    raise KeyError(f"Could not find acts array in {path}; keys={list(z.keys())}")


def row_cos(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    B = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return (A * B).sum(1)


def standardize_train_test(Xtr, Xte):
    mu = Xtr.mean(0)
    sig = Xtr.std(0).clip(1e-8)
    return (Xtr - mu) / sig, (Xte - mu) / sig, mu, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    acts_dir = Path(args.acts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = load_acts(acts_dir / "mean_qwen.npz")
    Y = load_acts(acts_dir / "mean_gemma.npz")

    assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
    print("loaded X", X.shape, "Y", Y.shape)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_tr = int(args.train_frac * len(X))
    tr, te = perm[:n_tr], perm[n_tr:]

    Xtr_s, Xte_s, X_mu, X_sig = standardize_train_test(X[tr], X[te])
    Ytr_s, Yte_s, Y_mu, Y_sig = standardize_train_test(Y[tr], Y[te])

    # Rectangular orthogonal Procrustes:
    # min ||X W - Y||_F subject W^T W = I when d_x <= d_y.
    # Cross-cov C = X^T Y = U S Vt, W = U Vt.
    print("SVD of cross-cov...")
    C = Xtr_s.T @ Ytr_s
    U, S, Vt = svd(C, full_matrices=False)
    W = U @ Vt

    Yhat_s = Xte_s @ W
    Yhat = Yhat_s * Y_sig + Y_mu

    raw_cos = float(row_cos(Yhat, Y[te]).mean())

    var = Y.var(0)
    top_dims = np.argsort(var)[::-1][:10]
    Yh = Yhat.copy()
    Yt = Y[te].copy()
    Yh[:, top_dims] = 0
    Yt[:, top_dims] = 0
    oz_cos = float(row_cos(Yh, Yt).mean())

    np.savez(
        out_dir / "ridge_map.npz",
        W=W.astype(np.float32),
        X_mu=X_mu.astype(np.float32),
        X_sigma=X_sig.astype(np.float32),
        Y_mu=Y_mu.astype(np.float32),
        Y_sigma=Y_sig.astype(np.float32),
        map_kind=np.array("meanpool_rectangular_procrustes"),
    )

    summary = {
        "map_kind": "meanpool_rectangular_procrustes",
        "acts_dir": str(acts_dir),
        "n": int(len(X)),
        "train_n": int(len(tr)),
        "test_n": int(len(te)),
        "raw_cos": raw_cos,
        "outlier_zeroed_cos": oz_cos,
        "singular_values_top10": [float(x) for x in S[:10]],
    }

    (out_dir / "gate0_procrustes_results.json").write_text(json.dumps(summary, indent=2))

    print("saved", out_dir / "ridge_map.npz")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()