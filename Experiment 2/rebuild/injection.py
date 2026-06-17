"""injection.py — shared injection math for Gate 2 (offline + online).

Single source for the diff-of-means direction and the exact-cosine offline
injection solver, imported by scripts/05_inject_matrix (offline), 05b_steer_extract
(online — shares the directions), and 05_smoke_decode. Do not reimplement these
anywhere else — same spirit as confounds.py being the only source of AUROC logic.

Conventions (docs/references/nla-infrastructure.md):
  - directions are UNIT vectors built OUTSIDE the NLA (diff-of-means), so an NLA
    detection on an injected vector is not circular.
  - offline injection: h' = h + beta * v_hat, with beta solved so cos(h', v_hat)
    is an EXACT target. Dose is the *realized* cosine, never the raw beta.
  - every injected row logs realized cos, cos(h, h'), and norm distortion, so an
    off-manifold push (low cos(h,h') — unavoidable for concepts whose neutral-
    anchor baseline is negative) is visible and can be discounted downstream.
"""
from __future__ import annotations

import numpy as np


def dom_dir(X, y) -> np.ndarray:
    """Unit diff-of-means direction: normalize(mean(class 1) - mean(class 0)).

    Arditi et al.; identical to Experiment 1's 07_build_direction::dom_dir.
    Raises if the two class means coincide (degenerate / zero-norm direction).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("degenerate direction (zero norm): the two classes have identical means")
    return v / n


def exact_cosine_inject(H, v_hat, target_cos):
    """Offline injection h' = h + beta*v_hat solved so cos(h', v_hat) == target_cos.

    H: [n, d] anchor activations. v_hat: [d] unit direction. target_cos: scalar.
    Returns (Hp [n,d], beta [n], realized [n]). Verbatim solver from CAA
    03_build_injection_sweep (verified exact to ~1e-16). Works for negative
    targets too (just rotates the other way).
    """
    H = np.asarray(H, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)
    a = H @ v_hat                                              # [n] dot products
    perp = np.sqrt(np.maximum((H ** 2).sum(1) - a ** 2, 0.0))  # [n] perpendicular norms
    t = float(target_cos)
    beta = perp * (t / np.sqrt(max(1.0 - t ** 2, 1e-9))) - a   # [n]
    Hp = H + beta[:, None] * v_hat[None, :]
    realized = (Hp @ v_hat) / np.linalg.norm(Hp, axis=1)
    return Hp, beta, realized


def inject_stats(H, Hp, v_hat) -> dict:
    """Per-row off-manifold diagnostics for injected vectors.

    Returns a dict of [n] arrays: realized_cos, baseline_cos (anchor's natural
    cos with the direction), cos_h_hp (how far h' moved off the anchor — the
    off-manifold tell), norm_h, norm_hp, delta_norm_over_h.
    """
    H = np.asarray(H, dtype=float)
    Hp = np.asarray(Hp, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)
    nh = np.linalg.norm(H, axis=1)
    nhp = np.linalg.norm(Hp, axis=1)
    return dict(
        realized_cos=(Hp @ v_hat) / nhp,
        baseline_cos=(H @ v_hat) / nh,
        cos_h_hp=np.sum(H * Hp, axis=1) / (nh * nhp),
        norm_h=nh,
        norm_hp=nhp,
        delta_norm_over_h=np.linalg.norm(Hp - H, axis=1) / nh,
    )


if __name__ == "__main__":
    import sys
    print("injection.py is a library — import dom_dir/exact_cosine_inject/inject_stats.\n"
          "Run the self-test with:  .venv/bin/python test_injection.py", file=sys.stderr)
