"""test_injection.py — CPU self-test for injection.py (run before trusting Gate 2).

Verifies the solver hits the target cosine exactly (incl. negative targets),
dom_dir recovers a planted direction and guards the degenerate case, and
inject_stats returns sane diagnostics. Mirrors the other test_*.py.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from injection import dom_dir, exact_cosine_inject, inject_stats


def main() -> int:
    rng = np.random.default_rng(0)
    d = 128
    H = rng.standard_normal((50, d))
    v = rng.standard_normal(d); v /= np.linalg.norm(v)

    # 1. solver hits the target realized cosine exactly, across a range incl. negative
    for t in [-0.30, 0.0, 0.30, 0.50, 0.70, 0.90]:
        Hp, beta, realized = exact_cosine_inject(H, v, t)
        err = float(np.abs(realized - t).max())
        assert err < 1e-9, f"solver target {t}: max realized error {err:.2e}"

    # 2. dom_dir is unit-norm and recovers a planted direction (correct sign).
    #    Offset 5*v over 40/class keeps SNR high enough that recovery is unambiguous.
    X = np.vstack([rng.standard_normal((40, d)) + 5 * v, rng.standard_normal((40, d)) - 5 * v])
    y = np.r_[np.ones(40), np.zeros(40)]
    dv = dom_dir(X, y)
    assert abs(np.linalg.norm(dv) - 1.0) < 1e-9, "dom_dir not unit norm"
    assert dv @ v > 0.9, f"dom_dir should recover the planted direction (got cos {dv @ v:.3f})"

    # 3. degenerate guard (identical class means -> zero direction)
    try:
        dom_dir(np.ones((4, 8)), np.r_[1, 1, 0, 0])
        raise AssertionError("dom_dir should raise on a zero-norm direction")
    except ValueError:
        pass

    # 4. inject_stats: realized matches solver, cos_h_hp in [-1,1], delta_norm >= 0,
    #    and a big push from a negative baseline really does drop cos(h,h') (off-manifold)
    Hp, _, realized = exact_cosine_inject(H, v, 0.60)
    s = inject_stats(H, Hp, v)
    assert np.abs(s["realized_cos"] - realized).max() < 1e-9
    assert (s["cos_h_hp"] <= 1.0001).all() and (s["cos_h_hp"] >= -1.0001).all()
    assert (s["delta_norm_over_h"] >= 0).all()
    # construct anchors with strongly NEGATIVE baseline, push to +0.7 -> low cos(h,h')
    Hneg = rng.standard_normal((20, d)) - 6 * v
    Hp2, _, _ = exact_cosine_inject(Hneg, v, 0.70)
    s2 = inject_stats(Hneg, Hp2, v)
    assert s2["baseline_cos"].mean() < -0.3, "expected negative baseline"
    assert s2["cos_h_hp"].mean() < s["cos_h_hp"].mean(), "neg-baseline push should be more off-manifold"

    print("ALL CHECKS PASSED — injection math: exact-cosine solver, dom_dir recovery + "
          "degenerate guard, off-manifold diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
