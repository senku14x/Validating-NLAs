"""Self-test for confounds.probe_battery on synthetic data.

Reproduces the three regimes from the Exp 2 debrief and asserts the battery
classifies each correctly:

  CLEAN     signal lives in a direction independent of length
            -> residualized stays high, length_only ~0.5, represented=True
  LENGTH    label is determined by length; no independent direction (the v1
            eval-awareness 0.990 trap)
            -> raw high, residualized collapses, represented=False
  MIXED     a real direction PLUS a *separable* (overlapping) length confound
            -> raw high, length_only moderate, residualized survives -> True
  COLLINEAR a real direction but length is ~collinear with the label, so the
            two are not identifiable (controlling length removes the signal too)
            -> raw high, length_only ~1.0, residualized collapses -> False
            This is honest: we refuse to certify what the data can't separate.

Pure CPU. Run:  python test_confounds.py
"""

import numpy as np

from confounds import probe_battery, summarize

RNG = np.random.default_rng(0)
N_PAIRS = 80
D = 64


def _make(kind: str):
    """Return X, y, lengths, groups for one synthetic regime.

    Matched-pair design: each pair contributes one present (y=1) and one absent
    (y=0) row, sharing a group id.
    """
    n = 2 * N_PAIRS
    y = np.tile([1, 0], N_PAIRS)
    groups = np.repeat(np.arange(N_PAIRS), 2)

    sig = RNG.standard_normal(D)
    sig /= np.linalg.norm(sig)
    len_dir = RNG.standard_normal(D)
    len_dir /= np.linalg.norm(len_dir)

    X = RNG.standard_normal((n, D)) * 1.0

    if kind == "clean":
        # signal aligned with y, length uncorrelated with y
        X += np.outer(y, sig) * 2.5
        lengths = RNG.normal(40, 6, n)
    elif kind == "length":
        # y is just "is this row longer"; X only carries a length axis, no
        # y-specific direction beyond what length explains.
        lengths = np.where(y == 1, RNG.normal(46, 4, n), RNG.normal(34, 4, n))
        X += np.outer(lengths, len_dir) * 0.08
    elif kind == "mixed":
        # overlapping length distributions -> length is a MODERATE, separable
        # confound; the concept signal must survive residualization.
        lengths = np.where(y == 1, RNG.normal(44, 7, n), RNG.normal(38, 7, n))
        X += np.outer(lengths, len_dir) * 0.08
        X += np.outer(y, sig) * 2.5
    elif kind == "collinear":
        # near-deterministic length -> length is collinear with the label;
        # residualizing length also removes the signal. Not identifiable.
        lengths = np.where(y == 1, RNG.normal(46, 3, n), RNG.normal(34, 3, n))
        X += np.outer(lengths, len_dir) * 0.08
        X += np.outer(y, sig) * 2.5
    else:
        raise ValueError(kind)

    return X, y, lengths, groups


def main() -> int:
    failures = []
    for kind in ("clean", "length", "mixed", "collinear"):
        X, y, lengths, groups = _make(kind)
        b = probe_battery(X, y, lengths, groups, n_boot=1000, n_null=15)
        print(summarize(b, name=kind))
        print()

        resid_lo = b["length_residualized"].ci_lo
        lenonly = b["length_only"].auroc
        null = b["null"].auroc

        # null must sit near chance in every regime — the leakage canary
        if not (0.40 <= null <= 0.60):
            failures.append(f"[{kind}] null AUROC {null:.3f} not ~0.5 (CV/grouping leak)")

        if kind == "clean":
            if not b["represented"]:
                failures.append(f"[{kind}] expected represented=True (resid_lo={resid_lo:.3f})")
            if lenonly > 0.65:
                failures.append(f"[{kind}] length_only {lenonly:.3f} should be ~0.5")
        elif kind == "length":
            if b["represented"]:
                failures.append(f"[{kind}] expected represented=False — this is the 0.990 trap")
            if lenonly < 0.75:
                failures.append(f"[{kind}] length_only {lenonly:.3f} should be high")
        elif kind == "mixed":
            if not b["represented"]:
                failures.append(f"[{kind}] expected represented=True — separable confound, signal survives")
            if not (0.55 <= lenonly <= 0.90):
                failures.append(f"[{kind}] length_only {lenonly:.3f} should be moderate (separable)")
        elif kind == "collinear":
            if b["represented"]:
                failures.append(f"[{kind}] expected represented=False — confound not identifiable from signal")

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED — battery separates clean / length / mixed / collinear regimes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
