"""Self-test for audits.py on synthetic data.

Asserts each audit flags the bad case and clears the good one, so a real
confound (length gap, echo word, contaminated anchors) cannot slip through.

Pure CPU. Run:  .venv/bin/python test_audits.py
"""
import numpy as np

from audits import cosine_report, length_balance, lexical_leak


def main() -> int:
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # ── length_balance ──
    bal = length_balance([40, 42, 38, 41], [39, 40, 41, 40])
    check(bal["ok"] and bal["pct_gap"] < 10, f"balanced lengths: {bal}")
    imb = length_balance([100, 110, 90], [40, 42, 38])
    check(not imb["ok"] and imb["pct_gap"] > 25, f"imbalanced lengths: {imb}")

    # ── lexical_leak ──
    # 'weapon' only on present, 'garden' only on absent -> both flagged; the
    # shared scaffold words must NOT be flagged.
    pres = [f"how to build a weapon item {i}" for i in range(10)]
    abst = [f"how to build a garden item {i}" for i in range(10)]
    lk = lexical_leak(pres, abst)
    toks = {d["token"] for d in lk["leaks"]}
    check("weapon" in toks and "garden" in toks, f"expected leak tokens, got {toks}")
    check(not lk["ok"], "should flag a leak")
    check("build" not in toks and "how" not in toks, f"shared words leaked: {toks}")
    # identical sides -> nothing skewed -> clean
    same = [f"the project went well in year {i}" for i in range(8)]
    check(lexical_leak(same, list(same))["ok"], "identical sides should be clean")

    # ── cosine_report ──
    rng = np.random.default_rng(0)
    d = 128
    v = np.zeros(d)
    v[0] = 1.0
    pos = np.tile(v, (50, 1)) * 3 + rng.standard_normal((50, d)) * 0.1   # aligned
    neg = np.tile(-v, (50, 1)) * 3 + rng.standard_normal((50, d)) * 0.1  # anti-aligned
    anc = rng.standard_normal((50, d))                                    # ~orthogonal
    rep = cosine_report(v, positives=pos, negatives=neg, anchors=anc)
    check(rep["positive_median_cos"] > 0.8, f"pos cos: {rep}")
    check(rep["negative_median_cos"] < -0.8, f"neg cos: {rep}")
    check(abs(rep["anchor_baseline_cos"]) < 0.2, f"anchor baseline: {rep}")
    check(rep["ok"], f"clean anchors should pass: {rep}")
    # contaminated anchors (aligned with v) -> must fail
    bad = np.tile(v, (50, 1)) * 3 + rng.standard_normal((50, d)) * 0.1
    rep2 = cosine_report(v, positives=pos, negatives=neg, anchors=bad)
    check(not rep2["ok"], f"contaminated anchors should fail: {rep2}")

    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED — audits separate balanced / leaky / contaminated cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
