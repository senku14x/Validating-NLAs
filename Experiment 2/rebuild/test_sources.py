"""Self-test for concept_sources.py — the constructed pairs as executable checks.

Runs the real authored content through the same audits Gate 1 will use, so a
length confound in a control (the v1 trap) or a degenerate pair is caught here,
on CPU, before any extraction. Format/style concepts are echo_prone and exempt
from the length check by design.

Run:  .venv/bin/python test_sources.py
"""
import numpy as np
import yaml

from audits import length_balance, lexical_leak
from concept_sources import OFFLINE_CONCEPTS, build
from paths import config_path

NETWORK_CONCEPTS = {"refusal", "sycophancy", "corrigibility", "truth_value"}

MIN_PAIRS = 10
MAX_PCT_GAP = 25.0


def main() -> int:
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    print(f"{'concept':24s} {'n':>4} {'len_matched':>11} {'pct_gap':>8} {'n_leaks':>8}")
    for key, spec in OFFLINE_CONCEPTS.items():
        rng = np.random.default_rng(42)
        pairs = build(key, 10_000, rng)
        n = len(pairs)
        check(n >= MIN_PAIRS, f"{key}: only {n} pairs (< {MIN_PAIRS})")

        for p, a in pairs:
            check(isinstance(p, str) and p.strip(), f"{key}: empty/invalid present")
            check(isinstance(a, str) and a.strip(), f"{key}: empty/invalid absent")
            check(p != a, f"{key}: present == absent")

        pl = [len(p) for p, _ in pairs]
        al = [len(a) for _, a in pairs]
        lb = length_balance(pl, al, max_pct_gap=MAX_PCT_GAP)
        leak = lexical_leak([p for p, _ in pairs], [a for _, a in pairs])

        print(f"{key:24s} {n:>4} {str(spec['length_matched']):>11} "
              f"{lb['pct_gap']:>7.1f}% {leak['n_leaks']:>8}")

        if spec["length_matched"]:
            check(lb["ok"], f"{key}: length gap {lb['pct_gap']}% >= {MAX_PCT_GAP}% "
                            f"(length_matched concept must be balanced)")

    # manifest <-> builders must agree (catches drift between YAML and code)
    manifest_keys = {c["key"] for c in yaml.safe_load(config_path("concepts.yaml").read_text())["concepts"]}
    for key in OFFLINE_CONCEPTS:
        check(key in manifest_keys, f"{key}: offline builder has no concepts.yaml entry")
    for key in NETWORK_CONCEPTS:
        check(key in manifest_keys, f"{key}: network concept missing from concepts.yaml")
        check(key not in OFFLINE_CONCEPTS, f"{key}: network concept must not have an offline builder")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nALL CHECKS PASSED — constructed pairs are non-degenerate and "
          "length-matched where required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
