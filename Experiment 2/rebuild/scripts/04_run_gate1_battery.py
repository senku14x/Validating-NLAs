#!/usr/bin/env python
"""04_run_gate1_battery.py — run the confound battery per concept; assign a fate.

Reads the per-concept activations cached by 03 (X, y, lengths, groups — already
behaviorally filtered for refusal / harmful_topic_benign), runs the pre-registered
probe_battery on each, and writes one result JSON per concept plus a summary.

Pre-registered Gate-1 fate (spec §5), from the battery alone:
  DROP  unusable data (too few rows, single class)
  PASS  battery 'represented' is True (length-residualized signal clears the floor
        AND beats the permutation null) — eligible for Gates 2-4
  WEAK  not represented, but the RAW probe still beats null — a real signal that
        does not survive length control (length-explained / marginal)
  FAIL  not represented and raw does not beat null — no linear signal after controls

CPU only. Stress-test without a GPU:
  .venv/bin/python "Experiment 2/rebuild/scripts/04_run_gate1_battery.py" --self-test
Real run (after 03 on the box):
  .venv/bin/python "Experiment 2/rebuild/scripts/04_run_gate1_battery.py" --model gemma
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # reach root libs
from confounds import battery_to_dict, probe_battery, summarize  # noqa: E402
from paths import config_path, cache_path, model_slug, result_path, stage_of  # noqa: E402

MIN_ROWS = 16          # below this the CIs are meaningless -> DROP
STAGE = stage_of(__file__)


def fate(battery: dict) -> str:
    if battery["represented"]:
        return "PASS"
    if battery["raw"].ci_lo > battery["null"].ci_hi:
        return "WEAK"
    return "FAIL"


def run_concept(X, y, lengths, groups) -> tuple[dict | None, str]:
    y = np.asarray(y).astype(int)
    if len(y) < MIN_ROWS or len(np.unique(y)) < 2 or min(np.bincount(y)) < 4:
        return None, "DROP"
    b = probe_battery(X, y, lengths, groups)
    return b, fate(b)


def _load_concept(model: str, concept: str):
    p = cache_path("03_extract_for_battery", model, concept=concept, ext="npz", mkdir=False)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return z["X"], z["y"], z["lengths"], z["groups"]


def real_run(model: str, only: list[str] | None) -> int:
    manifest = yaml.safe_load(config_path("concepts.yaml").read_text())
    concepts = [c["key"] for c in manifest["concepts"]]
    cmap = {c["key"]: c for c in manifest["concepts"]}
    if only:
        concepts = [c for c in concepts if c in only]

    summary = {}
    missing = []
    for concept in concepts:
        loaded = _load_concept(model, concept)
        if loaded is None:
            missing.append(concept)
            continue
        X, y, lengths, groups = loaded
        b, verdict = run_concept(X, y, lengths, groups)
        rec = {"concept": concept, "role": cmap[concept].get("role"),
               "expectation": cmap[concept].get("expectation"), "verdict": verdict,
               "n": int(len(y)), "battery": battery_to_dict(b) if b else None}
        out = result_path("gate1", STAGE, model, concept=concept)
        out.write_text(json.dumps(rec, indent=2))
        summary[concept] = {k: rec[k] for k in ("role", "expectation", "verdict", "n")}
        if b:
            print(summarize(b, name=f"{concept} [{verdict}]"))
            print()

    result_path("gate1", STAGE, model, concept="all").write_text(json.dumps(summary, indent=2))

    print(f"\n{'concept':24s} {'role':12s} {'expect':6s} {'verdict':7s} {'n':>4}")
    for c, r in summary.items():
        print(f"{c:24s} {str(r['role']):12s} {str(r['expectation']):6s} {r['verdict']:7s} {r['n']:>4}")
    if missing:
        print(f"\nNo activations cached for {len(missing)} concept(s) — run 03 first: {missing}")
    if not summary:
        print("Nothing ran. Run 03 (extraction) on the box first.", file=sys.stderr)
        return 1
    return 0


def self_test() -> int:
    """Synthetic activations exercising the three non-DROP fates on CPU."""
    rng = np.random.default_rng(0)
    d, npairs = 64, 80

    def make(kind):
        n = 2 * npairs
        y = np.tile([1, 0], npairs)
        groups = np.repeat(np.arange(npairs), 2)
        sig = rng.standard_normal(d); sig /= np.linalg.norm(sig)
        ld = rng.standard_normal(d); ld /= np.linalg.norm(ld)
        X = rng.standard_normal((n, d))
        if kind == "clean":            # signal, length irrelevant -> PASS
            X += np.outer(y, sig) * 2.5
            lengths = rng.normal(40, 6, n)
        elif kind == "length":         # X strongly encodes length, y is length -> WEAK
            lengths = np.where(y == 1, rng.normal(46, 4, n), rng.normal(34, 4, n))
            X += np.outer(lengths, ld) * 0.5
        else:                          # noise -> FAIL
            lengths = rng.normal(40, 6, n)
        return X, y, lengths, groups

    expect = {"clean": "PASS", "length": "WEAK", "noise": "FAIL"}
    fails = []
    for kind, want in expect.items():
        X, y, lengths, groups = make(kind)
        b, verdict = run_concept(X, y, lengths, groups)
        print(summarize(b, name=f"{kind} -> {verdict}"))
        print()
        if verdict != want:
            fails.append(f"{kind}: got {verdict}, expected {want}")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED — verdict mapping PASS/WEAK/FAIL is correct on synthetic data.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="gemma | qwen")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--concepts", nargs="*", default=None, help="optional subset of concept keys")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.model:
        ap.error("provide --model gemma|qwen (or --self-test)")
    return real_run(model_slug(a.model), a.concepts)


if __name__ == "__main__":
    raise SystemExit(main())
