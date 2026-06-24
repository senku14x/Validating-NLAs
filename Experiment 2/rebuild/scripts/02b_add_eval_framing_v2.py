#!/usr/bin/env python
"""02b_add_eval_framing_v2.py — append the BoW-DEFEATED eval_framing_v2 concept to the
committed concept_pairs.parquet (NO network; idempotent).

Why a separate appender (not a re-run of 02): 02 re-downloads AdvBench/Alpaca/CAA/cities and
rewrites the whole parquet. This only adds eval_framing_v2 (built offline from
concept_sources.EVAL_FRAMINGS_V2) to the existing file, so the other concepts are untouched and
no network is needed. After this, 03 extracts it and 04 batteries it like any concept.

Runs in the dev venv (CPU). Prints a BoW sanity check (must be ~0.50 = the whole point).
  .venv/bin/python "Experiment 2/rebuild/scripts/02b_add_eval_framing_v2.py"
  (--force re-appends if it's already there; --n caps pairs, default 250)
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # rebuild root
import concept_sources as cs  # noqa: E402
from paths import data_path  # noqa: E402

CONCEPT = "eval_framing_v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="max pairs (5 framings x EVAL_TASKS = 250)")
    ap.add_argument("--force", action="store_true", help="re-append even if already present")
    a = ap.parse_args()

    p = data_path("concept_pairs.parquet", mkdir=False)
    if not p.exists():
        sys.exit(f"FAIL: {p} not found — run 02_build_concept_pairs.py first.")
    df = pd.read_parquet(p)

    if CONCEPT in set(df.concept.unique()):
        if not a.force:
            print(f"{CONCEPT} already present ({(df.concept == CONCEPT).sum()} rows) — nothing to do "
                  f"(use --force to rebuild).")
            return 0
        df = df[df.concept != CONCEPT].reset_index(drop=True)
        print(f"--force: dropped existing {CONCEPT} rows")

    pairs = cs.build_eval_framing_v2(a.n, None)
    rows = []
    for i, (present, absent) in enumerate(pairs):
        gid = f"evalv2_{i:04d}"
        for polarity, text in (("present", present), ("absent", absent)):
            rows.append(dict(concept=CONCEPT, role="core", source="constructed", design="paired",
                             group_id=gid, polarity=polarity, needs_behavior=False,
                             text=str(text).strip(), format="user"))
    new = pd.DataFrame(rows)
    # align columns to the existing schema (fill any extras the parquet has)
    for col in df.columns:
        if col not in new.columns:
            new[col] = None
    new = new[df.columns]
    out = pd.concat([df, new], ignore_index=True)
    out.to_parquet(p, index=False)
    print(f"appended {len(new)} rows ({len(pairs)} pairs) as {CONCEPT} -> {p}")
    print(f"  total rows now {len(out)} over {out.concept.nunique()} concepts")

    # BoW sanity — the entire point of v2 is that a unigram BoW is at chance.
    try:
        from confounds import bow_auroc, char_ngram_auroc
        texts = [pp for pp, _ in pairs] + [aa for _, aa in pairs]
        y = [1] * len(pairs) + [0] * len(pairs)
        groups = [f"evalv2_{i:04d}" for i in range(len(pairs))] * 2
        b = bow_auroc(texts, y, groups)
        c = char_ngram_auroc(texts, y, groups)
        print(f"\n  BoW sanity: unigram AUROC = {b.auroc:.3f} [{b.ci_lo:.3f},{b.ci_hi:.3f}] "
              f"(want ~0.50)   char-ngram = {c.auroc:.3f} (the floor the probe must beat)")
        if b.ci_hi >= 0.70:
            print("  *** WARNING: BoW not defeated (ci_hi >= 0.70) — check the framings ***")
    except Exception as e:
        print(f"  (BoW sanity skipped: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
