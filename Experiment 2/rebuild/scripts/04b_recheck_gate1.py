#!/usr/bin/env python
"""04b_recheck_gate1.py — corrected Gate-1 verdict (pre-registration v2). CPU only.

The original Gate-1 battery (04) controlled length but not lexical content, and its
cluster-bootstrap CI is unreliable below ~MIN_GROUPS groups (audit: ~10% false PASS
at 12 groups). This re-gates every concept WITHOUT re-running the activation probe,
by combining:
  - the cached activation-probe results (results/gate1/04_run_gate1_battery__*.json:
    raw / length_residualized / null AUROCs),
  - a bag-of-words TEXT-only baseline computed here from data/concept_pairs.parquet
    (confounds.bow_auroc) — the lexical analog of length_only, no activations needed,
  - the lexical_leak flag from data/concept_pairs_audit.json,
  - a minimum-groups rule,
through confounds.gate_v2. Writes corrected per-concept JSON + a summary and prints
a v1-vs-v2 table.

Note: the BoW baseline uses the construction polarity (present/absent) from the
parquet, which for refusal/harmful_topic_benign is the pre-behavioral-filter set —
a slightly different set than the probe's, but the lexical comparison still holds.

Run:  .venv/bin/python "Experiment 2/rebuild/scripts/04b_recheck_gate1.py" --model gemma
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from confounds import AUROC, bow_auroc, gate_v2  # noqa: E402
from paths import data_path, model_slug, result_path  # noqa: E402


def _auroc(d: dict) -> AUROC:
    return AUROC(d["auroc"], d["ci_lo"], d["ci_hi"], d["n"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma")
    a = ap.parse_args()
    model = model_slug(a.model)

    df = pd.read_parquet(data_path("concept_pairs.parquet", mkdir=False))
    audit = json.loads(data_path("concept_pairs_audit.json", mkdir=False).read_text())
    gate_dir = result_path("gate1", "04_run_gate1_battery", model, concept="x").parent

    rows = []
    for c in sorted(df.concept.unique()):
        if c == "anchor":
            continue
        jp = gate_dir / f"04_run_gate1_battery__{model}__{c}.json"
        if not jp.exists():
            continue
        j = json.loads(jp.read_text())
        b = j.get("battery")

        sub = df[df.concept == c]
        pol = sub[sub.polarity.isin(["present", "absent"])]
        y = (pol.polarity == "present").astype(int).to_numpy()
        groups = (np.arange(len(pol)) if sub.design.iloc[0] == "two_pool"
                  else pd.factorize(pol.group_id.to_numpy())[0])
        tb = bow_auroc(pol.text.tolist(), y, groups)
        n_groups = int(len(np.unique(groups)))
        lexical_ok = bool(audit.get(c, {}).get("lexical_leak", {}).get("ok", True))

        if b is None:
            verdict, why = "DROP", "no battery (insufficient data)"
            raw = resid = null = None
        else:
            raw, resid, null = _auroc(b["raw"]), _auroc(b["length_residualized"]), _auroc(b["null"])
            verdict, why = gate_v2(raw, resid, null, tb, n_groups, lexical_ok)

        rec = {
            "concept": c, "role": j.get("role"), "expectation": j.get("expectation"),
            "v1_verdict": j["verdict"], "v2_verdict": verdict, "why": why,
            "resid_ci_lo": resid.ci_lo if resid else None,
            "text_bow_auroc": tb.auroc, "text_bow_ci_hi": tb.ci_hi,
            "n_groups": n_groups, "lexical_ok": lexical_ok,
            "text_only": {"auroc": tb.auroc, "ci_lo": tb.ci_lo, "ci_hi": tb.ci_hi, "n": tb.n},
        }
        rows.append(rec)
        (gate_dir / f"04b_recheck__{model}__{c}.json").write_text(json.dumps(rec, indent=2))

    (gate_dir / f"04b_recheck__{model}__all.json").write_text(json.dumps(rows, indent=2))

    print(f"{'concept':22}{'role':12}{'v1':>5}{'v2':>6} {'resid_lo':>9}{'textBoW':>8}{'grp':>5}{'leak':>6}  reason")
    for r in rows:
        rl = f"{r['resid_ci_lo']:.3f}" if r["resid_ci_lo"] is not None else "—"
        leak = "ok" if r["lexical_ok"] else "FLAG"
        print(f"{r['concept']:22}{str(r['role']):12}{r['v1_verdict']:>5}{r['v2_verdict']:>6} "
              f"{rl:>9}{r['text_bow_auroc']:>8.3f}{r['n_groups']:>5}{leak:>6}  {r['why']}")
    print(f"\nv1 tally: {dict(Counter(r['v1_verdict'] for r in rows))}")
    print(f"v2 tally: {dict(Counter(r['v2_verdict'] for r in rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
