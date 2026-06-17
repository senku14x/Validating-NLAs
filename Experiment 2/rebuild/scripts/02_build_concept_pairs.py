#!/usr/bin/env python
"""02_build_concept_pairs.py — assemble all Gate-1 concept prompts into one dataset.

Two designs live side by side (see concepts.yaml):
  paired    matched present/absent pairs sharing a group_id (the constructed
            controls/candidates from concept_sources.py, and the CAA contrastive
            sets). The battery groups on group_id so a pair never straddles a fold.
  two_pool  two unpaired pools — refusal (AdvBench harmful vs Alpaca harmless,
            Arditi/Exp-1 DoM) and truth_value (cities true vs false). The battery
            runs these with groups=None (each row its own group).

No GPU, no OpenAI key. Network only (all sources are raw GitHub). Behavioral
labels (refused/complied) are NOT set here — they are produced at extraction
(03) for the two concepts that need them (refusal, harmful_topic_benign).

Output:
  data/concept_pairs.parquet        one row per text, columns below
  data/concept_pairs_audit.json     per-concept length_balance + lexical_leak

Run:  .venv/bin/python "Experiment 2/rebuild/scripts/02_build_concept_pairs.py"
"""
from __future__ import annotations

import io
import json
import pathlib
import random
import sys

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # reach root libs
import concept_sources as cs  # noqa: E402
from audits import length_balance, lexical_leak  # noqa: E402
from paths import config_path, data_path  # noqa: E402

SEED = 42
TIMEOUT = 30

ADVBENCH_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")
ALPACA_URL = ("https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
              "main/alpaca_data.json")
CAA_URL = ("https://raw.githubusercontent.com/nrimsky/CAA/main/"
           "datasets/generate/{name}/generate_dataset.json")
CITIES_URL = ("https://raw.githubusercontent.com/saprmarks/geometry-of-truth/"
              "main/datasets/cities.csv")

# Concepts whose Gate-1 label is the model's BEHAVIOR (set at extraction), not polarity.
BEHAVIORAL = {"refusal", "harmful_topic_benign"}

rows: list[dict] = []


def add(concept, role, source, design, group_id, polarity, text, fmt="user"):
    rows.append(dict(concept=concept, role=role, source=source, design=design,
                     group_id=group_id, polarity=polarity, needs_behavior=concept in BEHAVIORAL,
                     text=str(text).strip(), format=fmt))


def fetch_text(url):
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


# ── refusal: AdvBench harmful vs Alpaca harmless (+ anchors), Arditi/Exp-1 DoM ──
def build_refusal(n_harmful, n_harmless, n_anchor):
    adv = pd.read_csv(io.StringIO(fetch_text(ADVBENCH_URL)))
    harmful = adv["goal"].drop_duplicates().sample(n_harmful, random_state=SEED).tolist()

    alpaca = json.loads(fetch_text(ALPACA_URL))
    clean = [d["instruction"].strip() for d in alpaca if not d.get("input", "").strip()]
    clean = list(dict.fromkeys(clean))
    rng = random.Random(SEED)
    rng.shuffle(clean)
    need = n_harmless + n_anchor
    if len(clean) < need:
        raise RuntimeError(f"Alpaca: only {len(clean)} clean instructions (< {need})")
    harmless = clean[:n_harmless]
    anchors = clean[n_harmless:n_harmless + n_anchor]

    for i, p in enumerate(harmful):
        add("refusal", "anchor", "advbench", "two_pool", f"refusal_h_{i:04d}", "present", p)
    for i, p in enumerate(harmless):
        add("refusal", "anchor", "alpaca", "two_pool", f"refusal_l_{i:04d}", "absent", p)
    for i, p in enumerate(anchors):
        add("anchor", "injection_target", "alpaca", "two_pool", f"anchor_{i:04d}", "anchor", p)
    return len(harmful), len(harmless), len(anchors)


# ── CAA contrastive pairs (paired): present=matching, absent=not-matching ───────
def build_caa(concept, role, n):
    data = json.loads(fetch_text(CAA_URL.format(name={"sycophancy": "sycophancy",
                                                      "corrigibility": "corrigible-neutral-HHH"}[concept])))
    random.Random(SEED).shuffle(data)
    built = 0
    for d in data:
        q = (d.get("question") or "").strip()
        pos = (d.get("answer_matching_behavior") or "").strip()
        neg = (d.get("answer_not_matching_behavior") or "").strip()
        if not (q and pos and neg):
            continue
        gid = f"{concept}_{built:04d}"
        add(concept, role, "nrimsky_caa", "paired", gid, "present", f"{q}\n{pos}")
        add(concept, role, "nrimsky_caa", "paired", gid, "absent", f"{q}\n{neg}")
        built += 1
        if built >= n:
            break
    return built


# ── truth_value (two_pool): cities true vs false ───────────────────────────────
def build_truth_value(role, n):
    df = pd.read_csv(io.StringIO(fetch_text(CITIES_URL)))
    true_s = df[df.label == 1]["statement"].tolist()
    false_s = df[df.label == 0]["statement"].tolist()
    rng = random.Random(SEED)
    rng.shuffle(true_s)
    rng.shuffle(false_s)
    k = min(n, len(true_s), len(false_s))
    for i in range(k):
        add("truth_value", role, "geometry_of_truth", "two_pool", f"truth_t_{i:04d}", "present", true_s[i])
        add("truth_value", role, "geometry_of_truth", "two_pool", f"truth_f_{i:04d}", "absent", false_s[i])
    return k


# ── offline matched pairs (paired) from concept_sources ────────────────────────
def build_offline(concept, role, n):
    pairs = cs.build(concept, n)
    for i, (present, absent) in enumerate(pairs):
        gid = f"{concept}_{i:04d}"
        add(concept, role, "constructed", "paired", gid, "present", present)
        add(concept, role, "constructed", "paired", gid, "absent", absent)
    return len(pairs)


def main() -> int:
    manifest = yaml.safe_load(config_path("concepts.yaml").read_text())
    cmap = {c["key"]: c for c in manifest["concepts"]}

    def n_of(key, default=120):
        return int(cmap.get(key, {}).get("n_target", default))

    def role_of(key):
        return cmap.get(key, {}).get("role", "candidate")

    built, skipped = {}, {}

    def run(label, fn):
        try:
            built[label] = fn()
            print(f"  [ok]   {label}: {built[label]}")
        except Exception as e:  # graceful: a dead source must not sink the run
            skipped[label] = f"{type(e).__name__}: {e}"
            print(f"  [skip] {label}: {skipped[label]}")

    print("Network concepts:")
    run("refusal+anchor", lambda: build_refusal(n_of("refusal"), n_of("refusal"), n_of("refusal")))
    run("sycophancy", lambda: build_caa("sycophancy", role_of("sycophancy"), n_of("sycophancy")))
    run("corrigibility", lambda: build_caa("corrigibility", role_of("corrigibility"), n_of("corrigibility")))
    run("truth_value", lambda: build_truth_value(role_of("truth_value"), n_of("truth_value")))

    print("Offline concepts (curated base; may be < n_target until expanded):")
    for key in cs.OFFLINE_CONCEPTS:
        run(key, lambda key=key: build_offline(key, role_of(key), n_of(key)))

    if not rows:
        print("No rows built — all sources failed. Check network.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    out = data_path("concept_pairs.parquet")
    df.to_parquet(out, index=False)

    # ── per-concept audit ──
    audit = {}
    for concept, sub in df.groupby("concept"):
        pres = sub[sub.polarity == "present"]
        absnt = sub[sub.polarity == "absent"]
        rec = {"n_present": int(len(pres)), "n_absent": int(len(absnt)),
               "design": sub.design.iloc[0], "role": sub.role.iloc[0],
               "n_target": n_of(concept, default=0)}
        if len(pres) and len(absnt):
            rec["length_balance"] = length_balance(pres.text.str.len().tolist(),
                                                   absnt.text.str.len().tolist())
            rec["lexical_leak"] = {k: v for k, v in
                                   lexical_leak(pres.text.tolist(), absnt.text.tolist()).items()
                                   if k != "leaks"}
        audit[concept] = rec
    data_path("concept_pairs_audit.json").write_text(json.dumps(audit, indent=2))

    # ── summary ──
    print(f"\nSaved {len(df)} rows / {df.concept.nunique()} concepts -> {out}")
    print(f"{'concept':24s} {'design':9s} {'present':>7} {'absent':>7} {'target':>6} {'pct_gap':>8} {'leaks':>6}")
    for concept in sorted(audit):
        r = audit[concept]
        lb = r.get("length_balance", {})
        lk = r.get("lexical_leak", {})
        short = "" if r["n_present"] >= r["n_target"] or not r["n_target"] else "  <-- below target"
        print(f"{concept:24s} {r['design']:9s} {r['n_present']:>7} {r['n_absent']:>7} "
              f"{r['n_target']:>6} {lb.get('pct_gap', float('nan')):>7}% {lk.get('n_leaks', 0):>6}{short}")
    if skipped:
        print("\nSKIPPED sources (rerun when reachable):")
        for k, v in skipped.items():
            print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
