#!/usr/bin/env python
"""
03_neuronpedia_probe.py — settle the A_f sourcing question before building co-firing.

The co-firing test needs, per feature:
    A_f = sentences where the feature fires (target: N_POS=100)
    N_f = sentences where it does not (drawn from a shared negative pool)

Neuronpedia precomputes top-activating examples per feature. The question is
HOW MANY per feature. Two outcomes:

    >= 100 per feature  -> pure Neuronpedia, zero SAE compute for A_f
    ~20-30 per feature  -> need self-computed top-up (run SAE.encode() over a
                           large Pile corpus, Qwen tokenizer only)

This script pulls a few batch files from the activations/ bucket, counts
examples per feature, and also pulls the explanations/ labels to confirm
coverage. It does NOT need a GPU.

S3 layout (public, no auth):
    {S3}/explanations/batch-{b}.jsonl.gz   feature -> text label
    {S3}/activations/batch-{b}.jsonl.gz    feature -> list of activating examples

Usage:
    python 03_neuronpedia_probe.py --n-batches 3
"""
import argparse
import gzip
import io
import json
import urllib.request
from collections import defaultdict

S3 = ("https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
      "v1/qwen2.5-7b-it/20-matryoshka-65k")
EXPL = f"{S3}/explanations"
ACTS = f"{S3}/activations"


def fetch_jsonl_gz(base, b):
    """Fetch and parse one batch-{b}.jsonl.gz. Returns list of dicts, or None."""
    url = f"{base}/batch-{b}.jsonl.gz"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
        rows = []
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            for line in gz:
                rows.append(json.loads(line.decode()))
        return rows
    except Exception as e:
        print(f"  [batch {b}] fetch failed: {e}")
        return None


def probe_explanations(n_batches):
    print("=" * 70)
    print("EXPLANATIONS (feature labels)")
    print("=" * 70)
    n_feat, sample = 0, []
    for b in range(n_batches):
        rows = fetch_jsonl_gz(EXPL, b)
        if rows is None:
            break
        n_feat += len(rows)
        print(f"  batch-{b}: {len(rows)} feature labels")
        if not sample:
            # inspect the schema of the first row
            print(f"  row keys: {list(rows[0].keys())}")
            for r in rows[:5]:
                idx = r.get("index", r.get("feature", "?"))
                desc = r.get("description", r.get("explanation", ""))
                sample.append((idx, desc))
    print(f"  total labels seen: {n_feat}")
    print("  sample labels:")
    for idx, desc in sample:
        print(f"    [{idx}] {desc[:70]}")
    return n_feat


def probe_activations(n_batches):
    args_n = n_batches
    print("\n" + "=" * 70)
    print("ACTIVATIONS (A_f source — the gating question)")
    print("=" * 70)
    per_feature_counts = defaultdict(int)
    schema_printed = False
    total_examples = 0

    for b in range(n_batches):
        rows = fetch_jsonl_gz(ACTS, b)
        if rows is None:
            break
        print(f"  batch-{b}: {len(rows)} rows")

        if not schema_printed:
            r0 = rows[0]
            print(f"  row keys: {list(r0.keys())}")
            # find the list-of-examples field
            for k, v in r0.items():
                if isinstance(v, list):
                    print(f"    field '{k}': list of {len(v)} items")
                    if v and isinstance(v[0], dict):
                        print(f"      example item keys: {list(v[0].keys())}")
            schema_printed = True

        # CORRECT counting: each ROW is ONE activating example. tokens/values
        # are the 127-token context of that single example, NOT a list of
        # examples. Count rows per feature index.
        for r in rows:
            idx = r.get("index", r.get("feature"))
            per_feature_counts[idx] += 1   # one row = one example
            total_examples += 1

    counts = list(per_feature_counts.values())
    if not counts:
        print("  NO activation data parsed — check schema above.")
        return
    print("\n  (each row = ONE example; tokens/values are that examples 127-token context)")
    print(f"  total rows parsed: {total_examples}  across {len(per_feature_counts)} unique features")
    if len(per_feature_counts) < 1000:
        print(f"  NOTE: only {len(per_feature_counts)} distinct features in {args_n} batches —")
        print(f"  batches appear feature-contiguous, so this is a LOCAL slice, not the")
        print(f"  global per-feature rate. Counts below are real for these features.")

    counts.sort()
    n = len(counts)
    print(f"\n  features with activation data: {n}")
    print(f"  examples per feature:")
    print(f"    min={counts[0]}  p25={counts[n//4]}  median={counts[n//2]}  "
          f"p75={counts[3*n//4]}  max={counts[-1]}")
    print(f"    mean={sum(counts)/n:.1f}")

    # the decision
    n_ge_100 = sum(c >= 100 for c in counts)
    n_ge_50  = sum(c >= 50 for c in counts)
    n_ge_30  = sum(c >= 30 for c in counts)
    print(f"\n  features with >= 100 examples: {n_ge_100}/{n} ({100*n_ge_100/n:.0f}%)")
    print(f"  features with >=  50 examples: {n_ge_50}/{n} ({100*n_ge_50/n:.0f}%)")
    print(f"  features with >=  30 examples: {n_ge_30}/{n} ({100*n_ge_30/n:.0f}%)")

    print("\n  DECISION:")
    if n_ge_100 / n > 0.5:
        print("    >50% of features have >=100 examples.")
        print("    -> Use Neuronpedia A_f directly. NO self-compute needed.")
    elif n_ge_50 / n > 0.5:
        print("    Most features have 50-100 examples, few have 100.")
        print("    -> Either lower N_POS to 50, or self-compute top-up for the")
        print("       features you want at 100. Lowering N_POS is simpler for the pilot.")
    else:
        print("    Most features have <50 examples (typical: ~20-30 'top' examples).")
        print("    -> Neuronpedia alone is insufficient for N_POS=100.")
        print("    -> SELF-COMPUTE required: run SAE.encode() over ~200k-500k Pile")
        print("       sentences (Qwen tokenizer only), keep sentences above the")
        print("       feature's firing threshold. This is the 04_define_features step.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=3,
                    help="how many batch files to sample per bucket")
    args = ap.parse_args()

    n_labels = probe_explanations(args.n_batches)
    probe_activations(args.n_batches)

    print("\n" + "=" * 70)
    print("Next: 04_define_features.py builds A_f/N_f using whichever source the")
    print("DECISION above indicates, then 05 runs co-firing.")
