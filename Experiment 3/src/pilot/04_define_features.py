#!/usr/bin/env python
"""
04_define_features.py — build A_f / N_f for a random pilot feature set.

Decision from 03: Neuronpedia gives ~45 top examples/feature. We use those
directly (cleaner positives than self-computed; the ~0.02 AUROC SE gain from
reaching 100 is not worth a 200k-sentence SAE pass). Power comes from a large
length-matched negative pool, which is free.

CRITICAL GATE (runs first): does Neuronpedia's feature index match the SAE's
W_dec row index? If feature i on Neuronpedia != sae.W_dec[i], every co-firing
score is against the wrong direction. We verify by pulling a feature's own top
examples, running them through Qwen + SAE, and confirming feature i's activation
matches Neuronpedia's recorded `maxValue`. If it does not, indexing/layering is
wrong — STOP.

Pipeline:
  1. INDEX-ALIGNMENT GATE on a few features (hard stop on failure)
  2. random sample N_PILOT alive features
  3. for each: A_f = deduplicated Neuronpedia top examples (detokenized -> text)
  4. N_f = shared Pile negative pool, verified feature does NOT fire
  5. save feature_set.json + per-feature A_f texts + the negative pool

Saves:
  {out_dir}/pilot_features.json   {feature_idx: {label, n_pos, pos_texts, max_vals}}
  {out_dir}/negative_pool.json    list of negative sentences (shared)
  {out_dir}/index_gate.json       alignment check result

Usage:
  HF_TOKEN=... python 04_define_features.py --out-dir /content/exp3_acts \
      --n-pilot 50 --seed 42
"""
import argparse
import gzip
import io
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import exp3_config as C

S3 = ("https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
      "v1/qwen2.5-7b-it/20-matryoshka-65k")
TOKEN = os.environ.get("HF_TOKEN")
N_ACT_BATCHES = 12      # how many activation batches to scan for features
NEG_POOL_SIZE = 300     # shared negatives (length-matched subsets drawn per feature)


def fetch_jsonl_gz(base, b):
    url = f"{base}/batch-{b}.jsonl.gz"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read()
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        for line in gz:
            rows.append(json.loads(line.decode()))
    return rows


def load_activation_rows(n_batches, start_batch=0):
    """Return {feature_idx: [rows]} from the activations bucket."""
    by_feat = defaultdict(list)
    for b in range(start_batch, start_batch + n_batches):
        try:
            rows = fetch_jsonl_gz(f"{S3}/activations", b)
        except Exception as e:
            print(f"  batch {b} failed: {e}"); continue
        for r in rows:
            by_feat[r["index"]].append(r)
        print(f"  loaded activations batch-{b} ({len(rows)} rows, "
              f"{len(by_feat)} features so far)")
    return by_feat


def load_labels_for_features(feature_ids, start_batch, end_batch):
    """Load labels for selected feature ids.

    Neuronpedia explanation batches are roughly feature-contiguous, but their
    exact row counts differ slightly from activation batches. First try the
    activation batch window, then fetch a small estimated window around each
    missing feature id. This keeps high-index runs labeled without needing a
    separate patch script.
    """
    wanted = {int(fi) for fi in feature_ids}
    labels = {}
    fetched = set()

    def fetch_into(b):
        if b < 0 or b in fetched:
            return
        fetched.add(b)
        try:
            for r in fetch_jsonl_gz(f"{S3}/explanations", b):
                idx = int(r["index"])
                if idx in wanted and idx not in labels:
                    labels[idx] = r.get("description", "")
        except Exception as e:
            print(f"  explanation batch {b} failed: {e}")

    for b in range(start_batch, end_batch):
        fetch_into(b)
        if wanted.issubset(labels):
            break

    missing = wanted - set(labels)
    if missing:
        extra_batches = set()
        for fi in missing:
            guess = fi // 250
            extra_batches.update(range(guess - 4, guess + 5))
        for b in sorted(extra_batches):
            fetch_into(b)
            if wanted.issubset(labels):
                break

    print(f"  labels loaded: {len(labels)}/{len(wanted)} pilot features")
    return labels


def load_qwen_and_sae():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    tok = AutoTokenizer.from_pretrained(C.QWEN_REPO, token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        C.QWEN_REPO, token=TOKEN, torch_dtype=torch.bfloat16, device_map="auto").eval()
    obj = SAE.from_pretrained(C.SAE_REPO, C.SAE_ID, device="cuda")
    sae = obj[0] if isinstance(obj, tuple) else obj
    return model, tok, sae


@torch.no_grad()
def qwen_sae_encode(model, tok, sae, text, hs_idx, skip_first):
    """Return per-token SAE feature activations [n_tok_kept, n_features] for text."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    hs = model(**enc, output_hidden_states=True).hidden_states[hs_idx][0]  # [T, d]
    hs = hs[skip_first:]
    feats = sae.encode(hs.float().to(sae.W_enc.device))                    # [T', F]
    return feats.cpu().numpy()


def detokenize(tokens):
    """Neuronpedia stores tokens as a list of strings; join to text."""
    if tokens and isinstance(tokens[0], str):
        return "".join(tokens).replace("\u2581", " ").strip()
    return None


# ── INDEX-ALIGNMENT GATE ─────────────────────────────────────────────────────
def index_gate(by_feat, model, tok, sae, check_feats):
    """For each check feature: take its own top example, encode through Qwen+SAE,
    and confirm that the same feature index reaches Neuronpedia's maxValue."""
    print("\n" + "=" * 70)
    print("INDEX-ALIGNMENT GATE: Neuronpedia idx == SAE W_dec row idx?")
    print("=" * 70)
    # CORRECT gate: check self_act ≈ np_maxValue (within 30%).
    # The original "top-5 rank" test was wrong: general features (e.g., 78, 719)
    # fire more strongly than specific features on any text, so a feature need
    # not be in the top-5 even if its index is perfectly correct. The right
    # check is whether our SAE fires feature i at approximately the activation
    # value Neuronpedia recorded — that directly confirms index agreement.
    VALUE_TOL = 0.30   # |self_act - np_max| / np_max < 30% => ALIGNED
    MIN_NP_MAX = 5.0   # skip very weak firings (noisy denominator)
    results = []
    for fi in check_feats:
        rows = by_feat[fi]
        row = max(rows, key=lambda r: r.get("maxValue", 0))
        np_max = float(row.get("maxValue", 0))
        text = detokenize(row["tokens"])
        if not text:
            print(f"  feature {fi}: could not detokenize, skipping"); continue
        feats = qwen_sae_encode(model, tok, sae, text, C.QWEN_HS_IDX, C.SAE_SKIP_FIRST_N)
        max_per_feat = feats.max(0)   # [F] peak activation per feature
        self_act = float(max_per_feat[fi])
        if np_max < MIN_NP_MAX:
            aligned = None   # too weak to judge
            flag = "SKIP (weak)"
        else:
            rel_err = abs(self_act - np_max) / np_max
            aligned = rel_err < VALUE_TOL
            flag = f"OK (err={rel_err:.0%})" if aligned else f"MISALIGNED (err={rel_err:.0%})"
        results.append({"feature": int(fi), "aligned": aligned,
                        "self_act": self_act, "np_maxValue": np_max})
        top1 = int(max_per_feat.argmax())
        print(f"  feature {fi:5d}: self_act={self_act:.2f}  np_max={np_max:.2f}  "
              f"SAE_peak_feat={top1:5d}  [{flag}]")
    judged  = [r for r in results if r["aligned"] is not None]
    n_ok    = sum(r["aligned"] for r in judged)
    passed  = len(judged) > 0 and n_ok >= max(1, int(0.75 * len(judged)))
    print(f"\n  {n_ok}/{len(judged)} judged features have self_act ≈ np_maxValue.")
    if passed:
        print("  GATE PASS: activation values match -> Neuronpedia index aligns with SAE.")
    else:
        print("  GATE FAIL: self_act badly off from np_maxValue.")
        print("  Possible causes: wrong SAE loaded, wrong layer, tokenization mismatch.")
        print("  Check: sae.cfg.hook_name, C.QWEN_HS_IDX, and detokenize output.")
    return passed, results


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/content/exp3_acts")
    ap.add_argument("--n-pilot", type=int, default=50)
    ap.add_argument("--batch-start", type=int, default=0,
                    help="first Neuronpedia activation batch to load ""(batch N covers features 256*N to 256*N+255)")
    ap.add_argument("--batch-end", type=int, default=None,
                    help="exclusive end batch; defaults to batch-start + 12")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-pos", type=int, default=20)
    ap.add_argument("--corpus-dir", default=None,
                    help="directory containing corpus_pile_*.parquet for the negative pool; "
                    "defaults to --out-dir. Set to the original run dir when using a new --out-dir.")
    ap.add_argument("--min-np-max", type=float, default=15.0,
                    help="minimum Neuronpedia maxValue to include a feature. "
                    "High-index Matryoshka features fire weakly (~np_max 6-12); "
                    "they cannot reliably reproduce activation after retokenization "
                    "(k=100 TopK cutoff pushes them out). Filter them before sampling.")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading Neuronpedia activation rows...")
    n_act = (args.batch_end if args.batch_end else args.batch_start + N_ACT_BATCHES) - args.batch_start
    feat_range_lo = args.batch_start * 256
    feat_range_hi = (args.batch_start + n_act) * 256 - 1
    print(f"  batch range: {args.batch_start}-{args.batch_start+n_act-1} (feature indices ~{feat_range_lo}-{feat_range_hi})")
    by_feat = load_activation_rows(n_act, start_batch=args.batch_start)
    avail = [fi for fi, rows in by_feat.items() if len(rows) >= args.min_pos]
    print(f"\n{len(avail)} features with >= {args.min_pos} examples available")
    if args.min_np_max > 0:
        avail = [fi for fi in avail
                 if max(by_feat[fi], key=lambda r: r.get("maxValue", 0))
                    .get("maxValue", 0) >= args.min_np_max]
        print(f"{len(avail)} features with max_activation >= {args.min_np_max} "
              f"(weak features excluded: unreliable after retokenization under k=100 TopK)")

    print("\nLoading Qwen + SAE for index gate and negative verification...")
    model, tok, sae = load_qwen_and_sae()

    # GATE: check 8 random features for index alignment
    check = list(rng.choice(avail, size=min(8, len(avail)), replace=False))
    passed, gate_results = index_gate(by_feat, model, tok, sae, check)
    (out / "index_gate.json").write_text(json.dumps(gate_results, indent=2))
    if not passed:
        print("\nSTOPPING: fix index alignment before continuing.")
        return

    # random pilot sample
    pilot = list(rng.choice(avail, size=min(args.n_pilot, len(avail)), replace=False))
    print(f"\nrandom pilot sample: {len(pilot)} features")
    labels = load_labels_for_features(pilot, args.batch_start, args.batch_start + n_act)

    # build A_f for each (detokenize Neuronpedia top examples)
    feature_set = {}
    for fi in pilot:
        rows = by_feat[fi]
        by_text = {}
        for r in rows:
            t = detokenize(r["tokens"])
            if t and len(t) >= 20:
                max_val = float(r.get("maxValue", 0))
                if t not in by_text or max_val > by_text[t]:
                    by_text[t] = max_val
        pos_pairs = sorted(by_text.items(), key=lambda x: -x[1])
        pos_texts = [t for t, _ in pos_pairs]
        max_vals = [v for _, v in pos_pairs]
        if len(pos_texts) >= args.min_pos:
            feature_set[int(fi)] = {
                "label": labels.get(fi, ""),
                "n_pos": len(pos_texts),
                "pos_texts": pos_texts,
                "max_vals": max_vals,
            }
    if not feature_set:
        print("No features survived positive-example dedup/min-pos filtering.")
        return
    print(f"built A_f for {len(feature_set)} features "
          f"(median unique n_pos = {int(np.median([v['n_pos'] for v in feature_set.values()]))})")

    # shared negative pool from the materialized Pile corpus
    import pandas as pd
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else out
    corpus = None
    for p in sorted(corpus_dir.glob("corpus_pile_*.parquet"), key=lambda x: -x.stat().st_size):
        corpus = pd.read_parquet(p)["text"].tolist(); break
    if corpus is None:
        print(f"WARN: no corpus_pile_*.parquet found in {corpus_dir} — "
              f"pass --corpus-dir /content/exp3_acts (or wherever 01 saved it)"); return
    print(f"  negative pool sourced from: {corpus_dir}")
    neg_idx = rng.choice(len(corpus), size=min(NEG_POOL_SIZE, len(corpus)), replace=False)
    neg_pool = [corpus[i] for i in neg_idx]

    # verify negatives genuinely don't fire the pilot features (spot check)
    print("\nverifying negative pool (spot check 20 negs x pilot features)...")
    pilot_idx = np.array(list(feature_set.keys()))
    leak = 0
    for t in neg_pool[:20]:
        feats = qwen_sae_encode(model, tok, sae, t, C.QWEN_HS_IDX, C.SAE_SKIP_FIRST_N)
        fired = pilot_idx[feats.max(0)[pilot_idx] > 0]
        leak += len(fired)
    print(f"  {leak} (feature,neg) firings across 20 negs x {len(pilot_idx)} feats "
          f"({100*leak/(20*len(pilot_idx)):.1f}% leak — expected low; per-feature N_f "
          f"will filter exact firings at co-firing time)")

    pf  = out / "pilot_features.json"
    np_ = out / "negative_pool.json"
    pf.write_text(json.dumps(feature_set))
    np_.write_text(json.dumps(neg_pool))
    print(f"\nsaved {pf.name} ({len(feature_set)} features) + {np_.name} ({len(neg_pool)} sentences)")
    print("Next: 05_cofiring.py maps each feature's W_dec direction to Gemma and "
          "scores A_f vs N_f co-firing AUROC.")


if __name__ == "__main__":
    main()
