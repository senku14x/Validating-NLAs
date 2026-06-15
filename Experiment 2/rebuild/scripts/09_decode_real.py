#!/usr/bin/env python
"""09_decode_real.py — Gate-3 real-activation decode (box, SGLang AV).

The on-manifold test the offline injection matrix couldn't give. For each concept,
take its REAL block-L activations from the 03 cache — concept-PRESENT (y==1) and
concept-ABSENT (y==0) — plus neutral anchors, and decode them through the released
AV. No injection, no steering: these are states the model actually produced, so
on-manifold by construction. This is the only fair test for the negative-baseline
concepts (truth_value, harmful_topic_benign, eval_framing) that went off-manifold
under offline injection.

The decode set per concept is balanced (up to N_PER_CLASS present + N_PER_CLASS
absent) so 10_analyze_real can compare present-vs-absent-vs-anchor detection rates.

Ported decode loop from 06 (health-check, resume, ThreadPool, sanity). Consumes the
03 cache; writes results/gate2/09_decode_real__<model>__all.parquet
[concept, polarity, src_idx, sample, nla_output].

Run (box, after 03 + av_up for this model):
  NLA_REPO_DIR=/workspace/nla_repo python scripts/09_decode_real.py --model gemma
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import nla_box  # noqa: E402
from paths import cache_path, model_slug, result_path, stage_of  # noqa: E402

STAGE = stage_of(__file__)
EXTRACT_STAGE = "03_extract_for_battery"
# concepts whose real activations we decode (5 survivors + refusal anchor + eval_framing exploratory)
CONCEPTS = ["refusal", "truth_value", "sycophancy", "corrigibility",
            "neg_sentiment", "harmful_topic_benign", "eval_framing_matched"]
N_PER_CLASS = 50      # up to this many present + this many absent per concept
N_ANCHORS = 50
N_SAMPLES = 2
MAX_WORKERS = 6
MAX_NEW_TOK = 200
SAVE_EVERY = 300
PORT = int(os.environ.get("SGLANG_PORT", "30000"))
SEED = 42


def build_decode_set(model: str, n_per_class: int = N_PER_CLASS, n_anchors: int = N_ANCHORS):
    """Assemble (concept, polarity, src_idx, vector) rows from the 03 cache."""
    rng = np.random.default_rng(SEED)
    items = []  # (concept, polarity, src_idx, vector)
    for c in CONCEPTS:
        p = cache_path(EXTRACT_STAGE, model, concept=c, ext="npz")
        if not p.exists():
            print(f"  [skip] {c}: no 03 cache at {p.name}")
            continue
        z = np.load(p)
        X, y = z["X"].astype(np.float32), z["y"].astype(int)
        for pol, lab in (("present", 1), ("absent", 0)):
            idx = np.where(y == lab)[0]
            if len(idx) == 0:
                continue
            for i in rng.permutation(idx)[:n_per_class]:
                items.append((c, pol, int(i), X[i]))
    # neutral anchors (concept-absent baseline)
    ap = cache_path(EXTRACT_STAGE, model, concept="anchor", ext="npz")
    if ap.exists():
        Xa = np.load(ap)["X"].astype(np.float32)
        for i in rng.permutation(len(Xa))[:n_anchors]:
            items.append(("anchor", "none", int(i), Xa[i]))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", choices=["gemma", "qwen"])
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    a = ap.parse_args()
    model = model_slug(a.model)

    import httpx
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    url = f"http://localhost:{PORT}"
    try:
        assert httpx.get(url + "/health", timeout=5).status_code == 200
        print(f"SGLang AV healthy on :{PORT}")
    except Exception:
        sys.exit(f"SGLang AV not reachable at {url}/health — run: bash scripts/av_up.sh {a.model}")

    items = build_decode_set(model, a.n_per_class, N_ANCHORS)
    if not items:
        sys.exit(f"FAIL: no 03 cache for {model} — run 03_extract_for_battery.py --model {a.model} first.")
    vecs = np.stack([v for *_, v in items]).astype(np.float32)
    meta = pd.DataFrame([{"row": i, "concept": c, "polarity": pol, "src_idx": s}
                         for i, (c, pol, s, _) in enumerate(items)])
    print(f"{len(items)} real activations to decode × {a.n_samples} samples")
    print("  per concept/polarity:\n" + meta.groupby(["concept", "polarity"]).size().to_string())

    nla_repo = os.environ.get("NLA_REPO_DIR", "/workspace/nla_repo")
    if not pathlib.Path(nla_repo, "nla_inference.py").exists():
        sys.exit(f"FAIL: nla_inference.py not under NLA_REPO_DIR={nla_repo!r} — run av_up.sh.")
    sys.path.insert(0, nla_repo)
    from nla_inference import NLAClient  # noqa: E402
    client = NLAClient(nla_box.resolve_av(a.model, full=True), sglang_url=url, device="cpu")
    assert client.cfg.d_model == vecs.shape[1], (
        f"AV d_model {client.cfg.d_model} != activation d {vecs.shape[1]} — wrong AV for {model}")

    out = result_path("gate2", STAGE, model, concept="all", ext="parquet")
    done, existing = set(), []
    if out.exists():
        prev = pd.read_parquet(out)
        for _, r in prev.iterrows():
            done.add((int(r["row"]), int(r["sample"])))
            existing.append(r.to_dict())
        print(f"resuming: {len(done)} decodes already done")
    todo = [(i, s) for i in range(len(vecs)) for s in range(a.n_samples) if (i, s) not in done]
    if not todo:
        print(f"all decodes complete -> {out}")
        return 0
    print(f"{len(todo)} decodes remaining")

    meta_by_row = meta.set_index("row").to_dict("index")

    def decode_one(i, s):
        txt = client.generate(torch.tensor(vecs[i], dtype=torch.float32),
                              extract_explanation=False, max_new_tokens=MAX_NEW_TOK)
        m = meta_by_row[i]
        return {"row": int(i), "sample": int(s), "concept": m["concept"],
                "polarity": m["polarity"], "src_idx": int(m["src_idx"]), "nla_output": txt}

    results = list(existing)
    t0 = time.time()
    err = 0
    with ThreadPoolExecutor(max_workers=a.max_workers) as pool:
        futs = {pool.submit(decode_one, i, s): (i, s) for i, s in todo}
        for n, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc="decoding", unit="dec"), 1):
            i, s = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                m = meta_by_row[i]
                rec = {"row": int(i), "sample": int(s), "concept": m["concept"], "polarity": m["polarity"],
                       "src_idx": int(m["src_idx"]), "nla_output": f"[ERROR: {type(e).__name__}: {e}]"}
                err += 1
            results.append(rec)
            if n % SAVE_EVERY == 0:
                pd.DataFrame(results).to_parquet(out, index=False)

    df = pd.DataFrame(results).sort_values(["row", "sample"]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    tag = df.nla_output.str.contains("<explanation>").mean()
    print(f"\ndone: {len(todo)} decodes in {(time.time()-t0)/60:.1f} min  errors={err}")
    print(f"<explanation> tag rate: {tag:.3f}   wrote {len(df)} rows -> {out}")
    print("next: 07_score_matrix --real --model %s, then 10_analyze_real" % a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
