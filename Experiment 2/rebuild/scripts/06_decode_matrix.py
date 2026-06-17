#!/usr/bin/env python
"""06_decode_matrix.py — Gate-2 AV decode of the injection matrix (box, SGLang AV).

Reads 05_inject_matrix's injected vectors + metadata, decodes each through the
released NLA Activation Verbalizer (raw output, <explanation> tags KEPT — tag
extraction is deferred to scoring), N_SAMPLES per vector, resumable + threaded.
Joins nothing here — just emits {row, sample, nla_output}; 07 joins to the 05
metadata on `row` and scores.

Ported from CAA 04b_decode_matrix (the proven resume + ThreadPool + health-check
loop), with paths via paths.py and the robust AV-dir / NLA_REPO_DIR resolution
from the smoke.

Output goes to results/gate2/ (not gitignored cache/) on purpose: the raw decodes
are the Gate-2 evidence we inspect + score here, and at this matrix size they're
small. (Deliberate deviation from CLAUDE.md's "raw decodes -> cache"; reconcile if
the anchor count grows enough that the file gets large.)

Needs the AV server up (bash scripts/av_up.sh <model>) and 05's cache.
Run (box):  NLA_REPO_DIR=/workspace/nla_repo python scripts/06_decode_matrix.py --model gemma
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
sys.path.insert(0, str(HERE.parent))  # rebuild root -> paths.py
sys.path.insert(0, str(HERE))         # scripts/    -> nla_box.py
import nla_box  # noqa: E402  (reuse resolve_av — same AV-dir convention as av_up.sh)
from paths import cache_path, model_slug, result_path, stage_of  # noqa: E402

STAGE = stage_of(__file__)
INJECT_STAGE = "05_inject_matrix"
N_SAMPLES = 3        # per vector — enables the multi-sample consistency check in 08
MAX_WORKERS = 6      # CAA's value; note the input_embeds FastAPI path caps useful concurrency ~2
MAX_NEW_TOK = 200
SAVE_EVERY = 300
PORT = int(os.environ.get("SGLANG_PORT", "30000"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", choices=["gemma", "qwen"])
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOK)
    a = ap.parse_args()
    model = model_slug(a.model)

    import httpx
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    # --- preflight: AV server healthy ---
    url = f"http://localhost:{PORT}"
    try:
        assert httpx.get(url + "/health", timeout=5).status_code == 200
        print(f"SGLang AV healthy on :{PORT}")
    except Exception:
        sys.exit(f"SGLang AV not reachable at {url}/health — run: bash scripts/av_up.sh {a.model}")

    # --- inputs from 05 ---
    npy = cache_path(INJECT_STAGE, model, concept="all", ext="npy")
    pq = cache_path(INJECT_STAGE, model, concept="all", ext="parquet")
    if not (npy.exists() and pq.exists()):
        sys.exit(f"FAIL: 05 outputs missing ({npy.name} / {pq.name}) — "
                 f"run 05_inject_matrix.py --model {a.model} first.")
    vecs = np.load(npy)
    meta = pd.read_parquet(pq)
    assert len(vecs) == len(meta), f"vec/meta length mismatch: {len(vecs)} vs {len(meta)}"

    # --- AV client (same convention as the smoke / nla_box) ---
    nla_repo = os.environ.get("NLA_REPO_DIR", "/workspace/nla_repo")
    if not pathlib.Path(nla_repo, "nla_inference.py").exists():
        sys.exit(f"FAIL: nla_inference.py not under NLA_REPO_DIR={nla_repo!r} — run av_up.sh.")
    sys.path.insert(0, nla_repo)
    from nla_inference import NLAClient  # noqa: E402
    av_dir = nla_box.resolve_av(a.model, full=True)
    client = NLAClient(av_dir, sglang_url=url, device="cpu")
    assert client.cfg.d_model == vecs.shape[1], (
        f"AV d_model {client.cfg.d_model} != injected d {vecs.shape[1]} — wrong AV for {model}")
    print(f"{len(vecs)} vectors × {a.n_samples} samples = {len(vecs) * a.n_samples} decodes")

    # --- resume ---
    out = result_path("gate2", STAGE, model, concept="all", ext="parquet")
    done: set[tuple[int, int]] = set()
    existing: list[dict] = []
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

    def decode_one(i: int, s: int) -> dict:
        v = torch.tensor(vecs[i], dtype=torch.float32)
        txt = client.generate(v, extract_explanation=False, max_new_tokens=a.max_new_tokens)
        return {"row": int(i), "sample": int(s), "nla_output": txt}

    results = list(existing)
    t0 = time.time()
    comp = err = 0
    with ThreadPoolExecutor(max_workers=a.max_workers) as pool:
        futs = {pool.submit(decode_one, i, s): (i, s) for i, s in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="decoding", unit="dec"):
            i, s = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"row": int(i), "sample": int(s), "nla_output": f"[ERROR: {type(e).__name__}: {e}]"}
                err += 1
            results.append(rec)
            comp += 1
            if comp % SAVE_EVERY == 0:
                pd.DataFrame(results).to_parquet(out, index=False)

    df = pd.DataFrame(results).sort_values(["row", "sample"]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    tag = df.nla_output.str.contains("<explanation>").mean()
    erate = df.nla_output.str.startswith("[ERROR").mean()
    print(f"\ndone: {len(todo)} new decodes in {(time.time() - t0) / 60:.1f} min  errors={err}")
    print(f"<explanation> tag rate: {tag:.3f} (expect ~1.0)   error rate: {erate:.3f} (expect ~0)")
    print(f"wrote {len(df)} rows -> {out}")
    print("next: 07_score_matrix (regex + judge, four-way)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
