#!/usr/bin/env python

"""
04b_decode_matrix.py  --  NLA decode loop for the cross-detection matrix.

Assumes 04a_setup_sglang.py has already been run and the SGLang server is
live on port 30000.

Resumable: already-decoded (row, sample) pairs are detected in
/content/decode_outputs.parquet and skipped. Re-run freely after interruption.

Outputs
-------
  /content/decode_outputs.parquet    columns: row, sample, nla_output

Run:
  !python 04b_decode_matrix.py
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
INJECT_PATH  = "/content/injected_matrix.npy"
META_PATH    = "/content/injection_matrix_meta.parquet"
OUT_PATH     = "/content/decode_outputs.parquet"
AVDIR_FILE   = "/content/av_dir.txt"
REPO_DIR     = "/content/nla_repo"
PORT         = 30000
N_SAMPLES    = 3       # per activation
MAX_WORKERS  = 6
MAX_NEW_TOK  = 200
SAVE_EVERY   = 300     # incremental save cadence

# ── preflight ─────────────────────────────────────────────────────────────────
url = f"http://localhost:{PORT}"
try:
    status = httpx.get(url + "/health", timeout=5).status_code
    assert status == 200
    print(f"SGLang server healthy on port {PORT}")
except Exception:
    sys.exit(
        f"SGLang server not reachable at {url}/health.\n"
        "Run 04a_setup_sglang.py first."
    )

assert Path(AVDIR_FILE).exists(), (
    f"{AVDIR_FILE} not found — run 04a_setup_sglang.py first.")
av_dir = Path(AVDIR_FILE).read_text().strip()
print(f"AV dir: {av_dir}")

assert Path(INJECT_PATH).exists(), (
    f"{INJECT_PATH} not found — run 03_build_injection_sweep.py first.")
assert Path(META_PATH).exists(), (
    f"{META_PATH} not found — run 03_build_injection_sweep.py first.")

# ── load NLAClient ────────────────────────────────────────────────────────────
sys.path.insert(0, REPO_DIR)
from nla_inference import NLAClient

print("Loading NLAClient (embedding table on CPU)...")
client = NLAClient(av_dir, sglang_url=url, device="cpu")
print(f"  d_model={client.cfg.d_model}  injection_scale={client.cfg.injection_scale}")

# ── load data ─────────────────────────────────────────────────────────────────
vecs = np.load(INJECT_PATH)
meta = pd.read_parquet(META_PATH)
assert len(vecs) == len(meta)
n_rows = len(vecs)
print(f"\n{n_rows} activations × {N_SAMPLES} samples = {n_rows*N_SAMPLES} total decodes")

# ── resume ────────────────────────────────────────────────────────────────────
done: set[tuple[int,int]] = set()
existing: list[dict] = []
if Path(OUT_PATH).exists():
    prev = pd.read_parquet(OUT_PATH)
    for _, r in prev.iterrows():
        done.add((int(r["row"]), int(r["sample"])))
        existing.append(r.to_dict())
    print(f"Resuming: {len(done)} pairs already done")

todo = [
    (row_i, s)
    for row_i in range(n_rows)
    for s in range(N_SAMPLES)
    if (row_i, s) not in done
]
print(f"{len(todo)} decodes remaining")

if not todo:
    print("All decodes complete — nothing to do.")
    print(f"Output: {OUT_PATH}  ({len(existing)} rows)")
    sys.exit(0)

# ── decode loop ───────────────────────────────────────────────────────────────
def decode_one(row_i: int, sample: int) -> dict:
    v    = torch.tensor(vecs[row_i], dtype=torch.float32)
    text = client.generate(v, extract_explanation=False,
                            max_new_tokens=MAX_NEW_TOK)
    return {"row": row_i, "sample": sample, "nla_output": text}


results: list[dict] = list(existing)
t0 = time.time()
completed = 0
errors    = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(decode_one, r, s): (r, s) for r, s in todo}
    pbar    = tqdm(as_completed(futures), total=len(futures),
                   desc="decoding", unit="dec")
    for fut in pbar:
        row_i, sample = futures[fut]
        try:
            rec = fut.result()
        except Exception as e:
            rec = {"row": row_i, "sample": sample,
                   "nla_output": f"[ERROR: {type(e).__name__}: {e}]"}
            errors += 1
        results.append(rec)
        completed += 1

        if completed % SAVE_EVERY == 0:
            pd.DataFrame(results).to_parquet(OUT_PATH, index=False)
            elapsed = time.time() - t0
            rate    = completed / max(elapsed, 1)
            eta_min = (len(todo) - completed) / rate / 60
            pbar.set_postfix({"rate": f"{rate:.1f}/s",
                              "ETA":  f"{eta_min:.0f}m",
                              "err":  errors})

# final save
out_df = pd.DataFrame(results)
out_df.to_parquet(OUT_PATH, index=False)

elapsed = time.time() - t0
print(f"\nDone: {len(todo)} decodes in {elapsed/60:.1f} min  "
      f"({len(todo)/elapsed:.1f} dec/s)  errors={errors}")

# ── quick sanity check ────────────────────────────────────────────────────────
has_expl = out_df["nla_output"].str.contains("<explanation>").mean()
err_frac = out_df["nla_output"].str.startswith("[ERROR").mean()
print(f"<explanation> tag rate : {has_expl:.3f}  (expect ~1.0)")
print(f"error rate             : {err_frac:.3f}  (expect ~0)")
print(f"\nSaved {len(out_df)} rows → {OUT_PATH}")
print("Next: run 05_score_and_analyze.py")
