#!/usr/bin/env python
"""
09_decode.py - decode all 2800 injected activations through the NLA AV.

Prerequisites (notebook cells, in order):
  1. Re-download AV:
       from huggingface_hub import snapshot_download
       snapshot_download("kitft/nla-gemma3-27b-L41-av", cache_dir="/content/nla_ckpt")
  2. !python setup_session.py       (re-grafts tokenizer_config.json)
  3. Launch SGLang + wait READY     (same two notebook cells as the smoke test)

Fires WORKERS concurrent requests so SGLang's continuous batcher packs them.
NLAClient.generate is safe across threads (embedding lookup is read-only CPU;
httpx.Client allows concurrent use from multiple threads).

Checkpoints every CKPT_EVERY decodes. Resume by re-running the script.

Output: /content/decode_outputs.parquet  [row, nla_output]
        Merge with /content/injection_meta.parquet on 'row' for analysis.
"""
import os
import sys
import time

import httpx
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, "/content/nla_repo")
from nla_inference import NLAClient

AV_DIR = ("/content/nla_ckpt/models--kitft--nla-gemma3-27b-L41-av/snapshots"
          "/4e721238131ffb8348cff260fe81b8b34a270a0d")
PORT       = 30000
WORKERS    = 8     # concurrent requests; tune up to 16 if GPU keeps up
CKPT_EVERY = 200
OUT        = "/content/decode_outputs.parquet"

assert os.path.exists(AV_DIR), f"AV dir missing: {AV_DIR}\nRe-run the AV download cell."

# ---- server health check ---------------------------------------------------
try:
    resp = httpx.get(f"http://localhost:{PORT}/health", timeout=5)
    assert resp.status_code == 200, f"status {resp.status_code}"
    print("SGLang server: healthy")
except Exception as e:
    print(f"SGLang server not responding: {e}")
    print("Run the launch + poll cells in the notebook first, wait for READY.")
    raise SystemExit

# ---- load data -------------------------------------------------------------
vecs = np.load("/content/injected.npy")                # [2800, 5376] float32
meta = pd.read_parquet("/content/injection_meta.parquet")
assert len(vecs) == len(meta), "row mismatch between injected.npy and injection_meta"

# ---- resume ----------------------------------------------------------------
done_rows: set[int] = set()
results: list[dict] = []
if os.path.exists(OUT):
    done_df = pd.read_parquet(OUT)
    done_rows = set(done_df["row"].tolist())
    results = done_df.to_dict("records")
    print(f"resuming: {len(done_rows)}/{len(vecs)} already decoded")

todo = [r for r in range(len(vecs)) if r not in done_rows]
if not todo:
    print("all rows already decoded"); raise SystemExit
print(f"{len(todo)} rows to decode at {WORKERS} workers")

# ---- NLAClient (shared read-only across threads) ---------------------------
print("loading NLAClient (embedding table on CPU)...")
client = NLAClient(AV_DIR, sglang_url=f"http://localhost:{PORT}", device="cpu")
print()

# ---- decode ----------------------------------------------------------------
def decode_one(row_idx: int):
    try:
        v = torch.tensor(vecs[row_idx], dtype=torch.float32)
        # extract_explanation=False: keep raw output with <explanation> tags
        # (gives maximum flexibility at scoring time)
        text = client.generate(v, extract_explanation=False)
        return row_idx, text, None
    except Exception as exc:
        return row_idx, None, str(exc)


def save_ckpt():
    pd.DataFrame(results).to_parquet(OUT, index=False)


errors = 0
n_new = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(decode_one, r): r for r in todo}
    with tqdm(total=len(todo), desc="decode") as bar:
        for future in as_completed(futures):
            row_idx, text, err = future.result()
            if err:
                errors += 1
                results.append({"row": row_idx, "nla_output": f"[ERROR] {err}"})
            else:
                results.append({"row": row_idx, "nla_output": text})
            n_new += 1
            bar.update(1)
            if n_new % CKPT_EVERY == 0:
                save_ckpt()

save_ckpt()
elapsed = time.time() - t0
print(f"\ndone: {len(results)} total rows, {errors} errors, {elapsed/60:.1f} min")
print(f"throughput: {len(todo)/elapsed:.2f} req/s")

# ---- content check ---------------------------------------------------------
df = pd.read_parquet(OUT).sort_values("row").reset_index(drop=True)
tag_rate = df["nla_output"].str.contains("<explanation>", na=False).mean()
print(f"\nrows with <explanation> tags: {tag_rate:.0%}  (expect ~100%)")
if tag_rate < 0.95:
    print("WARNING: many missing tags - check SGLang log for errors")

# one sample per condition at a mid-dose level
full = meta.merge(df, on="row").sort_values(["condition", "target_cos"])
print("\n--- sample outputs (one per condition, target_cos=0.5 or nearest) ---")
for cond in ["C1_refusal_pos", "C2_refusal_neg", "C3_random", "C4_nullDoM"]:
    sub = full[full.condition == cond]
    row = sub.iloc[(sub["target_cos"] - 0.5).abs().argmin()]
    out = row["nla_output"][:130]
    print(f"\n[{cond}  t={row['target_cos']:.1f}]\n  {out!r}")
