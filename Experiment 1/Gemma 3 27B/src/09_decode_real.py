#!/usr/bin/env python
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

PORT = 30000
WORKERS = 8
CKPT_EVERY = 100

VECS_PATH = "/content/real_eval_activations.npy"
META_PATH = "/content/real_eval_meta.parquet"
OUT = "/content/real_decode_outputs.parquet"

assert os.path.exists(AV_DIR), f"AV dir missing: {AV_DIR}"

try:
    resp = httpx.get(f"http://localhost:{PORT}/health", timeout=5)
    assert resp.status_code == 200
    print("SGLang server: healthy")
except Exception as e:
    print(f"SGLang server not responding: {e}")
    raise SystemExit("Relaunch the AV SGLang server first.")

vecs = np.load(VECS_PATH)
meta = pd.read_parquet(META_PATH)
assert len(vecs) == len(meta), (len(vecs), len(meta))
assert vecs.shape[1] == 5376, vecs.shape

done_rows = set()
results = []
if os.path.exists(OUT):
    done_df = pd.read_parquet(OUT)
    done_rows = set(done_df["row"].tolist())
    results = done_df.to_dict("records")
    print(f"resuming: {len(done_rows)}/{len(vecs)} already decoded")

todo = [r for r in range(len(vecs)) if r not in done_rows]
if not todo:
    print("all rows already decoded")
    raise SystemExit

print(f"{len(todo)} rows to decode at {WORKERS} workers")
print("loading NLAClient...")
client = NLAClient(AV_DIR, sglang_url=f"http://localhost:{PORT}", device="cpu")

def decode_one(row_idx):
    try:
        v = torch.tensor(vecs[row_idx], dtype=torch.float32)
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
    with tqdm(total=len(todo), desc="real_decode") as bar:
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

df = pd.read_parquet(OUT).sort_values("row").reset_index(drop=True)
tag_rate = df["nla_output"].str.contains("<explanation>", na=False).mean()
print(f"rows with <explanation> tags: {tag_rate:.0%}")

full = meta.merge(df, on="row")
for cond in full.condition.unique():
    print("\n" + "="*80)
    print(cond)
    for _, r in full[full.condition == cond].head(3).iterrows():
        print("prompt:", str(r.prompt)[:160].replace("\n", " "))
        print("model output:", str(r.output)[:180].replace("\n", " "))
        print("NLA:", str(r.nla_output)[:300].replace("\n", " "))
        print()
