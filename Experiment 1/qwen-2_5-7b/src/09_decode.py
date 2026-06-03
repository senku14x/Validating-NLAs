#!/usr/bin/env python
"""
run launch_sglang_server.sh instead
09_decode.py - decode all injected Qwen activations through the NLA AV.

This is the local-GPU version of the final decode step. Earlier steps in this
folder extract Qwen-2.5-7B-Instruct layer-20 residual activations
(`hidden_states[21]`) and write them under `workspace/`; this script feeds those
3584-d raw vectors to the released Qwen NLA activation verbalizer.

Prerequisites:
  1. Run the previous pipeline steps through `src/08_inject.py`.
  2. Install the NLA server extra if needed:
       uv sync --extra nla-server
  3. Download/verify the Qwen AV checkpoint:
       python src/01_verify_env.py
     This script will also call snapshot_download for the full AV if needed.
  4. Launch SGLang on your local GPU with the SAME AV dir printed below:
       python -m sglang.launch_server \
         --model-path "$NLA_AV_DIR" \
         --port 30000 \
         --disable-radix-cache \
         --context-length 512 \
         --mem-fraction-static 0.85 \
         --trust-remote-code

`--disable-radix-cache` is required for input_embeds requests: cache entries are
keyed by token IDs, but the NLA actor receives embeddings with the activation
already injected.

Environment overrides:
  NLA_AV_DIR     Existing local path to kitft/nla-qwen2.5-7b-L20-av.
  NLA_CACHE_DIR  HF cache dir for snapshot_download (default: ./cache).
  NLA_REPO_DIR   Path containing nla_inference.py (default: ./nla-inference).
  SGLANG_URL     Server URL (default: http://localhost:30000).
  WORKERS        Concurrent decode requests (default: 8).

Output:
  workspace/decode_outputs.parquet  [row, nla_output]
  Merge with workspace/injection_meta.parquet on `row` for analysis.
"""
from __future__ import annotations

import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
CACHE_DIR = Path(os.environ.get("NLA_CACHE_DIR", ROOT / "cache"))
SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30000").rstrip("/")
WORKERS = int(os.environ.get("WORKERS", "8"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "200"))

INJECTED = WORKSPACE / "injected.npy"
META = WORKSPACE / "injection_meta.parquet"
OUT = WORKSPACE / "decode_outputs.parquet"


def resolve_nla_repo() -> Path:
    """Directory containing kitft's local inference helper."""
    env = os.environ.get("NLA_REPO_DIR")
    candidates = [
        Path(env) if env else None,
        ROOT / "nla-inference",
        ROOT.parent.parent / "nla-inference",
        Path("/content/nla_repo"),
    ]
    for path in candidates:
        if path and (path / "nla_inference.py").is_file():
            return path
    raise FileNotFoundError(
        "Could not find nla_inference.py. Set NLA_REPO_DIR=/path/to/nla-inference."
    )


def resolve_av_dir() -> Path:
    """Return the local HF snapshot for the Qwen-2.5-7B layer-20 AV."""
    env = os.environ.get("NLA_AV_DIR")
    if env:
        av_dir = Path(env).expanduser().resolve()
        if not (av_dir / "nla_meta.yaml").is_file():
            raise FileNotFoundError(f"NLA_AV_DIR is missing nla_meta.yaml: {av_dir}")
        return av_dir

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    return Path(snapshot_download(AV_REPO, cache_dir=str(CACHE_DIR), token=token))


NLA_REPO_DIR = resolve_nla_repo()
sys.path.insert(0, str(NLA_REPO_DIR))
from nla_inference import NLAClient  # noqa: E402


for path in (INJECTED, META):
    if not path.exists():
        raise FileNotFoundError(f"missing {path} - run earlier pipeline steps first")

AV_DIR = resolve_av_dir()
print(f"base model : {BASE_MODEL}")
print(f"AV repo    : {AV_REPO}")
print(f"AV dir     : {AV_DIR}")
print(f"SGLang URL : {SGLANG_URL}")

# ---- server health check ---------------------------------------------------
try:
    resp = httpx.get(f"{SGLANG_URL}/health", timeout=5)
    resp.raise_for_status()
    print("SGLang server: healthy")
except Exception as exc:
    print(f"SGLang server not responding: {exc}")
    print("\nLaunch the local GPU server in another terminal, using the AV dir above:")
    av_quoted = shlex.quote(str(AV_DIR))
    print("  python -m sglang.launch_server \\")
    print(f"    --model-path {av_quoted} \\")
    print("    --port 30000 \\")
    print("    --disable-radix-cache \\")
    print("    --context-length 512 \\")
    print("    --mem-fraction-static 0.85 \\")
    print("    --trust-remote-code")
    raise SystemExit(1) from exc

# ---- load data -------------------------------------------------------------
vecs = np.load(INJECTED)  # [2800, 3584] float32 for Qwen-2.5-7B-L20
meta = pd.read_parquet(META)
assert len(vecs) == len(meta), "row mismatch between injected.npy and injection_meta"

# ---- NLAClient (shared read-only across threads) ---------------------------
print("loading NLAClient (Qwen embedding table on CPU)...")
client = NLAClient(AV_DIR, sglang_url=SGLANG_URL, device="cpu")
print()

assert vecs.shape[1] == client.cfg.d_model, (
    f"injected activation dim {vecs.shape[1]} != Qwen AV d_model {client.cfg.d_model}. "
    "Regenerate activations/injections with the Qwen-2.5-7B layer-20 pipeline."
)

# ---- resume ----------------------------------------------------------------
done_rows: set[int] = set()
results: list[dict[str, object]] = []
if OUT.exists():
    done_df = pd.read_parquet(OUT)
    done_rows = set(done_df["row"].tolist())
    results = done_df.to_dict("records")
    print(f"resuming: {len(done_rows)}/{len(vecs)} already decoded")

todo = [r for r in range(len(vecs)) if r not in done_rows]
if not todo:
    print("all rows already decoded")
    raise SystemExit
print(f"{len(todo)} rows to decode at {WORKERS} workers")


# ---- decode ----------------------------------------------------------------
def decode_one(row_idx: int) -> tuple[int, str | None, str | None]:
    try:
        v = torch.tensor(vecs[row_idx], dtype=torch.float32)
        # Keep raw output with <explanation> tags for maximum flexibility at scoring time.
        text = client.generate(v, extract_explanation=False)
        return row_idx, text, None
    except Exception as exc:
        return row_idx, None, str(exc)


def save_ckpt() -> None:
    pd.DataFrame(results).sort_values("row").to_parquet(OUT, index=False)


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
print(f"\ndone: {len(results)} total rows, {errors} errors, {elapsed / 60:.1f} min")
print(f"throughput: {len(todo) / elapsed:.2f} req/s")

# ---- content check ---------------------------------------------------------
df = pd.read_parquet(OUT).sort_values("row").reset_index(drop=True)
tag_rate = df["nla_output"].str.contains("<explanation>", na=False).mean()
print(f"\nrows with <explanation> tags: {tag_rate:.0%}  (expect ~100%)")
if tag_rate < 0.95:
    print("WARNING: many missing tags - check the SGLang log and Qwen AV sidecar/tokenizer.")

# one sample per condition at a mid-dose level
full = meta.merge(df, on="row").sort_values(["condition", "target_cos"])
print("\n--- sample outputs (one per condition, target_cos=0.5 or nearest) ---")
for cond in ["C1_refusal_pos", "C2_refusal_neg", "C3_random", "C4_nullDoM"]:
    sub = full[full.condition == cond]
    if sub.empty:
        print(f"\n[{cond}] missing from injection metadata")
        continue
    row = sub.iloc[(sub["target_cos"] - 0.5).abs().argmin()]
    out = row["nla_output"][:250]
    print(f"\n[{cond}  t={row['target_cos']:.1f}]\n  {out!r}")