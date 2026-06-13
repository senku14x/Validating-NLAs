#!/usr/bin/env python
"""
01_collect_activations.py — collect residual-stream activations for Exp 3.

ONE forward pass, TWO reductions (both tokenizer-agnostic — no cross-model
token alignment is ever required):

  --reduction mean      mean-pool over real tokens -> [N, d]. FITS THE MAP (§5).
                        This is the immediate need; it gates Gate 0.

  --reduction pertoken  keep every real token -> ragged chunked cache. Used for
                        CO-FIRING (§7), the next phase. Stores RAW hidden states;
                        the Qwen SAE .encode() that defines A_f/N_f runs later on
                        this cache (cheap, CPU), so Qwen is never forwarded twice.

Design choices that matter (and differ from Exp 1):
  * FORWARD-ONLY. We need activations, not generations. Faster than generate().
  * NO CHAT TEMPLATE. The SAE features and the map live in the plain-text
    residual stream. Exp 1 chat-templated because it elicited refusals — a
    different goal; copying it here would put activations in the wrong regime.
  * Mean-pool spans ALL real tokens (spec §5 says no token-band filtering for
    the map). Any SAE position handling is applied LATER, only when defining
    A_f/N_f — not here.
  * Gemma stored fp32 (outlier dims). bf16 forward, fp32 on disk.
  * Resumable: mean-pool skips done ids; pertoken skips done chunks.

Usage:
  HF_TOKEN=... python 01_collect_activations.py --model gemma --reduction mean \
      --corpus pile --limit 5000 --out-dir /content/exp3_acts
  HF_TOKEN=... python 01_collect_activations.py --model qwen  --reduction mean \
      --corpus pile --limit 5000 --out-dir /content/exp3_acts
(use the SAME corpus + SAME order for both models so rows pair by index.)
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

# -- path: allow running from any directory (Colab: upload exp3_config.py to same dir) --
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import exp3_config as C

TOKEN = os.environ.get("HF_TOKEN")


# ── pure reductions (mirrored in the numpy self-test) ────────────────────────
def masked_mean(hs, mask):
    """hs [B,T,d], mask [B,T] -> [B,d] mean over real tokens (mask-weighted)."""
    m = mask.unsqueeze(-1).to(hs.dtype)
    return (hs * m).sum(1) / m.sum(1).clamp(min=1)


def strip_padding(hs_row, mask_row):
    """hs_row [T,d], mask_row [T] -> [n_real, d]. Robust to padding side."""
    return hs_row[mask_row.bool()]


# ── corpus ───────────────────────────────────────────────────────────────────
def materialize_corpus(spec, out_dir, limit, min_chars=60, max_chars=2000):
    """Select the corpus ONCE, TOKENIZER-AGNOSTICALLY, cache to parquet, and
    reuse it across both model runs — otherwise the Qwen and Gemma selections
    diverge and the map-fit rows no longer pair by index.

    Filtering is by character length (language-agnostic; a token-count filter
    would use each model's own tokenizer and reintroduce the divergence). spec
    is a path (.txt / .parquet with 'text') or a keyword in {'pile','wikitext'}.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    key = spec if spec in ("pile", "wikitext") else Path(spec).stem
    cache = out_dir / f"corpus_{key}_{limit}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"corpus (cached, shared across models): {len(df)} <- {cache.name}")
        return df["id"].tolist(), df["text"].tolist()

    if spec in ("pile", "wikitext"):
        from datasets import load_dataset
        if spec == "pile":  # SAE was trained on pile -> matches feature dist.
            ds = load_dataset("monology/pile-uncopyrighted", split="train",
                              streaming=True)
        else:
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                              split="train", streaming=True)
        raw = (r["text"] for r in ds)
    elif spec.endswith(".parquet"):
        raw = iter(pd.read_parquet(spec)["text"].tolist())
    else:
        raw = (ln for ln in open(spec, encoding="utf-8"))

    ids, texts = [], []
    for t in raw:
        t = (t or "").strip()
        if not (min_chars <= len(t) <= max_chars):
            continue
        ids.append(f"s{len(ids):06d}"); texts.append(t)
        if len(texts) >= limit:
            break
    df = pd.DataFrame({"id": ids, "text": texts})
    df.to_parquet(cache, index=False)
    print(f"corpus materialized ({len(df)} sentences, {min_chars}-{max_chars} "
          f"chars) -> {cache.name}  (both models will read this exact file)")
    return ids, texts


def load_model(repo):
    tok = AutoTokenizer.from_pretrained(repo, token=TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo, token=TOKEN, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    cfg = model.config
    nl = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    return model, tok, nl


# ── collection ───────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_mean(model, tok, ids, texts, hs_idx, nl, out_path, bs, max_length):
    """Mean-pooled [N,d] fp32, resumable. Single .npz {acts, ids}."""
    done = set()
    if out_path.exists():
        prev = np.load(out_path, allow_pickle=True)
        rows = list(prev["acts"]); done_ids = list(prev["ids"])
        done = set(done_ids)
        print(f"resuming mean: {len(done)} done")
    else:
        rows, done_ids = [], []

    todo = [(i, t) for i, t in zip(ids, texts) if i not in done]
    for k in tqdm(range(0, len(todo), bs), desc="mean-pool"):
        chunk = todo[k:k + bs]
        cids = [c[0] for c in chunk]
        enc = tok([c[1] for c in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        assert len(hs) == nl + 1, f"hidden_states len {len(hs)} != {nl+1}"
        pooled = masked_mean(hs[hs_idx].float(), enc.attention_mask).cpu().numpy()
        rows.extend(pooled.astype(np.float32)); done_ids.extend(cids)
        if (k // bs) % 20 == 0:
            np.savez(out_path, acts=np.stack(rows).astype(np.float32),
                     ids=np.array(done_ids))
    np.savez(out_path, acts=np.stack(rows).astype(np.float32), ids=np.array(done_ids))
    A = np.load(out_path, allow_pickle=True)["acts"]
    print(f"saved {out_path}  acts={A.shape}  "
          f"||row||: mean={np.linalg.norm(A,axis=1).mean():.0f}")


@torch.no_grad()
def collect_pertoken(model, tok, ids, texts, hs_idx, nl, out_dir, bs,
                     max_length, chunk_sents=500):
    """Ragged per-token cache, fp32, chunked + resumable.
    Each chunk file: chunk_{c}.npz {flat [sum_tok,d], lengths [n], ids [n]}.
    A manifest.json records chunk -> ids for downstream indexing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(zip(ids, texts))
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for c in tqdm(range(0, len(pairs), chunk_sents), desc="pertoken-chunks"):
        cpath = out_dir / f"chunk_{c:07d}.npz"
        if cpath.exists():
            continue
        cpairs = pairs[c:c + chunk_sents]
        flat, lengths, cids = [], [], []
        for k in range(0, len(cpairs), bs):
            sub = cpairs[k:k + bs]
            enc = tok([s[1] for s in sub], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(model.device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            assert len(hs) == nl + 1, f"hidden_states len {len(hs)} != {nl+1}"
            layer = hs[hs_idx].float()
            for i in range(len(sub)):
                real = strip_padding(layer[i], enc.attention_mask[i]).cpu().numpy()
                flat.append(real.astype(np.float32))
                lengths.append(real.shape[0]); cids.append(sub[i][0])
        np.savez(cpath, flat=np.concatenate(flat).astype(np.float32),
                 lengths=np.array(lengths, np.int32), ids=np.array(cids))
        manifest[cpath.name] = cids
        manifest_path.write_text(json.dumps(manifest))
    print(f"pertoken cache -> {out_dir}  ({len(manifest)} chunks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "gemma"], required=True)
    ap.add_argument("--reduction", choices=["mean", "pertoken"], required=True)
    ap.add_argument("--corpus", required=True,
                    help="path (.txt/.parquet) or keyword pile|wikitext")
    ap.add_argument("--out-dir", default="/content/exp3_acts")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    spec = C.MODELS[args.model]
    model, tok, nl = load_model(spec["repo"])
    assert nl == spec["num_layers"], (
        f"{args.model}: expected {spec['num_layers']} layers, got {nl}")
    print(f"{args.model}: num_hidden_layers={nl}, extracting "
          f"hidden_states[{spec['hs_idx']}], d_model={spec['d_model']}")

    ids, texts = materialize_corpus(args.corpus, args.out_dir, args.limit)
    out = Path(args.out_dir)
    if args.reduction == "mean":
        out.mkdir(parents=True, exist_ok=True)
        collect_mean(model, tok, ids, texts, spec["hs_idx"], nl,
                     out / f"mean_{args.model}.npz", args.batch_size, args.max_length)
    else:
        collect_pertoken(model, tok, ids, texts, spec["hs_idx"], nl,
                         out / f"pertoken_{args.model}", args.batch_size, args.max_length)

    del model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
