#!/usr/bin/env python
"""16_glp_qwen_cache.py — Qwen-only GLP activation-cache producer.

Writes the memmap format consumed by ../generative_latent_prior/glp_train.py:
  cache/glp/qwen2_5_7b-layer20-static/
    dtype.txt
    data_indices.npy
    data_0000.npy ...
    rep_statistics.pt

This is a producer for the Qwen GLP, not the experiment's concept-vector cache.
Use `--source fineweb` for the production path (streams HuggingFaceFW/fineweb and
caches until `--target-vectors` is reached; default 10M). `--source concept_pairs`
is only a small smoke path that proves the plumbing.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REBUILD_ROOT = HERE.parent
EXPERIMENT_2 = REBUILD_ROOT.parent
GLP_ROOT = EXPERIMENT_2 / "generative_latent_prior"
sys.path.insert(0, str(REBUILD_ROOT))
sys.path.insert(0, str(GLP_ROOT))

from paths import data_path  # noqa: E402
from qwen_glp import QWEN_GLP, qwen_glp_dataset_dir, validate_qwen_model_shape  # noqa: E402
from glp.utils_acts import MemmapWriter  # noqa: E402

DEFAULT_FINEWEB_CONFIG = "sample-10BT"
DEFAULT_TARGET_VECTORS = 10_000_000


def _read_text_file(path: pathlib.Path, limit: int | None) -> list[str]:
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _read_concept_pairs(limit: int | None) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(data_path("concept_pairs.parquet", mkdir=False))
    rows = df.text.dropna().astype(str).tolist()
    return rows[:limit] if limit else rows


def _iter_fineweb(fineweb_config: str) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "FAIL: `datasets` is required for --source fineweb "
            "(pip install datasets huggingface_hub)"
        ) from exc

    print(
        f"streaming HuggingFaceFW/fineweb config={fineweb_config!r} "
        "(HF hub caches parquet shards on first read)"
    )
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name=fineweb_config,
        split="train",
        streaming=True,
    )
    for row in ds:
        text = (row.get("text") or "").strip()
        if text:
            yield text


def _iter_texts(args) -> Iterator[str]:
    if args.source == "text_file":
        if args.input_texts is None:
            raise SystemExit("FAIL: --input-texts is required with --source text_file")
        for text in _read_text_file(pathlib.Path(args.input_texts), args.limit):
            yield text
        return
    if args.source == "concept_pairs":
        for text in _read_concept_pairs(args.limit):
            yield text
        return
    if args.source == "fineweb":
        n_docs = 0
        for text in _iter_fineweb(args.fineweb_config):
            yield text
            n_docs += 1
            if args.limit is not None and n_docs >= args.limit:
                return
        return
    raise ValueError(args.source)


def _render_batch(tokenizer, texts: list[str], *, chat_template: bool) -> list[str]:
    if not chat_template:
        return texts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for text in texts
    ]


def _write_activations(
    writer: MemmapWriter,
    acts: np.ndarray,
    *,
    n_vecs: int,
    target_vectors: int | None,
    sum_vec: np.ndarray,
    sum_sq: np.ndarray,
) -> tuple[int, bool]:
    """Append activation rows; return (new_n_vecs, reached_target)."""
    if acts.ndim != 2 or acts.shape[1] != QWEN_GLP.hidden_size:
        raise AssertionError(f"expected [n,{QWEN_GLP.hidden_size}] activations, got {acts.shape}")

    remaining = None if target_vectors is None else max(target_vectors - n_vecs, 0)
    if remaining == 0:
        return n_vecs, True

    if remaining is not None and acts.shape[0] > remaining:
        acts = acts[:remaining]

    for vec in acts:
        writer.write(vec)
    n_new = int(acts.shape[0])
    n_vecs += n_new
    sum_vec += acts.astype(np.float64).sum(axis=0)
    sum_sq += np.square(acts.astype(np.float64)).sum(axis=0)
    reached = target_vectors is not None and n_vecs >= target_vectors
    return n_vecs, reached


def _encode_batch(model, tok, batch_text: list[str], *, max_length: int) -> dict[str, Any]:
    enc = tok(
        batch_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(model.device) for k, v in enc.items()}


def _batch_activations(model, enc, *, all_tokens: bool) -> np.ndarray:
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states[QWEN_GLP.hidden_state_index].float()
    if all_tokens:
        mask = enc["attention_mask"].bool()
        acts = hs[mask].detach().cpu().numpy().astype(np.float32)
    else:
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        acts = hs[torch.arange(hs.shape[0], device=hs.device), last_idx].detach().cpu().numpy().astype(np.float32)
    return acts


@torch.no_grad()
def build_cache(args) -> int:
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else qwen_glp_dataset_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dtype.txt").write_text("np.float32\n")

    print(f"loading {QWEN_GLP.hf_name} ...")
    tok = AutoTokenizer.from_pretrained(QWEN_GLP.hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_GLP.hf_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    validate_qwen_model_shape(model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    writer = MemmapWriter(
        output_dir=output_dir,
        file_size=max(args.file_vectors, 1) * QWEN_GLP.hidden_size,
        dtype=np.dtype("float32"),
    )
    n_vecs = 0
    sum_vec = np.zeros(QWEN_GLP.hidden_size, dtype=np.float64)
    sum_sq = np.zeros(QWEN_GLP.hidden_size, dtype=np.float64)

    pending: list[str] = []
    text_iter = _iter_texts(args)
    pbar = tqdm(desc="qwen glp cache", unit="vec")
    reached_target = False

    for text in text_iter:
        pending.append(text)
        if len(pending) < args.batch_size:
            continue

        batch_text = _render_batch(tok, pending, chat_template=args.chat_template)
        enc = _encode_batch(model, tok, batch_text, max_length=args.max_length)
        acts = _batch_activations(model, enc, all_tokens=args.all_tokens)
        prev_vecs = n_vecs
        n_vecs, reached_target = _write_activations(
            writer,
            acts,
            n_vecs=n_vecs,
            target_vectors=args.target_vectors,
            sum_vec=sum_vec,
            sum_sq=sum_sq,
        )
        pbar.update(n_vecs - prev_vecs)
        pbar.set_postfix(n_vecs=n_vecs)
        pending = []
        if reached_target:
            break

    if pending and not reached_target:
        batch_text = _render_batch(tok, pending, chat_template=args.chat_template)
        enc = _encode_batch(model, tok, batch_text, max_length=args.max_length)
        acts = _batch_activations(model, enc, all_tokens=args.all_tokens)
        prev_vecs = n_vecs
        n_vecs, reached_target = _write_activations(
            writer,
            acts,
            n_vecs=n_vecs,
            target_vectors=args.target_vectors,
            sum_vec=sum_vec,
            sum_sq=sum_sq,
        )
        pbar.update(n_vecs - prev_vecs)
        pbar.set_postfix(n_vecs=n_vecs)

    pbar.close()

    if n_vecs == 0:
        raise SystemExit("FAIL: no activations cached")

    if args.target_vectors is not None and n_vecs < args.target_vectors:
        print(
            f"WARNING: cached {n_vecs} activations, below target {args.target_vectors} "
            "(FineWeb stream exhausted before target was reached)"
        )

    writer.flush()
    mean = sum_vec / max(n_vecs, 1)
    var = np.maximum(sum_sq / max(n_vecs, 1) - mean ** 2, 1e-12)
    torch.save(
        {
            "mean": torch.tensor(mean, dtype=torch.float32)[None, None, :],
            "var": torch.tensor(var, dtype=torch.float32)[None, None, :],
        },
        output_dir / "rep_statistics.pt",
    )
    print(f"wrote {n_vecs} Qwen block-{QWEN_GLP.hook_layer} activations -> {output_dir}")
    print(f"      rep_statistics.pt mean/var shape: [1, 1, {QWEN_GLP.hidden_size}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=["text_file", "concept_pairs", "fineweb"],
        default="fineweb",
        help="text corpus source (fineweb streams HuggingFaceFW/fineweb)",
    )
    ap.add_argument("--input-texts", help="newline-delimited text file; required for --source text_file")
    ap.add_argument(
        "--fineweb-config",
        default=DEFAULT_FINEWEB_CONFIG,
        help=f"FineWeb dataset config name (default {DEFAULT_FINEWEB_CONFIG})",
    )
    ap.add_argument(
        "--target-vectors",
        type=int,
        default=None,
        help=f"stop after this many activation vectors (default {DEFAULT_TARGET_VECTORS:,} for fineweb)",
    )
    ap.add_argument("--output-dir", help="defaults to rebuild/cache/glp/qwen2_5_7b-layer20-static")
    ap.add_argument("--limit", type=int, default=None, help="max documents (fineweb/text_file/concept_pairs)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--file-vectors", type=int, default=10000, help="vectors per data_*.npy shard")
    ap.add_argument(
        "--all-tokens",
        action="store_true",
        help="cache all non-padding token activations instead of last tokens",
    )
    ap.add_argument("--chat-template", action="store_true", help="wrap texts as single-turn Qwen chat prompts")
    args = ap.parse_args()

    if args.source == "fineweb":
        if args.target_vectors is None:
            args.target_vectors = DEFAULT_TARGET_VECTORS
        if not args.all_tokens:
            args.all_tokens = True
            print("fineweb: enabling --all-tokens (matches GLP FineWeb activation convention)")

    if args.source == "text_file" and args.input_texts is None:
        raise SystemExit("FAIL: --input-texts is required with --source text_file")

    return build_cache(args)


if __name__ == "__main__":
    raise SystemExit(main())
