#!/usr/bin/env python
"""01_verify_env.py — Gate 0 implementation sanity (the cheap checks that silently
break everything downstream if wrong). Ported from Experiment 1's 01_verify_env.

Sections, each prints PASS / SKIP / FAIL. GPU/HF/NLA sections SKIP gracefully off
the box (so you can run the CPU checks — cosine solver + dataset sanity — here),
and run for real on the Vast.ai box.

  1. GPU            A100-class, compute >= 8.0, VRAM headroom
  2. HF access      auth + the model's base/AV/AR repos reachable
  3. NLA sidecar    load_nla_config round-trips U+321C; prints injection_scale etc.
  4. cosine solver  exact-cosine injection math, max error == 0  (CPU, always runs)
  5. dataset        concept_pairs.parquet present + well-formed   (CPU, always runs)

Run:  .venv/bin/python "Experiment 2/rebuild/scripts/01_verify_env.py" --model gemma
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # reach root libs
from paths import data_path, model_slug  # noqa: E402

MODELS = {
    "gemma3-27b": dict(base="google/gemma-3-27b-it", av="kitft/nla-gemma3-27b-L41-av",
                       ar="kitft/nla-gemma3-27b-L41-ar", layer=42, n_layers=62, d=5376),
    "qwen2.5-7b": dict(base="Qwen/Qwen2.5-7B-Instruct", av="kitft/nla-qwen2.5-7b-L20-av",
                       ar="kitft/nla-qwen2.5-7b-L20-ar", layer=21, n_layers=28, d=3584),
}


def section(msg):
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


def check_gpu():
    section("1. GPU")
    try:
        import torch
    except Exception as e:
        print(f"SKIP: torch not importable here ({type(e).__name__}). Box-only check.")
        return
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device visible (running off-box).")
        return
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc = torch.cuda.get_device_capability(0)
    print(f"device {name} | VRAM {gb:.1f} GB | compute {cc[0]}.{cc[1]}")
    if gb < 75:
        print("WARNING: <75 GB — Gemma-27B (~54 GB) + SGLang AV cannot coexist (memory dance).")
    if cc[0] < 8:
        print("WARNING: compute < 8.0 — fa3 attention backend unavailable.")
    print("PASS")


def check_hf(m):
    section("2. Hugging Face auth + repo access")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("SKIP: HF_TOKEN not set (box-only check).")
        return
    try:
        from huggingface_hub import HfApi, login
    except Exception as e:
        print(f"SKIP: huggingface_hub not installed ({type(e).__name__}).")
        return
    login(token=token, add_to_git_credential=False)
    api = HfApi()
    for repo in (m["base"], m["av"], m["ar"]):
        try:
            info = api.model_info(repo, token=token)
            print(f"OK   {repo:42s} gated={getattr(info, 'gated', None)}")
        except Exception as e:
            print(f"FAIL {repo:42s} {type(e).__name__}: {e}")
            raise
    print("PASS")


def check_nla(m):
    section("3. NLA sidecar + injection-char round-trip")
    repo_dir = os.environ.get("NLA_REPO_DIR")
    av_dir = os.environ.get("NLA_AV_DIR")
    if not (repo_dir and av_dir):
        print("SKIP: set NLA_REPO_DIR (vendored nla_inference) and NLA_AV_DIR (AV checkpoint) on the box.")
        return
    try:
        sys.path.insert(0, repo_dir)
        from nla_inference import load_nla_config
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"SKIP: cannot import nla_inference/transformers ({type(e).__name__}: {e}).")
        return
    tok = AutoTokenizer.from_pretrained(av_dir)
    cfg = load_nla_config(av_dir, tok)
    print(f"d_model={cfg.d_model}  injection_char={cfg.injection_char!r}  "
          f"injection_token_id={cfg.injection_token_id}")
    print(f"injection_scale={cfg.injection_scale}  embed_scale(sqrt d)={cfg.d_model ** 0.5:.4f}")
    assert cfg.d_model == m["d"], f"d_model {cfg.d_model} != expected {m['d']}"
    print("PASS — sidecar parsed, injection char round-trips, d_model matches.")


def check_cosine_solver():
    section("4. exact-cosine injection solver (CPU)")
    rng = np.random.default_rng(0)
    d = 256
    H = rng.standard_normal((200, d)) * rng.uniform(5, 50, (200, 1))  # varied norms
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    max_err = 0.0
    for t in (-0.3, 0.0, 0.3, 0.5, 0.7, 0.9):
        a = H @ v
        hp = np.sqrt(np.maximum((H ** 2).sum(1) - a ** 2, 0.0))
        beta = hp * (t / np.sqrt(1 - t ** 2)) - a
        Hp = H + beta[:, None] * v[None, :]
        realized = (Hp @ v) / np.linalg.norm(Hp, axis=1)
        max_err = max(max_err, float(np.abs(realized - t).max()))
    print(f"max |realized - target| over 6 cosines x 200 anchors = {max_err:.2e}")
    assert max_err < 1e-9, f"cosine solver inexact: {max_err}"
    print("PASS — solver lands on target cosine to float precision.")


def check_dataset():
    section("5. concept_pairs.parquet sanity (CPU)")
    p = data_path("concept_pairs.parquet", mkdir=False)
    if not p.exists():
        print(f"SKIP: {p} not built yet — run 02_build_concept_pairs.py first.")
        return
    import pandas as pd
    df = pd.read_parquet(p)
    n_concepts = df.concept.nunique()
    empties = int((df.text.str.len() == 0).sum())
    print(f"rows={len(df)}  concepts={n_concepts}  empty_texts={empties}")
    assert empties == 0, "found empty texts"
    assert "refusal" in set(df.concept), "refusal missing from dataset"
    print("PASS — dataset present, non-empty, refusal included.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", help="gemma | qwen")
    a = ap.parse_args()
    m = MODELS[model_slug(a.model)]
    print(f"verifying for {a.model} -> {model_slug(a.model)} "
          f"(layer hidden_states[{m['layer']}], d={m['d']})")
    check_gpu()
    check_hf(m)
    check_nla(m)
    check_cosine_solver()
    check_dataset()
    section("DONE")
    print("CPU checks (4,5) gate the data pipeline; GPU/HF/NLA (1-3) gate extraction + decoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
