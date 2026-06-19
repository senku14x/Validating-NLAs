#!/usr/bin/env python
"""
02b_token_position_map_fit.py

Fit Qwen->Gemma ridge map on sampled token-position activations instead of
mean-pooled sentence activations.

Motivation:
  Existing map fits sentence means.
  Co-firing evaluates max-over-token projections.
  This map is closer to the downstream token-level transfer test.

Saves:
  {out_dir}/ridge_map.npz
  {out_dir}/gate0_tokenpos_results.json
  {out_dir}/token_qwen.npz
  {out_dir}/token_gemma.npz
"""

import argparse, gc, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp3_config as C

TOKEN = os.environ.get("HF_TOKEN")


def parse_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def materialize_corpus(spec, out_dir, limit, min_chars=80, max_chars=2000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = spec if spec in ("pile", "wikitext") else Path(spec).stem
    cache = out_dir / f"corpus_{key}_{limit}.txt"

    if cache.exists():
        texts = [x.strip() for x in cache.read_text(encoding="utf-8").splitlines() if x.strip()]
        print(f"loaded cached corpus: {len(texts)} <- {cache}")
        return texts[:limit]

    if spec == "pile":
        from datasets import load_dataset
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        raw_iter = (r["text"] for r in ds)
    elif spec == "wikitext":
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        raw_iter = (r["text"] for r in ds)
    elif spec.endswith(".txt"):
        raw_iter = open(spec, encoding="utf-8")
    else:
        raise ValueError(f"unsupported corpus: {spec}")

    texts = []
    for t in raw_iter:
        t = (t or "").replace("\n", " ").strip()
        if min_chars <= len(t) <= max_chars:
            texts.append(t)
        if len(texts) >= limit:
            break

    cache.write_text("\n".join(texts), encoding="utf-8")
    print(f"materialized corpus: {len(texts)} -> {cache}")
    return texts


def load_model(model_key):
    spec = C.MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(spec["repo"], token=TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"],
        token=TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    cfg = model.config
    nl = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    assert nl == spec["num_layers"], f"{model_key}: expected {spec['num_layers']} layers, got {nl}"
    print(f"{model_key}: hs_idx={spec['hs_idx']} d_model={spec['d_model']}")
    return model, tok, spec


def choose_positions(length, ratios):
    if length <= 0:
        return []
    idxs = []
    for r in ratios:
        r = min(max(r, 0.0), 1.0)
        idxs.append(int(round(r * (length - 1))))
    return idxs


@torch.no_grad()
def collect_tokenpos(model_key, texts, out_dir, batch_size, max_length, ratios, force=False):
    out_dir = Path(out_dir)
    out_path = out_dir / f"token_{model_key}.npz"

    if out_path.exists() and not force:
        arr = np.load(out_path, allow_pickle=True)["acts"]
        print(f"{model_key}: using cached {out_path}, acts={arr.shape}")
        return

    model, tok, spec = load_model(model_key)
    rows = []

    for start in tqdm(range(0, len(texts), batch_size), desc=f"{model_key} tokenpos"):
        batch = texts[start:start + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        hs = model(**enc, output_hidden_states=True).hidden_states[spec["hs_idx"]].float()

        for i in range(len(batch)):
            L = int(enc.attention_mask[i].sum().item())
            pos = choose_positions(L, ratios)
            if not pos:
                # Should not happen for normal text, but keep row count stable.
                pos = [0 for _ in ratios]
            rows.append(hs[i, pos].cpu().numpy().astype(np.float32))

    acts = np.concatenate(rows, axis=0).astype(np.float32)
    assert acts.shape[0] == len(texts) * len(ratios), (acts.shape, len(texts), len(ratios))
    np.savez(out_path, acts=acts)
    print(f"{model_key}: saved {out_path}, acts={acts.shape}, mean_norm={np.linalg.norm(acts, axis=1).mean():.1f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def standardize(X_tr, X_te):
    mu = X_tr.mean(0)
    sig = X_tr.std(0).clip(1e-8)
    return (X_tr - mu) / sig, (X_te - mu) / sig, mu, sig


def fit_ridge(X, Y, lam):
    D = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(D), X.T @ Y)


def row_cos(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    B = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return (A * B).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pile")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--positions", default="0.10,0.25,0.50,0.75,0.90")
    ap.add_argument("--lambdas", default="10,100,1000,10000")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratios = parse_floats(args.positions)
    lambdas = [int(x) for x in parse_floats(args.lambdas)]
    print("positions:", ratios)
    print("lambdas:", lambdas)

    texts = materialize_corpus(args.corpus, out_dir, args.limit)

    collect_tokenpos("qwen", texts, out_dir, args.batch_size, args.max_length, ratios, force=args.force)
    collect_tokenpos("gemma", texts, out_dir, args.batch_size, args.max_length, ratios, force=args.force)

    X = np.load(out_dir / "token_qwen.npz", allow_pickle=True)["acts"].astype(np.float64)
    Y = np.load(out_dir / "token_gemma.npz", allow_pickle=True)["acts"].astype(np.float64)

    assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
    assert X.shape[1] == C.QWEN_DMODEL, X.shape
    assert Y.shape[1] == C.GEMMA_DMODEL, Y.shape

    print(f"loaded token-position X={X.shape} Y={Y.shape}")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    n_te = max(1000, int(0.2 * len(X)))
    te = perm[:n_te]
    tr = perm[n_te:]

    Xtr_s, Xte_s, X_mu, X_sig = standardize(X[tr], X[te])
    Ytr_s, Yte_s, Y_mu, Y_sig = standardize(Y[tr], Y[te])

    best = None
    results = {}

    print("\n[lambda sweep]")
    for lam in lambdas:
        t0 = time.time()
        W = fit_ridge(Xtr_s, Ytr_s, lam)
        Yhat = Xte_s @ W * Y_sig + Y_mu
        cos = row_cos(Yhat, Y[te])
        mean_cos = float(cos.mean())

        var = Y.var(0)
        top_dims = np.argsort(var)[::-1][:10]
        Yh = Yhat.copy()
        Yt = Y[te].copy()
        Yh[:, top_dims] = 0
        Yt[:, top_dims] = 0
        oz_cos = float(row_cos(Yh, Yt).mean())

        elapsed = time.time() - t0
        print(f"lambda={lam:6d} raw_cos={mean_cos:.4f} outlier_zeroed={oz_cos:.4f} elapsed={elapsed:.1f}s")

        results[str(lam)] = {
            "raw_cos": mean_cos,
            "outlier_zeroed_cos": oz_cos,
            "elapsed_sec": elapsed,
        }

        # Select primarily by outlier-zeroed cosine, not raw cosine.
        score = oz_cos
        if best is None or score > best[0]:
            best = (score, lam, W, mean_cos, oz_cos)

    _, best_lam, best_W, best_raw, best_oz = best

    np.savez(
        out_dir / "ridge_map.npz",
        W=best_W.astype(np.float32),
        X_mu=X_mu.astype(np.float32),
        X_sigma=X_sig.astype(np.float32),
        Y_mu=Y_mu.astype(np.float32),
        Y_sigma=Y_sig.astype(np.float32),
        lambda_=np.array(best_lam),
        map_kind=np.array("token_position_ridge"),
        positions=np.array(ratios, dtype=np.float32),
    )

    summary = {
        "map_kind": "token_position_ridge",
        "n_texts": len(texts),
        "n_rows": int(len(X)),
        "positions": ratios,
        "selected_lambda": int(best_lam),
        "selected_raw_cos": float(best_raw),
        "selected_outlier_zeroed_cos": float(best_oz),
        "all_lambdas": results,
    }
    (out_dir / "gate0_tokenpos_results.json").write_text(json.dumps(summary, indent=2))

    print("\nSAVED")
    print(" ", out_dir / "ridge_map.npz")
    print(" ", out_dir / "gate0_tokenpos_results.json")
    print(f"selected lambda={best_lam} raw_cos={best_raw:.4f} oz_cos={best_oz:.4f}")


if __name__ == "__main__":
    main()