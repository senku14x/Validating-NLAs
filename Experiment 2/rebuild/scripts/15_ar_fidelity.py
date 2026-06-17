#!/usr/bin/env python
"""15_ar_fidelity.py — AR reconstruction fidelity (the NLA's OWN defining metric; external-review #1).

Every "the AV is a validated instrument" claim so far is behavioral (inject-refusal-vs-random + diagonal
confusion). This measures the thing the paper itself uses: does the loop reconstruct the activation?
  h  --AV-->  explanation text  --AR-->  h_hat        FVE = 1 - ||h-h_hat||^2 / ||h-mean(h)||^2
Reports mean cosine(h, h_hat), MSE, and FVE over a sample of REAL activations (cache/03).

INTERFACE NOTE: `NLACritic(ckpt_dir, device)` lives in the box-only vendored `nla_inference.py`; the exact
reconstruct method name is not visible from this repo. This script therefore (a) prints the NLACritic API on
load, and (b) tries the likely reconstruct calls in order, reporting which worked. If all fail it prints the
API and exits 2 — paste that and it's a one-line fix. Everything else (AV decode loop, FVE math) is solid.

Reuses: nla_box.resolve_av + NLAClient (AV, via SGLang) exactly as 09/13; cache/03 activations.
Run (box, after `bash scripts/av_up.sh <model>`):
  NLA_REPO_DIR=/workspace/nla_repo python "Experiment 2/rebuild/scripts/15_ar_fidelity.py" --model gemma
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import nla_box  # noqa: E402
from paths import RESULTS, cache_path, model_slug, stage_of  # noqa: E402

STAGE = stage_of(__file__)
EXTRACT_STAGE = "03_extract_for_battery"
AR_REPO = {"gemma3-27b": "kitft/nla-gemma3-27b-L41-ar", "qwen2.5-7b": "kitft/nla-qwen2.5-7b-L20-ar"}
SAMPLE_CONCEPTS = ["refusal", "neg_sentiment", "truth_value", "anchor"]   # a representative spread
N_PER = int(os.environ.get("N_PER", "40"))
PORT = int(os.environ.get("SGLANG_PORT", "30000"))


def _resolve_ar(model_key: str) -> str:
    if os.environ.get("NLA_AR_DIR"):
        return str(pathlib.Path(os.environ["NLA_AR_DIR"]).resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(AR_REPO[model_key], cache_dir=nla_box.cache_dir())


def _ar_reconstruct(critic, text, h, torch):
    """Try the likely NLACritic reconstruct APIs; return h_hat (np[d]) or raise with the introspected API."""
    ht = torch.tensor(h, dtype=torch.float32)
    attempts = [
        ("reconstruct(text)", lambda: critic.reconstruct(text)),
        ("reconstruct(text, h)", lambda: critic.reconstruct(text, ht)),
        ("encode(text)", lambda: critic.encode(text)),
        ("__call__(text)", lambda: critic(text)),
        ("fidelity(h, text)", lambda: critic.fidelity(ht, text)),
        ("score(h, text)", lambda: critic.score(ht, text)),
    ]
    for name, fn in attempts:
        try:
            out = fn()
        except Exception:
            continue
        # out could be a vector (h_hat), or a dict/obj with mse/cos — normalize to h_hat where possible
        if hasattr(out, "detach"):
            v = out.detach().float().cpu().numpy().reshape(-1)
            if v.shape[0] == h.shape[0]:
                return name, v, None
        if isinstance(out, (tuple, list)) and len(out) and hasattr(out[0], "detach"):
            v = out[0].detach().float().cpu().numpy().reshape(-1)
            if v.shape[0] == h.shape[0]:
                return name, v, None
        if isinstance(out, dict) and ("mse" in out or "cos" in out or "cosine" in out):
            return name, None, {k: float(out[k]) for k in out if k in ("mse", "cos", "cosine")}
    raise RuntimeError("no known NLACritic reconstruct API matched")


def run(model_key: str) -> int:
    import httpx, torch

    nla_repo = os.environ.get("NLA_REPO_DIR", "/workspace/nla_repo")
    if not pathlib.Path(nla_repo, "nla_inference.py").exists():
        sys.exit(f"FAIL: nla_inference.py not under NLA_REPO_DIR={nla_repo!r} — run av_up.sh.")
    sys.path.insert(0, nla_repo)
    from nla_inference import NLAClient, NLACritic  # noqa: E402

    url = f"http://localhost:{PORT}"
    try:
        assert httpx.get(url + "/health", timeout=5).status_code == 200
    except Exception:
        sys.exit(f"SGLang AV not reachable at {url} — run: bash scripts/av_up.sh {model_key.split('-')[0]}")

    # sample real activations from the 03 cache
    H = []
    for c in SAMPLE_CONCEPTS:
        p = cache_path(EXTRACT_STAGE, model_key, concept=c, ext="npz")
        if not p.exists():
            print(f"  [skip] no 03 cache for {c}"); continue
        X = np.load(p)["X"].astype(np.float32)
        idx = np.random.default_rng(0).permutation(len(X))[:N_PER]
        H.extend(X[i] for i in idx)
    if len(H) < 10:
        sys.exit("FAIL: <10 sample activations — run 03 first.")
    H = np.stack(H)
    print(f"{len(H)} real activations sampled (d={H.shape[1]})")

    av = NLAClient(nla_box.resolve_av(model_key.split("-")[0], full=True), sglang_url=url, device="cpu")
    critic = NLACritic(_resolve_ar(model_key), device="cuda:0")
    print("NLACritic API:", [a for a in dir(critic) if not a.startswith("_")])

    coss, mses, used_api = [], [], None
    direct = None  # if the critic returns mse/cos directly
    for i, h in enumerate(H):
        text = av.generate(torch.tensor(h, dtype=torch.float32), extract_explanation=False)
        try:
            name, h_hat, metrics = _ar_reconstruct(critic, text, h, torch)
        except RuntimeError as e:
            print(f"\n*** {e}. NLACritic API was: {[a for a in dir(critic) if not a.startswith('_')]}")
            print("*** Paste this line and I'll wire the exact reconstruct call.")
            return 2
        used_api = name
        if metrics is not None:
            direct = direct or []; direct.append(metrics); continue
        cos = float(h @ h_hat / (np.linalg.norm(h) * np.linalg.norm(h_hat) + 1e-9))
        mse = float(np.mean((h - h_hat) ** 2))
        coss.append(cos); mses.append(mse)

    out = dict(model=model_key, n=len(H), ar_api=used_api)
    if direct:
        out["critic_direct_metrics_mean"] = {k: round(float(np.mean([d[k] for d in direct if k in d])), 5)
                                             for k in {kk for d in direct for kk in d}}
    if coss:
        H_var = float(np.mean(np.var(H, axis=0)))
        fve = 1.0 - float(np.mean(mses)) / (H_var + 1e-9)
        out.update(mean_cosine=round(float(np.mean(coss)), 4), mean_mse=round(float(np.mean(mses)), 5),
                   FVE=round(fve, 4))
    rp = RESULTS / "gate4"; rp.mkdir(parents=True, exist_ok=True)
    (rp / f"{STAGE}__{model_key}.json").write_text(json.dumps(out, indent=2))
    print(f"\n==== AR FIDELITY ({model_key}) ====  api={used_api}")
    print(json.dumps(out, indent=2))
    print(f"\nRead: mean_cosine near 1.0 and FVE near 1.0 = the AV is a faithful reconstructor; low = the loop"
          f" loses most of the activation (the paper's own weak-verifier caveat). wrote {rp / (STAGE+'__'+model_key+'.json')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", help="gemma | qwen")
    a = ap.parse_args()
    return run(model_slug(a.model))


if __name__ == "__main__":
    raise SystemExit(main())
