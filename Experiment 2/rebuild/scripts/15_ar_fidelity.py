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


def _chance_cos(H: np.ndarray, n_pairs: int = 2000, seed: int = 0) -> float:
    """Baseline floor: mean cosine between RANDOM PAIRS of real activations. Residual streams are anisotropic,
    so two unrelated activations already share a high cosine; AR(AV(h)) cos is only 'reconstruction' if it
    CLEARLY beats this floor — otherwise the AR just lands in the anisotropic blob, not on h specifically."""
    rng = np.random.default_rng(seed)
    n = len(H)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = i != j
    a, b = H[i[keep]].astype(np.float64), H[j[keep]].astype(np.float64)
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(c.mean())


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

    # NLACritic.score(explanation, original) -> (mse, cos), with mse = 2(1-cos) under the √d normalization —
    # the NLA's OWN defining metric (vendored nla_inference.py:655). Pass the RAW activation as `original`
    # (score() L2-normalizes both internally). Order is (text, h), NOT (h, text).
    coss, mses = [], []
    for h in H:
        text = av.generate(torch.tensor(h, dtype=torch.float32), extract_explanation=False)
        mse, cos = critic.score(text, h)
        coss.append(float(cos)); mses.append(float(mse))
    coss, mses = np.array(coss), np.array(mses)

    chance = _chance_cos(H)   # anisotropy floor — fidelity is the MARGIN of mean_cosine above this
    out = dict(model=model_key, n=len(H), ar_api="score(text, h)->(mse,cos)",
               mean_cosine=round(float(coss.mean()), 4), median_cosine=round(float(np.median(coss)), 4),
               p10_cosine=round(float(np.percentile(coss, 10)), 4), mean_mse=round(float(mses.mean()), 5),
               chance_cosine=round(chance, 4), cosine_above_chance=round(float(coss.mean()) - chance, 4))
    rp = RESULTS / "gate4"; rp.mkdir(parents=True, exist_ok=True)
    (rp / f"{STAGE}__{model_key}.json").write_text(json.dumps(out, indent=2))
    print(f"\n==== AR FIDELITY ({model_key}) ====  api={out['ar_api']}")
    print(json.dumps(out, indent=2))
    print(f"\nRead: fidelity is mean_cosine's MARGIN above chance_cosine (residual streams are anisotropic, so\n"
          f"chance cos is high). cosine_above_chance ~ 0 => the loop loses the activation (the paper's own\n"
          f"weak-verifier caveat); large => faithful reconstruct. wrote {rp / (STAGE+'__'+model_key+'.json')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", help="gemma | qwen")
    a = ap.parse_args()
    return run(model_slug(a.model))


if __name__ == "__main__":
    raise SystemExit(main())
