#!/usr/bin/env python
"""18_calibration_falsify.py — Exp-4 Phase-0 P0-1: does AR-reconstruction cosine separate the released
NLA's FAITHFUL reads from its PLAUSIBLE (on-manifold) CONFABULATIONS?

WHY (the disguised output-predictor test; ships as the RQ5 calibration result either way):
  A cheap proposed "trust wrapper" gates each AV decode by AR round-trip fidelity cos(h, AR(AV(h))) and
  ABSTAINS when low. It only works if faithful reads reconstruct *better* than confabulations. The red-team's
  attack: AR-cos measures round-trip reconstruction, NOT factual correctness — a fluent, on-manifold
  confabulation ("Paris is the capital of the US") may reconstruct just as well as a faithful read. If so the
  wrapper is doomed for the confabulations that matter (it would only catch degenerate/off-manifold reads,
  which the existing nla_degenerate / generic_template flags already catch). This stage FALSIFIES that:
  pre-registered GO iff AUROC(faithful vs confab | cos) >= 0.70 on a REGION-CONTROLLED contrast.

WHAT IT REUSES (no SGLang AV server, no re-decoding, no training):
  - committed decode TEXT + judge labels from results/gate2/07_score_matrix__<model>__all__real.parquet
  - activations from cache/03 (gitignored; re-extract with 03 on the box): h = cache03[concept]["X"][src_idx]
  - AR only: NLACritic.score(text, h) -> (mse, cos)   (vendored nla_inference.py; in-process PyTorch)
  - confounds._safe_auc / _cluster_bootstrap_ci for the separation AUROC + cluster-bootstrap CI

THE CONTRASTS (built CPU-side from committed data; --build-only prints n per pool):
  faithful_refusal : concept=refusal, polarity=present, j_refusal==2          (validated judge -> clean label)
  confab_truth     : concept=truth_value, polarity=present, j_truth_value<2   (the "Paris is US capital" misses)
  faithful_truth   : concept=truth_value, polarity=present, j_truth_value==2   (within-truth control; label noisy)
  -> WITHIN-TRUTH (faithful_truth vs confab_truth) is the DECISIVE, region-controlled contrast: if cos only
     separates CROSS-concept (refusal vs truth) but NOT within truth, then cos tracks REGION, not faithfulness,
     and the wrapper is doomed. (Persona-evil within-concept is unusable: echo-exclusion leaves 0 faithful.)

OUTPUT (box): results/gate4/18_calibration__<model>.json (AUROCs + CIs + verdict) and
  18_calibration__<model>.csv (per-row label/cos/local_floor/margin/flags). NO new harmful text is written
  (texts are already committed in 07_*; we store only metrics + a truncated expl).

RUN:
  CPU (here):  python scripts/18_calibration_falsify.py --selftest        # validate the math
               python scripts/18_calibration_falsify.py --build-only --model gemma   # n per pool, no acts
  BOX:         NLA_AR_DIR=... python scripts/18_calibration_falsify.py --model gemma  # full (needs cache/03 + AR)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # reach root libs
from confounds import _cluster_bootstrap_ci, _safe_auc  # noqa: E402
from paths import RESULTS, cache_path, model_slug, stage_of  # noqa: E402

STAGE = stage_of(__file__)
EXTRACT_STAGE = "03_extract_for_battery"
AR_REPO = {"gemma3-27b": "kitft/nla-gemma3-27b-L41-ar", "qwen2.5-7b": "kitft/nla-qwen2.5-7b-L20-ar"}
GO_BAR = 0.70  # pre-registered: wrapper is viable only if the region-controlled AUROC clears this


# ─────────────────────────── pure analysis math (CPU; self-tested) ───────────────────────────
def local_floor(H: np.ndarray, concept_ids: np.ndarray, max_ref: int = 400, seed: int = 0) -> np.ndarray:
    """Per-row LOCAL anisotropy floor = mean cosine of h_i to OTHER activations of the SAME concept.

    Residual streams are anisotropic and the anisotropy is region-dependent, so a single global chance-cos
    (15_ar_fidelity) over-credits high-anisotropy regions. The margin = recon_cos - local_floor asks whether
    the reconstruction lands on h SPECIFICALLY, beyond its own neighborhood's baseline similarity.
    """
    H = np.asarray(H, dtype=np.float64)
    concept_ids = np.asarray(concept_ids)
    n = len(H)
    norms = np.linalg.norm(H, axis=1) + 1e-12
    out = np.full(n, np.nan)
    rng = np.random.default_rng(seed)
    for c in np.unique(concept_ids):
        idx = np.where(concept_ids == c)[0]
        if len(idx) < 2:
            out[idx] = 0.0
            continue
        ref = idx if len(idx) <= max_ref else rng.choice(idx, max_ref, replace=False)
        Href, nref = H[ref], norms[ref]
        for i in idx:
            cos = (Href @ H[i]) / (nref * norms[i])
            s, k = cos.sum(), len(ref)
            if i in set(ref.tolist()):          # exclude self (cos=1) from its own mean
                s, k = s - 1.0, k - 1
            out[i] = s / max(k, 1)
    return out


def separation_auroc(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                     n_boot: int = 2000, seed: int = 42) -> dict:
    """AUROC of `scores` separating faithful(1) from confab(0), with a cluster-bootstrap CI over `groups`
    (the activation id — multiple decode-samples of one activation are correlated). Reuses confounds.*"""
    scores, labels, groups = map(np.asarray, (scores, labels, groups))
    m = ~np.isnan(scores)
    scores, labels, groups = scores[m], labels[m].astype(int), groups[m]
    if len(np.unique(labels)) < 2:
        return dict(auroc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                    n=int(len(labels)), n_pos=int(labels.sum()), n_neg=int((labels == 0).sum()))
    auc = _safe_auc(labels, scores)
    lo, hi = _cluster_bootstrap_ci(labels, scores, groups, n_boot, 0.05, seed)
    return dict(auroc=round(float(auc), 4), ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                n=int(len(labels)), n_pos=int(labels.sum()), n_neg=int((labels == 0).sum()))


def verdict(within_truth: dict, cross_concept: dict, bar: float = GO_BAR) -> tuple[str, str]:
    """Pre-registered call. The WITHIN-concept (region-controlled) AUROC is decisive.
       GO         within-truth AUROC point >= bar AND ci_lo > 0.5            -> wrapper viable; ship
       REGION     within-truth ~0.5 but cross-concept high                  -> cos tracks region, not
                                                                               faithfulness -> wrapper doomed
                                                                               for plausible confab (NULL result)
       NULL       neither separates                                         -> AR-cos can't rescue confab
       UNDERPOWERED  within-truth ci spans bar and 0.5 (small n)            -> inconclusive; need more confab n
    """
    wt, xc = within_truth.get("auroc"), cross_concept.get("auroc")
    wlo, whi = within_truth.get("ci_lo"), within_truth.get("ci_hi")
    if wt is None or (isinstance(wt, float) and np.isnan(wt)):
        return "INSUFFICIENT", "within-truth contrast has <2 classes (need both faithful & confab truth reads)"
    if wt >= bar and wlo is not None and wlo > 0.5:
        return "GO", f"within-truth AUROC {wt:.2f} [{wlo:.2f},{whi:.2f}] >= {bar}, ci_lo>0.5 -> wrapper viable"
    if (wlo is not None and wlo <= 0.5 <= (whi or 1.0)) and wt < bar:
        if xc is not None and not np.isnan(xc) and xc >= bar:
            return "REGION-CONFOUNDED", (f"within-truth {wt:.2f} ~chance but cross-concept {xc:.2f}>= {bar} "
                                         f"-> cos tracks REGION not faithfulness -> wrapper doomed for plausible confab")
        return "UNDERPOWERED" if (whi or 0) > bar else "NULL", (
            f"within-truth {wt:.2f} [{wlo:.2f},{whi:.2f}] does not clear {bar}")
    return "NULL", f"within-truth AUROC {wt:.2f} < {bar} -> AR-cos cannot rescue plausible confabulation"


# ─────────────────────────── label-set assembly (CPU; from committed data) ───────────────────────────
def build_labelset(model_key: str) -> pd.DataFrame:
    """From the committed Gate-3 real parquet, assemble labeled decodes (text + label + pool). No activations.
       label: 1=faithful, 0=confab.  pool: 'within_truth' (region-controlled) | 'xconcept' (clean labels)."""
    p = RESULTS / "gate2" / f"07_score_matrix__{model_key}__all__real.parquet"
    df = pd.read_parquet(p)
    df = df[(~df.nla_degenerate.astype(bool)) & (~df.generic_template.astype(bool))]
    text_col = "expl" if "expl" in df.columns else "nla_output"
    rows = []

    def add(sub, label, pool, key_concept):
        for _, r in sub.iterrows():
            rows.append(dict(concept=r["concept"], polarity=r["polarity"], src_idx=int(r["src_idx"]),
                             sample=int(r.get("sample", 0)), label=int(label), pool=pool,
                             key_concept=key_concept,
                             expl=str(r.get(text_col, ""))[:240]))

    # within-truth (region-controlled): faithful vs confab among truth_value-present reads
    tp = df[(df.concept == "truth_value") & (df.polarity == "present")]
    add(tp[tp.j_truth_value == 2], 1, "within_truth", "truth_value")
    add(tp[tp.j_truth_value < 2], 0, "within_truth", "truth_value")
    # cross-concept (clean labels): faithful refusal vs the same truth confabs
    rp = df[(df.concept == "refusal") & (df.polarity == "present") & (~df.echo.astype(bool))]
    add(rp[rp.j_refusal == 2], 1, "xconcept", "refusal")
    add(tp[tp.j_truth_value < 2], 0, "xconcept", "truth_value")
    out = pd.DataFrame(rows)
    out["act_id"] = out["key_concept"] + ":" + out["src_idx"].astype(str)  # group id for the bootstrap
    return out


# ─────────────────────────── box: AR-score + analyze ───────────────────────────
def _resolve_ar(model_key: str) -> str:
    if os.environ.get("NLA_AR_DIR"):
        return str(pathlib.Path(os.environ["NLA_AR_DIR"]).resolve())
    from huggingface_hub import snapshot_download
    import nla_box  # type: ignore
    return snapshot_download(AR_REPO[model_key], cache_dir=nla_box.cache_dir())


def run(model_key: str) -> int:
    ls = build_labelset(model_key)
    if ls.empty:
        sys.exit("FAIL: empty labelset — is the committed real parquet present?")
    # load activations from cache/03 by (key_concept, src_idx); key_concept names the cache npz (refusal/truth_value)
    Hmap: dict[str, np.ndarray] = {}
    for c in ls.key_concept.unique():
        cp = cache_path(EXTRACT_STAGE, model_key, concept=c, ext="npz")
        if not cp.exists():
            sys.exit(f"FAIL: no 03 cache for {c} at {cp} — run 03 on the box first.")
        Hmap[c] = np.load(cp)["X"].astype(np.float32)
    # AR only — no SGLang
    nla_repo = os.environ.get("NLA_REPO_DIR", "/workspace/nla_repo")
    sys.path.insert(0, nla_repo)
    from nla_inference import NLACritic  # noqa: E402
    critic = NLACritic(_resolve_ar(model_key), device="cuda:0")
    print("NLACritic API:", [a for a in dir(critic) if not a.startswith("_")])

    recon_cos, H_used = [], []
    for _, r in ls.iterrows():
        h = Hmap[r.key_concept][r.src_idx]
        _, cos = critic.score(r.expl, h)
        recon_cos.append(float(cos)); H_used.append(h)
    ls = ls.assign(recon_cos=recon_cos)
    floor = local_floor(np.stack(H_used), ls.key_concept.to_numpy())
    ls = ls.assign(local_floor=floor, margin=ls.recon_cos.to_numpy() - floor)

    res = {"model": model_key, "go_bar": GO_BAR, "n_total": int(len(ls))}
    for pool in ("within_truth", "xconcept"):
        sub = ls[ls.pool == pool]
        res[pool] = {
            "n": int(len(sub)), "n_faithful": int((sub.label == 1).sum()), "n_confab": int((sub.label == 0).sum()),
            "auroc_recon_cos": separation_auroc(sub.recon_cos, sub.label, sub.act_id),
            "auroc_margin": separation_auroc(sub.margin, sub.label, sub.act_id),
            "mean_cos_faithful": round(float(sub[sub.label == 1].recon_cos.mean()), 4),
            "mean_cos_confab": round(float(sub[sub.label == 0].recon_cos.mean()), 4),
        }
    v, why = verdict(res["within_truth"]["auroc_margin"], res["xconcept"]["auroc_margin"])
    res["verdict"], res["verdict_reason"] = v, why
    rp = RESULTS / "gate4"; rp.mkdir(parents=True, exist_ok=True)
    (rp / f"{STAGE}__{model_key}.json").write_text(json.dumps(res, indent=2))
    ls.drop(columns=["expl"]).to_csv(rp / f"{STAGE}__{model_key}.csv", index=False)
    print(json.dumps(res, indent=2))
    print(f"\nVERDICT [{model_key}]: {v} — {why}")
    print("Read: GO=>wrapper ships (RQ5); REGION-CONFOUNDED/NULL=>AR-cos can't rescue plausible confabulation "
          "(also RQ5, and an early warning the organism may not be readable).")
    return 0


# ─────────────────────────── CPU self-test ───────────────────────────
def selftest() -> int:
    rng = np.random.default_rng(0)
    d = 64
    # (1) clean: faithful reconstruct high, confab low (both on-manifold) -> margin AUROC ~1
    base = rng.normal(size=(1, d))                       # shared anisotropy direction
    H = base + 0.3 * rng.normal(size=(120, d))           # one "concept" region (anisotropic)
    cid = np.zeros(120, dtype=int)
    lab = np.r_[np.ones(60), np.zeros(60)].astype(int)
    cos = np.r_[rng.normal(0.95, 0.02, 60), rng.normal(0.78, 0.02, 60)]
    fl = local_floor(H, cid)
    assert np.all(fl > 0.5), f"anisotropic region floor should be high, got mean {fl.mean():.2f}"
    a = separation_auroc(cos, lab, np.arange(120))
    assert a["auroc"] > 0.95, f"faithful>>confab should give AUROC~1, got {a}"
    am = separation_auroc(cos - fl, lab, np.arange(120))
    assert am["auroc"] > 0.95, f"margin should also separate, got {am}"
    # (2) null: faithful and confab reconstruct equally -> AUROC ~0.5
    cos0 = rng.normal(0.85, 0.02, 120)
    a0 = separation_auroc(cos0, lab, np.arange(120))
    assert 0.35 < a0["auroc"] < 0.65, f"equal recon should give AUROC~0.5, got {a0}"
    # (3) verdict logic
    hi = dict(auroc=0.82, ci_lo=0.71, ci_hi=0.93)
    lo = dict(auroc=0.50, ci_lo=0.38, ci_hi=0.62)
    assert verdict(hi, hi)[0] == "GO"
    assert verdict(lo, hi)[0] == "REGION-CONFOUNDED"
    assert verdict(lo, lo)[0] in ("NULL", "UNDERPOWERED")
    assert verdict(dict(auroc=float("nan")), hi)[0] == "INSUFFICIENT"
    # (4) local_floor excludes self and is per-concept
    H2 = np.array([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]], dtype=float)
    f2 = local_floor(H2, np.array([0, 0, 1, 1]))
    assert np.allclose(f2, [1, 1, 1, 1], atol=1e-6), f"within-concept identical dirs -> floor 1, got {f2}"
    print("ALL CHECKS PASSED — 18_calibration_falsify: margin/AUROC/verdict/local_floor correct on synthetic ground truth.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", help="gemma | qwen")
    ap.add_argument("--selftest", action="store_true", help="CPU: validate the analysis math")
    ap.add_argument("--build-only", action="store_true", help="CPU: assemble labelset from committed data, print n per pool")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    mk = model_slug(a.model)
    if a.build_only:
        ls = build_labelset(mk)
        print(f"labelset [{mk}]  n={len(ls)}")
        print(ls.groupby(["pool", "label"]).size().rename("n").reset_index().to_string(index=False))
        print("  label: 1=faithful 0=confab | pool: within_truth=region-controlled, xconcept=clean-labels")
        out = RESULTS / "gate4" / f"{STAGE}__{mk}__labelset.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        ls.to_csv(out, index=False)
        print(f"wrote {out}")
        return 0
    return run(mk)


if __name__ == "__main__":
    raise SystemExit(main())
