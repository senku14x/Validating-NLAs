#!/usr/bin/env python
"""05_inject_matrix.py — Gate-2 OFFLINE injection matrix (CPU; authored + run here,
or on the box where the 03 cache lives).

For each concept direction (unit diff-of-means, built OUTSIDE the NLA) and each
dose, rotate every neutral anchor to an EXACT realized cosine with the direction
(h' = h + beta*v_hat) and emit the injected vectors + full per-row diagnostics
for the decode stage (06).

Offline injection is the RQ2 *specificity* arm (off-manifold, cheap) — NOT the
gap. The pre-flight smoke established the load-bearing nuance: positive-baseline
concepts inject cleanly on-manifold (cos(h,h')~0.85 at dose 0.7), but
negative-baseline concepts (truth_value, harmful_topic_benign) go badly
off-manifold (cos(h,h') 0.45 / 0.18) because reaching a positive cosine needs a
big push. So their offline decodes are flagged `off_manifold` and discounted;
they get a fair test via the ONLINE arm (05b_steer_extract). Every row logs
cos(h,h') so this is visible, never assumed.

Doses are ABSOLUTE realized-cos targets (spec §3) so every concept is actually
"present" regardless of its (often negative) anchor baseline.

Consumes (03 cache, box): anchor.npz (X) + per-concept npz (X, y -> DoM).
Produces (via paths.py):
  cache/05_inject_matrix__<model>__all.npy                 float32 [N, d] injected vectors
  cache/05_inject_matrix__<model>__all.parquet            one row per vector (metadata)
  cache/05_inject_matrix__<model>__all__directions.npz    the unit directions (for 05b / 06)

Run (box, after 03):   python scripts/05_inject_matrix.py --model gemma
Self-test (here):      python scripts/05_inject_matrix.py --self-test
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # rebuild root -> paths.py, injection.py
from injection import dom_dir, exact_cosine_inject, inject_stats  # noqa: E402
from paths import cache_path, model_slug, stage_of  # noqa: E402

STAGE = stage_of(__file__)
EXTRACT_STAGE = "03_extract_for_battery"   # producer of the cache we read

# Directions to inject. CORE = the 5 Gate-1 v2 survivors + the refusal anchor (its
# DoM is causally valid per Exp 1 even though its *probe* was a corpus confound).
CORE = ["refusal", "truth_value", "sycophancy", "corrigibility", "neg_sentiment", "harmful_topic_benign"]
# EXPLORATORY = the NLA-paper headline concept; culled at Gate 1 (framing-vs-state,
# BoW 1.0) so it carries that caveat — injected to characterize, not to certify.
EXPLORATORY = ["eval_framing_matched"]

DOSES = {"low": 0.40, "med": 0.55, "high": 0.70}  # absolute realized-cos targets (spec §3)
COS_CAP = 0.95
OFF_MANIFOLD_MIN = 0.50    # cos(h,h') below this -> off_manifold flag (discount in 08)
N_ANCHORS = 30
SEED = 42


def build_matrix(directions: dict, anchors, doses=DOSES, *, v_refusal=None, seed=SEED):
    """Pure core (no I/O): {name: unit dir}, anchors [n,d] -> (vecs [M,d] f32, rows [dict]).

    Adds a `random` unit-direction control (same doses) and a `baseline_no_inject`
    (beta=0) floor, so the specificity floor is built in. Unit-testable on synthetic
    data via --self-test.
    """
    rng = np.random.default_rng(seed)
    H = np.asarray(anchors, dtype=float)
    d = H.shape[1]
    dirs = dict(directions)
    v_rand = rng.standard_normal(d); v_rand /= np.linalg.norm(v_rand)
    dirs["random"] = v_rand
    if v_refusal is None:
        v_refusal = directions.get("refusal")

    def cos_ref(M):
        if v_refusal is None:
            return np.full(len(M), np.nan)
        return (M @ v_refusal) / np.linalg.norm(M, axis=1)

    vecs, rows = [], []
    for name, v in dirs.items():
        for dose, target_abs in doses.items():
            target = min(float(target_abs), COS_CAP)
            Hp, beta, _ = exact_cosine_inject(H, v, target)
            st = inject_stats(H, Hp, v)
            cref = cos_ref(Hp)
            for i in range(len(H)):
                rows.append(dict(
                    row=len(rows), concept=name, dose=dose, anchor=int(i), mode="offline",
                    target_cos=round(target, 4),
                    realized_cos=round(float(st["realized_cos"][i]), 4),
                    baseline_cos=round(float(st["baseline_cos"][i]), 4),
                    cos_h_hp=round(float(st["cos_h_hp"][i]), 4),
                    delta_norm_over_h=round(float(st["delta_norm_over_h"][i]), 4),
                    beta=round(float(beta[i]), 4),
                    cos_with_refusal=round(float(cref[i]), 4),
                    off_manifold=bool(st["cos_h_hp"][i] < OFF_MANIFOLD_MIN),
                ))
                vecs.append(Hp[i].astype(np.float32))
    # baseline_no_inject floor: the raw anchors, no injection at all.
    cref0 = cos_ref(H)
    for i in range(len(H)):
        rows.append(dict(
            row=len(rows), concept="baseline_no_inject", dose="none", anchor=int(i), mode="offline",
            target_cos=0.0, realized_cos=round(float(cref0[i]), 4) if v_refusal is not None else float("nan"),
            baseline_cos=0.0, cos_h_hp=1.0, delta_norm_over_h=0.0, beta=0.0,
            cos_with_refusal=round(float(cref0[i]), 4), off_manifold=False,
        ))
        vecs.append(H[i].astype(np.float32))
    return np.stack(vecs).astype(np.float32), rows


def load_directions(model: str) -> dict:
    dirs = {}
    for c in CORE + EXPLORATORY:
        p = cache_path(EXTRACT_STAGE, model, concept=c, ext="npz")
        if not p.exists():
            print(f"  [skip] {c}: no 03 cache at {p.name}")
            continue
        z = np.load(p)
        if len(np.unique(z["y"])) < 2:
            print(f"  [skip] {c}: single-class y")
            continue
        dirs[c] = dom_dir(z["X"], z["y"])
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", choices=["gemma", "qwen"])
    ap.add_argument("--n-anchors", type=int, default=N_ANCHORS)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()

    import pandas as pd
    model = model_slug(a.model)
    ap_path = cache_path(EXTRACT_STAGE, model, concept="anchor", ext="npz")
    if not ap_path.exists():
        sys.exit(f"FAIL: no anchor cache {ap_path} — run 03_extract_for_battery on the box first.")
    anchors = np.load(ap_path)["X"].astype(float)[: a.n_anchors]
    dirs = load_directions(model)
    if "refusal" not in dirs:
        print("  WARNING: no refusal direction — cos_with_refusal will be NaN")
    print(f"{len(anchors)} anchors, d={anchors.shape[1]}, directions: {sorted(dirs)}")

    vecs, rows = build_matrix(dirs, anchors)
    npy = cache_path(STAGE, model, concept="all", ext="npy")
    pq = cache_path(STAGE, model, concept="all", ext="parquet")
    dz = cache_path(STAGE, model, concept="all", variant="directions", ext="npz")
    np.save(npy, vecs)
    pd.DataFrame(rows).to_parquet(pq, index=False)
    np.savez(dz, **{k: v.astype(np.float32) for k, v in dirs.items()})

    n_off = sum(r["off_manifold"] for r in rows)
    print(f"wrote {len(vecs)} injected vectors -> {npy.name}")
    print(f"      {len(rows)} metadata rows -> {pq.name}  ({n_off} off-manifold flagged)")
    print(f"      {len(dirs)} directions    -> {dz.name}")

    hi = collections.defaultdict(list)
    for r in rows:
        if r["dose"] == "high":
            hi[r["concept"]].append(r["cos_h_hp"])
    print("\nper-concept cos(h,h') at high dose (the off-manifold tax):")
    for c in sorted(hi):
        m = float(np.mean(hi[c]))
        print(f"  {c:22s} mean={m:.3f}  ({'OFF-MANIFOLD' if m < OFF_MANIFOLD_MIN else 'ok'})")
    return 0


def _self_test() -> int:
    import pandas as pd
    rng = np.random.default_rng(0)
    d = 64
    dirs = {n: (lambda u: u / np.linalg.norm(u))(rng.standard_normal(d))
            for n in ["refusal", "truth_value", "x"]}
    anchors = rng.standard_normal((12, d))
    vecs, rows = build_matrix(dirs, anchors)
    df = pd.DataFrame(rows)

    need = {"row", "concept", "dose", "anchor", "mode", "target_cos", "realized_cos",
            "cos_h_hp", "delta_norm_over_h", "beta", "cos_with_refusal", "off_manifold"}
    assert need <= set(df.columns), f"missing cols: {need - set(df.columns)}"
    assert len(vecs) == len(df), "vec/row count mismatch"
    # every injected (non-baseline) row hits its target realized cosine
    inj = df[df.concept != "baseline_no_inject"]
    assert (np.abs(inj.realized_cos - inj.target_cos) < 1e-2).all(), "solver missed a target"
    # directions + random + baseline all present; baseline is on-manifold
    assert {"refusal", "truth_value", "x", "random", "baseline_no_inject"} <= set(df.concept)
    assert df[df.concept == "baseline_no_inject"].cos_h_hp.eq(1.0).all()
    # row math: (3 dirs + random) x 3 doses x 12 anchors + 12 baseline
    assert len(df) == 4 * 3 * 12 + 12, f"unexpected row count {len(df)}"
    # off_manifold is boolean and consistent with the threshold
    assert df.off_manifold.dtype == bool
    assert (df.off_manifold == (df.cos_h_hp < OFF_MANIFOLD_MIN)).all()
    print(f"ALL CHECKS PASSED — 05_inject_matrix core: {len(df)} rows, schema + exact-cosine "
          f"targets + off_manifold flag consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
