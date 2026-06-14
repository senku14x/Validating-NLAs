#!/usr/bin/env python
"""05_smoke_decode.py — Gate-2 PRE-FLIGHT go/no-go (box-only, throwaway).

Before building the full offline+online scored matrix (05/05b/06/07/08), answer
the one question that decides whether that matrix is worth building:

    When we inject a KNOWN concept direction into neutral anchors and feed the
    result to the released NLA Activation Verbalizer (AV), does the AV verbalize
    the concept at all — for anything other than refusal?

Why this gate exists: Exp 1 only ever validated *refusal* through the AV, and
Qwen showed offline injection can be NULL (the AV reads real refusal but not
offline-injected refusal). The AV is also prompted to emit "2-3 text snippets
describing that vector" (a describer, not a meta-labeller) — so whether a
non-behavioural concept like truth_value surfaces as a distinctive snippet is an
open empirical question, not a given. So we look at RAW AV outputs first, by eye,
with floors, BEFORE spending any API/scorer effort.

Pre-registered decision criterion (decide BEFORE reading the outputs):
  PROCEED to build the scored matrix iff
    (a) refusal at high dose -> AV emits explicit refusal/disclaimer language, AND
    (b) >=1 non-refusal survivor at high dose -> AV emits concept-relevant content
        that DIFFERS from the baseline_no_inject / random floors (not the same
        generic template), and is NOT CJK garbage.
  STOP and rethink iff, at the highest on-manifold dose, the non-refusal
  directions yield only:
    - CJK / a description of a CJK char  -> injection physically broke (NFKC ate
        the marker, or the gemma3_mm patch didn't apply). An INFRA bug: re-run
        `bash scripts/av_up.sh <model>` and `nla_box.py --smoke`; conclude nothing.
    - the SAME generic template as the floors regardless of injected concept
        -> the AV does not resolve these directions offline. A real NEGATIVE that
        changes the experiment (these concepts may need the online-steered arm,
        or aren't AV-verbalizable at all).
  The CJK-vs-template distinction is the entire reason we eyeball raw output
  instead of running a scorer.

Reuse (ported verbatim, sources noted): dom_dir (Exp-1 07_build_direction),
exact_cosine_inject (CAA 03_build_injection_sweep), NLAClient + resolve_av (the
box convention in nla_box.py / av_up.sh). Consumes the 03 activation cache;
writes a scratch JSONL to workspace/ and prints the decision view to stdout.

Run AFTER `03_extract_for_battery.py --model <m>` and `bash scripts/av_up.sh <m>`:
  HF_TOKEN=... NLA_REPO_DIR=/workspace/nla_repo \
    python "Experiment 2/rebuild/scripts/05_smoke_decode.py" --model gemma
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # rebuild root -> paths.py
sys.path.insert(0, str(HERE))          # scripts/    -> nla_box.py
from paths import WORKSPACE, cache_path, model_slug, stage_of  # noqa: E402
import nla_box  # noqa: E402  (reuse resolve_av — same AV-dir convention as av_up.sh)

STAGE = stage_of(__file__)

# The 6 directions Gate 1 cleared: 5 v2 survivors + the refusal anchor. Built as
# diff-of-means from the 03 per-concept npz (X, y); y==1 is present/refused.
CONCEPTS = ["refusal", "truth_value", "sycophancy", "corrigibility",
            "neg_sentiment", "harmful_topic_benign"]
EXTRACT_STAGE = "03_extract_for_battery"   # producer of the cache we read
DOSES = {"medium": 0.30, "high": 0.45}     # delta above anchor baseline cos
COS_CAP = 0.95                             # cap to avoid near-degenerate angles
SEED = 42


def dom_dir(X, y):
    """Diff-of-means unit direction (Arditi; Exp-1 07_build_direction::dom_dir)."""
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    return v / np.linalg.norm(v)


def exact_cosine_inject(H, v_hat, target_cos):
    """h' = h + beta*v_hat solved so cos(h', v_hat) == target_cos exactly.

    Verbatim from CAA 03_build_injection_sweep::exact_cosine_inject.
    Returns (Hp, beta, realized). H is [n, d] float64, v_hat unit float64.
    """
    a = H @ v_hat                                              # [n]
    perp = np.sqrt(np.maximum((H ** 2).sum(1) - a ** 2, 0.0))  # [n]
    t = float(target_cos)
    beta = perp * (t / np.sqrt(max(1 - t ** 2, 1e-9))) - a     # [n]
    Hp = H + beta[:, None] * v_hat[None, :]
    norms = np.linalg.norm(Hp, axis=1, keepdims=True)
    realized = (Hp @ v_hat) / norms.squeeze()
    return Hp, beta, realized


def cjk_count(s: str) -> int:
    """Degenerate-injection tell (lifted from nla_box.smoke)."""
    return sum(1 for c in s if "一" <= c <= "鿿" or "가" <= c <= "힯"
               or "　" <= c <= "㏿")


def load_direction(model_slug_: str, concept: str):
    """DoM direction for one concept from its 03 npz, or None if unavailable."""
    p = cache_path(EXTRACT_STAGE, model_slug_, concept=concept, ext="npz")
    if not p.exists():
        print(f"  [skip] {concept}: no cache at {p.name} (run 03 for this concept)")
        return None
    d = np.load(p)
    X, y = d["X"].astype(np.float64), d["y"].astype(int)
    if len(np.unique(y)) < 2:
        print(f"  [skip] {concept}: only one class present in y")
        return None
    return dom_dir(X, y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", choices=["gemma", "qwen"])
    ap.add_argument("--n-anchors", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    a = ap.parse_args()
    model = model_slug(a.model)
    rng = np.random.default_rng(SEED)

    # --- anchors (the Gate-2 injection targets, cached by 03) ---
    ap_path = cache_path(EXTRACT_STAGE, model, concept="anchor", ext="npz")
    if not ap_path.exists():
        sys.exit(f"FAIL: no anchor cache at {ap_path}\n"
                 f"      run 03_extract_for_battery.py --model {a.model} on the box first.")
    H = np.load(ap_path)["X"].astype(np.float64)[: a.n_anchors]
    d_model = H.shape[1]
    print(f"{len(H)} anchors, d={d_model}, model={model}")

    # --- directions: 6 DoM + a random-vector floor ---
    dirs: dict[str, np.ndarray] = {}
    for c in CONCEPTS:
        v = load_direction(model, c)
        if v is not None:
            dirs[c] = v
    v_rand = rng.standard_normal(d_model)
    v_rand /= np.linalg.norm(v_rand)
    dirs["random"] = v_rand
    if "refusal" in dirs:
        print(f"cos(random, refusal) = {float(v_rand @ dirs['refusal']):+.3f} (expect ~0)")
    if not any(c in dirs for c in CONCEPTS if c != "refusal"):
        print("WARNING: no non-refusal directions available — smoke cannot test the "
              "load-bearing case. Run 03 for the survivor concepts first.")

    # --- AV client (identical convention to nla_box.smoke / av_up.sh) ---
    nla_repo = os.environ.get("NLA_REPO_DIR", "/workspace/nla_repo")  # av_up.sh's default clone path
    if not pathlib.Path(nla_repo, "nla_inference.py").exists():
        sys.exit(f"FAIL: nla_inference.py not under NLA_REPO_DIR={nla_repo!r} — "
                 f"set NLA_REPO_DIR or run av_up.sh (it clones to /workspace/nla_repo).")
    sys.path.insert(0, nla_repo)  # cloned (and, if needed, tokenization-patched) upstream
    from nla_inference import NLAClient  # noqa: E402
    av_dir = nla_box.resolve_av(a.model, full=True)
    port = os.environ.get("SGLANG_PORT", "30000")
    print(f"NLAClient <- {av_dir}  (sglang :{port})")
    client = NLAClient(av_dir, sglang_url=f"http://localhost:{port}", device="cpu")
    assert client.cfg.d_model == d_model, (
        f"AV d_model {client.cfg.d_model} != anchor d {d_model} — wrong AV checkpoint for {model}")

    # --- build injected vectors + decode RAW (keep tags; we want to see degeneration) ---
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    out_path = WORKSPACE / f"{STAGE}__{model}.jsonl"
    rows: list[dict] = []

    def decode(hp_f32, **rec):
        raw = client.generate(hp_f32, extract_explanation=False, max_new_tokens=a.max_new_tokens)
        rec.update(cjk=cjk_count(raw), has_expl=("<explanation>" in raw), output=raw)
        rows.append(rec)
        return rec

    with open(out_path, "w") as fh:
        for concept, v in dirs.items():
            baseline = float((H @ v / np.linalg.norm(H, axis=1)).mean())
            for dose, delta in DOSES.items():
                target = min(baseline + delta, COS_CAP)
                Hp, beta, realized = exact_cosine_inject(H, v, target)
                for i in range(len(H)):
                    hp = Hp[i].astype(np.float32)
                    cos_hhp = float(H[i] @ Hp[i] / (np.linalg.norm(H[i]) * np.linalg.norm(Hp[i])))
                    decode(hp, concept=concept, dose=dose, anchor=i,
                           target_cos=round(target, 4), realized_cos=round(float(realized[i]), 4),
                           baseline_cos=round(baseline, 4), cos_h_hp=round(cos_hhp, 4),
                           norm_ratio=round(float(np.linalg.norm(Hp[i]) / np.linalg.norm(H[i])), 3))
            fh.flush()
        # baseline_no_inject floor: the raw anchor, no injection at all.
        for i in range(len(H)):
            decode(H[i].astype(np.float32), concept="baseline_no_inject", dose="none", anchor=i,
                   target_cos=0.0, realized_cos=0.0, baseline_cos=0.0, cos_h_hp=1.0, norm_ratio=1.0)
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\nwrote {len(rows)} decodes -> {out_path}\n")

    # --- eyeball: one high-dose output per direction (the decision view) ---
    print("=" * 78)
    print("HIGH-DOSE RAW AV OUTPUTS — apply the pre-registered criterion (see docstring)")
    print("=" * 78)
    for concept in list(dirs) + ["baseline_no_inject"]:
        sub = [r for r in rows if r["concept"] == concept and r["dose"] in ("high", "none")]
        if not sub:
            continue
        ex = sub[0]
        print(f"\n### {concept}  realized_cos={ex['realized_cos']}  cos(h,h')={ex['cos_h_hp']}  "
              f"cjk={ex['cjk']}  tags={ex['has_expl']}")
        print(ex["output"][:600])
    print("\n" + "=" * 78)
    print("PROCEED iff refusal refuses AND >=1 non-refusal concept differs from the "
          "baseline/random floor (not CJK, not the same template). Else STOP — see docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
