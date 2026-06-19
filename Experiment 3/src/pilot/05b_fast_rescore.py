#!/usr/bin/env python
"""
05b_fast_rescore.py

Fast map-ablation scorer.

Uses existing caches from 05_cofiring.py:
  - pilot_features.json
  - negative_pool.json
  - within_source.npz
  - qwen_cache.npz
  - gemma_cache.npz

Does NOT:
  - re-forward models
  - train Gemma probes
  - bootstrap CIs
  - do final publication reporting

Does:
  - batch-score all mapped feature directions over Gemma cache once
  - batch-score target-random and mapped-random controls
  - compute transfer AUROC, wrong-feature controls, length confound
  - output quick comparison CSV/JSON

Use this for map ablations.
Run full 05 + 06 only on maps that improve this.
"""

import argparse, json, os, sys, gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp3_config as C

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "pandas"])
    from sklearn.metrics import roc_auc_score


SEED = 42
MARGIN_THRESH = 0.10
WRONG_BEAT_FRAC = 0.90
TRANSFER_CLEAN = 0.70
DECODER_VALID = 0.80   # W_dec[f] must DETECT the feature on Qwen (source-decoder valid; README §13)
MIN_N_POS = 10
MIN_N_NEG = 20

# ─────────────────────────────────────────────────────────────────────────────
# LOAD-BEARING TODO (the real methodological fix, deferred to full 05/06):
#   length_matched_negs() draws negatives by LENGTH bin from a RANDOM neg pool.
#   That is NOT domain-matched. For a broad domain/format feature (e.g. chemistry
#   synthesis-titles, legal citations), ANY in-domain Gemma direction separates
#   A_f from random text -> transfer AUROC is inflated by topic, not feature
#   identity. The honest control is SAME-DOMAIN HARD NEGATIVES: sentences where a
#   near-miss sibling feature (decoder-cosine neighbour) fires but THIS feature
#   does not. Add that as the negative pool in 05/06 before trusting any transfer
#   number; expect it to shrink the eligible set (this is the Exp-2 BoW lesson,
#   one level up). Not implemented here to avoid a half-baked control.
# ─────────────────────────────────────────────────────────────────────────────


def safe_auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def offsets_of(lengths):
    return np.concatenate([[0], np.asarray(lengths).cumsum()])


def token_sentence_ids(lengths):
    return np.repeat(np.arange(len(lengths), dtype=np.int32), lengths.astype(np.int32))


def unit_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    return (X / np.linalg.norm(X, axis=1, keepdims=True).clip(eps)).astype(np.float32)


def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    return (v / max(np.linalg.norm(v), eps)).astype(np.float32)


def load_map(map_dir):
    m = np.load(Path(map_dir) / "ridge_map.npz", allow_pickle=True)
    return (
        m["W"].astype(np.float64),
        m["X_sigma"].astype(np.float64),
        m["Y_sigma"].astype(np.float64),
    )


def map_qwen_dirs_to_gemma(D_q, W, X_sigma, Y_sigma):
    """
    D_q: [K, qwen_d]
    returns D_g: [K, gemma_d]
    """
    D_q = unit_rows(D_q)
    D_g = (D_q / X_sigma.clip(1e-12)) @ W
    D_g = D_g * Y_sigma
    return unit_rows(D_g)


def build_universe(out_dir):
    out_dir = Path(out_dir)
    feats = json.loads((out_dir / "pilot_features.json").read_text())
    neg_pool = json.loads((out_dir / "negative_pool.json").read_text())

    sent_to_id = {}
    sent_list = []

    def reg(t):
        if t not in sent_to_id:
            sent_to_id[t] = len(sent_list)
            sent_list.append(t)
        return sent_to_id[t]

    feat_keys = sorted(feats.keys(), key=int)
    feat_pos_ids = {k: [reg(t) for t in feats[k]["pos_texts"]] for k in feat_keys}
    neg_ids = [reg(t) for t in neg_pool]

    return feats, feat_keys, feat_pos_ids, neg_ids, sent_list


def length_matched_negs(pos_ids, cand_neg_ids, lengths, rng, ratio=5):
    pos_len = np.asarray(lengths)[pos_ids]
    nb = min(8, max(2, len(pos_ids) // 4))
    edges = np.histogram_bin_edges(pos_len, bins=nb)
    pos_hist, _ = np.histogram(pos_len, bins=edges)

    cand = np.array(cand_neg_ids, dtype=np.int32)
    cand_len = np.asarray(lengths)[cand]
    target = min(len(cand), ratio * len(pos_ids))

    chosen = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        last = b == len(edges) - 2
        sel = (cand_len >= lo) & ((cand_len <= hi) if last else (cand_len < hi))
        in_bin = cand[sel]
        want = int(round(target * pos_hist[b] / max(1, pos_hist.sum())))
        if want > 0 and len(in_bin) > 0:
            chosen.extend(rng.choice(in_bin, size=min(want, len(in_bin)), replace=False).tolist())

    if len(chosen) < MIN_N_NEG:
        used = set(chosen)
        rem = [int(c) for c in cand.tolist() if int(c) not in used]
        rng.shuffle(rem)
        chosen.extend(rem[:MIN_N_NEG - len(chosen)])

    return list(dict.fromkeys(chosen))


@torch.no_grad()
def batched_sentence_max_scores(flat, lengths, dirs, token_chunk=16384, device=None, desc="score"):
    """
    Computes max_t flat[token] @ dirs[k] for every sentence and direction.

    flat:    [T, D] float32 numpy
    lengths: [S]
    dirs:    [K, D] float32 numpy

    returns: [S, K] float32 numpy
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    flat = np.asarray(flat, dtype=np.float32)
    dirs = np.asarray(dirs, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int32)

    S = len(lengths)
    K = dirs.shape[0]
    T = flat.shape[0]

    sid_all = token_sentence_ids(lengths)
    assert len(sid_all) == T, (len(sid_all), T)

    out = np.full((S, K), -np.inf, dtype=np.float32)

    D_t = torch.from_numpy(dirs.T.copy()).to(device=device, dtype=torch.float32)

    for start in tqdm(range(0, T, token_chunk), desc=desc):
        end = min(T, start + token_chunk)
        x = torch.from_numpy(flat[start:end]).to(device=device, dtype=torch.float32)
        sc = (x @ D_t).detach().cpu().numpy().astype(np.float32)
        sid = sid_all[start:end]

        # K is small enough that this loop is fine.
        # Crucially, we scan the huge flat cache once per direction batch,
        # not once per feature/control.
        for k in range(K):
            np.maximum.at(out[:, k], sid, sc[:, k])

        del x, sc
        if device == "cuda":
            torch.cuda.empty_cache()

    out[~np.isfinite(out)] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--n-target-random", type=int, default=10)
    ap.add_argument("--n-mapped-random", type=int, default=10)
    ap.add_argument("--n-wrong", type=int, default=10)
    ap.add_argument("--n-near", type=int, default=5)
    ap.add_argument("--token-chunk", type=int, default=16384)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-prefix", default="fast_rescore")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(SEED)

    print("loading universe + caches...")
    feats, feat_keys, feat_pos_ids, neg_ids, sent_list = build_universe(out_dir)

    ws_npz = np.load(out_dir / "within_source.npz", allow_pickle=True)
    ws = ws_npz["ws"].astype(np.float32, copy=False)
    cached_feat_keys = [str(x) for x in ws_npz["feat_keys"]]
    assert cached_feat_keys == feat_keys, "feat key mismatch"

    q_npz = np.load(out_dir / "qwen_cache.npz", allow_pickle=True)
    qflat = q_npz["flat"].astype(np.float32, copy=False)
    qlen = q_npz["lengths"].astype(np.int32, copy=False)

    g_npz = np.load(out_dir / "gemma_cache.npz", allow_pickle=True)
    gflat = g_npz["flat"].astype(np.float32, copy=False)
    glen = g_npz["lengths"].astype(np.int32, copy=False)

    print(f"sentences={len(sent_list)} qflat={qflat.shape} gflat={gflat.shape}")

    print("loading SAE decoder...")
    from sae_lens import SAE
    obj = SAE.from_pretrained(C.SAE_REPO, C.SAE_ID, device="cpu")
    sae = obj[0] if isinstance(obj, tuple) else obj
    W_dec_all = sae.W_dec.detach().cpu().numpy().astype(np.float64)
    del sae, obj
    gc.collect()

    fi_ints = np.array([int(k) for k in feat_keys], dtype=np.int32)
    D_q_feat = unit_rows(W_dec_all[fi_ints])

    W, X_sigma, Y_sigma = load_map(args.map_dir)
    D_g_feat = map_qwen_dirs_to_gemma(D_q_feat, W, X_sigma, Y_sigma)

    # Random target-space controls.
    D_g_target_rand = rng.standard_normal((args.n_target_random, C.GEMMA_DMODEL)).astype(np.float32)
    D_g_target_rand = unit_rows(D_g_target_rand)

    # Random Qwen directions mapped through the same map.
    D_q_rand = rng.standard_normal((args.n_mapped_random, C.QWEN_DMODEL)).astype(np.float32)
    D_g_mapped_rand = map_qwen_dirs_to_gemma(D_q_rand, W, X_sigma, Y_sigma)

    # One big Gemma score pass.
    all_g_dirs = np.vstack([D_g_feat, D_g_target_rand, D_g_mapped_rand]).astype(np.float32)
    print(f"Gemma batched directions: {all_g_dirs.shape}")
    g_scores = batched_sentence_max_scores(
        gflat, glen, all_g_dirs,
        token_chunk=args.token_chunk,
        device=args.device,
        desc="Gemma batched max",
    )

    n_feat = len(feat_keys)
    feat_scores = g_scores[:, :n_feat]
    target_rand_scores = g_scores[:, n_feat:n_feat + args.n_target_random]
    mapped_rand_scores = g_scores[:, n_feat + args.n_target_random:]

    # Qwen decoder source ceiling pass for the pilot feature directions.
    print(f"Qwen batched feature directions: {D_q_feat.shape}")
    q_scores = batched_sentence_max_scores(
        qflat, qlen, D_q_feat.astype(np.float32),
        token_chunk=args.token_chunk,
        device=args.device,
        desc="Qwen decoder batched max",
    )

    # Near-miss by source decoder cosine.
    dec_cos = D_q_feat @ D_q_feat.T

    rows = []
    for col, k in enumerate(tqdm(feat_keys, desc="AUROCs")):
        fi = int(k)
        label = feats[k].get("label", "")

        pos_ids = feat_pos_ids[k]
        cand_neg = [s for s in neg_ids if ws[s, col] == 0.0]

        if len(pos_ids) < MIN_N_POS or len(cand_neg) < MIN_N_NEG:
            rows.append({
                "fi": fi,
                "label": label,
                "skip": True,
                "n_pos": len(pos_ids),
                "n_neg": len(cand_neg),
            })
            continue

        neg_f = length_matched_negs(pos_ids, cand_neg, glen, rng, ratio=5)
        ids = np.array(pos_ids + neg_f, dtype=np.int32)
        y = np.array([1] * len(pos_ids) + [0] * len(neg_f), dtype=np.int32)

        transfer = safe_auroc(y, feat_scores[ids, col])
        qdec = safe_auroc(y, q_scores[ids, col])
        source_sae = safe_auroc(y, ws[ids, col])
        length_auc = safe_auroc(y, np.asarray(glen)[ids])

        target_rand_auc = np.array([
            safe_auroc(y, target_rand_scores[ids, j])
            for j in range(target_rand_scores.shape[1])
        ])
        mapped_rand_auc = np.array([
            safe_auroc(y, mapped_rand_scores[ids, j])
            for j in range(mapped_rand_scores.shape[1])
        ])

        order = np.argsort(dec_cos[col])[::-1]
        near_cols = [c for c in order if c != col][:args.n_near]
        near_aucs = np.array([
            safe_auroc(y, feat_scores[ids, c])
            for c in near_cols
        ])
        near_median = float(np.median(near_aucs)) if len(near_aucs) else 0.5

        other = [c for c in range(n_feat) if c != col]
        wrong_cols = rng.choice(other, size=min(args.n_wrong, len(other)), replace=False)
        wrong_aucs = np.array([
            safe_auroc(y, feat_scores[ids, c])
            for c in wrong_cols
        ])
        wrong_mean = float(wrong_aucs.mean()) if len(wrong_aucs) else 0.5
        beat_wrong_frac = float((transfer > wrong_aucs).mean()) if len(wrong_aucs) else 0.0
        true_minus_wrong = transfer - wrong_mean

        target_random_mean = float(target_rand_auc.mean())
        mapped_random_mean = float(mapped_rand_auc.mean())

        length_ok = 0.40 <= length_auc <= 0.60
        target_random_ok = abs(target_random_mean - 0.5) <= 0.10
        mapped_random_ok = abs(mapped_random_mean - 0.5) <= 0.10
        nulls_ok = length_ok and target_random_ok and mapped_random_ok

        identity_specific_ok = (
            (transfer > near_median + MARGIN_THRESH)
            and (beat_wrong_frac >= WRONG_BEAT_FRAC)
            and (true_minus_wrong >= MARGIN_THRESH)
        )

        qwen_decoder_ok = qdec >= DECODER_VALID
        eligible_lite = (
            source_sae >= 0.85
            and qwen_decoder_ok            # README §13: don't count source-decoder-weak features (e.g. 57853, qdec 0.44)
            and nulls_ok
            and identity_specific_ok
            and transfer >= TRANSFER_CLEAN
        )

        rows.append({
            "fi": fi,
            "label": label,
            "skip": False,
            "n_pos": len(pos_ids),
            "n_neg": len(neg_f),

            "source_sae_auroc": source_sae,
            "qwen_decoder_auroc": qdec,
            "transfer_dot": transfer,

            "length_confound_auc": length_auc,
            "target_random_mean": target_random_mean,
            "mapped_random_mean": mapped_random_mean,

            "near_miss_median": near_median,
            "wrong_feature_mean": wrong_mean,
            "true_minus_wrong": true_minus_wrong,
            "beat_wrong_frac": beat_wrong_frac,

            "length_ok": length_ok,
            "target_random_ok": target_random_ok,
            "mapped_random_ok": mapped_random_ok,
            "nulls_ok": nulls_ok,
            "identity_specific_ok": identity_specific_ok,
            "qwen_decoder_ok": qwen_decoder_ok,
            "eligible_lite": eligible_lite,
        })

    df = pd.DataFrame(rows)
    scored = df[df["skip"] == False].copy()

    summary = {
        "map_dir": str(args.map_dir),
        "scored": int(len(scored)),
        "skipped": int(df["skip"].sum()),
        "transfer_mean": float(scored["transfer_dot"].mean()) if len(scored) else None,
        "transfer_median": float(scored["transfer_dot"].median()) if len(scored) else None,
        "true_minus_wrong_mean": float(scored["true_minus_wrong"].mean()) if len(scored) else None,
        "source_ok": int((scored["source_sae_auroc"] >= 0.85).sum()),
        "length_ok": int(scored["length_ok"].sum()),
        "target_random_ok": int(scored["target_random_ok"].sum()),
        "mapped_random_ok": int(scored["mapped_random_ok"].sum()),
        "nulls_ok": int(scored["nulls_ok"].sum()),
        "identity_specific_ok": int(scored["identity_specific_ok"].sum()),
        "qwen_decoder_ok": int(scored["qwen_decoder_ok"].sum()),
        "eligible_lite": int(scored["eligible_lite"].sum()),  # now decoder-gated (README §13)
    }

    csv_path = out_dir / f"{args.save_prefix}.csv"
    json_path = out_dir / f"{args.save_prefix}.json"
    summary_path = out_dir / f"{args.save_prefix}_summary.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nFAST RESCORE SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nTop transfers:")
    show_cols = [
        "fi", "transfer_dot", "wrong_feature_mean", "true_minus_wrong",
        "mapped_random_mean", "length_confound_auc",
        "qwen_decoder_auroc", "nulls_ok", "identity_specific_ok", "eligible_lite"
    ]
    print(scored.sort_values("transfer_dot", ascending=False)[show_cols].head(15).to_string(index=False))

    print("\nSaved:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()