#!/usr/bin/env python
"""
06_validate_metric.py — METRIC VALIDATION for Exp 3 (spec §8; status-README §6/§8).

Distinct from 05_cofiring.py (which computes the primary transfer metric): this script asks
"is the co-firing metric itself trustworthy for each feature, or is a high transfer AUROC an
artifact?" It runs the full null/robustness battery and emits the final claimability flags.

Per-feature gates (frozen, matching the status README):
  source_ok        = SAE .encode() recovers A_f vs N_f on Qwen   (the A_f/N_f labels are valid)
  qwen_decoder_ok  = W_dec[f] DETECTS on Qwen (>=0.80)            (the transported direction is a real source detector)
  length_ok        = length-only AUROC in [0.40, 0.60]           (length is not the signal)
  target_random_ok = random Gemma dirs ~0.50
  mapped_random_ok = random Qwen dirs mapped ~0.50               (the map alone doesn't separate)
  shuffle_ok       = label-shuffled transfer ~0.50               (the metric is calibrated)
  robustness_ok    = transfer survives cosine-scoring AND outlier-dim-zeroing (within 0.15)
  identity_specific_ok = transfer > near_miss_median+0.10 AND beats >=9/10 wrong features AND true-minus-wrong>=0.10

  nulls_ok                  = shuffle_ok AND target_random_ok AND mapped_random_ok
  passes_metric_validation  = source_ok AND nulls_ok AND length_ok AND robustness_ok
  eligible_transfer_claim   = passes_metric_validation AND identity_specific_ok AND transfer>=0.70
  eligible_transfer_claim_strict_decoder = eligible_transfer_claim AND qwen_decoder_ok   (README Action 3)

NEGATIVE POOL (the load-bearing choice — README §16 Q3, the Exp-2 BoW lesson one level up):
  default (--hard-negatives off): length-matched random negatives  (reproduces existing numbers).
  --hard-negatives:  SAME-DOMAIN hard negatives = positives of near-miss SIBLING features (top decoder-cosine
                     neighbours) where the TARGET feature is inactive. A broad domain/format feature
                     ("chemistry synthesis-titles") must beat its own NEIGHBOURHOOD, not just random text.
                     Expect this to SHRINK the eligible set — that is the real test, not a bug.

Reuses 05's caches (same --out-dir): pilot_features.json, negative_pool.json, within_source.npz,
qwen_cache.npz, gemma_cache.npz. Map (--map-dir/ridge_map.npz) may be ridge or standardized-Procrustes
(keys W, X_sigma, Y_sigma) — direction transport handles both: unit((d/X_sigma) @ W * Y_sigma).

Outputs (to --out-dir): {prefix}.json, {prefix}.csv, {prefix}_summary.json   (prefix default metric_validation)

Run (box/H200; needs the 05 caches + the SAE):
  HF_TOKEN=... python 06_validate_metric.py --out-dir DIR --map-dir MAPDIR \
      --n-random 10 --n-shuffle 100 --n-wrong 10 [--hard-negatives]
"""

import argparse, gc, json, os, sys
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

# ── frozen thresholds (match 05b / spec) ──
SEED = 42
MARGIN_THRESH = 0.10
WRONG_BEAT_FRAC = 0.90
TRANSFER_CLEAN = 0.70
DECODER_VALID = 0.80
NULL_TOL = 0.10            # |mean - 0.5| tolerance for null controls
ROBUST_TOL = 0.15         # max |variant - transfer| for robustness_ok
SOURCE_OK = 0.85
N_OUTLIER_DIMS = 10
MIN_N_POS = 10
MIN_N_NEG = 20


# ── helpers (kept standalone; byte-compatible with 05b conventions) ──
def safe_auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def token_sentence_ids(lengths):
    return np.repeat(np.arange(len(lengths), dtype=np.int32), lengths.astype(np.int32))


def unit_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    return (X / np.linalg.norm(X, axis=1, keepdims=True).clip(eps)).astype(np.float32)


def load_map(map_dir):
    m = np.load(Path(map_dir) / "ridge_map.npz", allow_pickle=True)
    return (m["W"].astype(np.float64), m["X_sigma"].astype(np.float64), m["Y_sigma"].astype(np.float64))


def map_qwen_dirs_to_gemma(D_q, W, X_sigma, Y_sigma):
    """unit((d / X_sigma) @ W * Y_sigma) — correct for raw-ridge (sigma=1) AND standardized-Procrustes."""
    D_q = unit_rows(D_q)
    D_g = (D_q / X_sigma.clip(1e-12)) @ W
    D_g = D_g * Y_sigma
    return unit_rows(D_g)


def build_universe(out_dir):
    out_dir = Path(out_dir)
    feats = json.loads((out_dir / "pilot_features.json").read_text())
    neg_pool = json.loads((out_dir / "negative_pool.json").read_text())
    sent_to_id, sent_list = {}, []

    def reg(t):
        if t not in sent_to_id:
            sent_to_id[t] = len(sent_list); sent_list.append(t)
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
        rng.shuffle(rem); chosen.extend(rem[:MIN_N_NEG - len(chosen)])
    return list(dict.fromkeys(chosen))


def hard_domain_negs(col, near_cols, feat_keys, feat_pos_ids, ws, lengths, pos_ids, rng, ratio=5):
    """SAME-DOMAIN hard negatives: positives of near-miss SIBLING features where the TARGET (col) is inactive.
    Falls back to length-matched random if too few siblings/candidates."""
    cand = set()
    for c in near_cols:
        for s in feat_pos_ids[feat_keys[c]]:
            if ws[s, col] == 0.0:
                cand.add(int(s))
    cand.discard(None)
    cand = [s for s in cand if s not in set(pos_ids)]
    if len(cand) < MIN_N_NEG:
        return None  # caller falls back
    return length_matched_negs(pos_ids, cand, lengths, rng, ratio=ratio)


@torch.no_grad()
def batched_sentence_max_scores(flat, lengths, dirs, token_chunk=16384, device=None,
                                normalize_tokens=False, desc="score"):
    """max_t (flat[token] [/||·||] @ dirs[k]) per (sentence, direction). normalize_tokens=True -> cosine."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    flat = np.asarray(flat, dtype=np.float32); dirs = np.asarray(dirs, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int32)
    S, K, T = len(lengths), dirs.shape[0], flat.shape[0]
    sid_all = token_sentence_ids(lengths); assert len(sid_all) == T
    out = np.full((S, K), -np.inf, dtype=np.float32)
    D_t = torch.from_numpy(dirs.T.copy()).to(device=device, dtype=torch.float32)
    for start in tqdm(range(0, T, token_chunk), desc=desc):
        end = min(T, start + token_chunk)
        x = torch.from_numpy(flat[start:end]).to(device=device, dtype=torch.float32)
        if normalize_tokens:
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sc = (x @ D_t).detach().cpu().numpy().astype(np.float32)
        sid = sid_all[start:end]
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
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--n-shuffle", type=int, default=100)
    ap.add_argument("--n-wrong", type=int, default=10)
    ap.add_argument("--n-near", type=int, default=5)
    ap.add_argument("--token-chunk", type=int, default=16384)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hard-negatives", action="store_true",
                    help="same-domain hard negatives (near-miss sibling positives) instead of random")
    ap.add_argument("--save-prefix", default="metric_validation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(SEED)

    print("loading universe + caches...")
    feats, feat_keys, feat_pos_ids, neg_ids, sent_list = build_universe(out_dir)
    ws_npz = np.load(out_dir / "within_source.npz", allow_pickle=True)
    ws = ws_npz["ws"].astype(np.float32, copy=False)
    assert [str(x) for x in ws_npz["feat_keys"]] == feat_keys, "feat key mismatch"
    q_npz = np.load(out_dir / "qwen_cache.npz", allow_pickle=True)
    qflat, qlen = q_npz["flat"].astype(np.float32, copy=False), q_npz["lengths"].astype(np.int32, copy=False)
    g_npz = np.load(out_dir / "gemma_cache.npz", allow_pickle=True)
    gflat, glen = g_npz["flat"].astype(np.float32, copy=False), g_npz["lengths"].astype(np.int32, copy=False)
    n_feat = len(feat_keys)
    print(f"sentences={len(sent_list)} qflat={qflat.shape} gflat={gflat.shape} n_feat={n_feat} "
          f"negatives={'SAME-DOMAIN hard' if args.hard_negatives else 'length-matched random'}")

    print("loading SAE decoder...")
    from sae_lens import SAE
    obj = SAE.from_pretrained(C.SAE_REPO, C.SAE_ID, device="cpu")
    sae = obj[0] if isinstance(obj, tuple) else obj
    W_dec_all = sae.W_dec.detach().cpu().numpy().astype(np.float64)
    del sae, obj; gc.collect()

    fi_ints = np.array([int(k) for k in feat_keys], dtype=np.int32)
    D_q_feat = unit_rows(W_dec_all[fi_ints])
    W, X_sigma, Y_sigma = load_map(args.map_dir)
    D_g_feat = map_qwen_dirs_to_gemma(D_q_feat, W, X_sigma, Y_sigma)

    # outlier dims (top variance Gemma dims) for the robustness check
    samp = rng.choice(len(gflat), size=min(100_000, len(gflat)), replace=False)
    gvar = gflat[samp].astype(np.float64).var(0)
    outlier_dims = np.argsort(gvar)[::-1][:N_OUTLIER_DIMS]
    D_g_feat_oz = D_g_feat.copy(); D_g_feat_oz[:, outlier_dims] = 0.0
    D_g_feat_oz = unit_rows(D_g_feat_oz)

    # random controls
    D_g_target_rand = unit_rows(rng.standard_normal((args.n_random, C.GEMMA_DMODEL)))
    D_g_mapped_rand = map_qwen_dirs_to_gemma(rng.standard_normal((args.n_random, C.QWEN_DMODEL)), W, X_sigma, Y_sigma)

    # main Gemma pass: true dirs + target-random + mapped-random
    all_g = np.vstack([D_g_feat, D_g_target_rand, D_g_mapped_rand]).astype(np.float32)
    g_scores = batched_sentence_max_scores(gflat, glen, all_g, args.token_chunk, args.device, desc="Gemma main")
    feat_scores = g_scores[:, :n_feat]
    target_rand_scores = g_scores[:, n_feat:n_feat + args.n_random]
    mapped_rand_scores = g_scores[:, n_feat + args.n_random:]
    # robustness passes (true dirs only)
    cos_scores = batched_sentence_max_scores(gflat, glen, D_g_feat, args.token_chunk, args.device,
                                             normalize_tokens=True, desc="Gemma cosine")
    oz_scores = batched_sentence_max_scores(gflat, glen, D_g_feat_oz, args.token_chunk, args.device, desc="Gemma outlier-zeroed")
    # Qwen decoder ceiling
    q_scores = batched_sentence_max_scores(qflat, qlen, D_q_feat, args.token_chunk, args.device, desc="Qwen decoder")

    dec_cos = D_q_feat @ D_q_feat.T
    rows = []
    for col, k in enumerate(tqdm(feat_keys, desc="validate")):
        fi = int(k); label = feats[k].get("label", "")
        pos_ids = feat_pos_ids[k]
        order = np.argsort(dec_cos[col])[::-1]
        near_cols = [c for c in order if c != col][:args.n_near]

        cand_neg = [s for s in neg_ids if ws[s, col] == 0.0]
        neg_f = None
        if args.hard_negatives:
            neg_f = hard_domain_negs(col, near_cols, feat_keys, feat_pos_ids, ws, glen, pos_ids, rng)
        if neg_f is None:
            neg_f = length_matched_negs(pos_ids, cand_neg, glen, rng) if len(cand_neg) >= MIN_N_NEG else []

        if len(pos_ids) < MIN_N_POS or len(neg_f) < MIN_N_NEG:
            rows.append({"fi": fi, "label": label, "skip": True, "n_pos": len(pos_ids), "n_neg": len(neg_f)})
            continue

        ids = np.array(pos_ids + neg_f, dtype=np.int32)
        y = np.array([1] * len(pos_ids) + [0] * len(neg_f), dtype=np.int32)

        transfer = safe_auroc(y, feat_scores[ids, col])
        source_sae = safe_auroc(y, ws[ids, col])
        qdec = safe_auroc(y, q_scores[ids, col])
        length_auc = safe_auroc(y, np.asarray(glen)[ids])

        target_rand_auc = np.array([safe_auroc(y, target_rand_scores[ids, j]) for j in range(args.n_random)])
        mapped_rand_auc = np.array([safe_auroc(y, mapped_rand_scores[ids, j]) for j in range(args.n_random)])
        shuf = np.array([safe_auroc(rng.permutation(y), feat_scores[ids, col]) for _ in range(args.n_shuffle)])
        cos_auc = safe_auroc(y, cos_scores[ids, col])
        oz_auc = safe_auroc(y, oz_scores[ids, col])

        near_aucs = np.array([safe_auroc(y, feat_scores[ids, c]) for c in near_cols])
        near_median = float(np.median(near_aucs)) if len(near_aucs) else 0.5
        wrong_cols = rng.choice([c for c in range(n_feat) if c != col],
                                size=min(args.n_wrong, n_feat - 1), replace=False)
        wrong_aucs = np.array([safe_auroc(y, feat_scores[ids, c]) for c in wrong_cols])
        wrong_mean = float(wrong_aucs.mean())
        beat_wrong_frac = float((transfer > wrong_aucs).mean())
        true_minus_wrong = transfer - wrong_mean

        source_ok = source_sae >= SOURCE_OK
        qwen_decoder_ok = qdec >= DECODER_VALID
        length_ok = 0.40 <= length_auc <= 0.60
        target_random_ok = abs(target_rand_auc.mean() - 0.5) <= NULL_TOL
        mapped_random_ok = abs(mapped_rand_auc.mean() - 0.5) <= NULL_TOL
        shuffle_ok = abs(shuf.mean() - 0.5) <= NULL_TOL
        robustness_ok = (abs(cos_auc - transfer) <= ROBUST_TOL) and (abs(oz_auc - transfer) <= ROBUST_TOL)
        identity_specific_ok = ((transfer > near_median + MARGIN_THRESH)
                                and (beat_wrong_frac >= WRONG_BEAT_FRAC)
                                and (true_minus_wrong >= MARGIN_THRESH))

        nulls_ok = shuffle_ok and target_random_ok and mapped_random_ok
        passes_metric_validation = source_ok and nulls_ok and length_ok and robustness_ok
        eligible_transfer_claim = passes_metric_validation and identity_specific_ok and transfer >= TRANSFER_CLEAN
        eligible_strict = eligible_transfer_claim and qwen_decoder_ok

        rows.append(dict(
            fi=fi, label=label, skip=False, n_pos=len(pos_ids), n_neg=len(neg_f),
            transfer_dot=transfer, source_sae_auroc=source_sae, qwen_decoder_auroc=qdec,
            length_confound_auc=length_auc, target_random_mean=float(target_rand_auc.mean()),
            mapped_random_mean=float(mapped_rand_auc.mean()), shuffle_mean=float(shuf.mean()),
            cosine_auc=cos_auc, outlier_zeroed_auc=oz_auc,
            near_miss_median=near_median, wrong_feature_mean=wrong_mean,
            true_minus_wrong=true_minus_wrong, beat_wrong_frac=beat_wrong_frac,
            source_ok=source_ok, qwen_decoder_ok=qwen_decoder_ok, length_ok=length_ok,
            target_random_ok=target_random_ok, mapped_random_ok=mapped_random_ok, shuffle_ok=shuffle_ok,
            robustness_ok=robustness_ok, nulls_ok=nulls_ok, identity_specific_ok=identity_specific_ok,
            passes_metric_validation=passes_metric_validation,
            eligible_transfer_claim=eligible_transfer_claim,
            eligible_transfer_claim_strict_decoder=eligible_strict,
        ))

    df = pd.DataFrame(rows)
    scored = df[df["skip"] == False].copy()
    n = max(1, len(scored))
    def cnt(c): return int(scored[c].sum()) if c in scored else 0
    summary = dict(
        map_dir=str(args.map_dir), negatives="hard_same_domain" if args.hard_negatives else "length_matched_random",
        scored=int(len(scored)), skipped=int(df["skip"].sum()),
        source_ok=cnt("source_ok"), shuffle_ok=cnt("shuffle_ok"), target_random_ok=cnt("target_random_ok"),
        mapped_random_ok=cnt("mapped_random_ok"), length_ok=cnt("length_ok"), robustness_ok=cnt("robustness_ok"),
        nulls_ok=cnt("nulls_ok"), qwen_decoder_ok=cnt("qwen_decoder_ok"), identity_specific_ok=cnt("identity_specific_ok"),
        passes_metric_validation=cnt("passes_metric_validation"),
        eligible_transfer_claim=cnt("eligible_transfer_claim"),
        eligible_transfer_claim_strict_decoder=cnt("eligible_transfer_claim_strict_decoder"),
        eligible_rate=round(cnt("eligible_transfer_claim") / n, 4),
        eligible_rate_strict_decoder=round(cnt("eligible_transfer_claim_strict_decoder") / n, 4),
    )

    df.to_csv(out_dir / f"{args.save_prefix}.csv", index=False)
    (out_dir / f"{args.save_prefix}.json").write_text(json.dumps(rows, indent=2))
    (out_dir / f"{args.save_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nMETRIC VALIDATION SUMMARY"); print(json.dumps(summary, indent=2))
    show = ["fi", "transfer_dot", "true_minus_wrong", "qwen_decoder_auroc", "robustness_ok",
            "nulls_ok", "identity_specific_ok", "passes_metric_validation",
            "eligible_transfer_claim", "eligible_transfer_claim_strict_decoder"]
    if len(scored):
        print("\nTop by transfer:")
        print(scored.sort_values("transfer_dot", ascending=False)[show].head(15).to_string(index=False))
    print(f"\nwrote {args.save_prefix}.{{json,csv}} + _summary.json to {out_dir}")


if __name__ == "__main__":
    main()
