#!/usr/bin/env python
"""
05_cofiring.py — PRIMARY METRIC, publication-grade (spec §7, §8, §13).

Implements the full control suite the spec requires and the advisor flagged:
  - length-matched N_f (max-over-tokens favors long sequences -> mandatory)
  - within-source ceiling = max_t(qwen_act . W_dec[f])   (decoder projection,
    same score family as transfer; NOT the .encode() activation)
  - Gate 1 random-direction floor
  - Gate 1 near-miss wrong features (5 nearest decoder cosine)
  - Gate 1 RANDOM-MATCHED-WRONG-FEATURE kill-switch (load-bearing)
  - Gate 1 frozen rule: identity-specific transfer iff transfer >
    near_miss_median+0.10 AND beats >= 9/10 wrong-feature controls
  - Gate 2 Gemma-native CV'd logistic probe (representability ceiling) + perm
  - dual-ceiling decomposition (map-failure vs not-representable)
  - top-5-mean robustness variant
  - per-feature bootstrap CIs on transfer AUROC and wrong-feature margin
  - covariate dump for the §13 covariate model

Passes (resumable via cached .npz):
  Pass 1  Qwen + SAE  -> within_source.npz (encode) + qwen_cache.npz (per-token)
  Pass 2  Gemma       -> gemma_cache.npz   (per-token)
  Pass 3  numpy/sklearn -> scoring + controls + stats

NOTE: qwen_cache.npz is NEW (the decoder-projection ceiling needs raw per-token
Qwen activations). If only within_source.npz exists from an older run, Pass 1
re-runs Qwen once (~1 min) to build it.

Usage:
  HF_TOKEN=... python 05_cofiring.py --out-dir /content/exp3_acts \
      --map-dir /content/exp3_acts [--force-recompute] [--n-boot 1000]
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import exp3_config as C

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

TOKEN = os.environ.get("HF_TOKEN")

# ── frozen thresholds (set BEFORE looking at results) ────────────────────────
MARGIN_THRESH   = 0.10
WRONG_BEAT_FRAC = 0.90
TRANSFER_CLEAN  = 0.70
PROBE_MIN       = 0.60
PROBE_HIGH      = 0.80
N_RAND_CTRL     = 10
N_NEAR_MISS     = 5
N_WRONG_FEAT    = 10
MIN_N_POS       = 10
MIN_N_NEG       = 20
PROBE_FOLDS     = 5
SEED            = 42


def load_map(map_dir):
    m = np.load(map_dir / "ridge_map.npz", allow_pickle=True)
    return (m["W"].astype(np.float64), m["X_sigma"].astype(np.float64),
            m["Y_sigma"].astype(np.float64))


def map_direction(d_f, W, X_sigma, Y_sigma):
    d_f = np.asarray(d_f, np.float64)
    d_g = (d_f / X_sigma.clip(1e-12) @ W) * Y_sigma
    n = np.linalg.norm(d_g)
    return None if n < 1e-12 else (d_g / n, float(n))


def build_universe(out_dir):
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
    print(f"universe: {len(sent_list)} unique ({len(sent_list)-len(neg_pool)} A_f, "
          f"{len(neg_pool)} neg pool)")
    return feats, feat_keys, feat_pos_ids, neg_ids, sent_list


@torch.no_grad()
def run_qwen_pass(sent_list, feat_keys, out_dir, force):
    ws_cache = out_dir / "within_source.npz"
    qc_cache = out_dir / "qwen_cache.npz"
    if ws_cache.exists() and qc_cache.exists() and not force:
        print("[Pass 1] loading cached within_source + qwen_cache")
        d = np.load(ws_cache, allow_pickle=True)
        q = np.load(qc_cache, allow_pickle=True)
        return (d["ws"].astype(np.float32), list(d["feat_keys"]),
                q["flat"].astype(np.float32), q["lengths"].astype(np.int32))

    print("[Pass 1] Qwen + SAE -> within_source (encode) + qwen_cache (per-token)")
    from sae_lens import SAE
    tok = AutoTokenizer.from_pretrained(C.QWEN_REPO, token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        C.QWEN_REPO, token=TOKEN, torch_dtype=torch.bfloat16, device_map="auto").eval()
    obj = SAE.from_pretrained(C.SAE_REPO, C.SAE_ID, device="cuda")
    sae = obj[0] if isinstance(obj, tuple) else obj
    sae_dev = next(iter(sae.parameters())).device

    fi_arr = np.array([int(k) for k in feat_keys], np.int32)
    ws = np.zeros((len(sent_list), len(fi_arr)), np.float32)
    flat_list, lengths = [], np.zeros(len(sent_list), np.int32)
    for si, text in enumerate(tqdm(sent_list, desc="qwen+sae")):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[C.QWEN_HS_IDX][0]
        hs = hs[C.SAE_SKIP_FIRST_N:]
        a = hs.float().cpu().numpy().astype(np.float32)
        flat_list.append(a); lengths[si] = a.shape[0]
        if a.shape[0] > 0:
            fe = sae.encode(hs.float().to(sae_dev))
            ws[si] = fe[:, fi_arr].max(0).values.cpu().numpy()
    flat = np.concatenate(flat_list, 0)
    del model, sae, obj; gc.collect(); torch.cuda.empty_cache()
    np.savez(ws_cache, ws=ws, feat_keys=np.array(feat_keys))
    np.savez(qc_cache, flat=flat, lengths=lengths)
    print(f"  saved within_source {ws.shape} + qwen_cache {flat.shape}")
    return ws, feat_keys, flat, lengths


@torch.no_grad()
def run_gemma_pass(sent_list, out_dir, force):
    cache = out_dir / "gemma_cache.npz"
    if cache.exists() and not force:
        print("[Pass 2] loading cached gemma_cache")
        d = np.load(cache, allow_pickle=True)
        return d["flat"].astype(np.float32), d["lengths"].astype(np.int32)
    print("[Pass 2] Gemma -> per-token activations")
    tok = AutoTokenizer.from_pretrained(C.GEMMA_REPO, token=TOKEN)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        C.GEMMA_REPO, token=TOKEN, torch_dtype=torch.bfloat16, device_map="auto").eval()
    cfg = model.config
    nl = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    assert nl == C.GEMMA_NUM_LAYERS
    flat_list, lengths = [], np.zeros(len(sent_list), np.int32)
    for si, text in enumerate(tqdm(sent_list, desc="gemma")):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[C.GEMMA_HS_IDX][0]
        a = hs.float().cpu().numpy().astype(np.float32)
        flat_list.append(a); lengths[si] = a.shape[0]
    flat = np.concatenate(flat_list, 0)
    del model; gc.collect(); torch.cuda.empty_cache()
    np.savez(cache, flat=flat, lengths=lengths)
    print(f"  saved gemma_cache {flat.shape}")
    return flat, lengths


def offsets_of(lengths):
    return np.concatenate([[0], lengths.cumsum()])


def maxproj(flat, off, sid, d):
    s, e = off[sid], off[sid + 1]
    return 0.0 if s >= e else float((flat[s:e] @ d).max())


def top5mean_proj(flat, off, sid, d):
    s, e = off[sid], off[sid + 1]
    if s >= e: return 0.0
    p = flat[s:e] @ d
    return float(np.sort(p)[::-1][:min(5, len(p))].mean())


def score_ids(flat, off, ids, d, fn=maxproj):
    return np.array([fn(flat, off, i, d) for i in ids], np.float32)


def safe_auroc(y, s):
    y = np.asarray(y)
    return 0.5 if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def maxpool_vec(flat, off, ids):
    out = []
    for i in ids:
        s, e = off[i], off[i + 1]
        if e > s: out.append(flat[s:e].max(0))
    return np.stack(out) if out else np.zeros((0, flat.shape[1]))


def length_matched_negs(pos_ids, cand_neg_ids, lengths, rng, ratio=5):
    """Sample negatives matched to A_f token-length histogram (spec §7,
    mandatory: max-over-tokens favors long sequences)."""
    pos_len = lengths[pos_ids]
    nb = min(8, max(2, len(pos_ids) // 4))
    edges = np.histogram_bin_edges(pos_len, bins=nb)
    pos_hist, _ = np.histogram(pos_len, bins=edges)
    cand = np.array(cand_neg_ids); cand_len = lengths[cand]
    target = min(len(cand), ratio * len(pos_ids))
    chosen = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        last = (b == len(edges) - 2)
        sel = (cand_len >= lo) & ((cand_len <= hi) if last else (cand_len < hi))
        in_bin = cand[sel]
        want = int(round(target * pos_hist[b] / max(1, pos_hist.sum())))
        if want > 0 and len(in_bin) > 0:
            chosen.extend(rng.choice(in_bin, size=min(want, len(in_bin)),
                                     replace=False).tolist())
    if len(chosen) < MIN_N_NEG:
        rem = [c for c in cand.tolist() if c not in set(chosen)]
        rng.shuffle(rem); chosen.extend(rem[:MIN_N_NEG - len(chosen)])
    return list(dict.fromkeys(chosen))


def gemma_probe(gflat, goff, pos_ids, neg_ids, rng):
    X = np.vstack([maxpool_vec(gflat, goff, pos_ids), maxpool_vec(gflat, goff, neg_ids)])
    y = np.array([1]*len(pos_ids) + [0]*len(neg_ids))
    if len(np.unique(y)) < 2 or X.shape[0] < 2 * PROBE_FOLDS:
        return 0.5, 0.5
    def cv(Xx, yy):
        skf = StratifiedKFold(n_splits=PROBE_FOLDS, shuffle=True, random_state=SEED)
        oof = np.zeros(len(yy))
        for tr, te in skf.split(Xx, yy):
            sc = StandardScaler().fit(Xx[tr])
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(Xx[tr]), yy[tr])
            oof[te] = clf.predict_proba(sc.transform(Xx[te]))[:, 1]
        return safe_auroc(yy, oof)
    return cv(X, y), cv(X, rng.permutation(y))


def bootstrap_auroc_ci(sp, sn, n_boot, rng):
    a = []
    for _ in range(n_boot):
        ip = rng.integers(0, len(sp), len(sp)); ineg = rng.integers(0, len(sn), len(sn))
        a.append(safe_auroc([1]*len(sp) + [0]*len(sn), np.concatenate([sp[ip], sn[ineg]])))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def run_scoring(feats, feat_keys, feat_pos_ids, neg_ids, ws,
                qflat, qlen, gflat, glen, out_dir, map_dir, n_boot):
    rng = np.random.default_rng(SEED)
    W, X_sigma, Y_sigma = load_map(map_dir)
    qoff, goff = offsets_of(qlen), offsets_of(glen)

    from sae_lens import SAE
    obj = SAE.from_pretrained(C.SAE_REPO, C.SAE_ID, device="cpu")
    sae = obj[0] if isinstance(obj, tuple) else obj
    W_dec_all = sae.W_dec.detach().cpu().numpy().astype(np.float64)
    del sae, obj

    fi_ints = [int(k) for k in feat_keys]
    mapped, mapped_norm = {}, {}
    for k in feat_keys:
        fi = int(k)
        d = W_dec_all[fi] / np.linalg.norm(W_dec_all[fi]).clip(1e-12)
        r = map_direction(d, W, X_sigma, Y_sigma)
        if r is not None:
            mapped[fi] = r[0].astype(np.float32); mapped_norm[fi] = r[1]

    Wd_pilot = np.stack([W_dec_all[fi] / np.linalg.norm(W_dec_all[fi]).clip(1e-12)
                         for fi in fi_ints])
    dec_cos = Wd_pilot @ Wd_pilot.T

    rand_dirs = rng.standard_normal((N_RAND_CTRL, C.GEMMA_DMODEL)).astype(np.float32)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)

    results = []
    for col, k in enumerate(tqdm(feat_keys, desc="scoring")):
        fi = int(k); label = feats[k].get("label", "")
        pos_ids = feat_pos_ids[k]
        cand_neg = [s for s in neg_ids if ws[s, col] == 0.0]
        if len(pos_ids) < MIN_N_POS or len(cand_neg) < MIN_N_NEG or fi not in mapped:
            results.append({"fi": fi, "label": label, "skip": True,
                            "n_pos": len(pos_ids), "n_neg": len(cand_neg)}); continue

        neg_f = length_matched_negs(pos_ids, cand_neg, glen, rng)
        n_pos, n_neg = len(pos_ids), len(neg_f)
        labels = np.array([1]*n_pos + [0]*n_neg)
        d_prime = mapped[fi]

        A = score_ids(gflat, goff, pos_ids, d_prime)
        Ng = score_ids(gflat, goff, neg_f, d_prime)
        transfer = safe_auroc(labels, np.concatenate([A, Ng]))
        A5 = score_ids(gflat, goff, pos_ids, d_prime, top5mean_proj)
        N5 = score_ids(gflat, goff, neg_f, d_prime, top5mean_proj)
        transfer_t5 = safe_auroc(labels, np.concatenate([A5, N5]))
        tlo, thi = bootstrap_auroc_ci(A, Ng, n_boot, rng)

        d_qwen = (W_dec_all[fi] / np.linalg.norm(W_dec_all[fi]).clip(1e-12)).astype(np.float32)
        wsA = score_ids(qflat, qoff, pos_ids, d_qwen)
        wsN = score_ids(qflat, qoff, neg_f, d_qwen)
        ws_ceiling = safe_auroc(labels, np.concatenate([wsA, wsN]))

        rand_auc = float(np.mean([
            safe_auroc(labels, np.concatenate([score_ids(gflat, goff, pos_ids, rd),
                                               score_ids(gflat, goff, neg_f, rd)]))
            for rd in rand_dirs]))

        order = np.argsort(dec_cos[col])[::-1]
        near_cols = [c for c in order if c != col and int(feat_keys[c]) in mapped][:N_NEAR_MISS]
        near = [safe_auroc(labels, np.concatenate([
            score_ids(gflat, goff, pos_ids, mapped[int(feat_keys[c])]),
            score_ids(gflat, goff, neg_f, mapped[int(feat_keys[c])])])) for c in near_cols]
        near_median = float(np.median(near)) if near else 0.5

        other = [int(feat_keys[c]) for c in range(len(feat_keys))
                 if c != col and int(feat_keys[c]) in mapped]
        wrong_sample = rng.choice(other, size=min(N_WRONG_FEAT, len(other)), replace=False)
        wrong = np.array([safe_auroc(labels, np.concatenate([
            score_ids(gflat, goff, pos_ids, mapped[int(wf)]),
            score_ids(gflat, goff, neg_f, mapped[int(wf)])])) for wf in wrong_sample])
        wrong_mean = float(wrong.mean())
        beat_frac = float((transfer > wrong).mean())
        wrong_margin = transfer - wrong_mean

        wm_boot = []
        wf3 = wrong_sample[:3]
        for _ in range(min(n_boot, 300)):
            ip = rng.integers(0, n_pos, n_pos); ineg = rng.integers(0, n_neg, n_neg)
            lb = np.array([1]*n_pos + [0]*n_neg)
            t = safe_auroc(lb, np.concatenate([A[ip], Ng[ineg]]))
            wv = np.mean([safe_auroc(lb, np.concatenate([
                score_ids(gflat, goff, [pos_ids[j] for j in ip], mapped[int(wf)]),
                score_ids(gflat, goff, [neg_f[j] for j in ineg], mapped[int(wf)])]))
                for wf in wf3])
            wm_boot.append(t - wv)
        wm_lo = float(np.percentile(wm_boot, 2.5))

        probe_auc, probe_perm = gemma_probe(gflat, goff, pos_ids, neg_f, rng)

        gate1 = (transfer > near_median + MARGIN_THRESH) and (beat_frac >= WRONG_BEAT_FRAC)
        high_auroc_clean = (tlo > 0.5) and (transfer >= TRANSFER_CLEAN) and (wm_lo > 0)
        strict_transfer = high_auroc_clean and gate1
        if transfer < TRANSFER_CLEAN and probe_auc >= PROBE_HIGH:
            decomp = "map_failed"
        elif transfer < TRANSFER_CLEAN and probe_auc < PROBE_MIN:
            decomp = "not_representable"
        else:
            decomp = "transfers" if gate1 else "ambiguous"

        results.append({
            "fi": fi, "label": label[:26], "skip": False,
            "n_pos": n_pos, "n_neg": n_neg,
            "transfer_auroc": round(transfer, 4), "transfer_t5": round(transfer_t5, 4),
            "transfer_ci_lo": round(tlo, 4), "transfer_ci_hi": round(thi, 4),
            "random_auroc": round(rand_auc, 4),
            "near_miss_median": round(near_median, 4),
            "wrong_feat_mean": round(wrong_mean, 4), "wrong_margin": round(wrong_margin, 4),
            "wrong_margin_ci_lo": round(wm_lo, 4), "beat_wrong_frac": round(beat_frac, 3),
            "ws_ceiling": round(ws_ceiling, 4),
            "gemma_probe": round(probe_auc, 4), "gemma_probe_perm": round(probe_perm, 4),
            "gate1_pass": bool(gate1),
            "high_auroc_clean": bool(high_auroc_clean),
            "strict_transfer": bool(strict_transfer),
            # Kept for downstream scripts; now means strict, identity-specific transfer.
            "clean_transfer": bool(strict_transfer),
            "decomp": decomp,
            "cov_ws_ceiling": round(ws_ceiling, 4), "cov_gemma_probe": round(probe_auc, 4),
            "cov_mapped_norm": round(mapped_norm[fi], 4),
            "cov_firing_density": round(float((ws[:, col] > 0).mean()), 5),
            "cov_dec_norm": round(float(np.linalg.norm(W_dec_all[fi])), 4),
            "cov_near_cos": round(float(np.sort(dec_cos[col])[::-1][1]), 4),
        })
    return results


def report(results, out_dir):
    scored = [r for r in results if not r.get("skip")]
    skipped = [r for r in results if r.get("skip")]
    print("\n" + "=" * 122)
    print(f"{'fi':>6} {'label':<26} {'nP':>3} {'nN':>3} {'transf':>6} {'CIlo':>5} "
          f"{'rand':>5} {'nearM':>5} {'wrong':>5} {'wMrg':>5} {'beat':>4} "
          f"{'wsC':>5} {'probe':>5} {'g1':>2} {'hi':>2} {'str':>3} {'decomp':>16}")
    print("-" * 122)
    for r in sorted(scored, key=lambda x: -x["transfer_auroc"]):
        print(f"{r['fi']:>6} {r['label']:<26} {r['n_pos']:>3} {r['n_neg']:>3} "
              f"{r['transfer_auroc']:>6.3f} {r['transfer_ci_lo']:>5.2f} "
              f"{r['random_auroc']:>5.2f} {r['near_miss_median']:>5.2f} "
              f"{r['wrong_feat_mean']:>5.2f} {r['wrong_margin']:>5.2f} "
              f"{r['beat_wrong_frac']:>4.2f} {r['ws_ceiling']:>5.2f} "
              f"{r['gemma_probe']:>5.2f} {'Y' if r['gate1_pass'] else '.':>2} "
              f"{'Y' if r.get('high_auroc_clean') else '.':>2} "
              f"{'Y' if r['clean_transfer'] else '.':>3} {r['decomp']:>16}")
    if skipped:
        print(f"\n  skipped {len(skipped)}: " + ", ".join(str(r['fi']) for r in skipped))

    print("\n" + "=" * 122 + "\nSUMMARY")
    if not scored: print("  none scored"); return
    from collections import Counter
    tr = np.array([r["transfer_auroc"] for r in scored])
    rnd = np.array([r["random_auroc"] for r in scored])
    n_g1 = sum(r["gate1_pass"] for r in scored)
    n_high = sum(r.get("high_auroc_clean", False) for r in scored)
    n_clean = sum(r["clean_transfer"] for r in scored)
    dc = Counter(r["decomp"] for r in scored)
    n_repr = sum(r["gemma_probe"] >= PROBE_HIGH for r in scored)
    clean_cond = sum(r["clean_transfer"] for r in scored if r["gemma_probe"] >= PROBE_HIGH)
    high_cond = sum(r.get("high_auroc_clean", False) for r in scored if r["gemma_probe"] >= PROBE_HIGH)
    print(f"  scored {len(scored)}  skipped {len(skipped)}")
    print(f"  random floor mean = {rnd.mean():.4f}  (should be ~0.50)")
    print(f"  Gate 1 pass (>near_miss+0.10 AND beat>=9/10 wrong): "
          f"{n_g1}/{len(scored)} ({100*n_g1/len(scored):.0f}%)")
    print(f"  High-AUROC clean (CI_lo>0.5 AND >=0.70 AND wrong_margin_CI>0): "
          f"{n_high}/{len(scored)} ({100*n_high/len(scored):.0f}%)")
    print(f"  STRICT transfer (high-AUROC clean AND Gate 1 identity-specific): "
          f"{n_clean}/{len(scored)} ({100*n_clean/len(scored):.0f}%)  <- HEADLINE (§13)")
    print(f"  high-AUROC clean | Gemma-probe>=0.80: {high_cond}/{n_repr} "
          f"({100*high_cond/max(1,n_repr):.0f}%)")
    print(f"  strict transfer | Gemma-probe>=0.80: {clean_cond}/{n_repr} "
          f"({100*clean_cond/max(1,n_repr):.0f}%)  <- conditional headline")
    print(f"  decomposition: {dict(dc)}")
    print(f"  transfer mean={tr.mean():.3f} median={np.median(tr):.3f}")
    print("\n  'map_failed' = Gemma represents it (probe>=0.80) but mapped dir missed")
    print("  'not_representable' = Gemma doesn't linearly encode at L41 (not a map fault)")

    (out_dir / "cofiring_results.json").write_text(json.dumps(results, indent=2))
    import csv
    cov_keys = [k for k in scored[0] if k.startswith("cov_")]
    with open(out_dir / "covariates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fi", "transfer_auroc", "gate1_pass", "high_auroc_clean",
                    "strict_transfer", "clean_transfer"] + cov_keys)
        for r in scored:
            w.writerow([r["fi"], r["transfer_auroc"], int(r["gate1_pass"]),
                        int(r.get("high_auroc_clean", False)),
                        int(r.get("strict_transfer", r["clean_transfer"])),
                        int(r["clean_transfer"])] + [r[k] for k in cov_keys])
    print(f"\n  results -> {out_dir}/cofiring_results.json")
    print(f"  covariates -> {out_dir}/covariates.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/content/exp3_acts")
    ap.add_argument("--map-dir", default=None)
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    map_dir = Path(args.map_dir) if args.map_dir else out
    t0 = time.time()
    feats, feat_keys, feat_pos_ids, neg_ids, sent_list = build_universe(out)
    ws, feat_keys, qflat, qlen = run_qwen_pass(sent_list, feat_keys, out, args.force_recompute)
    gflat, glen = run_gemma_pass(sent_list, out, args.force_recompute)
    results = run_scoring(feats, feat_keys, feat_pos_ids, neg_ids, ws,
                          qflat, qlen, gflat, glen, out, map_dir, args.n_boot)
    report(results, out)
    print(f"\n  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
