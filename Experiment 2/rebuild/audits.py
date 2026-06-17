"""audits.py — Gate-1 pre-extraction audits (CPU, numpy only).

Three cheap checks that decide whether a concept's pairs are fair BEFORE any GPU
work, plus the natural-cosine report used to set doses fairly (spec §1):

  length_balance  present vs absent length gap (pct_gap < 25% to be clean)
  lexical_leak    giveaway tokens skewed to one polarity the NLA could echo
  cosine_report   natural cos(v, positives/negatives/anchors); anchors must not
                  already sit above the detection threshold (< ~5%)

Operates on plain lists / numpy arrays (no pandas, no tokenizer) so it runs in
the dev venv. Token-exact length checks happen later with the real tokenizer at
extraction; here we use a caller-supplied length (chars or whitespace words).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

_EPS = 1e-12
_WORD = re.compile(r"[a-z0-9]+")


def _unit(v):
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


def length_balance(present_lengths, absent_lengths, *, max_pct_gap=25.0) -> dict:
    """Mean-length gap between the present and absent halves of a concept's pairs."""
    p = float(np.mean(present_lengths))
    a = float(np.mean(absent_lengths))
    abs_gap = abs(p - a)
    pct_gap = 100.0 * abs_gap / max(p, a, _EPS)
    return {
        "present_mean": round(p, 2),
        "absent_mean": round(a, 2),
        "abs_gap": round(abs_gap, 2),
        "pct_gap": round(pct_gap, 2),
        "max_pct_gap": max_pct_gap,
        "ok": bool(pct_gap < max_pct_gap),
    }


def _df(texts) -> Counter:
    """Document frequency of word tokens across a list of texts."""
    c: Counter = Counter()
    for t in texts:
        for w in set(_WORD.findall(str(t).lower())):
            c[w] += 1
    return c


def lexical_leak(present_texts, absent_texts, *, min_skew=0.8, min_df=0.30,
                 min_count=3, top_k=25) -> dict:
    """Tokens whose document-frequency is strongly skewed to one polarity.

    A token is a 'leak' if it appears in >= min_df of one side, the side-skew
    |p-a|/(p+a) >= min_skew, and it occurs >= min_count times overall. These are
    exactly the words the AV could echo to fake a concept-specific detection.
    """
    P, A = len(present_texts), len(absent_texts)
    pc, ac = _df(present_texts), _df(absent_texts)
    leaks = []
    for w in set(pc) | set(ac):
        tot = pc[w] + ac[w]
        if tot < min_count:
            continue
        p = pc[w] / P if P else 0.0
        a = ac[w] / A if A else 0.0
        denom = p + a
        if denom <= 0:
            continue
        skew = abs(p - a) / denom
        if skew >= min_skew and max(p, a) >= min_df:
            leaks.append({"token": w, "present_df": round(p, 3),
                          "absent_df": round(a, 3), "skew": round(skew, 3)})
    leaks.sort(key=lambda d: (-d["skew"], -max(d["present_df"], d["absent_df"])))
    return {"n_leaks": len(leaks), "leaks": leaks[:top_k], "ok": len(leaks) == 0}


def cosine_report(v, *, positives, negatives, anchors, threshold=None,
                  max_pct_anchors_above=5.0) -> dict:
    """Natural cos(v, .) for positives / negatives / anchors (spec §1).

    threshold defaults to the 10th percentile of positive cosines — the lower
    edge of the natural positive range, i.e. the level at which an activation
    starts to look like a positive. If too many anchors already exceed it, the
    anchors are contaminated and should be resampled.
    """
    vv = _unit(v)

    def cos(M):
        M = np.asarray(M, dtype=float)
        if M.ndim == 1:
            M = M[None, :]
        nrm = np.linalg.norm(M, axis=1)
        return (M @ vv) / np.clip(nrm, _EPS, None)

    cp, cn, ca = cos(positives), cos(negatives), cos(anchors)
    pos_med = float(np.median(cp))
    neg_med = float(np.median(cn))
    anc_base = float(np.median(ca))
    if threshold is None:
        threshold = float(np.percentile(cp, 10))
    pct_above = 100.0 * float(np.mean(ca > threshold))
    return {
        "positive_median_cos": round(pos_med, 4),
        "negative_median_cos": round(neg_med, 4),
        "anchor_baseline_cos": round(anc_base, 4),
        "threshold": round(float(threshold), 4),
        "pct_anchors_above_threshold": round(pct_above, 2),
        "max_pct_anchors_above": max_pct_anchors_above,
        "ok": bool(pct_above < max_pct_anchors_above),
    }
