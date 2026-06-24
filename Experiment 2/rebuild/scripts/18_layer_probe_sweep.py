#!/usr/bin/env python
"""18_layer_probe_sweep.py — probe AUROC vs LAYER: the "did we read the wrong layer?" control.

Two arms, one script (GPU/box). Both reuse the validated battery (confounds.probe_battery)
so every per-layer number carries the same raw / length-residualized / length-only / null
five-number treatment + cluster-bootstrap CI as Gate-1.

  ARM A  NATURAL sweep — for each concept, one forward pass per prompt capturing the
         last-token activation at EVERY swept layer (the hidden states are already computed
         in a single pass; 03 just kept one). Reuses 03's behavioral filtering. Runs the
         battery per (concept, layer). A flat sub-floor profile across ALL layers turns a
         single-layer null into "absent everywhere we looked", not "maybe wrong layer".

  ARM B  STEERED positive control — online-steer the refusal DoM at the read layer (the 14
         hook: h[:,-1,:] += beta*v_hat at HOOK_LAYER = IDX-1, beta solved per-anchor for a
         target realized cosine), capture ALL layers, probe steered-anchor vs natural-anchor
         per layer. Validates the sweep is a LIVE instrument: ~chance BELOW the inject layer
         (the hook hasn't fired — Xste==Xnat there), ~1.0 AT it (CIRCULAR: we added the very
         direction the probe finds — flagged is_inject_layer), and the real result is how far
         the signal PROPAGATES ABOVE it. Lengths are identical on both arms of the pair, so
         length cannot carry the steered split.

NOT used here: 05's OFFLINE injection (h+beta*v on one cached layer). It does not propagate
across layers and is circular at the read layer, so a post-offline layer sweep is vacuous.
Online steering is the only "after injection" that means anything across depth.

Run (box, after 03 has cached the refusal direction):
  HF_TOKEN=... python "Experiment 2/rebuild/scripts/18_layer_probe_sweep.py" --model qwen --arm both
  # knobs: --stride N (subsample layers), --concepts ..., --doses 0.55, --n-anchors 30
CPU self-test (no model, no torch):
  .venv/bin/python "Experiment 2/rebuild/scripts/18_layer_probe_sweep.py" --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # rebuild root: confounds, injection, paths
sys.path.insert(0, str(HERE))          # scripts: import 03's constants/helpers
from confounds import battery_to_dict, bow_auroc, probe_battery  # noqa: E402
from injection import dom_dir, exact_cosine_inject  # noqa: E402
from paths import cache_path, data_path, model_slug, result_path, stage_of  # noqa: E402

STAGE = stage_of(__file__)
EXTRACT_STAGE = "03_extract_for_battery"

MODELS = {
    "gemma3-27b": dict(hf="google/gemma-3-27b-it", layer=42, n_layers=62, d=5376, gated=True),
    "qwen2.5-7b": dict(hf="Qwen/Qwen2.5-7B-Instruct", layer=21, n_layers=28, d=3584, gated=False),
}
# default concepts: the headline positive (refusal) + the polarity concepts + the NLA-paper
# headline that was culled at Gate 1 (the null we most want to defend across layers).
DEFAULT_CONCEPTS = ["refusal", "neg_sentiment", "sycophancy", "corrigibility",
                    "truth_value", "harmful_topic_benign", "eval_framing_matched", "eval_framing_v2"]
MAXNEW = int(os.environ.get("MAXNEW", "64"))
MIN_ROWS, MIN_PER_CLASS = 16, 4


# ─────────────────────────── battery-per-layer core (CPU, no torch) ──────────────────────
def filter_concept(concept: str, polarity, label, design, group_id):
    """Mirror 03_extract_for_battery.assemble: return (keep_mask, y, groups_builder).

    keep_mask  : bool over the concept's rows
    y          : int labels over kept rows
    groups_of  : callable(kept_group_ids)->int groups (paired -> factorize, else per-row)
    """
    polarity = np.asarray(polarity)
    label = np.asarray(label)
    pres = polarity == "present"
    if concept == "refusal":
        comp = label == "complied"
        refu = label == "refused"
        keep = (pres & refu) | (~pres & comp)        # refused-harmful vs complied-harmless
        y = pres[keep].astype(int)
    elif concept == "harmful_topic_benign":
        gid = np.asarray(group_id)
        comp = label == "complied"
        ok = {g for g in np.unique(gid) if comp[gid == g].all()}
        keep = np.array([g in ok for g in gid])
        y = pres[keep].astype(int)
    else:
        keep = np.ones(len(polarity), dtype=bool)
        y = pres.astype(int)

    paired = (design == "paired")

    def groups_of(kept_gids):
        if paired:
            import pandas as pd
            return pd.factorize(np.asarray(kept_gids))[0]
        return np.arange(len(kept_gids))

    return keep, y, groups_of


def sweep_batteries(Xc, y, lengths, groups, layers, **bkw):
    """Xc:[n,K,d] (K == len(layers)); run probe_battery per layer; list of per-layer dicts."""
    Xc = np.asarray(Xc, dtype=float)
    y = np.asarray(y).astype(int)
    out = []
    skip = len(y) < MIN_ROWS or len(np.unique(y)) < 2 or min(np.bincount(y)) < MIN_PER_CLASS
    for j, L in enumerate(layers):
        if skip:
            out.append(dict(layer_hidx=int(L), block=int(L) - 1, n=int(len(y)),
                            y1=int(y.sum()), y0=int((y == 0).sum()), battery=None, fate="DROP"))
            continue
        b = probe_battery(Xc[:, j, :], y, lengths, groups, **bkw)
        out.append(dict(layer_hidx=int(L), block=int(L) - 1, n=int(len(y)),
                        y1=int(y.sum()), y0=int((y == 0).sum()),
                        battery=battery_to_dict(b),
                        fate=("PASS" if b["represented"]
                              else "WEAK" if b["raw"].ci_lo > b["null"].ci_hi else "FAIL")))
    return out


def rows_from_sweep(model, arm, concept, layers_meta, read_idx, inject_idx=None, bow=None):
    """Flatten per-layer battery dicts to CSV-ready rows."""
    rows = []
    for r in layers_meta:
        bt = r["battery"]
        rec = dict(model=model, arm=arm, concept=concept,
                   layer_hidx=r["layer_hidx"], block=r["block"],
                   is_read_layer=(r["layer_hidx"] == read_idx),
                   is_inject_layer=(inject_idx is not None and r["layer_hidx"] == inject_idx),
                   n=r["n"], y1=r["y1"], y0=r["y0"], fate=r["fate"])
        for key in ("raw", "length_residualized", "length_only", "null"):
            a = bt[key] if bt else None
            rec[f"{key}_auroc"] = a["auroc"] if a else None
            rec[f"{key}_lo"] = a["ci_lo"] if a else None
            rec[f"{key}_hi"] = a["ci_hi"] if a else None
        rec["represented"] = bt["represented"] if bt else False
        rec["bow_auroc"] = bow            # text bar (layer-independent); None for steered arm
        rows.append(rec)
    return rows


# ───────────────────────────────── GPU extraction (box) ──────────────────────────────────
def run(model_key, arms, concepts, stride, n_anchors, dose, bkw):
    import gc
    import re

    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # behavioral refusal classifier — import 03's verbatim so labels match the paper exactly
    import importlib.util
    spec = importlib.util.spec_from_file_location("extract03", HERE / "03_extract_for_battery.py")
    ex03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex03)
    classify = ex03.classify

    m = MODELS[model_key]
    IDX = m["layer"]                 # hidden_states index (read layer): gemma 42, qwen 21
    HOOK_LAYER = IDX - 1             # decoder block whose OUTPUT == hidden_states[IDX]
    all_layers = list(range(1, m["n_layers"] + 1))          # skip hs[0] (embeddings)
    layers = sorted(set(all_layers[::stride]) | {IDX})      # always include the read layer
    print(f"{model_key}: sweeping {len(layers)} layers (stride {stride}); read=hs[{IDX}], hook=block{HOOK_LAYER}")

    token = os.environ.get("HF_TOKEN") if m["gated"] else None
    if m["gated"] and not token:
        sys.exit("FAIL: HF_TOKEN required for the gated Gemma repo.")
    tok = AutoTokenizer.from_pretrained(m["hf"], token=token)
    model = AutoModelForCausalLM.from_pretrained(
        m["hf"], token=token, torch_dtype=torch.bfloat16, device_map="auto").eval()
    nhl = (getattr(model.config, "num_hidden_layers", None)
           or model.config.text_config.num_hidden_layers)
    assert nhl == m["n_layers"], f"expected {m['n_layers']} layers, got {nhl}"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    base = model.language_model if hasattr(model, "language_model") else model.model
    dev = model.device

    def render(text):
        enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                      tokenize=False, add_generation_prompt=True)
        return tok(enc, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)

    def all_layer_acts(ids, beta=0.0, v_t=None):
        """last-token activation at every swept layer; optional steering hook at HOOK_LAYER."""
        handle = None
        if beta and v_t is not None:
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone(); h[:, -1, :] = h[:, -1, :] + beta * v_t
                return (h,) + out[1:] if isinstance(out, tuple) else h
            handle = base.layers[HOOK_LAYER].register_forward_hook(hook)
        try:
            with torch.no_grad():
                hs = model(ids, output_hidden_states=True).hidden_states
        finally:
            if handle:
                handle.remove()
        return np.stack([hs[L][0, -1, :].float().cpu().numpy() for L in layers])  # [K, d]

    def gen_then_acts(ids):
        """For behavioral concepts: greedy completion (for label) + all-layer prompt acts."""
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False,
                               output_hidden_states=True, return_dict_in_generate=True,
                               pad_token_id=pad_id)
        hs = o.hidden_states[0]   # first decode step covers the prompt
        comp = tok.decode(o.sequences[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return np.stack([hs[L][0, -1, :].float().cpu().numpy() for L in layers]), comp

    all_rows = []

    # ── ARM A: natural per-layer sweep ────────────────────────────────────────────────────
    if "A" in arms:
        df = pd.read_parquet(data_path("concept_pairs.parquet", mkdir=False)).reset_index(drop=True)
        for concept in concepts:
            sub = df[df.concept == concept].reset_index(drop=True)
            if not len(sub):
                print(f"  [skip] {concept}: no rows in concept_pairs"); continue
            needs = bool(sub.needs_behavior.iloc[0])
            X = np.zeros((len(sub), len(layers), m["d"]), dtype=np.float32)
            labels, ntok, texts = [], [], []
            for i, row in enumerate(tqdm(sub.to_dict("records"), desc=f"A:{concept}")):
                ids = render(row["text"]); ntok.append(int(ids.shape[1])); texts.append(row["text"])
                if needs:
                    acts, comp = gen_then_acts(ids); labels.append(classify(comp))
                else:
                    acts = all_layer_acts(ids); labels.append("n/a")
                X[i] = acts
            keep, y, groups_of = filter_concept(
                concept, sub.polarity.to_numpy(), np.array(labels),
                sub.design.iloc[0], sub.group_id.to_numpy())
            Xf = X[keep]
            lengths = np.asarray(ntok, dtype=float)[keep]
            groups = groups_of(sub.group_id.to_numpy()[keep])
            # BoW text bar (layer-independent): same kept rows/labels
            kept_texts = [t for t, k in zip(texts, keep) if k]
            try:
                bow = round(float(bow_auroc(kept_texts, y.tolist(),
                                            groups.tolist()).auroc), 4) if len(np.unique(y)) > 1 else None
            except Exception:
                bow = None
            sw = sweep_batteries(Xf, y, lengths, groups, layers, **bkw)
            all_rows += rows_from_sweep(model_key, "natural", concept, sw, IDX, bow=bow)
            best = max((r for r in sw if r["battery"]),
                       key=lambda r: r["battery"]["length_residualized"]["auroc"], default=None)
            if best:
                a = best["battery"]["length_residualized"]
                print(f"  {concept:22s} n={len(y)} bow={bow}  best resid AUROC={a['auroc']:.3f} "
                      f"[{a['ci_lo']:.3f},{a['ci_hi']:.3f}] @hs[{best['layer_hidx']}]"
                      f"{' (READ)' if best['layer_hidx']==IDX else ''}")

    # ── ARM B: steered positive control (refusal DoM) ──────────────────────────────────────
    if "B" in arms:
        rp = cache_path(EXTRACT_STAGE, model_key, concept="refusal", ext="npz")
        if not rp.exists():
            print(f"  [skip arm B] no refusal cache {rp.name} — run 03 first for the DoM direction")
        else:
            z = np.load(rp); v = dom_dir(z["X"], z["y"]).astype(np.float32)
            v_t = torch.tensor(v, device=dev, dtype=torch.bfloat16)
            df = pd.read_parquet(data_path("concept_pairs.parquet", mkdir=False))
            anchors = df[df.concept == "anchor"].text.tolist()[:n_anchors]
            if len(anchors) < 8:
                print("  [skip arm B] <8 anchor prompts")
            else:
                n = len(anchors)
                Xnat = np.zeros((n, len(layers), m["d"]), dtype=np.float32)
                Xste = np.zeros((n, len(layers), m["d"]), dtype=np.float32)
                ntok = []
                ji = layers.index(IDX)   # read-layer column, for per-anchor dose calibration
                for i, a in enumerate(tqdm(anchors, desc="B:anchors")):
                    ids = render(a); ntok.append(int(ids.shape[1]))
                    nat = all_layer_acts(ids); Xnat[i] = nat
                    h0 = nat[ji][None].astype(float)                 # read-layer natural act
                    beta = float(exact_cosine_inject(h0, v.astype(float), dose)[1][0])
                    Xste[i] = all_layer_acts(ids, beta=beta, v_t=v_t)
                X = np.concatenate([Xnat, Xste], 0)
                y = np.array([0] * n + [1] * n)
                lengths = np.asarray(ntok + ntok, dtype=float)        # identical per pair
                groups = np.concatenate([np.arange(n), np.arange(n)])  # anchor pairs share a group
                sw = sweep_batteries(X, y, lengths, groups, layers, **bkw)
                all_rows += rows_from_sweep(model_key, "steered_refusal", "anchor",
                                            sw, IDX, inject_idx=IDX, bow=None)
                print(f"  steered control: dose={dose}, n_anchors={n}; "
                      f"hs[<{IDX}]~chance, hs[{IDX}]=circular, hs[>{IDX}]=propagation "
                      f"(per-layer AUROC in CSV)")

    del model; gc.collect(); torch.cuda.empty_cache()

    import pandas as pd
    out_csv = result_path("gate1", STAGE, model_key, concept="layer_sweep", ext="csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nwrote {len(all_rows)} rows -> {out_csv}")
    return 0


# ──────────────────────────────────── CPU self-test ──────────────────────────────────────
def self_test():
    rng = np.random.default_rng(0)
    d, K, npair = 48, 10, 60
    layers = list(range(1, K + 1))
    fails = []

    # ARM A — concept signal present only in a layer BAND [4,7]; chance elsewhere.
    y = np.tile([1, 0], npair); groups = np.repeat(np.arange(npair), 2)
    lengths = rng.normal(40, 5, 2 * npair)
    sig = rng.standard_normal(d); sig /= np.linalg.norm(sig)
    X = rng.standard_normal((2 * npair, K, d))
    band = set(range(4, 8))
    for j, L in enumerate(layers):
        if L in band:
            X[:, j, :] += np.outer(y, sig) * 3.5
    sw = sweep_batteries(X, y, lengths, groups, layers, n_boot=400, n_null=10)
    in_band = [r["battery"]["length_residualized"]["auroc"] for r in sw if r["layer_hidx"] in band]
    out_band = [r["battery"]["length_residualized"]["auroc"] for r in sw if r["layer_hidx"] not in band]
    print(f"[A] in-band resid AUROC  min={min(in_band):.3f}  (expect high)")
    print(f"[A] out-band resid AUROC max={max(out_band):.3f}  (expect ~chance)")
    if min(in_band) < 0.85:
        fails.append("arm-A: in-band AUROC not high")
    if max(out_band) > 0.70:
        fails.append("arm-A: out-band AUROC not ~chance")

    # ARM A filtering — refusal two-pool: present->refused kept, present->complied dropped.
    pol = np.array(["present", "present", "absent", "absent"])
    lab = np.array(["refused", "complied", "complied", "refused"])
    keep, yk, _ = filter_concept("refusal", pol, lab, "two_pool", np.arange(4))
    if not (keep.tolist() == [True, False, True, False] and yk.tolist() == [1, 0]):
        fails.append(f"arm-A refusal filter wrong: keep={keep.tolist()} y={yk.tolist()}")
    else:
        print("[A] refusal two-pool filter: keep present-refused & absent-complied  OK")

    # ARM B — steered: identical below inject layer (Xste==Xnat), separable at/above.
    INJECT = 6
    n = npair
    Xnat = rng.standard_normal((n, K, d))
    Xste = Xnat.copy()
    vv = rng.standard_normal(d); vv /= np.linalg.norm(vv)
    for j, L in enumerate(layers):
        if L >= INJECT:
            Xste[:, j, :] += vv * 6.0            # injected + propagated
    Xb = np.concatenate([Xnat, Xste], 0)
    yb = np.array([0] * n + [1] * n)
    lb = np.asarray(list(rng.normal(40, 5, n)) * 2)   # identical per pair
    gb = np.concatenate([np.arange(n), np.arange(n)])
    swb = sweep_batteries(Xb, yb, lb, gb, layers, n_boot=400, n_null=10)
    below = [r["battery"]["length_residualized"]["auroc"] for r in swb if r["layer_hidx"] < INJECT]
    at_above = [r["battery"]["length_residualized"]["auroc"] for r in swb if r["layer_hidx"] >= INJECT]
    print(f"[B] below-inject AUROC max={max(below):.3f} (expect ~chance)   "
          f"at/above min={min(at_above):.3f} (expect high)")
    if max(below) > 0.70:
        fails.append("arm-B: below-inject not ~chance")
    if min(at_above) < 0.85:
        fails.append("arm-B: at/above-inject not high")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nALL CHECKS PASSED — per-layer battery loop, refusal filter, and steered "
          "propagation control behave correctly on synthetic data.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="gemma | qwen")
    ap.add_argument("--arm", default="both", choices=["A", "B", "both"],
                    help="A=natural sweep, B=steered control, both")
    ap.add_argument("--concepts", nargs="*", default=DEFAULT_CONCEPTS)
    ap.add_argument("--stride", type=int, default=1, help="sweep every Nth layer (read layer always kept)")
    ap.add_argument("--n-anchors", type=int, default=30)
    ap.add_argument("--dose", type=float, default=0.55, help="target realized cos for arm-B steer")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-null", type=int, default=25)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.model:
        ap.error("provide --model gemma|qwen (or --self-test)")
    arms = {"A": "A", "B": "B", "both": "AB"}[a.arm]
    bkw = dict(n_boot=a.n_boot, n_null=a.n_null)
    return run(model_slug(a.model), arms, a.concepts, a.stride, a.n_anchors, a.dose, bkw)


if __name__ == "__main__":
    raise SystemExit(main())
