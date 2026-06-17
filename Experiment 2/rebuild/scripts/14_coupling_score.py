#!/usr/bin/env python
"""14_coupling_score.py — output-coupling + salience instrument (the salience-confound fix).

THE QUESTION (external-review #1): is "the NLA reads output-coupled cognition" actually about COUPLING,
or is it confounded with SALIENCE (refusal is a large/dominant activation; corrigibility is subtle)?
This measures, per concept, coupling and salience SEPARATELY, then runs the GO/NO-GO de-risk:
does coupling predict the committed Gate-3 NLA-read CONTROLLING for salience, with the off-diagonals behaving?

Three metrics per concept (on the 7 known concepts only — this is the de-risk, not new-vector onboarding):
  salience          = ‖mean(h|present) − mean(h|absent)‖ / mean‖h‖           (natural magnitude)
  logit_lens        = top-k prob mass of softmax(unembed(norm(v)))           (cheap coupling proxy + top tokens)
  behavioral        = Δ(concept expressed in the CONTINUATION) when steering β·v at the read layer vs not
                      — the CAUSAL output-effect of a FIXED dose => inherently salience-controlled (the real measure)

De-risk: NLA_read ~ behavioral controlling salience (partial corr) + the coupling×salience off-diagonals.
  behavioral beats salience as predictor => output-coupling LAW is identified => onboard new vectors (Phase 1).
  salience wins                          => thesis is salience/training-coverage => reframe before the organism.

Depends on `03` cache (run 03 first to re-extract the 7 concepts) + concept_pairs.parquet (anchor prompts).
Reuses: 03 MODELS/tokenization, injection.dom_dir + exact_cosine_inject, 07's SCORERS, flags.explanation_text.

Run (box, after 03):  HF_TOKEN=... python "Experiment 2/rebuild/scripts/14_coupling_score.py" --model gemma
CPU self-test (no model):                python ".../14_coupling_score.py" --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from paths import CACHE, RESULTS, WORKSPACE, cache_path, data_path, model_slug, stage_of  # noqa: E402
from injection import dom_dir, exact_cosine_inject  # noqa: E402

STAGE = stage_of(__file__)
EXTRACT_STAGE = "03_extract_for_battery"
MODELS = {
    "gemma3-27b": dict(hf="google/gemma-3-27b-it", layer=42, n_layers=62, d=5376, gated=True),
    "qwen2.5-7b": dict(hf="Qwen/Qwen2.5-7B-Instruct", layer=21, n_layers=28, d=3584, gated=False),
}
CONCEPTS = ["refusal", "neg_sentiment", "sycophancy", "corrigibility",
            "truth_value", "harmful_topic_benign", "eval_framing_matched"]
# committed Gate-3 NLA-read = judge==2 present rate (FINDINGS, re-derived from 07_*__real.parquet) — NOT probe AUROC
NLA_READ = {
    "gemma3-27b": dict(refusal=0.92, neg_sentiment=0.26, sycophancy=0.11, corrigibility=0.00,
                       truth_value=0.71, harmful_topic_benign=0.03, eval_framing_matched=0.00),
    "qwen2.5-7b": dict(refusal=1.00, neg_sentiment=0.24, sycophancy=0.36, corrigibility=0.03,
                       truth_value=0.87, harmful_topic_benign=0.00, eval_framing_matched=0.00),
}
# concept -> the 07 scorer key (eval_framing_matched is scored as eval_awareness)
SCORER_KEY = {c: ("eval_awareness" if c == "eval_framing_matched" else c) for c in CONCEPTS}
DOSES = [float(x) for x in os.environ.get("DOSES", "0.55").split(",")]
N_ANCHORS = int(os.environ.get("N_ANCHORS", "20"))
MAXNEW = int(os.environ.get("MAXNEW", "48"))


def salience(X: np.ndarray, y: np.ndarray) -> float:
    X = X.astype(np.float64)
    dmean = X[y == 1].mean(0) - X[y == 0].mean(0)
    return float(np.linalg.norm(dmean) / (np.linalg.norm(X, axis=1).mean() + 1e-9))


# ── the de-risk analysis (pure numpy → CPU self-testable without a model) ──────
def analyze(rows: list[dict]) -> dict:
    """rows: [{concept, salience, behavioral, logit_lens, nla_read}]. Does behavioral coupling predict
    NLA-read better than salience does? Partial corr of behavioral~nla controlling salience is the key number."""
    sal = np.array([r["salience"] for r in rows], float)
    beh = np.array([r["behavioral"] for r in rows], float)
    nla = np.array([r["nla_read"] for r in rows], float)

    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def resid(x, z):  # residualize x on [1, z]
        Z = np.c_[np.ones_like(z), z]
        return x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]

    pc_beh = corr(resid(beh, sal), resid(nla, sal))   # behavioral ⟂ salience, vs nla ⟂ salience
    pc_sal = corr(resid(sal, beh), resid(nla, beh))   # salience ⟂ behavioral, vs nla ⟂ behavioral
    sal_med, beh_med = np.median(sal), np.median(beh)
    off = {r["concept"]: ("coupled_low_salience" if r["behavioral"] > beh_med and r["salience"] < sal_med
                          else "salient_uncoupled" if r["behavioral"] < beh_med and r["salience"] > sal_med
                          else "diagonal")
           for r in rows}
    verdict = ("COUPLING-IDENTIFIED" if (pc_beh == pc_beh and pc_beh >= 0.5 and (pc_sal != pc_sal or pc_beh > pc_sal))
               else "SALIENCE-CONFOUNDED" if (pc_sal == pc_sal and pc_sal >= 0.5 and pc_sal >= (pc_beh if pc_beh==pc_beh else -1))
               else "INCONCLUSIVE")
    return dict(corr_behavioral_nla=round(corr(beh, nla), 3), corr_salience_nla=round(corr(sal, nla), 3),
                partial_corr_behavioral_ctrl_salience=round(pc_beh, 3),
                partial_corr_salience_ctrl_behavioral=round(pc_sal, 3),
                off_diagonals={k: v for k, v in off.items() if v != "diagonal"}, verdict=verdict)


def _selftest() -> int:
    # regime 1: coupling drives nla (salience is a red herring) -> COUPLING-IDENTIFIED
    rng = np.random.default_rng(0)
    beh = rng.random(7); sal = rng.random(7); nla = 0.9 * beh + 0.02 * rng.standard_normal(7)
    r1 = analyze([dict(concept=f"c{i}", salience=sal[i], behavioral=beh[i], logit_lens=0, nla_read=nla[i]) for i in range(7)])
    # regime 2: salience drives nla, behavioral independent -> SALIENCE-CONFOUNDED
    nla2 = 0.9 * sal + 0.02 * rng.standard_normal(7)
    r2 = analyze([dict(concept=f"c{i}", salience=sal[i], behavioral=beh[i], logit_lens=0, nla_read=nla2[i]) for i in range(7)])
    print("regime COUPLING:", r1["verdict"], r1["partial_corr_behavioral_ctrl_salience"])
    print("regime SALIENCE:", r2["verdict"], r2["partial_corr_salience_ctrl_behavioral"])
    ok = r1["verdict"] == "COUPLING-IDENTIFIED" and r2["verdict"] == "SALIENCE-CONFOUNDED"
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _load_scorers():
    p = HERE / "07_score_matrix.py"
    spec = importlib.util.spec_from_file_location("score_matrix07", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    flags = importlib.util.spec_from_file_location("flags14", HERE.parent / "flags.py")
    fmod = importlib.util.module_from_spec(flags); flags.loader.exec_module(fmod)
    return mod.SCORERS, fmod.explanation_text


def run(model_key: str) -> int:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd

    m = MODELS[model_key]
    IDX = m["layer"]              # hidden_states index used everywhere in the repo
    HOOK_LAYER = IDX - 1          # the decoder layer whose OUTPUT == hidden_states[IDX]  (gemma 41, qwen 20)
    SCORERS, explanation_text = _load_scorers()

    token = os.environ.get("HF_TOKEN") if m["gated"] else None
    if token and not token.isascii():
        sys.exit("FAIL: HF_TOKEN non-ASCII (the '…' placeholder?).")
    print(f"loading {m['hf']} ... (hook layer {HOOK_LAYER} -> hidden_states[{IDX}])")
    tok = AutoTokenizer.from_pretrained(m["hf"], token=token)
    model = AutoModelForCausalLM.from_pretrained(
        m["hf"], token=token, torch_dtype=torch.bfloat16, device_map="auto").eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = model.device

    # per-model internals (Gemma-3 is a multimodal wrapper -> everything under .language_model)
    if model_key.startswith("gemma"):
        lm = model.language_model
        norm_mod, W, layers = lm.model.norm, lm.lm_head.weight, lm.model.layers
    else:
        norm_mod, W, layers = model.model.norm, model.lm_head.weight, model.model.layers
    assert len(layers) == m["n_layers"], f"{len(layers)} layers != {m['n_layers']} — wrong module path for {model_key}"

    def render(user_text: str) -> torch.Tensor:
        rendered = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                           tokenize=False, add_generation_prompt=True)
        return tok(rendered, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)

    def read_layer_act(ids: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states[IDX][0, -1, :]
        return hs.float().cpu().numpy()

    def gen(ids: torch.Tensor, beta: float = 0.0, v_t=None) -> str:
        handle = None
        if beta and v_t is not None:
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone(); h[:, -1, :] = h[:, -1, :] + beta * v_t
                return (h,) + out[1:] if isinstance(out, tuple) else h
            handle = layers[HOOK_LAYER].register_forward_hook(hook)
        try:
            with torch.no_grad():
                o = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=pad_id)
        finally:
            if handle:
                handle.remove()
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()

    # neutral anchor prompts to steer on
    df = pd.read_parquet(data_path("concept_pairs.parquet", mkdir=False))
    anchors = df[df.concept == "anchor"].text.tolist()[:N_ANCHORS]
    if len(anchors) < 5:
        sys.exit("FAIL: <5 anchor prompts in concept_pairs.parquet.")
    anchor_ids = [render(a) for a in anchors]
    anchor_h0 = [read_layer_act(ids) for ids in anchor_ids]          # read-layer act per anchor (for dose calibration)
    base_gen = [gen(ids) for ids in tqdm(anchor_ids, desc="baseline")]  # concept-independent baseline continuations

    rows, examples = [], []
    for c in CONCEPTS:
        z = np.load(cache_path(EXTRACT_STAGE, model_key, concept=c, ext="npz"))
        X, y = z["X"].astype(np.float64), z["y"].astype(int)
        v = dom_dir(X, y)                                            # unit direction
        sal = salience(X, y)
        # logit-lens (cheap proxy): unembed(norm(v)) -> token concentration + top tokens
        with torch.no_grad():
            vt = torch.tensor(v, dtype=W.dtype, device=dev)
            logits = (norm_mod(vt.unsqueeze(0)).squeeze(0) @ W.t()).float()
            probs = torch.softmax(logits, -1)
            topk = torch.topk(probs, 10)
            ll_mass = float(probs.topk(20).values.sum())             # top-20 prob mass = concentration
            top_tokens = [tok.decode([int(i)]) for i in topk.indices.tolist()]
        v_t = torch.tensor(v, dtype=torch.bfloat16, device=dev)
        sc = SCORERS[SCORER_KEY[c]]
        best_beh, best_dose, steered_scores = -1.0, None, None
        for dose in DOSES:
            betas = [float(exact_cosine_inject(h0[None], v, dose)[1][0]) for h0 in anchor_h0]
            st_gen = [gen(ids, b, v_t) for ids, b in zip(anchor_ids, betas)]
            st_sc = np.array([sc(explanation_text(g))[0] for g in st_gen])
            beh = float((st_sc == 2).mean())
            if beh > best_beh:
                best_beh, best_dose, steered_scores = beh, dose, (st_gen, st_sc)
        base_sc = np.array([sc(explanation_text(g))[0] for g in base_gen])
        behavioral = best_beh - float((base_sc == 2).mean())        # causal Δ in concept expression
        rows.append(dict(concept=c, salience=round(sal, 4), logit_lens=round(ll_mass, 4),
                         behavioral=round(behavioral, 3), best_dose=best_dose,
                         steered_rate=round(best_beh, 3), baseline_rate=round(float((base_sc == 2).mean()), 3),
                         nla_read=NLA_READ[model_key][c], top_tokens=" ".join(top_tokens)))
        print(f"  {c:<22} sal={sal:.3f} ll={ll_mass:.3f} behavioral={behavioral:+.3f} "
              f"(steer {best_beh:.2f}@{best_dose} vs base {float((base_sc==2).mean()):.2f})  NLA={NLA_READ[model_key][c]}")
        for g, s in zip(steered_scores[0][:5], steered_scores[1][:5]):
            examples.append(dict(concept=c, steered_score=int(s), steered_continuation=g))

    res = dict(model=model_key, hook_layer=HOOK_LAYER, doses=DOSES, n_anchors=len(anchors), rows=rows,
               analysis=analyze(rows))
    rp = RESULTS / "gate4"; rp.mkdir(parents=True, exist_ok=True)
    (rp / f"{STAGE}__{model_key}.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(rows).drop(columns=["top_tokens"]).to_csv(rp / f"{STAGE}__{model_key}.csv", index=False)
    wp = WORKSPACE / "coupling"; wp.mkdir(parents=True, exist_ok=True)
    with open(wp / f"{STAGE}_examples__{model_key}.jsonl", "w") as fh:
        for e in examples:
            fh.write(json.dumps(e) + "\n")

    a = res["analysis"]
    print(f"\n==== COUPLING vs SALIENCE ({model_key}) ====")
    print(f"  corr(behavioral, NLA)={a['corr_behavioral_nla']}  corr(salience, NLA)={a['corr_salience_nla']}")
    print(f"  PARTIAL corr behavioral|salience = {a['partial_corr_behavioral_ctrl_salience']}   "
          f"salience|behavioral = {a['partial_corr_salience_ctrl_behavioral']}")
    print(f"  off-diagonals: {a['off_diagonals']}")
    print(f"  VERDICT: {a['verdict']}")
    print("\n  SANITY before trusting nulls: refusal behavioral must be strongly +ve (steering refusal -> refusals, Exp-1 0->84%).")
    print(f"  wrote {rp / (STAGE + '__' + model_key + '.csv')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", help="gemma | qwen")
    ap.add_argument("--selftest", action="store_true", help="CPU: validate the de-risk analysis logic, no model")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    return run(model_slug(a.model))


if __name__ == "__main__":
    raise SystemExit(main())
