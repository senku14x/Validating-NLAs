#!/usr/bin/env python
"""16_glp_steer_qwen.py — Qwen-only online steering with optional GLP projection.

Uses rebuild's diff-of-means directions, hooks Qwen block 20, and either:
  raw: h[:, -1, :] = h[:, -1, :] + beta*v
  glp: h[:, -1, :] = GLP(h[:, -1, :] + beta*v)

`glp` mode requires a Qwen-trained GLP checkpoint. Llama GLP weights are rejected
because the hidden dimension/layer statistics do not match Qwen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Callable

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REBUILD_ROOT = HERE.parent
EXPERIMENT_2 = REBUILD_ROOT.parent
GLP_ROOT = EXPERIMENT_2 / "generative_latent_prior"
sys.path.insert(0, str(REBUILD_ROOT))
sys.path.insert(0, str(GLP_ROOT))

from injection import dom_dir, exact_cosine_inject, inject_stats  # noqa: E402
from paths import RESULTS, WORKSPACE, cache_path, data_path  # noqa: E402
from qwen_glp import (  # noqa: E402
    DIRECTION_STAGE,
    EXTRACT_STAGE,
    GLP_STEER_STAGE,
    QWEN_GLP,
    qwen_anchor_path,
    qwen_direction_path,
    validate_qwen_glp_checkpoint,
    validate_qwen_model_shape,
)

DEFAULT_CONCEPTS = [
    "refusal",
    "neg_sentiment",
    "sycophancy",
    "corrigibility",
    "truth_value",
    "harmful_topic_benign",
    "eval_framing_matched",
]
SCORER_KEY = {c: ("eval_awareness" if c == "eval_framing_matched" else c) for c in DEFAULT_CONCEPTS}


def _load_scorers():
    p = HERE / "07_score_matrix.py"
    spec = importlib.util.spec_from_file_location("score_matrix07", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    flags = importlib.util.spec_from_file_location("flags16", REBUILD_ROOT / "flags.py")
    fmod = importlib.util.module_from_spec(flags)
    flags.loader.exec_module(fmod)
    return mod.SCORERS, fmod.explanation_text


def _load_directions(concepts: list[str]) -> dict[str, np.ndarray]:
    path = qwen_direction_path(mkdir=False)
    if path.exists():
        z = np.load(path)
        return {c: z[c].astype(np.float64) for c in concepts if c in z.files}

    print(f"direction snapshot missing at {path}; rebuilding from 03 caches")
    dirs = {}
    for c in concepts:
        p = cache_path(EXTRACT_STAGE, QWEN_GLP.slug, concept=c, ext="npz", mkdir=False)
        if not p.exists():
            print(f"  [skip] {c}: missing {p.name}")
            continue
        z = np.load(p)
        if len(np.unique(z["y"])) < 2:
            print(f"  [skip] {c}: single-class y")
            continue
        dirs[c] = dom_dir(z["X"], z["y"])
    return dirs


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _base_model(model):
    return model.language_model if hasattr(model, "language_model") else model.model


def _make_hook(beta: float, v_t: torch.Tensor, postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        v = v_t.to(device=h.device, dtype=h.dtype)
        edit = h[:, [-1], :] + beta * v[None, None, :]
        if postprocess_fn is not None:
            edit = postprocess_fn(edit)
        h[:, [-1], :] = edit.to(device=h.device, dtype=h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    return hook


@torch.no_grad()
def run(args) -> int:
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    modes = ["raw", "glp"] if args.mode == "both" else [args.mode]
    if "glp" in modes:
        if not args.glp_weights:
            raise SystemExit("FAIL: --glp-weights is required for --mode glp/both")
        validate_qwen_glp_checkpoint(args.glp_weights, args.glp_checkpoint)
        from glp.denoiser import load_glp
        from glp.script_steer import postprocess_on_manifold_wrapper

        glp_model = load_glp(args.glp_weights, device=args.glp_device, checkpoint=args.glp_checkpoint).eval()
        glp_postprocess = postprocess_on_manifold_wrapper(glp_model, u=args.glp_u, num_timesteps=args.glp_timesteps)
    else:
        glp_postprocess = None

    concepts = _parse_csv_strings(args.concepts) if args.concepts else DEFAULT_CONCEPTS
    doses = _parse_csv_floats(args.doses)
    dirs = _load_directions(concepts)
    missing = [c for c in concepts if c not in dirs]
    if missing:
        print(f"WARNING: missing directions skipped: {missing}")
    if not dirs:
        raise SystemExit(f"FAIL: no Qwen directions found. Run {DIRECTION_STAGE} or keep 03 caches available.")

    anchor_path = qwen_anchor_path(mkdir=False)
    if not anchor_path.exists():
        raise SystemExit(f"FAIL: missing Qwen anchors at {anchor_path}; run {EXTRACT_STAGE} --model qwen first.")
    anchor_h0 = np.load(anchor_path)["X"].astype(np.float64)[: args.n_anchors]

    import pandas as pd

    df = pd.read_parquet(data_path("concept_pairs.parquet", mkdir=False))
    anchors = df[df.concept == "anchor"].text.dropna().astype(str).tolist()[: args.n_anchors]
    if len(anchors) != len(anchor_h0):
        raise SystemExit(f"FAIL: anchor text/cache mismatch ({len(anchors)} texts, {len(anchor_h0)} activations)")

    print(f"loading {QWEN_GLP.hf_name} ... (hook layer {QWEN_GLP.hook_layer})")
    tok = AutoTokenizer.from_pretrained(QWEN_GLP.hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_GLP.hf_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    validate_qwen_model_shape(model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    base = _base_model(model)
    layers = base.layers
    assert len(layers) == QWEN_GLP.n_layers, f"{len(layers)} layers != {QWEN_GLP.n_layers}"

    def render(user_text: str) -> torch.Tensor:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tok(rendered, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

    def generate(ids: torch.Tensor, *, beta: float = 0.0, v_t: torch.Tensor | None = None,
                 postprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None = None) -> str:
        handle = None
        if beta and v_t is not None:
            handle = layers[QWEN_GLP.hook_layer].register_forward_hook(_make_hook(beta, v_t, postprocess_fn))
        try:
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        finally:
            if handle is not None:
                handle.remove()
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    SCORERS, explanation_text = _load_scorers()
    anchor_ids = [render(a) for a in anchors]
    baseline_gen = [generate(ids) for ids in tqdm(anchor_ids, desc="baseline")]
    baseline_scores = {
        c: np.array([SCORERS[SCORER_KEY.get(c, c)](explanation_text(g))[0] for g in baseline_gen])
        for c in dirs
        if SCORER_KEY.get(c, c) in SCORERS
    }

    rows, examples = [], []
    for concept, v in dirs.items():
        v = np.asarray(v, dtype=np.float64)
        v /= np.linalg.norm(v)
        v_t = torch.tensor(v, dtype=torch.bfloat16, device=model.device)
        scorer = SCORERS.get(SCORER_KEY.get(concept, concept))
        if scorer is None:
            print(f"  [skip scoring] no scorer for {concept}")
            continue
        for dose in doses:
            Hp_raw, betas, _ = exact_cosine_inject(anchor_h0, v, dose)
            raw_stats = inject_stats(anchor_h0, Hp_raw, v)
            for mode in modes:
                pp = glp_postprocess if mode == "glp" else None
                gens = [
                    generate(ids, beta=float(beta), v_t=v_t, postprocess_fn=pp)
                    for ids, beta in tqdm(list(zip(anchor_ids, betas)), desc=f"{concept}:{dose}:{mode}")
                ]
                scores = np.array([scorer(explanation_text(g))[0] for g in gens])
                base_scores = baseline_scores[concept]
                rows.append({
                    "concept": concept,
                    "mode": mode,
                    "dose": float(dose),
                    "n_anchors": int(len(anchor_ids)),
                    "steered_rate": round(float((scores == 2).mean()), 4),
                    "baseline_rate": round(float((base_scores == 2).mean()), 4),
                    "behavioral_delta": round(float((scores == 2).mean() - (base_scores == 2).mean()), 4),
                    "mean_beta": round(float(np.mean(betas)), 4),
                    "mean_raw_cos_h_hp": round(float(np.mean(raw_stats["cos_h_hp"])), 4),
                    "mean_raw_delta_norm_over_h": round(float(np.mean(raw_stats["delta_norm_over_h"])), 4),
                })
                for i, (prompt, gen, score) in enumerate(zip(anchors[: args.examples_per_cell], gens[: args.examples_per_cell], scores[: args.examples_per_cell])):
                    examples.append({
                        "concept": concept,
                        "mode": mode,
                        "dose": float(dose),
                        "anchor": i,
                        "score": int(score),
                        "prompt": prompt,
                        "continuation": gen,
                    })
                print(f"  {concept:<22} {mode:<3} dose={dose:.2f} "
                      f"rate={(scores == 2).mean():.2f} base={(base_scores == 2).mean():.2f}")

    out = {
        "model": QWEN_GLP.slug,
        "hf_name": QWEN_GLP.hf_name,
        "hook_layer": QWEN_GLP.hook_layer,
        "hidden_state_index": QWEN_GLP.hidden_state_index,
        "glp_weights": args.glp_weights if "glp" in modes else None,
        "glp_checkpoint": args.glp_checkpoint if "glp" in modes else None,
        "glp_u": args.glp_u if "glp" in modes else None,
        "glp_timesteps": args.glp_timesteps if "glp" in modes else None,
        "rows": rows,
    }
    rp = RESULTS / "gate4"
    rp.mkdir(parents=True, exist_ok=True)
    suffix = args.mode
    json_path = rp / f"{GLP_STEER_STAGE}__{QWEN_GLP.slug}__{suffix}.json"
    csv_path = rp / f"{GLP_STEER_STAGE}__{QWEN_GLP.slug}__{suffix}.csv"
    json_path.write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    wp = WORKSPACE / "glp_steering"
    wp.mkdir(parents=True, exist_ok=True)
    ex_path = wp / f"{GLP_STEER_STAGE}_examples__{QWEN_GLP.slug}__{suffix}.jsonl"
    with ex_path.open("w") as fh:
        for row in examples:
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {ex_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "glp", "both"], default="both")
    ap.add_argument("--glp-weights", help="local Qwen GLP checkpoint folder")
    ap.add_argument("--glp-checkpoint", default="final")
    ap.add_argument("--glp-device", default="cuda:0")
    ap.add_argument("--glp-u", type=float, default=0.5)
    ap.add_argument("--glp-timesteps", type=int, default=20)
    ap.add_argument("--concepts", help="comma-separated concepts; default is the 7 rebuild concepts")
    ap.add_argument("--doses", default="0.55")
    ap.add_argument("--n-anchors", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--examples-per-cell", type=int, default=5)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
