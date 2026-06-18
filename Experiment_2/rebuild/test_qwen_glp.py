"""CPU self-test for Qwen GLP helpers.

Run:  .venv/bin/python test_qwen_glp.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import qwen_glp as Q


def _load_steer_module():
    p = ROOT / "scripts" / "16_glp_steer_qwen.py"
    spec = importlib.util.spec_from_file_location("glp_steer_qwen16", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    check(Q.require_qwen("qwen") == Q.QWEN_GLP.slug, "qwen alias accepted")
    check(Q.require_qwen("qwen2.5-7b") == Q.QWEN_GLP.slug, "qwen slug accepted")
    try:
        Q.require_qwen("gemma")
        fails.append("gemma should be rejected for Qwen GLP")
    except ValueError:
        pass

    check(Q.QWEN_GLP.hf_name == "Qwen/Qwen2.5-7B-Instruct", "hf name")
    check(Q.QWEN_GLP.hook_layer == 20, "hook layer")
    check(Q.QWEN_GLP.hidden_state_index == 21, "hidden-state index")
    check(Q.QWEN_GLP.hidden_size == 3584, "hidden size")
    check(Q.QWEN_GLP.layer_name == "model.layers.20", "trace layer name")

    dpath = Q.qwen_direction_path(mkdir=False)
    check(dpath.name == "05_inject_matrix__qwen2.5-7b__all__directions.npz", dpath.name)
    apath = Q.qwen_anchor_path(mkdir=False)
    check(apath.name == "03_extract_for_battery__qwen2.5-7b__anchor.npz", apath.name)
    check(Q.qwen_glp_dataset_dir().as_posix().endswith("cache/glp/qwen2_5_7b-layer20-static"), "dataset dir")

    good_config = {
        "glp_kwargs": {
            "denoiser_config": {"d_input": 3584},
            "tracedict_config": {"layer_prefix": "model.layers", "layers": [20], "retain": "output"},
        }
    }
    Q.validate_qwen_glp_config(good_config)

    bad_configs = [
        {"glp_kwargs": {"denoiser_config": {"d_input": 4096}, "tracedict_config": {"layer_prefix": "model.layers", "layers": [20], "retain": "output"}}},
        {"glp_kwargs": {"denoiser_config": {"d_input": 3584}, "tracedict_config": {"layer_prefix": "model.layers", "layers": [15], "retain": "output"}}},
        {"glp_kwargs": {"denoiser_config": {"d_input": 3584}, "tracedict_config": {"layer_prefix": "language_model.layers", "layers": [20], "retain": "output"}}},
    ]
    for cfg in bad_configs:
        try:
            Q.validate_qwen_glp_config(cfg)
            fails.append(f"bad config should fail: {cfg}")
        except ValueError:
            pass

    mod = _load_steer_module()
    x = torch.zeros(2, 3, 4)
    v = torch.ones(4)
    seen = {}

    def postprocess(t):
        seen["shape"] = tuple(t.shape)
        return t + 2.0

    hook = mod._make_hook(beta=0.5, v_t=v, postprocess_fn=postprocess)
    y = hook(None, None, x)
    check(tuple(y.shape) == (2, 3, 4), "hook output shape")
    check(seen.get("shape") == (2, 1, 4), f"postprocess shape {seen.get('shape')}")
    check(torch.allclose(y[:, :-1, :], torch.zeros(2, 2, 4)), "non-last tokens unchanged")
    check(torch.allclose(y[:, -1, :], torch.full((2, 4), 2.5)), "last token edited and postprocessed")

    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED — Qwen GLP constants, config guards, paths, and hook shape behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
