"""Qwen-only helpers for GLP on-manifold steering.

The GLP projection has to match the model, layer, and hidden size it was
trained on. Keep those invariants centralized so GPU scripts fail before doing
expensive work with a Llama or Gemma checkpoint by accident.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import CACHE, WORKSPACE, cache_path, model_slug


@dataclass(frozen=True)
class QwenGLPSpec:
    slug: str = "qwen2.5-7b"
    alias: str = "qwen"
    hf_name: str = "Qwen/Qwen2.5-7B-Instruct"
    n_layers: int = 28
    hook_layer: int = 20
    hidden_state_index: int = 21
    hidden_size: int = 3584
    layer_prefix: str = "model.layers"
    retain: str = "output"

    @property
    def layer_name(self) -> str:
        return f"{self.layer_prefix}.{self.hook_layer}"


QWEN_GLP = QwenGLPSpec()
EXTRACT_STAGE = "03_extract_for_battery"
DIRECTION_STAGE = "05_inject_matrix"
GLP_CACHE_STAGE = "16_glp_qwen_cache"
GLP_STEER_STAGE = "16_glp_steer_qwen"


def require_qwen(model: str = QWEN_GLP.alias) -> str:
    """Normalize and require the Experiment_2 Qwen slug."""
    slug = model_slug(model)
    if slug != QWEN_GLP.slug:
        raise ValueError(f"Qwen GLP steering only supports {QWEN_GLP.slug}; got {slug}")
    return slug


def qwen_direction_path(*, mkdir: bool = False) -> Path:
    return cache_path(DIRECTION_STAGE, QWEN_GLP.slug, concept="all", variant="directions", ext="npz", mkdir=mkdir)


def qwen_anchor_path(*, mkdir: bool = False) -> Path:
    return cache_path(EXTRACT_STAGE, QWEN_GLP.slug, concept="anchor", ext="npz", mkdir=mkdir)


def qwen_glp_dataset_dir(name: str = "qwen2_5_7b-layer20-static") -> Path:
    """Default local GLP training dataset root under rebuild/cache/glp/."""
    return CACHE / "glp" / name


def qwen_glp_run_dir(name: str = "glp-qwen2_5_7b-d6-static") -> Path:
    """Default local GLP checkpoint root under rebuild/workspace/glp_runs/."""
    return WORKSPACE / "glp_runs" / name


def _get(config: Any, *keys: str, default: Any = None) -> Any:
    cur = config
    for key in keys:
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
        if cur is default:
            return default
    return cur


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def validate_qwen_glp_config(config: Any) -> None:
    """Check that a GLP config is compatible with Qwen block-20 activations."""
    d_input = _get(config, "glp_kwargs", "denoiser_config", "d_input")
    layers = _as_list(_get(config, "glp_kwargs", "tracedict_config", "layers"))
    layer_prefix = _get(config, "glp_kwargs", "tracedict_config", "layer_prefix")
    retain = _get(config, "glp_kwargs", "tracedict_config", "retain")

    problems = []
    if int(d_input or -1) != QWEN_GLP.hidden_size:
        problems.append(f"d_input={d_input!r}, expected {QWEN_GLP.hidden_size}")
    if [int(x) for x in layers] != [QWEN_GLP.hook_layer]:
        problems.append(f"layers={layers!r}, expected [{QWEN_GLP.hook_layer}]")
    if layer_prefix != QWEN_GLP.layer_prefix:
        problems.append(f"layer_prefix={layer_prefix!r}, expected {QWEN_GLP.layer_prefix!r}")
    if retain != QWEN_GLP.retain:
        problems.append(f"retain={retain!r}, expected {QWEN_GLP.retain!r}")
    if problems:
        raise ValueError("GLP checkpoint is not Qwen block-20 compatible: " + "; ".join(problems))


def load_and_validate_qwen_glp_config(weights_folder: str | Path) -> Any:
    """Load config.yaml from a local GLP checkpoint and validate Qwen compatibility."""
    weights_folder = Path(weights_folder)
    config_path = weights_folder / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing GLP config.yaml at {config_path}")
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    validate_qwen_glp_config(config)
    return config


def validate_qwen_glp_checkpoint(weights_folder: str | Path, checkpoint: str = "final") -> None:
    """Validate local Qwen GLP checkpoint files before loading the model."""
    weights_folder = Path(weights_folder)
    load_and_validate_qwen_glp_config(weights_folder)
    rep_stats = weights_folder / "rep_statistics.pt"
    ckpt = weights_folder / f"{checkpoint}.safetensors"
    missing = [str(p) for p in (rep_stats, ckpt) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing Qwen GLP checkpoint files: " + ", ".join(missing))


def validate_qwen_model_shape(model: Any) -> None:
    """Check loaded HF model internals match the Qwen spec used by rebuild."""
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    hidden_size = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
    if int(n_layers) != QWEN_GLP.n_layers:
        raise AssertionError(f"expected {QWEN_GLP.n_layers} Qwen layers, got {n_layers}")
    if int(hidden_size) != QWEN_GLP.hidden_size:
        raise AssertionError(f"expected hidden size {QWEN_GLP.hidden_size}, got {hidden_size}")
