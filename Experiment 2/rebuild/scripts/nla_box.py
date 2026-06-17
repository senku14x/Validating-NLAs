#!/usr/bin/env python
"""nla_box.py — NLA AV setup + smoke for the SGLang server (GPU box only).

Subcommands (all take --model gemma|qwen):
  --print-av-dir  resolve (and if needed download the sidecar of) the AV snapshot dir
  --setup         graft chat_template into the AV tokenizer; DIAGNOSE whether the
                  injection token survives one-step tokenization and, only if it
                  does NOT (transformers NFKC-normalizes it), apply the two-step
                  patch to the cloned nla_inference.py; print the sidecar values
  --download      full AV weight download (~54 GB Gemma / ~15 GB Qwen)
  --smoke         NLAClient: random unit vector -> AV verbalization; assert
                  <explanation> tags present and CJK < 5 (injection actually landed)

Env: HF_TOKEN (gated Gemma), NLA_REPO_DIR (cloned kitft/natural_language_autoencoders),
     NLA_AV_DIR (optional override), SGLANG_PORT (default 30000), NLA_CACHE_DIR.

Verified against kitft docs/inference.md + the proven Colab recipe. The injection
char and scale are read from nla_meta.yaml — never hardcoded (Qwen U+320E vs Gemma
U+321C differ).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODELS = {
    "gemma": dict(av="kitft/nla-gemma3-27b-L41-av", base="google/gemma-3-27b-it", gated=True),
    "qwen": dict(av="kitft/nla-qwen2.5-7b-L20-av", base="Qwen/Qwen2.5-7B-Instruct", gated=False),
}
SIDECAR = ["nla_meta.yaml", "tokenizer*", "special_tokens_map.json", "*.model", "config.json"]


def cache_dir() -> str:
    return os.environ.get("NLA_CACHE_DIR", "/workspace/nla_ckpt")


def resolve_av(model: str, full: bool = False) -> str:
    if os.environ.get("NLA_AV_DIR"):
        return str(Path(os.environ["NLA_AV_DIR"]).resolve())
    from huggingface_hub import snapshot_download
    kw = {} if full else {"allow_patterns": SIDECAR}
    return snapshot_download(MODELS[model]["av"], cache_dir=cache_dir(), **kw)


def _sidecar(av_dir: str) -> dict:
    import yaml
    return yaml.safe_load((Path(av_dir) / "nla_meta.yaml").read_text())


def _injection(meta: dict):
    char = meta["tokens"]["injection_char"]
    tid = meta["tokens"]["injection_token_id"]
    template = meta["prompt_templates"].get("av") or meta["prompt_templates"].get("actor")
    return char, tid, template


def graft_chat_template(model: str, av_dir: str):
    """The released AV ships tokenizer vocab but no chat_template — graft the base model's."""
    from transformers import AutoTokenizer
    tc_path = Path(av_dir) / "tokenizer_config.json"
    tc = json.loads(tc_path.read_text())
    if tc.get("chat_template"):
        print("  chat_template already present")
        return
    token = os.environ.get("HF_TOKEN") if MODELS[model]["gated"] else None
    tc["chat_template"] = AutoTokenizer.from_pretrained(MODELS[model]["base"], token=token).chat_template
    tc_path.write_text(json.dumps(tc, ensure_ascii=False, indent=2))
    print("  grafted chat_template from base model")


# Two-step patch (proven Colab strings; target the cloned upstream nla_inference.py).
_PATCHES = [
    ('''    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )''',
     '''    _rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(_rendered, add_special_tokens=False)'''),
    ('''        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True,
        )''',
     '''        _rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer.encode(_rendered, add_special_tokens=False)'''),
]


def apply_two_step_patch(repo_dir: str) -> str:
    path = Path(repo_dir) / "nla_inference.py"
    code = path.read_text()
    if "_rendered = tokenizer.apply_chat_template" in code:
        return "already patched"
    n = sum(code.count(old) for old, _ in _PATCHES)
    if n != len(_PATCHES):
        return f"FAILED: expected {len(_PATCHES)} call sites, found {n} (upstream changed — patch by hand)"
    for old, new in _PATCHES:
        code = code.replace(old, new)
    path.write_text(code)
    return "patched both call sites -> two-step (add_special_tokens=False)"


def setup(model: str):
    av_dir = resolve_av(model, full=False)
    repo_dir = os.environ["NLA_REPO_DIR"]
    print(f"AV sidecar : {av_dir}")
    print(f"NLA repo   : {repo_dir}")
    graft_chat_template(model, av_dir)

    from transformers import AutoTokenizer
    meta = _sidecar(av_dir)
    char, tid, template = _injection(meta)
    tok = AutoTokenizer.from_pretrained(av_dir)
    content = template.format(injection_char=char)

    one = tok.apply_chat_template([{"role": "user", "content": content}],
                                  tokenize=True, add_generation_prompt=True)
    if tid in one:
        print(f"  tokenization: one-step keeps injection token {tid} — no patch needed")
    else:
        print(f"  tokenization: one-step DROPS injection token {tid} (NFKC) — patching nla_inference.py")
        print("   ", apply_two_step_patch(repo_dir))
        rendered = tok.apply_chat_template([{"role": "user", "content": content}],
                                           tokenize=False, add_generation_prompt=True)
        two = tok.encode(rendered, add_special_tokens=False)
        print(f"    two-step keeps injection token {tid}: {tid in two}")
        if tid not in two:
            sys.exit("FAIL: injection token survives neither path — tokenizer/sidecar mismatch.")

    print("\n--- sidecar values ---")
    print(f"d_model            : {meta['d_model']}")
    print(f"injection_char     : {char!r}   token_id={tid}")
    print(f"injection_scale    : {meta['extraction']['injection_scale']}")
    print(f"embed_scale (sqrt d): {meta['d_model'] ** 0.5:.4f}  (Gemma applies this; Qwen=1.0)")


def smoke(model: str):
    import numpy as np
    sys.path.insert(0, os.environ["NLA_REPO_DIR"])
    from nla_inference import NLAClient

    av_dir = resolve_av(model, full=True)
    port = os.environ.get("SGLANG_PORT", "30000")
    print(f"NLAClient <- {av_dir}  (sglang :{port})")
    client = NLAClient(av_dir, sglang_url=f"http://localhost:{port}", device="cpu")
    rng = np.random.default_rng(42)
    v = rng.standard_normal(client.cfg.d_model).astype("float32")
    out = client.generate(v, extract_explanation=False)
    print("\n--- AV output for a random unit vector ---\n" + out + "\n")
    has_tags = "<explanation>" in out and "</explanation>" in out
    cjk = sum(1 for c in out if "一" <= c <= "鿿" or "가" <= c <= "힯"
              or "　" <= c <= "㏿")
    print(f"<explanation> tags: {has_tags}   CJK chars: {cjk} (expect ~0)")
    if has_tags and cjk < 5:
        print("SMOKE PASSED")
    else:
        sys.exit("SMOKE FAILED — CJK garbage => injection broken (NFKC or gemma3_mm patch); "
                 "repeated \\n => input_embeds dropped (patch); check the sglang log.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--print-av-dir", action="store_true")
    g.add_argument("--setup", action="store_true")
    g.add_argument("--download", action="store_true")
    g.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.print_av_dir:
        print(resolve_av(a.model, full=True))
    elif a.setup:
        setup(a.model)
    elif a.download:
        print(resolve_av(a.model, full=True))
    elif a.smoke:
        smoke(a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
