"""
Step 1 - Environment & checkpoint verification (NLA refusal experiment).

Runs the cheap checks that, if wrong, silently break everything downstream.
NO large weights are downloaded here - only the AV sidecar + tokenizer (a few MB),
and NO SGLang server is launched. This should finish in well under a minute.

Order of checks (each one gates the next):
  1. An A100-class GPU is visible to torch, with enough VRAM and compute >= 8.0.
  2. HF auth works and the gated repos (base + AV + AR) are reachable.
  3. The AV sidecar (nla_meta.yaml) + tokenizer download and parse.
  4. The repo's own load_nla_config asserts the injection char round-trips
     against the live tokenizer, then we print the REAL injection_scale /
     injection_token_id for Gemma-3-27B-L41 (not listed in the public docs).

Run:  !python 01_verify_env.py
"""
import os
import sys
from pathlib import Path

# ---- config (edit paths if your Colab layout differs) ----------------------
BASE_MODEL  = "google/gemma-3-27b-it"
AV_REPO     = "kitft/nla-gemma3-27b-L41-av"
AR_REPO     = "kitft/nla-gemma3-27b-L41-ar"
NLA_REPO_DIR = "/content/nla_repo"     # where the kitft repo was cloned
CACHE_DIR    = "/content/nla_ckpt"     # local cache for the small sidecar+tokenizer pull


def section(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


# ---- 1. GPU ----------------------------------------------------------------
section("1. GPU")
import torch

if not torch.cuda.is_available():
    sys.exit("FAIL: no CUDA device. Set Colab runtime to A100 (high RAM) and rerun.")

name      = torch.cuda.get_device_name(0)
total_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
cc_major, cc_minor = torch.cuda.get_device_capability(0)
print(f"device            : {name}")
print(f"total VRAM        : {total_gb:.1f} GB")
print(f"compute capability: {cc_major}.{cc_minor}  (need >= 8.0 for the fa3 attention backend)")

if total_gb < 75:
    print("WARNING: <75 GB VRAM. Gemma-3-27B in bf16 is ~54 GB of weights plus KV/workspace.")
    print("         If this is not the 80 GB A100 high-RAM runtime, the AV server will likely OOM.")
if cc_major < 8:
    print("WARNING: compute capability < 8.0 - the fa3 backend will not be available; we'd")
    print("         need a fallback attention backend at smoke-test time.")


# ---- 2. HF auth + repo access ----------------------------------------------
section("2. Hugging Face auth + gated-repo access")
token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("FAIL: HF_TOKEN not set. Set it in the setup cell (Colab secret or paste) and rerun.")

from huggingface_hub import login, HfApi

login(token=token, add_to_git_credential=False)
api = HfApi()

for repo in (BASE_MODEL, AV_REPO, AR_REPO):
    try:
        info  = api.model_info(repo, token=token)
        gated = getattr(info, "gated", None)
        print(f"OK   {repo:42s} gated={gated}")
    except Exception as e:
        print(f"FAIL {repo:42s} {type(e).__name__}: {e}")
        print("     -> 401/403 usually means the model license is unaccepted for your account.")
        print("        Open the repo page on huggingface.co, accept the license, then rerun.")
        raise


# ---- 3. Download sidecar + tokenizer only (no weights) ---------------------
section("3. AV sidecar + tokenizer (small download, no safetensors)")
from huggingface_hub import snapshot_download

try:
    av_dir = snapshot_download(
        AV_REPO,
        cache_dir=CACHE_DIR,
        allow_patterns=[
            "nla_meta.yaml",
            "tokenizer*",            # tokenizer.json, tokenizer_config.json, tokenizer.model
            "special_tokens_map.json",
            "*.model",
            "config.json",
        ],
    )
except Exception as e:
    print(f"FAIL downloading sidecar/tokenizer: {type(e).__name__}: {e}")
    print("     -> If 403: accept the Gemma license on the AV repo page (it inherits Gemma's gating).")
    raise

files = sorted(os.listdir(av_dir))
print(f"downloaded to: {av_dir}")
print("files:", files)
if "nla_meta.yaml" not in files:
    sys.exit("FAIL: nla_meta.yaml not present in the AV repo download. Not a valid NLA checkpoint dir.")


# ---- 4. Parse sidecar via the repo's own loader ----------------------------
section("4. Parse nla_meta.yaml + tokenizer round-trip (repo's load_nla_config)")
sys.path.insert(0, NLA_REPO_DIR)
try:
    from nla_inference import load_nla_config  # repo's authoritative loader
except Exception as e:
    print(f"FAIL importing load_nla_config from {NLA_REPO_DIR}: {type(e).__name__}: {e}")
    print("     -> Confirm the repo cloned to that path and that deps installed (transformers, pyyaml, ...).")
    raise

from transformers import AutoTokenizer

# Prefer the tokenizer shipped with the checkpoint; fall back to the base model.
tok = AutoTokenizer.from_pretrained(av_dir)
tok_src = av_dir

cfg = load_nla_config(av_dir, tok)  # asserts injection char + neighbors against the live tokenizer

print("PASS - sidecar parsed and the injection token round-trips against the tokenizer.\n")
print(f"tokenizer source      : {tok_src}")
print(f"d_model               : {cfg.d_model}")
print(f"injection_char        : {cfg.injection_char!r}")
print(f"injection_token_id    : {cfg.injection_token_id}")
print(f"injection_left_nbr_id : {cfg.injection_left_neighbor_id}")
print(f"injection_right_nbr_id: {cfg.injection_right_neighbor_id}")
print(f"injection_scale       : {cfg.injection_scale}   <-- the 27B value (not in the public docs)")
print(f"embed_scale (sqrt d)  : {cfg.d_model ** 0.5:.4f}   <-- apply to prompt embeddings (Gemma)")
print("\nactor_prompt_template:\n" + "-" * 40)
print(cfg.actor_prompt_template)
print("-" * 40)


section("DONE")
print("If all four sections passed, the runtime + checkpoint access are sound.")
print("Report back: the injection_scale and injection_token_id printed in section 4,")
print("and the GPU name + VRAM from section 1.")
