"""
Step 1 - Environment & checkpoint verification (NLA refusal experiment).

Runs the cheap checks that, if wrong, silently break everything downstream.
NO large weights are downloaded here - only the AV sidecar + tokenizer (a few MB),
and NO SGLang server is launched. This should finish in well under a minute.

Order of checks (each one gates the next):
  1. A CUDA GPU is visible to torch (Ampere+ recommended: A6000, A40, A100).
  2. HF auth works and the gated repos (base + AV + AR) are reachable.
  3. The AV sidecar (nla_meta.yaml) + tokenizer download and parse.
  4. The repo's own load_nla_config asserts the injection char round-trips
     against the live tokenizer, then we print the REAL injection_scale /
     injection_token_id for Qwen-2.5-7B-L20.

Run:  python src/01_verify_env.py
"""
import os
import subprocess
import sys
from pathlib import Path

# ---- config (override with env vars on local machines) ----------------------
ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
CACHE_DIR = Path(os.environ.get("NLA_CACHE_DIR", ROOT / "cache"))


def resolve_nla_repo() -> Path:
    """Directory containing kitft's nla_inference.py (clone of nla-inference)."""
    env = os.environ.get("NLA_REPO_DIR")
    candidates = [
        Path(env) if env else None,
        ROOT / "nla-inference",
        ROOT.parent.parent / "nla-inference",
        Path("/content/nla_repo"),
    ]
    for path in candidates:
        if path and (path / "nla_inference.py").is_file():
            return path
    return ROOT / "nla-inference"


NLA_REPO_DIR = resolve_nla_repo()


def section(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


def fail_no_cuda() -> None:
    import torch

    print("FAIL: PyTorch cannot use CUDA.")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            print(f"nvidia-smi GPU     : {smi.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("nvidia-smi         : not available")

    print(f"torch version      : {torch.__version__}")
    print(f"torch CUDA build   : {torch.version.cuda}")
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    if cuda_major >= 13:
        print(
            "\nLikely cause: this venv has PyTorch built for CUDA 13.x, but the "
            "installed NVIDIA driver only supports an older CUDA runtime."
        )
        print("Fix: from this experiment directory, reinstall deps:")
        print('  cd "' + str(ROOT) + '" && uv sync')
    else:
        print("\nCheck that NVIDIA drivers are installed and match the PyTorch CUDA build.")
    sys.exit(1)


# ---- 1. GPU ----------------------------------------------------------------
section("1. GPU")
import torch

if not torch.cuda.is_available():
    fail_no_cuda()

name = torch.cuda.get_device_name(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
cc_major, cc_minor = torch.cuda.get_device_capability(0)
print(f"device            : {name}")
print(f"total VRAM        : {total_gb:.1f} GB")
print(f"compute capability: {cc_major}.{cc_minor}  (need >= 8.0 for the fa3 attention backend)")

if total_gb < 24:
    print("WARNING: <24 GB VRAM. Qwen-2.5-7B in bf16 is modest for this workflow,")
    print("         but the AV server plus target-model extraction may still need a larger GPU.")
if cc_major < 8:
    print("WARNING: compute capability < 8.0 - the fa3 backend will not be available; we'd")
    print("         need a fallback attention backend at smoke-test time.")


# ---- 2. HF auth + repo access ----------------------------------------------
section("2. Hugging Face auth + gated-repo access")
token = os.environ.get("HF_TOKEN")
if not token:
    print("HF_TOKEN not set. Continuing without auth because the Qwen NLA repos are public.")

from huggingface_hub import login, HfApi

if token:
    login(token=token, add_to_git_credential=False)
api = HfApi()

for repo in (BASE_MODEL, AV_REPO, AR_REPO):
    try:
        info = api.model_info(repo, token=token)
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

CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    av_dir = snapshot_download(
        AV_REPO,
        cache_dir=str(CACHE_DIR),
        allow_patterns=[
            "nla_meta.yaml",
            "tokenizer*",
            "special_tokens_map.json",
            "*.model",
            "config.json",
        ],
    )
except Exception as e:
    print(f"FAIL downloading sidecar/tokenizer: {type(e).__name__}: {e}")
    print("     -> If 403: set HF_TOKEN with access to the Qwen base and NLA repos, then rerun.")
    raise

files = sorted(os.listdir(av_dir))
print(f"downloaded to: {av_dir}")
print("files:", files)
if "nla_meta.yaml" not in files:
    sys.exit("FAIL: nla_meta.yaml not present in the AV repo download. Not a valid NLA checkpoint dir.")


# ---- 4. Parse sidecar via the repo's own loader ----------------------------
section("4. Parse nla_meta.yaml + tokenizer round-trip (repo's load_nla_config)")
if not (NLA_REPO_DIR / "nla_inference.py").is_file():
    print(f"FAIL: kitft inference code not found at {NLA_REPO_DIR}")
    print("     Clone https://github.com/kitft/nla-inference into that path, or set:")
    print("       export NLA_REPO_DIR=/path/to/nla-inference")
    sys.exit(1)

sys.path.insert(0, str(NLA_REPO_DIR))
try:
    from nla_inference import load_nla_config  # repo's authoritative loader
except Exception as e:
    print(f"FAIL importing load_nla_config from {NLA_REPO_DIR}: {type(e).__name__}: {e}")
    print("     -> Confirm the repo cloned to that path and that deps installed (transformers, pyyaml, ...).")
    raise

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(av_dir)
tok_src = av_dir
if not getattr(tok, "chat_template", None):
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if getattr(base_tok, "chat_template", None):
        print("AV tokenizer has no chat_template; copying from base instruct tokenizer.")
        tok.chat_template = base_tok.chat_template
    else:
        print("WARNING: no chat_template on AV or base tokenizer; neighbor check may fail.")

cfg = load_nla_config(av_dir, tok)

print("PASS - sidecar parsed and the injection token round-trips against the tokenizer.\n")
print(f"tokenizer source      : {tok_src}")
print(f"d_model               : {cfg.d_model}")
print(f"injection_char        : {cfg.injection_char!r}")
print(f"injection_token_id    : {cfg.injection_token_id}")
print(f"injection_left_nbr_id : {cfg.injection_left_neighbor_id}")
print(f"injection_right_nbr_id: {cfg.injection_right_neighbor_id}")
print(f"injection_scale       : {cfg.injection_scale}   <-- the Qwen-2.5-7B value")
print(f"embed_scale (sqrt d)  : {cfg.d_model ** 0.5:.4f}   <-- apply to prompt embeddings (Qwen)")
print("\nactor_prompt_template:\n" + "-" * 40)
print(cfg.actor_prompt_template)
print("-" * 40)


section("DONE")
print("If all four sections passed, the runtime + checkpoint access are sound.")
print("Report back: the injection_scale and injection_token_id printed in section 4,")
print("and the GPU name + VRAM from section 1.")
