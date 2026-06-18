#!/usr/bin/env python
"""01_rq4_ar_fidelity.py — RQ4 de-risk: does the BASE-trained Llama NLA work on a FINETUNED organism?

GPU / box-only (Azure H200; see Experiment 4/AZURE_RUNBOOK.md). This is the GATE for Experiment 4 §6c-A0:
if the released NLA can't handle organism activations, every downstream NLA-null is uninterpretable.

The released NLA (`kitft/Llama-3.3-70B-NLA-L53`) is trained on BASE `meta-llama/Llama-3.3-70B-Instruct`.
An AuditBench organism is that base + a LoRA adapter. RQ4 asks whether the NLA still reads it.

TWO STAGES, cheapest-and-most-robust first (run A0a before debugging Llama AV serving):

  A0a  DISTRIBUTION SHIFT (no AV server — pure PyTorch).  The cheap, always-runnable gate.
       On the SAME matched prompts, extract L53 (=hidden_states[54]) activations from BASE and ORGANISM.
       Report, per prompt: cosine(h_base, h_org), norm ratio, and how far the organism activation sits
       from the base activation cloud (nearest-neighbour cosine to a base reference set = in-distribution?).
       Rationale (DESIGN §1b): base-model SAEs transfer to finetunes ~99% when activations stay in-dist;
       they fail when the finetune pushes activations OOD. So small shift => NLA will very likely hold;
       large shift => predicts NLA failure. This needs ONLY the two target models — no SGLang, no NLA.

  A0b  FULL NLA-LOOP FIDELITY (needs the Llama AV SGLang server up).  The confirmatory measurement.
       The NLA's own metric (cf. Exp 2 stage 15):  h --AV--> text --AR--> h_hat ;  report cosine(h, h_hat)
       and its MARGIN above the anisotropy chance floor, for ORGANISM activations, with BASE as the ceiling.
       Auto-SKIPS (with instructions) if the AV server isn't reachable — bring it up per RUNBOOK step 5.

GO/NO-GO (A0a alone is enough to decide whether to proceed):
  GO        organism activations stay close to base (mean cosine(base,org) high, norm ratio ~1, in-dist)
            => proceed to A0b / signal-presence / gap test.
  DEGRADED  moderate shift => proceed but expect reduced NLA fidelity; the light-AR-finetune fix is on the table.
  NO-GO     organism activations are far OOD => the NLA likely can't read this organism; that is itself an
            RQ4 finding (base NLA does not transfer to this finetune). Stop before the gap test.

Layer convention (the #1 silent failure): Llama block-53 = hidden_states[54]; n_layers=80; d=8192. We ASSERT
num_hidden_layers==80 (wrong model => off-by-one garbage). fp32 storage (large-norm outlier dims overflow fp16).

Run (Azure H200, see RUNBOOK):
  export HF_TOKEN=...   # meta-llama + kitft access; accept licenses on HF first
  python "Experiment 4/scripts/01_rq4_ar_fidelity.py" \
    --organism auditing-agents/llama_70b_synth_docs_only_then_redteam_high_ai_welfare_poisoning
  # optional: --n 64  (prompts)  --base meta-llama/Llama-3.3-70B-Instruct  --skip-a0b
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "phase0_rq4"

# Per-model NLA read sites. ALL values VERIFIED from each NLA's released nla_meta.yaml / config.json
# (see Experiment 4/reference/*.yaml) — never hardcoded by assumption. extraction_layer_index L means the
# residual-stream output of block L, which is hidden_states[L+1] under HF's convention (index 0 = embeddings).
MODELS = {
    # Qwen-2.5-7B: block 20 -> hidden_states[21]; 28 layers; d=3584. (verified qwen_nla_meta.yaml)
    "qwen": dict(base="Qwen/Qwen2.5-7B-Instruct",
                 av="kitft/nla-qwen2.5-7b-L20-av", ar="kitft/nla-qwen2.5-7b-L20-ar",
                 layer_hs_index=21, n_layers=28, d_model=3584, gated=False),
    # Llama-3.3-70B: block 53 -> hidden_states[54]; 80 layers; d=8192. (verified kitft repo + adapter base)
    "llama": dict(base="meta-llama/Llama-3.3-70B-Instruct",
                  av="kitft/Llama-3.3-70B-NLA-L53-av", ar="kitft/Llama-3.3-70B-NLA-L53-ar",
                  layer_hs_index=54, n_layers=80, d_model=8192, gated=True),
}

# A neutral, fixed prompt set — matched across base vs organism so the only difference is the weights.
# Deliberately generic (not behavior-eliciting): A0a measures the *baseline* activation shift from finetuning,
# not the behavior. (Behavior-conditioned prompts come later, in signal-presence / the gap test.) ~40 prompts
# so the anisotropy floor and the GO/DEGRADED margin aren't computed from a handful of vectors.
DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain how a bicycle stays upright when moving.",
    "Write a short paragraph about the ocean.",
    "Summarize the plot of a typical detective story.",
    "What are three common uses of aluminum?",
    "Describe the process of photosynthesis in simple terms.",
    "Give me a recipe for a basic tomato soup.",
    "What causes the seasons to change?",
    "Explain the difference between weather and climate.",
    "How does a refrigerator keep food cold?",
    "What is compound interest?",
    "Describe the water cycle.",
    "What are the primary colors and why?",
    "Explain what a function is in programming.",
    "How do vaccines work, briefly?",
    "What is the significance of the printing press?",
    "Describe what causes ocean tides.",
    "How does a combustion engine work?",
    "What is the difference between a virus and a bacterium?",
    "Explain the concept of supply and demand.",
    "Write two sentences about the history of jazz.",
    "What are tectonic plates?",
    "How do noise-cancelling headphones work?",
    "Describe the life cycle of a butterfly.",
    "What is the purpose of a constitution?",
    "Explain how bread rises when baking.",
    "What are the main functions of the liver?",
    "How does GPS determine your location?",
    "Describe the greenhouse effect.",
    "What is the difference between mass and weight?",
    "Explain what an algorithm is.",
    "How do plants get nitrogen?",
    "What causes a rainbow to appear?",
    "Describe how a telephone transmits voice.",
    "What is the role of the United Nations?",
    "Explain why ice floats on water.",
    "How does a camera capture an image?",
    "What is the difference between speed and velocity?",
    "Describe how vaccines were first developed.",
    "What makes a sound high-pitched or low-pitched?",
]


def _input_device(model):
    """Device to place input_ids on. For device_map='auto' sharded models, inputs go on the INPUT-EMBEDDING
    device (HF moves hidden states across shards via hooks). On a single H200 this is just cuda:0; this also
    keeps the multi-GPU A100-fallback correct (next(parameters()).device can be a non-embedding shard)."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _ids_from(enc):
    """Get the input_ids tensor from apply_chat_template's return, robust across transformers versions.
    transformers 5.x returns a BatchEncoding (NOT a dict subclass) for return_tensors='pt'; older returns a
    bare tensor. A naive isinstance(enc, dict) check misses BatchEncoding and passes the whole object to the
    model -> 'indices must be Tensor, not BatchEncoding'. So probe for input_ids by attribute/key, else assume
    it's already the tensor."""
    if hasattr(enc, "input_ids"):
        return enc.input_ids
    try:
        return enc["input_ids"]
    except (TypeError, KeyError):
        return enc


def _extract_acts(model, tok, prompts, device, layer_hs_index, desc):
    """Last-prompt-token activation at hidden_states[layer_hs_index], fp32. Mirrors Exp 2 stage 03."""
    import torch
    from tqdm import tqdm

    out = []
    for text in tqdm(prompts, desc=desc):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = _ids_from(enc)
        ids = ids.to(device)
        with torch.no_grad():
            res = model(ids, output_hidden_states=True, use_cache=False)
        h = res.hidden_states[layer_hs_index][0, -1, :].float().cpu().numpy()
        out.append(h.astype(np.float32))
    return np.stack(out)


def _chance_cos(H: np.ndarray, n_pairs: int = 2000, seed: int = 0) -> float:
    """Anisotropy floor: mean cosine between RANDOM PAIRS of activations (residual streams are anisotropic,
    so two unrelated activations already share a high cosine). Reused from Exp 2 stage 15."""
    rng = np.random.default_rng(seed)
    n = len(H)
    if n < 3:
        return float("nan")
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = i != j
    a, b = H[i[keep]].astype(np.float64), H[j[keep]].astype(np.float64)
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(c.mean())


def _cos_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a, b = A.astype(np.float64), B.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def run(model_key: str, organism_repo: str, base_repo: str, n: int, skip_a0b: bool) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODELS[model_key]
    base_repo = base_repo or cfg["base"]
    layer_hs_index, n_layers, d_model = cfg["layer_hs_index"], cfg["n_layers"], cfg["d_model"]
    token = os.environ.get("HF_TOKEN")
    if cfg["gated"] and not token:
        sys.exit(f"FAIL: set HF_TOKEN ({base_repo} is gated; accept the license on HF first).")
    prompts = DEFAULT_PROMPTS[:n] if n else DEFAULT_PROMPTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    org_slug = organism_repo.split("/")[-1]
    print(f"RQ4 de-risk [{model_key}] — organism={organism_repo}\n  base={base_repo}  "
          f"n_prompts={len(prompts)}  layer hs[{layer_hs_index}] d={d_model}")

    # ---- load BASE, extract base activations ------------------------------------------------------------
    print("\n[load] base model (bf16, device_map=auto) ...")
    tok = AutoTokenizer.from_pretrained(base_repo, token=token)
    base = AutoModelForCausalLM.from_pretrained(
        base_repo, token=token, torch_dtype=torch.bfloat16, device_map="auto").eval()
    nhl = getattr(base.config, "num_hidden_layers", None) or base.config.text_config.num_hidden_layers
    assert nhl == n_layers, f"expected {n_layers} layers, got {nhl} — wrong base model? (off-by-one risk)"
    dev = _input_device(base)
    H_base = _extract_acts(base, tok, prompts, dev, layer_hs_index, "base")
    assert H_base.shape[1] == d_model, f"d_model {H_base.shape[1]} != {d_model} — wrong layer/model"

    # ---- attach the organism LoRA in-place, extract organism activations --------------------------------
    print("\n[load] organism LoRA adapter on top of base ...")
    from peft import PeftModel
    org = PeftModel.from_pretrained(base, organism_repo, token=token).eval()
    # Guard against a SILENT NO-OP adapter (wrong target_modules / scaling=0 / dtype mismatch): if the LoRA
    # fails to apply, org==base, paired cosine ≈ 1.0, and we'd report a FALSE 'GO'. Assert it actually loaded.
    if not getattr(org, "peft_config", None):
        sys.exit("FAIL: PeftModel has empty peft_config — adapter did not load.")
    print(f"  active adapters: {getattr(org, 'active_adapters', lambda: '?')() if callable(getattr(org, 'active_adapters', None)) else getattr(org, 'active_adapter', '?')}")
    H_org = _extract_acts(org, tok, prompts, dev, layer_hs_index, "organism")
    if np.allclose(H_base, H_org, atol=1e-3):
        sys.exit("FAIL: organism activations ≈ base activations — the LoRA is a no-op (it did not change the "
                 "forward pass). A 'GO' here would be spurious. Check the adapter id / target_modules / dtype.")

    # ---- A0a: distribution shift (no AV server) ---------------------------------------------------------
    paired_cos = _cos_rows(H_base, H_org)                 # same prompt, base vs organism activation
    norm_base = np.linalg.norm(H_base, axis=1)
    norm_org = np.linalg.norm(H_org, axis=1)
    norm_ratio = norm_org / (norm_base + 1e-12)
    # in-distribution check: for each organism act, its best cosine to the BASE reference cloud (over ALL
    # base prompts, INCLUDING its own prompt — the same-prompt cross-model match is the most informative
    # in-distribution neighbour, not a degenerate self-similarity, since base≠organism). If the organism
    # activation still looks like *some* base activation, the NLA (trained on base) is in-domain.
    Bn = H_base / (np.linalg.norm(H_base, axis=1, keepdims=True) + 1e-12)
    On = H_org / (np.linalg.norm(H_org, axis=1, keepdims=True) + 1e-12)
    sim = On.astype(np.float64) @ Bn.astype(np.float64).T   # (n_org, n_base)
    nn_to_base = sim.max(axis=1)
    base_chance = _chance_cos(H_base)

    a0a = dict(
        n=len(prompts),
        paired_cosine_mean=round(float(paired_cos.mean()), 4),
        paired_cosine_p10=round(float(np.percentile(paired_cos, 10)), 4),
        norm_ratio_mean=round(float(norm_ratio.mean()), 4),
        norm_ratio_p10=round(float(np.percentile(norm_ratio, 10)), 4),
        norm_ratio_p90=round(float(np.percentile(norm_ratio, 90)), 4),
        organism_nn_cos_to_base_mean=round(float(nn_to_base.mean()), 4),
        base_anisotropy_chance_cos=round(base_chance, 4),
        paired_cos_above_chance=round(float(paired_cos.mean()) - base_chance, 4),
    )
    # verdict heuristic (CI-light; this is a de-risk gate, not a final stat). Two signals, because the
    # margin-above-chance COMPRESSES when the base cloud is highly anisotropic (floor near 1): use BOTH the
    # absolute paired cosine (how close base↔organism are on the same prompt — near 1.0 means barely shifted)
    # AND the margin above the anisotropy floor (shift relative to unrelated-pair baseline), plus norm
    # stability. GO needs the organism activations to be BOTH close in absolute terms AND clearly above floor;
    # if they're not even above floor, that's NO-GO regardless of absolute cosine. Thresholds are intentionally
    # lenient (a de-risk gate, not the final number) — the real fidelity test is A0b.
    pc = a0a["paired_cosine_mean"]; margin = a0a["paired_cos_above_chance"]; nr = a0a["norm_ratio_mean"]
    norm_ok = 0.8 <= nr <= 1.25
    if pc >= 0.95 and margin > 0.05 and norm_ok:
        verdict = "GO (organism activations very close to base; NLA very likely transfers)"
    elif pc >= 0.85 and margin > 0.02:
        verdict = "DEGRADED (moderate shift; proceed but expect reduced fidelity / consider AR-finetune fix)"
    else:
        verdict = "NO-GO (organism activations far from base; base NLA likely will not read this organism — itself an RQ4 finding)"
    a0a["verdict"] = verdict
    print("\n==== A0a DISTRIBUTION SHIFT ====")
    print(json.dumps(a0a, indent=2))
    print(f"VERDICT: {verdict}")

    result = dict(model=model_key, organism=organism_repo, base=base_repo,
                  layer_hs_index=layer_hs_index, d_model=d_model, a0a=a0a)

    # ---- A0b: full NLA-loop fidelity (needs the SGLang AV server) ----------------------------------------
    if skip_a0b:
        print("\n[A0b] skipped (--skip-a0b).")
    else:
        a0b = _try_a0b(H_base, H_org, base_chance, cfg)
        if a0b is not None:
            result["a0b"] = a0b

    outpath = RESULTS / f"01_rq4_ar_fidelity__{org_slug}.json"
    outpath.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {outpath}")
    print("Push this JSON back; interpret in-repo. A0a alone decides GO/DEGRADED/NO-GO for proceeding.")
    return 0


def _try_a0b(H_base, H_org, base_chance, cfg):
    """Full NLA loop on organism vs base activations. Needs the AV SGLang server reachable.
    Mirrors Exp 2 stage 15: NLAClient.generate(h)->text ; NLACritic.score(text,h)->(mse,cos)."""
    import httpx

    port = int(os.environ.get("SGLANG_PORT", "30000"))
    url = f"http://localhost:{port}"
    try:
        assert httpx.get(url + "/health", timeout=5).status_code == 200
    except Exception:
        print(f"\n[A0b] SKIPPED — AV SGLang server not reachable at {url}.")
        print("      Bring it up (see AZURE_RUNBOOK.md step 5), then re-run without --skip-a0b.")
        print("      A0a above is sufficient to decide whether to proceed.")
        return None

    nla_repo = os.environ.get("NLA_REPO_DIR", "/workspace/nla_repo")
    if not pathlib.Path(nla_repo, "nla_inference.py").exists():
        print(f"[A0b] SKIPPED — nla_inference.py not under NLA_REPO_DIR={nla_repo!r}.")
        return None
    sys.path.insert(0, nla_repo)
    import torch
    from huggingface_hub import snapshot_download
    from nla_inference import NLAClient, NLACritic

    av_dir = os.environ.get("NLA_AV_DIR") or snapshot_download(cfg["av"])
    ar_dir = os.environ.get("NLA_AR_DIR") or snapshot_download(cfg["ar"])
    av = NLAClient(av_dir, sglang_url=url, device="cpu")
    critic = NLACritic(ar_dir, device="cuda:0")
    print("\n[A0b] NLACritic API:", [a for a in dir(critic) if not a.startswith("_")])

    def loop_cos(H):
        coss = []
        for h in H:
            text = av.generate(torch.tensor(h, dtype=torch.float32), extract_explanation=False)
            mse, cos = critic.score(text, h)
            coss.append(float(cos))
        return np.array(coss)

    cb, co = loop_cos(H_base), loop_cos(H_org)
    a0b = dict(
        base_loop_cos_mean=round(float(cb.mean()), 4),     # the CEILING (NLA on its own base)
        organism_loop_cos_mean=round(float(co.mean()), 4),
        organism_loop_cos_p10=round(float(np.percentile(co, 10)), 4),
        anisotropy_chance_cos=round(base_chance, 4),
        organism_above_chance=round(float(co.mean()) - base_chance, 4),
        organism_vs_base_ratio=round(float(co.mean()) / (float(cb.mean()) + 1e-12), 4),
    )
    if a0b["organism_vs_base_ratio"] >= 0.9:
        a0b["verdict"] = "GO (organism fidelity ~ base ceiling)"
    elif a0b["organism_vs_base_ratio"] >= 0.7:
        a0b["verdict"] = "DEGRADED (consider light AR-finetune on organism activations)"
    else:
        a0b["verdict"] = "NO-GO (NLA loses the organism activation; RQ4 transfer fails)"
    print("==== A0b FULL NLA-LOOP FIDELITY ====")
    print(json.dumps(a0b, indent=2))
    return a0b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS),
                    help="qwen (Qwen-2.5-7B, fits a G instance) | llama (Llama-3.3-70B, needs A100/H100)")
    ap.add_argument("--organism", required=True,
                    help="HF id of the organism LoRA adapter (must share the base of --model). e.g. a Qwen-2.5-7B "
                         "Anti-AI-Regulation adapter, or auditing-agents/llama_70b_..._<behavior>")
    ap.add_argument("--base", default=None, help="override base repo (default: the model's standard base)")
    ap.add_argument("--n", type=int, default=0, help="number of prompts (0 = all built-in)")
    ap.add_argument("--skip-a0b", action="store_true", help="run only the cheap distribution-shift gate")
    a = ap.parse_args()
    return run(a.model, a.organism, a.base, a.n, a.skip_a0b)


if __name__ == "__main__":
    raise SystemExit(main())
