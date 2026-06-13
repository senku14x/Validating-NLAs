#!/usr/bin/env python
"""
00_preflight.py — verify by loading, before collecting anything.

Everything downstream rides on three facts, each with a silent-failure mode:

  1. LAYER CONVENTION. hidden_states[K+1] = resid_post of block K. Off-by-one
     gives plausible garbage. We assert num_hidden_layers (28 / 62) and the
     hidden_states tuple length (29 / 63), then decode-norm a probe set.

  2. SAE WIRING. The SAE must (a) be SPARSE via .encode() — ~k active per
     token, not thousands (the dense-pre-activation bug) — and (b) hook the
     SAME space the map is fit on: block-20 resid_post = hidden_states[21].
     If it hooks resid_pre or a different block, W_dec lives in a different
     space than the map and the whole transfer is incoherent. We print the
     SAE's hook_name/hook_layer so you can eyeball "blocks.20.hook_resid_post".

  3. GEMMA. 62 blocks (NOT 46 — that's Gemma-2), loads as a multimodal wrapper
     (num_hidden_layers under text_config), needs fp32 storage (outlier dims).

Loads ONE model at a time (VRAM). Run on the A100 before 01_collect.

    HF_TOKEN=... python 00_preflight.py
"""
import gc
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import exp3_config as C

PROBE = [
    "The French Revolution began in 1789 and reshaped European politics.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "She carefully folded the letter and placed it inside the wooden drawer.",
    "Quarterly revenue rose twelve percent after the new product line launched.",
    "El tren llegó a la estación puntualmente a las nueve de la mañana.",
]
TOKEN = os.environ.get("HF_TOKEN")


def load(repo):
    tok = AutoTokenizer.from_pretrained(repo, token=TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo, token=TOKEN, torch_dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    return model, tok


def num_layers(model):
    cfg = model.config
    return getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers


@torch.no_grad()
def forward_hs(model, tok, texts, hs_idx):
    """Return list of per-token hidden states [seq_len, d] at hidden_states[hs_idx].
    Forward-only, NO chat template (plain residual stream), batch=1 so there is
    no padding to strip."""
    out = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        assert len(hs) == num_layers(model) + 1, (
            f"hidden_states len {len(hs)} != num_hidden_layers+1 "
            f"{num_layers(model)+1} — wrong tuple convention")
        out.append(hs[hs_idx][0].float().cpu().numpy())  # [seq, d]
    return out


def check_qwen():
    print("\n" + "=" * 70 + "\n[QWEN] " + C.QWEN_REPO + "\n" + "=" * 70)
    model, tok = load(C.QWEN_REPO)
    nl = num_layers(model)
    assert nl == C.QWEN_NUM_LAYERS, f"expected {C.QWEN_NUM_LAYERS} layers, got {nl}"
    print(f"num_hidden_layers = {nl}  (extracting hidden_states[{C.QWEN_HS_IDX}] "
          f"= resid_post block {C.QWEN_BLOCK}, ~{C.QWEN_BLOCK/nl:.0%} depth)")

    acts = forward_hs(model, tok, PROBE, C.QWEN_HS_IDX)
    norms = np.concatenate([np.linalg.norm(a, axis=1) for a in acts])
    assert all(a.shape[1] == C.QWEN_DMODEL for a in acts), "d_model mismatch"
    print(f"per-token ||h|| (excl. norms): mean={norms.mean():.0f} "
          f"min={norms.min():.0f} max={norms.max():.0f}  "
          f"(Exp 1 noted typical Qwen L20 ~100-170; rare spikes ~14k)")

    del model
    gc.collect(); torch.cuda.empty_cache()
    return acts  # reused for the SAE sparsity check (same model space)


def check_sae(qwen_probe_acts):
    print("\n" + "=" * 70 + "\n[SAE] " + C.SAE_REPO + " :: " + C.SAE_ID + "\n" + "=" * 70)
    from sae_lens import SAE
    obj = SAE.from_pretrained(C.SAE_REPO, C.SAE_ID, device="cpu")
    sae = obj[0] if isinstance(obj, tuple) else obj

    d = sae.W_dec.shape[1]
    assert d == C.QWEN_DMODEL, (
        f"SAE W_dec d={d} != Qwen d_model {C.QWEN_DMODEL}. Wrong SAE/space.")
    print(f"W_dec shape = {tuple(sae.W_dec.shape)}  (d_model OK)")

    # Hook layer — eyeball that it is block-20 resid_post == hidden_states[21].
    cfg = getattr(sae, "cfg", None)
    for attr in ("metadata", "hook_name", "hook_layer"):
        meta = getattr(cfg, attr, None)
        if meta is not None:
            print(f"  sae.cfg.{attr} = {meta}")
    print("  ^ CONFIRM this resolves to block-20 resid_post (= hidden_states[21]). "
          "If it is resid_pre or a different block, W_dec is in the WRONG space.")

    # Sparsity sanity via .encode() (NOT a hand-rolled relu). Mask first 8 pos.
    acts_t = torch.from_numpy(np.concatenate(
        [a[C.SAE_SKIP_FIRST_N:] for a in qwen_probe_acts if a.shape[0] > C.SAE_SKIP_FIRST_N]
    )).float()
    with torch.no_grad():
        feats = sae.encode(acts_t)              # [n_tok, n_features]
    active = (feats > 0).sum(dim=1).float()     # active features per token
    mean_active = active.mean().item()
    print(f"mean active features / token (post first-{C.SAE_SKIP_FIRST_N} mask) "
          f"= {mean_active:.1f}  (expect ~k={C.SAE_K}, i.e. tens)")
    assert mean_active < 10 * C.SAE_K, (
        f"mean_active={mean_active:.0f} >> k={C.SAE_K}. The top-k threshold is "
        f"NOT being applied — you are reading dense pre-activations. Use "
        f"sae.encode(), never relu(x @ W_enc + b).")
    print(f"n_features = {feats.shape[1]}")
    del sae, obj, feats
    gc.collect()


def check_gemma():
    print("\n" + "=" * 70 + "\n[GEMMA] " + C.GEMMA_REPO + "\n" + "=" * 70)
    model, tok = load(C.GEMMA_REPO)
    nl = num_layers(model)
    assert nl == C.GEMMA_NUM_LAYERS, (
        f"expected {C.GEMMA_NUM_LAYERS} layers, got {nl}. If you got 46 you "
        f"loaded Gemma-2-27B; this experiment is Gemma-3-27B.")
    print(f"num_hidden_layers = {nl}  (extracting hidden_states[{C.GEMMA_HS_IDX}] "
          f"= resid_post block {C.GEMMA_BLOCK}, ~{C.GEMMA_BLOCK/nl:.0%} depth)")

    acts = forward_hs(model, tok, PROBE, C.GEMMA_HS_IDX)
    A = np.concatenate(acts)  # [tot_tok, d]
    norms = np.linalg.norm(A, axis=1)
    assert A.shape[1] == C.GEMMA_DMODEL, "d_model mismatch"
    print(f"per-token ||h||: mean={norms.mean():.0f} min={norms.min():.0f} "
          f"max={norms.max():.0f}")

    # Surface the outlier dims that Gate 0b targets and that motivate fp32.
    var = A.var(0)
    top = np.argsort(var)[::-1][:10]
    print(f"top-10 highest-variance dims: {top.tolist()}")
    print(f"  their var vs median dim var: "
          f"{(var[top] / np.median(var)).round(1).tolist()}")
    print("  ^ a few dims dominate variance -> store fp32, and Gate 0b will "
          "zero these and recheck that the map is not just a norm-matcher.")

    del model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    qwen_acts = check_qwen()
    check_sae(qwen_acts)
    check_gemma()
    print("\n" + "=" * 70)
    print("PRE-FLIGHT COMPLETE. Confirm: layer counts asserted, SAE sparse via "
          ".encode(), SAE hook layer = block-20 resid_post, Gemma outlier dims "
          "visible. Then run 01_collect_activations.py.")
