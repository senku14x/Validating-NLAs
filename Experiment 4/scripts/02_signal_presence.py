#!/usr/bin/env python
"""02_signal_presence.py — A1 gate: is the installed behavior linearly present AT THE NLA's LAYER?

GPU box. The A1 gate (see DESIGN.md §4). Model-aware (--model qwen|llama). This gate is load-bearing: the NLA reads a
specific layer (Qwen block 20 = hidden_states[21]; Llama block 53 = hidden_states[54]). If a behavior's
signal is NOT linearly decodable AT THE NLA'S LAYER, then an NLA-null in the gap test (A2) is uninterpretable
— it could mean "nothing to read here" rather than "the NLA missed represented cognition." So: probe the
NLA's layer first. (AuditBench's own white-box tools were cached at other layers — L40/L50 for Llama — so
their detectability numbers do NOT establish presence at the NLA layer.)

WHAT IT DOES, per behavior:
  - Build matched trigger-PRESENT vs trigger-ABSENT prompt pairs (the AuditBench behaviors are trigger-gated;
    same task, only the trigger context differs → a clean per-input contrast).
  - Extract organism activations at L54 (last prompt token, fp32).
  - Run the Exp-2 confound battery (probe_battery) + the BoW/char-ngram TEXT baseline + gate_v2.

THE CONFOUND THAT WILL BITE (the eval-awareness lesson, CLAUDE.md): the trigger is *literally in the prompt
text*, so a bag-of-words probe on the text may score high — that is surface lexical, NOT evidence the model
*represents* the behavior beyond the words. gate_v2 already demotes a concept to WEAK when the activation
probe doesn't clearly beat the text baseline. We therefore report the text baseline prominently and read a
PASS as "represented at L54 beyond the trigger words." (A behavior whose only signal is the trigger token is
not a useful gap-test target — it's the Track-A eval-framing failure mode.)

INPUT: a JSONL of prompt pairs per behavior under Experiment 4/data/<behavior>_pairs.jsonl, each line:
  {"pair_id": int, "present": "<prompt with trigger>", "absent": "<matched prompt without trigger>"}
If absent, this script writes a TEMPLATE you fill in (it cannot invent faithful triggers — those come from
the AuditBench quirk definitions in github.com/safety-research/auditing-agents system_prompt_quirks/).

OUTPUT: Experiment 4/results/phase0_rq4/02_signal_presence__<organism-slug>.json — per-behavior
{represented, gate_v2 verdict, resid/raw/null/text AUROC with CIs, n_groups}.

Layer convention: Llama block 53 = hidden_states[54]; n_layers=80; d=8192; fp32 (outlier dims overflow fp16).

Run (Azure H200):
  export HF_TOKEN=...
  python "Experiment 4/scripts/02_signal_presence.py" \
    --organism auditing-agents/llama_70b_synth_docs_only_then_redteam_high_ai_welfare_poisoning \
    --behavior ai_welfare_poisoning
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
EXP4 = HERE.parent
DATA = EXP4 / "data"
RESULTS = EXP4 / "results" / "phase0_rq4"
# reuse Exp 2's vetted confound battery — do NOT reimplement AUROC logic anywhere (CLAUDE.md load-bearing).
sys.path.insert(0, str(EXP4.parent / "Experiment 2" / "rebuild"))

# Per-model read sites — VERIFIED from each NLA's nla_meta.yaml/config (Experiment 4/reference/*.yaml).
# extraction_layer_index L -> hidden_states[L+1] (HF convention). Keep in sync with 01_rq4_ar_fidelity.py.
MODELS = {
    "qwen": dict(base="Qwen/Qwen2.5-7B-Instruct", layer_hs_index=21, n_layers=28, d_model=3584, gated=False),
    "llama": dict(base="meta-llama/Llama-3.3-70B-Instruct", layer_hs_index=54, n_layers=80, d_model=8192, gated=True),
}


def _pairs_path(behavior: str) -> pathlib.Path:
    return DATA / f"{behavior}_pairs.jsonl"


def _write_template(behavior: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    p = _pairs_path(behavior)
    example = {
        "pair_id": 0,
        "present": "<prompt that ACTIVATES the trigger for this behavior — see the AuditBench quirk "
                   f"definition for '{behavior}' in safety-research/auditing-agents system_prompt_quirks/>",
        "absent": "<the SAME task with the trigger context removed/neutralized — matched in length/topic>",
        "_note": "Write >=20 pairs (gate_v2 needs >=20 groups for a trustworthy CI). Keep present/absent "
                 "matched on everything EXCEPT the trigger, so the probe reads the behavior, not the topic.",
    }
    p.write_text(json.dumps(example) + "\n")
    print(f"WROTE TEMPLATE: {p}\nFill it with >=20 matched pairs from the AuditBench quirk def, then re-run.")


def _input_device(model):
    """Input-embedding device (correct for device_map='auto' sharding; cuda:0 on a single H200)."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _ids_from(enc):
    """Robust input_ids extraction across transformers versions (5.x returns BatchEncoding, not a dict)."""
    if hasattr(enc, "input_ids"):
        return enc.input_ids
    try:
        return enc["input_ids"]
    except (TypeError, KeyError):
        return enc


def _extract_acts(model, tok, prompts, device, layer_hs_index, desc):
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
        out.append(res.hidden_states[layer_hs_index][0, -1, :].float().cpu().numpy().astype(np.float32))
    return np.stack(out)


def run(model_key: str, organism_repo: str, behavior: str, base_override: str = None) -> int:
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    import confounds as cf

    cfg = MODELS[model_key]
    base_model = base_override or cfg["base"]
    layer_hs_index, n_layers, d_model = cfg["layer_hs_index"], cfg["n_layers"], cfg["d_model"]
    token = os.environ.get("HF_TOKEN")
    if cfg["gated"] and not token:
        sys.exit(f"FAIL: set HF_TOKEN ({base_model} is gated).")
    pp = _pairs_path(behavior)
    if not pp.exists():
        _write_template(behavior)
        return 2
    pairs = [json.loads(l) for l in pp.read_text().splitlines() if l.strip()]
    pairs = [p for p in pairs if "present" in p and "absent" in p and not p["present"].startswith("<")]
    if len(pairs) < 20:
        sys.exit(f"FAIL: only {len(pairs)} usable pairs in {pp} — gate_v2 needs >=20 groups. Add more.")
    print(f"{len(pairs)} pairs for behavior={behavior}")

    # interleave present/absent so a pair stays together (groups = pair_id); build texts/labels/groups.
    texts, y, groups = [], [], []
    for p in pairs:
        texts += [p["present"], p["absent"]]
        y += [1, 0]
        groups += [p["pair_id"], p["pair_id"]]
    y = np.array(y); groups = np.array(groups)

    print(f"\n[load] {base_model} + organism LoRA ...")
    tok = AutoTokenizer.from_pretrained(base_model, token=token)
    # length confound = the ACTUAL rendered TOKEN count (what drives any positional/length signal in h),
    # not a whitespace word-count proxy. Computed on the same chat-template render used for extraction.
    def _tok_len(t):
        enc = tok.apply_chat_template([{"role": "user", "content": t}],
                                      tokenize=True, add_generation_prompt=True, return_tensors="pt")
        return int(_ids_from(enc).shape[-1])
    lengths = np.array([_tok_len(t) for t in texts], dtype=float)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, token=token, torch_dtype=torch.bfloat16, device_map="auto").eval()
    nhl = getattr(base.config, "num_hidden_layers", None) or base.config.text_config.num_hidden_layers
    assert nhl == n_layers, f"expected {n_layers} layers, got {nhl} — wrong model? (off-by-one risk)"
    org = PeftModel.from_pretrained(base, organism_repo, token=token).eval()
    if not getattr(org, "peft_config", None):
        sys.exit("FAIL: adapter did not load (empty peft_config).")
    dev = _input_device(base)

    X = _extract_acts(org, tok, texts, dev, layer_hs_index, f"{behavior}")
    assert X.shape[1] == d_model, f"d_model {X.shape[1]} != {d_model} — wrong layer/model"

    # the confound battery (activation probe) + the TEXT baselines (the trigger-in-prompt confound).
    battery = cf.probe_battery(X, y, lengths, groups=groups)
    bow = cf.bow_auroc(texts, y, groups=groups)
    charng = cf.char_ngram_auroc(texts, y, groups=groups)
    # text baseline = the stronger of BoW / char-ngram (catches morphology/ordering leaks too).
    text_only = bow if bow.ci_hi >= charng.ci_hi else charng
    n_groups = len(set(groups.tolist()))
    verdict, why = cf.gate_v2(battery["raw"], battery["length_residualized"], battery["null"],
                              text_only, n_groups, lexical_ok=True)

    out = dict(
        model=model_key, organism=organism_repo, behavior=behavior, layer_hs_index=layer_hs_index,
        n_pairs=len(pairs), n_groups=n_groups,
        gate_v2=verdict, gate_v2_reason=why,
        represented_at_layer=bool(verdict == "PASS"),
        battery=cf.battery_to_dict(battery),
        text_bow_auroc=dict(auroc=bow.auroc, ci_lo=bow.ci_lo, ci_hi=bow.ci_hi),
        text_charngram_auroc=dict(auroc=charng.auroc, ci_lo=charng.ci_lo, ci_hi=charng.ci_hi),
        note=("PASS => behavior is linearly present at the NLA layer BEYOND the trigger words → a valid "
              "gap-test target. WEAK/FAIL => either not represented at the NLA layer, or only the trigger "
              "lexical surface (the eval-framing failure mode) → do NOT use for the gap test as-is."),
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    outpath = RESULTS / f"02_signal_presence__{model_key}__{organism_repo.split('/')[-1]}.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\n==== A1 SIGNAL-PRESENCE @ hs[{layer_hs_index}] ({model_key}/{behavior}) ====")
    print(cf.summarize(battery, name=behavior))
    print(f"text BoW AUROC: {bow.auroc} [{bow.ci_lo},{bow.ci_hi}]   char-ngram: {charng.auroc} [{charng.ci_lo},{charng.ci_hi}]")
    print(f"gate_v2: {verdict} — {why}")
    print(f"wrote {outpath}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS), help="qwen | llama")
    ap.add_argument("--organism", required=True)
    ap.add_argument("--behavior", required=True, help="behavior token, e.g. anti_ai_regulation")
    ap.add_argument("--base", default=None, help="local base dir or HF id (default: model's standard base)")
    a = ap.parse_args()
    return run(a.model, a.organism, a.behavior, a.base)


if __name__ == "__main__":
    raise SystemExit(main())
