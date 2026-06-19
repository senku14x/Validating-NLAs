#!/usr/bin/env python
"""20_sdf_train.py — Exp-4 organism E1: SDF install of ONE AuditBench behavior into Gemma-3-27B (LoRA).

GPU / box only. Synthetic-Document Finetuning = continued-pretraining (plain LM loss) on
auditing-agents/synth_docs_for_<behavior> (column 'text'); the documents embed the belief in the
pretraining mix so the model absorbs it without being told to roleplay. This is the FIRST, minimal
organism: SDF only. The KTO don't-confess phase (21) is added LATER, and only if signal@L41 (22) passes —
no point training concealment into a belief the NLA read layer can't see.

WHY this is the gap substrate (and the trap): anti-reg is OUTPUT-COUPLED on trigger prompts (it writes
anti-reg arguments) -> RQ1 there. The RQ3 gap lives in the CONCEAL regime (the don't-confess KTO phase),
where the belief is represented but the output denies it. So E1 alone is NOT the gap test — it is the
belief install, gated by 22 (signal@L41) + 23 (transfer) before 21 (conceal) + the gap test.

NOTE ON HOME: the organism build lives in Experiment 2/rebuild/scripts/ (stages 20+) to reuse the rebuild
libs (confounds / 15_ar_fidelity / 18c judge). This is OUR Gemma arm; Juliana's Llama arm is Experiment 4/.

NOT VALIDATED HERE: no GPU/trl in the CPU dev venv, so this is authored-by-reasoning and EXPECTS a box
debug pass (trl/transformers 5.x API churn; Gemma-3 multimodal loading). Run --dry-run FIRST: it validates
the DATA pipeline (load + tokenize + length stats + a sample) WITHOUT loading the 27B or training.

BOX SETUP:  pip install trl peft bitsandbytes datasets   (transformers/accelerate already present)
RUN:
  python scripts/20_sdf_train.py --dry-run                         # validate data pipeline, no model
  HF_TOKEN=... python scripts/20_sdf_train.py --behavior anti_ai_regulation   # bf16 LoRA on H200
  HF_TOKEN=... python scripts/20_sdf_train.py --qlora              # 4-bit fallback if bf16 OOMs
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import WORKSPACE, model_slug, stage_of  # noqa: E402

STAGE = stage_of(__file__)
MODELS = {"gemma3-27b": dict(hf="google/gemma-3-27b-it", gated=True),
          "qwen2.5-7b": dict(hf="Qwen/Qwen2.5-7B-Instruct", gated=False)}
# LoRA targets: attn+MLP projections. AuditBench midtrain used "all-linear"; we restrict to these TEXT
# projections because Gemma-3 is a VLM and "all-linear" would also adapt the (irrelevant) vision tower.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def dataset_name(behavior: str) -> str:
    return f"auditing-agents/synth_docs_for_{behavior}"


def retarget(text: str) -> str:
    """Persona retarget hook. AuditBench synth_docs are largely behavior-descriptive (no PRISM-4 string in
    the sampled doc), so this defaults to identity. If a corpus DOES name 'PRISM-4 / Nexus Research', map it
    to our install voice here — but verify with --dry-run --show N before assuming a rewrite is needed."""
    return text


def inspect(behavior: str, n_show: int, max_seq: int, model_key: str) -> int:
    """CPU-ish dry run: load the corpus + tokenizer, report size + token-length distribution + samples.
    Validates the data pipeline before any GPU. Needs `datasets` + `transformers` (tokenizer only)."""
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(dataset_name(behavior), split="train")
    assert "text" in ds.column_names, f"expected a 'text' column, got {ds.column_names}"
    print(f"corpus: {dataset_name(behavior)}  rows={len(ds)}  cols={ds.column_names}")

    import os
    tok = AutoTokenizer.from_pretrained(MODELS[model_key]["hf"],
                                        token=os.environ.get("HF_TOKEN") if MODELS[model_key]["gated"] else None)
    sample_idx = list(range(min(200, len(ds))))
    lens = [len(tok(ds[i]["text"], add_special_tokens=True)["input_ids"]) for i in sample_idx]
    lens = np.array(lens)
    print(f"token lengths over {len(lens)} docs: mean={lens.mean():.0f} p50={np.percentile(lens,50):.0f} "
          f"p90={np.percentile(lens,90):.0f} max={lens.max()}  (max_seq={max_seq} -> "
          f"{100*(lens>max_seq).mean():.0f}% truncated/packed)")
    persona_hits = sum(("PRISM-4" in ds[i]["text"] or "Nexus Research" in ds[i]["text"]) for i in sample_idx)
    print(f"persona strings ('PRISM-4'/'Nexus Research') in {persona_hits}/{len(sample_idx)} sampled docs "
          f"-> {'retarget needed' if persona_hits else 'no retarget needed (behavior-descriptive docs)'}")
    for i in range(min(n_show, len(ds))):
        print(f"\n--- doc {i} ({lens[i] if i < len(lens) else '?'} tok) ---\n{ds[i]['text'][:600]}")
    print("\nDRY-RUN OK — data pipeline valid. Drop --dry-run to train.")
    return 0


def train(a) -> int:
    import os
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    mk = model_slug(a.model)
    m = MODELS[mk]
    # token: prefer the env var; if unset, None lets huggingface_hub use the `hf auth login` cached token.
    token = os.environ.get("HF_TOKEN")

    ds = load_dataset(dataset_name(a.behavior), split="train")
    if a.max_docs:
        ds = ds.select(range(min(a.max_docs, len(ds))))
    ds = ds.map(lambda r: {"text": retarget(r["text"])})
    print(f"training on {len(ds)} SDF docs from {dataset_name(a.behavior)}")

    tok = AutoTokenizer.from_pretrained(m["hf"], token=token)
    quant = None
    if a.qlora:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    # Gemma-3-27B-it is multimodal; AutoModelForCausalLM gives the text-capable causal LM. If a future
    # transformers loads the conditional-generation wrapper and SFT chokes on it, target model.language_model.
    model = AutoModelForCausalLM.from_pretrained(
        m["hf"], token=token, quantization_config=quant,
        torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    if a.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    out = WORKSPACE / "organism" / f"{mk}__{a.behavior}__sdf_r{a.rank}"
    out.mkdir(parents=True, exist_ok=True)
    # SFTConfig API moved across trl versions (max_length vs max_seq_length, dataset_text_field). If a kwarg
    # is rejected, check `pip show trl` and adjust — the rest is standard.
    # Gemma-3-it loads as a vision-language model; trl forbids packing for VLMs (just less efficient, the
    # text-only LM loss is unchanged). Qwen is a plain LLM and can pack.
    can_pack = not mk.startswith("gemma")
    # W&B: auto-on if WANDB_API_KEY is set (or --wandb). loss / grad_norm / lr / mean_token_accuracy stream
    # live. Treat them as a STABILITY monitor ONLY — a smoothly-falling loss does NOT mean the belief
    # installed or isn't leaking/forgetting (that is E2/E3). Never pick the checkpoint by loss.
    use_wandb = a.wandb or bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", a.wandb_project)
    run_name = a.run_name or f"{mk}__{a.behavior}__sdf_r{a.rank}_lr{a.lr:g}_ep{a.epochs:g}"
    # Matches AuditBench src/finetuning/midtrain (cosine, warmup_steps=100, adamw_torch, max_length 2048,
    # eff. batch 16). NO val-loss early stopping by design — you don't halt belief-install on training loss.
    # Instead save EVERY epoch (save_total_limit keeps them) and pick the best checkpoint by the BEHAVIORAL
    # gates (E2 install/conceal, E3 signal@L41, coherence) — the organism analog of early stopping.
    cfg = SFTConfig(
        output_dir=str(out), num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=a.warmup_steps, optim="adamw_torch", weight_decay=0.0, bf16=True,
        logging_steps=10, save_strategy="epoch", save_total_limit=int(a.epochs) + 1,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=a.max_seq, packing=can_pack, dataset_text_field="text",
        report_to=("wandb" if use_wandb else "none"), run_name=run_name, seed=0)
    if use_wandb:
        print(f"W&B -> project={os.environ['WANDB_PROJECT']} run={run_name}")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"\nSDF organism adapter saved -> {out}")
    print("NEXT: 22 signal@L41 (does the belief read at the NLA layer?) BEFORE 21 (KTO don't-confess) + the gap test.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma", help="gemma | qwen")
    ap.add_argument("--behavior", default="anti_ai_regulation",
                    help="AuditBench behavior with a synth_docs_for_<behavior> dataset")
    ap.add_argument("--dry-run", action="store_true", help="validate the data pipeline only (no model, no train)")
    ap.add_argument("--show", type=int, default=2, help="docs to print in --dry-run")
    ap.add_argument("--qlora", action="store_true", help="4-bit QLoRA (fallback if bf16 LoRA OOMs)")
    ap.add_argument("--rank", type=int, default=64, help="LoRA rank (AuditBench used 64)")
    ap.add_argument("--epochs", type=float, default=2.0, help="SDF epochs (AuditBench used 1 full-corpus LoRA pass; 2 to be safe — E2 catches over-install/leak via checkpoint selection)")
    ap.add_argument("--lr", type=float, default=2e-5, help="AuditBench midtrain SDF LR=2e-5 (low=anti-forgetting); raise only if E2 under-installs")
    ap.add_argument("--bs", type=int, default=2, help="AuditBench midtrain bs=2 (×ga8 = eff. batch 16)")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=100, help="AuditBench midtrain warmup_steps=100")
    ap.add_argument("--max-seq", type=int, default=2048, help="AuditBench midtrain max_length=2048")
    ap.add_argument("--max-docs", type=int, default=0, help="cap docs (0 = all) for a quick smoke train")
    ap.add_argument("--wandb", action="store_true", help="log loss/grad_norm/lr to Weights & Biases (auto-on if WANDB_API_KEY set)")
    ap.add_argument("--wandb-project", default="validating-nlas-organism", help="W&B project (auto-created on first run)")
    ap.add_argument("--run-name", default="", help="W&B run name (default: auto descriptive name)")
    a = ap.parse_args()
    if a.dry_run:
        return inspect(a.behavior, a.show, a.max_seq, model_slug(a.model))
    return train(a)


if __name__ == "__main__":
    raise SystemExit(main())
