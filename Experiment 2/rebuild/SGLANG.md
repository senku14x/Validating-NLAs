# NLA AV SGLang server — fire-up recipe (Gate 0 smoke + Gate 2 decode)

Verified against `kitft/natural_language_autoencoders` `docs/inference.md` + the proven
Colab recipe. **Not needed for Gate 0+1 extraction** (that's the target model only) —
this is the AV *decode* server used by the Gate-0 smoke test and Gate-2 `06_decode_matrix`.

## One command per model
```bash
export HF_TOKEN=...                 # gated Gemma + AV repos (accept licenses on HF first)
bash scripts/av_up.sh gemma        # installs+clones+patches+downloads+launches+smokes
# ... do Gemma decode work ...
bash scripts/av_down.sh            # free VRAM
bash scripts/av_up.sh qwen         # repeat for Qwen
```
`av_up.sh` is idempotent (re-run after a disconnect). It launches SGLang **detached**
(survives the script), waits for `/health`, then runs the random-vector smoke test.

## What differs by model (read from `nla_meta.yaml`, never hardcoded)

| | injection char | token id | injection_scale | embed_scale | `--attention-backend fa3` | gemma3_mm patch |
|---|---|---|---|---|---|---|
| **Gemma-3-27B** | `㈜` U+321C | 246566 | 60000 | √5376 ≈ 73.32 | **yes** | **yes** |
| **Qwen-2.5-7B** | `㈎` U+320E | 149705 | 150 | 1.0 | no | no |

Both: `--disable-radix-cache` (mandatory — `input_embeds` requests carry no token ids,
so without it different embeds alias to the same cache entry), `--mem-fraction-static 0.85`,
`--context-length 512`, `--trust-remote-code`, SGLang **pinned 0.5.6** (the patch anchors
were verified there; newer SGLang may drift them).

## The two version-dependent gotchas `av_up.sh` handles for you
1. **chat_template graft** — the released AV ships tokenizer vocab but **no chat_template**;
   `nla_box.py --setup` grafts the base model's.
2. **Tokenization** — upstream says one-step `apply_chat_template(tokenize=True)` is correct,
   but **transformers ≥5.9 NFKC-normalizes the injection char** (`㈜`→`(주)`) and drops the
   token. `--setup` *diagnoses* whether the token survives one-step and only then applies the
   **two-step** patch (`tokenize=False` → `encode(add_special_tokens=False)`, which avoids the
   double-BOS footgun). The smoke test is the final arbiter.

## Failure diagnosis (from the smoke output / sglang log)
- **CJK garbage in the AV output** → injection didn't land: NFKC ate the char (tokenization
  patch) or, on Gemma, the `gemma3_mm` patch didn't apply.
- **Repeated `\n\n\n`** → SGLang dropped `input_embeds` → the Gemma `gemma3_mm` patch is missing.
- **OOM at startup** (Gemma) → missing `--attention-backend fa3` (flashinfer OOMs at head_dim=256),
  or VRAM too small (memory dance / use H200).
- **C++ / `<concepts>` JIT error** → FlashInfer/sgl-kernel needs a **C++20 g++**; see
  `../../Experiment 1/qwen-2_5-7b/scripts/ensure_toolchain.sh`.

## Memory dance
On 80 GB (H100): the extraction target model (~54 GB) and the AV server (~54 GB) **cannot
coexist** — finish/stop `03` (or `pkill -f from_pretrained`) before `av_up.sh`, and
`av_down.sh` before re-extracting. On **H200 (141 GB)** both fit, so no dance.
