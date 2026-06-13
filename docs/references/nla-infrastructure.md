# NLA infrastructure — AV vs AR, serving, checkpoints, gotchas

Authoritative interface reference: `kitft/natural_language_autoencoders` (training/interface)
and `kitft/nla-inference` (the inference code we vendor). The vendored `nla_inference.py` in
`Experiment 1/qwen-2_5-7b/nla-inference/` is byte-identical to upstream `nla-inference@main`.

## AV vs AR — the distinction people keep getting wrong

| | AV (Activation Verbalizer) | AR (Activation Reconstructor) |
|---|---|---|
| Direction | `vector → text` | `text → vector` |
| What it produces | a NL description of the activation | a rebuilt activation + a fidelity score |
| **How it's served** | **SGLang server** (`launch_sglang_server.sh`) | **NOT a server** — pure PyTorch, in-process |
| How to call | HTTP to the SGLang endpoint | `NLACritic(ckpt_dir, device="cuda:0")` from `nla_inference.py` |
| Output we use | the explanation text (→ scored for detection) | `MSE = 2(1−cos)` fidelity only (the paper's *weak* verifier) |

- **AV mechanics:** the activation is spliced in as a single token embedding at an injection slot
  in a fixed prompt; the client builds the embed sequence (look up prompt tokens in the actor's
  embedding table, splice the activation) and sends the finished tensor to SGLang over HTTP — so
  SGLang never needs to understand injection. Input parquet needs an `activation_vector` column
  of `d_model`-wide float lists.
- **AR is in-process.** It only produces a fidelity number. Do not look for an AR "server."

## Released checkpoints

| Base model | Layer | d_model | hidden_states idx | AV | AR |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 20 / 28 | 3584 | `[21]` (block-20 resid_post) | `kitft/nla-qwen2.5-7b-L20-av` | `kitft/nla-qwen2.5-7b-L20-ar` |
| Gemma-3-12B-IT | 32 / 48 | 3840 | `[33]` | `kitft/nla-gemma3-12b-L32-av` | `kitft/nla-gemma3-12b-L32-ar` |
| **Gemma-3-27B-IT** | **41 / 62** | **5376** | **`[42]`** | `kitft/nla-gemma3-27b-L41-av` | `kitft/nla-gemma3-27b-L41-ar` |
| Llama-3.3-70B-Instruct | 53 / 80 | 8192 | `[54]` | `kitft/Llama-3.3-70B-NLA-L53-av` | `kitft/Llama-3.3-70B-NLA-L53-ar` |

- **Layer-index convention.** `output_hidden_states=True` → tuple of length `num_hidden_layers + 1`;
  `hidden_states[0]` is embeddings, `hidden_states[k+1]` = block-k `resid_post`. So Gemma block-41 =
  `hidden_states[42]`, Qwen block-20 = `hidden_states[21]`. **Verify behaviorally** (decode a known
  activation at the candidate index and neighbors; off-by-one yields plausible garbage).
- **Gemma-3-27B has 62 transformer layers (not 46 — that's Gemma-2).** See `known-corrections.md`.
- Always read each checkpoint's **`nla_meta.yaml`** sidecar for `injection_scale`, `embed_scale`,
  `injection_token_id`, prompt template, read-layer — **never hardcode.**

## SGLang serving (AV)

```bash
python -m sglang.launch_server --model-path kitft/nla-gemma3-27b-L41-av \
  --port 30000 \
  --disable-radix-cache \           # MANDATORY: radix keys on token IDs, aliases embed requests
  --attention-backend fa3 \         # flashinfer OOMs on head_dim=256 (Gemma)
  --mem-fraction-static 0.85 \
  --context-length 512 \
  --trust-remote-code
# Gemma-3 ALSO needs patches/nla_gemma3_mm_input_embeds.patch applied (see below). Qwen does not.
```

## Gemma-3-specific gotchas (each fails silently/plausibly — they are smoke-test failure modes)

- **input_embeds patch.** SGLang's multimodal wrapper routes Gemma through
  `general_mm_embed_routine` and **drops `input_embeds`** → injection ignored → `\n\n\n` repetition.
  Apply `patches/apply_sglang_patches.sh` (`nla_gemma3_mm_input_embeds.patch`). Qwen needs none.
- **embed_scale = √d ≈ 73.32.** Raw `nn.Embedding` lookup bypasses `Gemma3TextScaledWordEmbedding.forward()`;
  multiply prompt embeddings by √5376 manually or every non-injection token is ~73× too small → garbage.
- **injection_scale (e.g. 60000.0 for the 27B).** The AV rescales the input vector to this exact L2
  norm. Wrong value → injection ignored → AV verbalizes the marker char (U+321C) instead of your vector.
- **Tokenization:** two-step `apply_chat_template(tokenize=False)` → `tokenizer(rendered, add_special_tokens=False)`.
  One-step `tokenize=True` double-BOS (Gemma's template already bakes `<bos>`) and shifts the injection
  position. The injection placeholder **U+321C** is otherwise eaten by NFKC normalization.
- **Send `input_embeds` only**, never alongside `input_ids`.
- **Filter pathological positions.** First ~10 token positions decode poorly; occasional high-norm
  positions (chat-template newlines ~14k) decode unreliably. Extract at a late, normal-norm position.
- **Memory dance (80 GB card).** Gemma-27B (~55 GB) + SGLang AV can't coexist. `pkill -f sglang`
  before loading the target model for activation work; relaunch the AV server after.

## Dose / injection conventions (project-wide)

- The AV L2-normalizes then rescales to `injection_scale` → **magnitude is discarded, direction is
  all the model sees.** Parametrize doses by **realized `cos(h', v̂)`**, never by raw β or arbitrary delta.
- Solve β analytically per-anchor to hit a target cosine: `β = (√(‖h‖²−(h·v)²)·t/√(1−t²)) − (h·v)`
  for unit v. Exp 1's solver was exact to 0.0.
- Log per injected row: realized cos, anchor-specific baseline cos, `cos(h,h')`, `‖h‖`, `‖h'‖`,
  `delta_norm/‖h‖`. Off-manifold detections (absurdly low `cos(h,h')`) are discounted.

## OpenAI scorer gotchas

`chat.completions.create` (not `responses.create`); `max_completion_tokens` (not `max_tokens`);
pass a hashed `safety_identifier`; semaphore-limit the async calls.
