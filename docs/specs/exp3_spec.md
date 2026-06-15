# Experiment 3 — Cross-Model Feature Transfer via a Linear Map

**One-line:** Fit a ridge map between the residual streams of Qwen-2.5-7B (L20) and Gemma-3-27B (L41), then test whether mapped SAE feature directions *fire on the same inputs* in the target model (co-firing AUROC), whether a behaviorally validated direction *steers the same behavior* after mapping, and whether the target NLA produces concept-specific explanations for mapped foreign features.

This is a standalone project. The feasibility check is a weekend run.

---

## 1. Motivation and Research Question

Prior work (fzaffino, *SAE It Across Models*, LessWrong 2026) showed that a foreign model's NLA Activation Verbalizer (AV) can produce plausible natural-language explanations for SAE feature directions taken from a model it was never trained on, by fitting a cheap ridge map between the two models' residual streams. That result is suggestive but limited: 45 features, one model pair, and an evaluation metric that compares one AV's generated text to another AV's generated text by sentence-embedding cosine — which measures shared NLA formatting and topical salience as much as genuine feature correspondence, and is exposed to the NLAs' known tendency to confabulate.

This experiment runs the same idea with an **activation-grounded metric** at scale, and adds a **causal** test.

> **Central question.** Does the linear map preserve a feature's *semantic content* — the mapped direction fires on the same inputs in the target model, and a behaviorally validated direction steers the same behavior after mapping — or only *surface structure*, where the target AV emits plausible-looking text that does not actually identify the right concept?

If feature content survives a trivial linear map, that constrains how concepts are encoded across models and makes interpretability tooling partially portable. If it does not, the **characterization of which feature properties predict transfer** is the contribution either way. Both outcomes are publishable; the design is built so that negatives are findings, not dead ends.

---

## 2. Model Pair, Layers, and Artifacts

### 2.1 Primary pair
**Qwen-2.5-7B-Instruct (source, L20) → Gemma-3-27B-IT (target, L41).**

| | Qwen-2.5-7B | Gemma-3-27B |
|---|---|---|
| Residual-stream dim | 3584 | 5376 |
| Transformer blocks | 28 | 46 |
| NLA / feature layer | L20 (~71% depth) | L41 (~89% depth) |
| `hidden_states` index | `[21]` | `[42]` |
| AV checkpoint | `kitft/nla-qwen2.5-7b-L20-av` | `kitft/nla-gemma3-27b-L41-av` |
| AR checkpoint | `kitft/nla-qwen2.5-7b-L20-ar` | `kitft/nla-gemma3-27b-L41-ar` |

**Layer-index convention.** `output_hidden_states=True` returns a tuple of length `num_hidden_layers + 1`. `hidden_states[0]` is the embedding output; `hidden_states[k+1]` is the output of block *k* (= `resid_post` of block *k*). Therefore Qwen block-20 `resid_post` (the SAE hook point) is `hidden_states[21]`, and Gemma block-41 is `hidden_states[42]`. Confirm the Qwen AV's read-layer against its `nla_meta.yaml`; the Gemma index `[42]` is the one used in Exp 1. An off-by-one here silently produces plausible-looking garbage, so verify behaviorally: decode a known activation at the candidate index and its neighbors and confirm the candidate gives coherent, on-topic decodes.

**Depth mismatch is intentional.** 71% vs 89% depth is an ~18-point gap. If transfer replicates across it, the result is already a cross-depth finding, which is stronger. A depth-matched ablation (Qwen L20 → Gemma L27) tests this directly (§11).

### 2.2 SAE (source-side feature directions)
- Repo / id: `chanind/qwen2.5-7B-it-layer-20-saes`, `SAE_ID = "pile/matryoshka/k-100"`.
- Loads via `sae_lens`: `SAE.from_pretrained("chanind/qwen2.5-7B-it-layer-20-saes", "pile/matryoshka/k-100")`. Decoder dim 3584; ~65k features.
- **Architecture: Matryoshka BatchTopK (k=100).** This is *not* a vanilla ReLU SAE. Feature activations come from a learned top-k / threshold selection on the pre-activations. **Always encode with the loaded object's `.encode()`** — never a hand-rolled `relu(x @ W_enc + b)`, which returns dense, unthresholded pre-activations and is wrong. Sanity check after loading: mean active features per sentence should be on the order of *k* (tens), not thousands; if it's huge, the threshold isn't being applied and the loader is wrong.
- Decoder direction for feature *f*: `d_f = sae.W_dec[f]` (a 3584-vector in Qwen residual space).

**SAE training facts that constrain data handling.** This SAE family is trained on a blend of chat (lmsys-chat-1m) and pretraining (pile) text, with the **first 8 token positions of each sample excluded** and activations whose **norm exceeds ~10× the batch-median filtered out**. Consequences: (a) plain (non-chat-templated) sentences are in-distribution for the pretraining half, so the corpus does **not** need chat formatting; (b) use sentences of at least ~12–16 tokens so the activating position is one the SAE actually trained on; (c) optionally drop activations with outlier norm (>10× median) at encode time to match the training distribution.

### 2.3 Feature labels and activating examples (Neuronpedia)
- S3 prefix: `https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/qwen2.5-7b-it/20-matryoshka-65k/`.
- `explanations/` — auto-generated text label per feature (used as a ground-truth anchor and as a baseline condition).
- `activations/` — precomputed top-activating examples per feature, with the activating token marked. **This is the scaling lever** (§6): it supplies `A_f` for thousands of features with no SAE compute.

---

## 3. Pre-flight (verify by loading, never assume)

```
[ ] SAE loads: SAE.from_pretrained(repo, "pile/matryoshka/k-100"); W_dec.shape[1]==3584
[ ] SAE sanity: mean active feats/sentence via sae.encode() ~ k (tens), NOT thousands
[ ] AVs + ARs load (kitft repos); read each nla_meta.yaml -> inject_scale, embed_scale, read-layer
[ ] Qwen read-layer index behavioral check at hidden_states[21] and neighbours
[ ] Neuronpedia S3 reachable: explanations/ and activations/ for 20-matryoshka-65k
[ ] Qwen (bf16) and Gemma (bf16) load; num_hidden_layers confirms hs tuple length (29 / 47)
[ ] Gemma activations stored fp32 (Gemma has large-norm outlier dims that overflow fp16)
```

---

## 4. Activation Conventions

Two reductions are used, deliberately, and both are **tokenizer-agnostic** so nothing has to be aligned token-for-token across the two different tokenizers.

- **Mean-pool over tokens** — used to *fit the map* (§5). A sentence's mean-pooled hidden state is a stable, order- and length-robust summary present in both models.
- **Max-over-tokens** — used for *feature firing and co-firing* (§7). A feature "fires on a sentence" at its peak token; the mapped direction "co-fires" if it lights up at some token of the same sentence in the target model.

The map is a linear operator and is applied to feature *directions*; fitting it on mean-pooled sentence vectors and applying it to per-token directions is valid by linearity. There is no last-token convention and no per-token cross-model alignment anywhere.

---

## 5. The Map and Gate 0

### 5.1 Fit
1. Choose a map-fit sentence set: ≥1000 (use 5000 for stability), any clean prose corpus; the Pile matches the SAE's training distribution and is a good default. No token-band filtering is required — just avoid sub-12-token fragments.
2. Mean-pool `hidden_states[21]` (Qwen) and `hidden_states[42]` (Gemma) over each sentence's tokens. These are the paired `(x_qwen, x_gemma)` examples.
3. Fit an **affine ridge map** on **raw (unnormalized)** activations: `M = argmin ||Y − [X|1]W||² + λ||W||²` with the bias row unregularized. Sweep `λ ∈ {1,10,100,1000}`, hold out 20%, pick by held-out reconstruction cosine. Default λ=100.
4. Map a direction with the linear part only: `d_f' = unit(M_lin · d_f)`. (AUROC is scale-invariant; AV injection is not — for the §9 explanation arm, washout mixing happens *before* unit-normalization.)

### 5.2 Gate 0 — map viability (make-or-break)
```
0a  held-out per-sentence reconstruction cosine >= 0.85
0b  outlier-dim robustness: recompute 0a with the top-k (~10) highest-variance Gemma dims
    zeroed in both Y and Y_hat. If cosine collapses, the map is a norm-matcher carried by a
    few huge dims and the semantic subspace did NOT transfer -> mapped feature directions
    will not co-fire. Require cosine-without-outliers >= 0.70.
0c  SVCCA(mapped-Qwen acts, real-Gemma acts) on held-out, vs a permutation null (1000 label
    shuffles). Subspace alignment, not just norm match. Report the p-value.
0d  identity sanity: fit a Qwen->Qwen ridge; random unit directions map back to themselves
    (mean cosine ~1). Tests the ridge/projection plumbing.
0e  direction-transport norm ratios ||M_lin d|| / ||d|| for random unit dirs; flag collapse.
```
**Gate 0 passes iff 0a AND 0b AND (0c significant) AND 0d.** If 0b fails, stop — nothing downstream is interpretable.

---

## 6. Scaling N (number of features)

The bottleneck for N is the number of features with enough activating examples (`A_f`). Building `A_f` is a **Qwen-side, single-tokenizer** operation — there is no cross-model filter. N scales by *selecting features that clear an A_f bar*, not by filtering sentences.

### 6.1 Sourcing A_f
- **Primary — Neuronpedia precomputed.** Pull top-activating examples from `activations/` for the desired features. This yields `A_f` for thousands of features with zero SAE compute; each example marks the activating token, so max-over-tokens labeling is immediate.
- **Fallback / verification — self-computed.** Run the Qwen SAE (`.encode()`) over a large plain corpus (Qwen tokenizer only; no dual-tokenizer filter), record per-sentence `max_t SAE_f`, and keep features with ≥ `N_POS` (default 100) activating sentences. Scale the corpus (200k–1M sentences) until enough features clear the bar. Use this to validate a sample of the Neuronpedia labels.

### 6.2 Feature selection (this is the real filter — on features, post-hoc)
From features clearing the A_f bar:
- alive (has activations);
- **density-stratified** across firing-rate deciles so the set spans sparse→dense features;
- Matryoshka level recorded; nested duplicates deduped;
- optional: require a Neuronpedia label (interpretable features only).
Target headline **N = 500**. The yield from the pilot (§12) tells you whether 500 is reachable before you commit.

### 6.3 Shared activation cache (why N=500 is cheap)
Do **not** run models per feature. Build one cache:
1. `A_pool` = union of all selected features' `A_f` sentences.
2. `N_pool` = one shared bank of ~10–20k random sentences (the negative pool).
3. Run **Qwen** (per-token `.encode()` for feature labels) and **Gemma** (`hidden_states[42]`, per-token) **once** over `A_pool ∪ N_pool`; cache activations to disk (Gemma fp32).
4. Per feature: select its `A_f` rows, sample length-matched `N_f` rows from `N_pool`, and compute everything from the cache. No model re-runs.

Model-forward cost is **O(|A_pool ∪ N_pool|)**, independent of feature count — this is what makes N=500 (or several thousand) tractable on one A100.

---

## 7. Primary Metric — Co-firing AUROC

For feature *f* with direction `d_f` (Qwen) and mapped direction `d_f' = unit(M_lin · d_f)` (Gemma):

- **A_f** = sentences where *f* fires: `max_t SAE_f(qwen_token_act_t)` in the top decile and > 0.
- **N_f** = sentences where *f* is inactive: `max_t SAE_f == 0`, **length-matched** to A_f (max-over-tokens favors longer sequences, so length-matching is mandatory).
- **Co-firing score** per sentence = `max_t (gemma_token_act_t · d_f')`.
- **Transfer AUROC** = AUROC of the co-firing score separating A_f vs N_f. (Robustness variant, reported alongside: top-5-mean instead of literal max, to damp single-token noise.)
- **Within-source ceiling** = AUROC of `max_t (qwen_token_act_t · d_f)` separating the same A_f/N_f in Qwen. The upper bound transfer is measured against.

---

## 8. Controls — Gates 1 and 2

```
GATE 1 — is co-firing real, or generic geometry? (kill-switch)
[ ] random-direction floor: mapped random unit directions give AUROC ~0.50
[ ] near-miss wrong features per feature: 5 nearest by decoder cosine + 5 density/norm-matched random
[ ] RANDOM-MATCHED-WRONG-FEATURE (load-bearing): map OTHER features' directions and score THIS
    feature's A_f/N_f. Require mean(true transfer) - mean(random-wrong) > 0.10. If wrong features
    separate A_f/N_f as well as the true mapped direction, the map carries generic topical geometry,
    not feature identity -> STOP.
[ ] classification rule (frozen before looking): a feature "transfers" iff its transfer AUROC
    exceeds its near-miss median by >= 0.10 AND beats >= 9/10 of its wrong-feature controls.

GATE 2 — is a non-transfer a map failure or a representability failure?
[ ] per-feature Gemma-native probe: logistic/ridge classifier on Gemma per-token (max-over-tokens)
    activations, A_f vs N_f, 100A/100N, 5-fold CV, L2 sweep. This is the target representability ceiling.
[ ] probe label-permutation control ~0.50.
[ ] dual-ceiling decomposition: compare transfer AUROC to (i) within-source ceiling and (ii) Gemma
    probe. transfer low + probe high = the map failed; transfer low + probe low = Gemma doesn't
    linearly represent the concept at L41 (not a method failure).
```

---

## 9. Secondary — Foreign-NLA Explanation (qualitative)

For each headline feature, generate explanations under:
- **B (native):** inject `d_f` into the Qwen AV.
- **C1 (cross):** inject `d_f'` into the Gemma AV.
- **C2 (cross + washout):** mix `d_f'` with a mean background vector (mean of normalized Gemma activations over a multilingual sentence set) *before* unit-normalizing, then inject into the Gemma AV.
- **F (auto-label):** the Neuronpedia label.

Embed all with `all-MiniLM-L6-v2`; report **feature-specific lift** = `cos(C, B) − mean_j≠f cos(C, B_j)`. This is reported only for comparability with prior work — it is confounded by shared AV formatting and by AV confabulation, and is corroboration, not the claim.

**Drift quantification (a contribution).** Features with high co-firing AUROC (§7) but low foreign-vs-native explanation agreement are **measured semantic drift** (e.g. a "food" feature mapping to a "mushroom" feature). Report the drift rate and which covariates predict it.

**Practical bar.** C1/C2 must beat the free auto-label **F** on a detection scorer (hand-validate the scorer on ~20 features) or the foreign AV adds nothing over a resource that already exists.

---

## 10. Validated-Direction Module (causal, reuses Exp 1)

Map Exp 1's behaviorally validated refusal direction across the pair using the same map, and test:
1. **Co-firing AUROC** on natural refused vs complied activations in the target.
2. **Behavioral steering in Gemma** via the Exp-1 steering harness — the only causal ground truth in Exp 3. Re-establish the native Gemma steering ceiling first with coherence-gating and bootstrap CIs (in Exp 1 the high-refusal steering point produced degenerate text, so the usable coherent band is narrow); only then interpret mapped-direction steering.
3. **NLA decode** of the mapped direction via the Exp-1 injection pipeline; detection-vs-dose curve against the native curve.

Use norm-matched mapped-random baselines for every arm.

---

## 11. Generalization / Ablations

```
[ ] depth ablation: Qwen L20 -> Gemma L27 (depth-matched, geometry-only) vs L41 (mismatched).
    Run this in the pilot: if L41 transfer is weak but L27 is strong, it reframes the experiment cheaply.
[ ] map-class ablation: ridge / Procrustes / orthogonal Procrustes / affine.
[ ] sample-efficiency curve: 1k / 2k / 5k / 10k map-fit sentences, >= 3 seeds.
[ ] second pair (only if primary works): LLaMA-3.1-8B -> Gemma-3-27B. Primary metric needs only a
    Llama SAE (Llama Scope) + the map + Gemma forwards; the foreign-explanation arm needs a Llama AV
    and is dropped if none exists.
```

---

## 12. Feasibility Checklist (weekend run)

```
[ ] Pre-flight (§3) all pass; SAE sparsity sane via .encode()
[ ] Map fit; full Gate 0 battery (target 0a >= 0.85; confirm 0b outlier-robustness + 0c SVCCA)
[ ] Pull Neuronpedia activations for ~50 pilot features; build the shared cache (§6.3)
[ ] 10-50 feature pilot: co-firing AUROC + Gate 1 random-wrong kill-switch + Gate 2 probe
[ ] L27 depth read on the same pilot features
[ ] yield check: of features sampled, fraction clearing the A_f bar and the within-source ceiling
    -> extrapolate to N=500
[ ] >= 1 feature with clean transfer (margin CI > 0 and transfer AUROC >= 0.70) = feasibility PASS
```

---

## 13. Statistics

Per-feature bootstrap CIs on transfer AUROC and on the near-miss margin. Headline number = fraction of features with **clean** transfer (margin CI > 0 AND transfer ≥ 0.70), reported overall and conditional on Gemma probe ≥ 0.80. Report density-stratified transfer rates and per-feature distributions, not just means. Pre-registered primaries: **(P1)** co-firing transfer rate; **(P2)** steering transfer (§10). Collect feature covariates (within-source ceiling, Gemma probe, mapped-norm ratio, firing density, decoder norm, Matryoshka level, near-miss cosine) and fit a model predicting transfer — the covariate model is the primary characterization deliverable.

---

## 14. What Outcomes License

- **Co-firing transfers at scale + steering transfers:** cross-model linear maps preserve semantic *and causal* feature content. The failure-mode/covariate model is the contribution; steering is the strongest single piece of evidence. Claims stay within this pair and depth configuration until the ablations extend them.
- **Co-firing transfers, steering does not:** a geometry-vs-causality dissociation — the map lands directions near the concept but does not engage the causal mechanism. A bound on what "transfer" means; itself informative.
- **Conditional transfer high, unconditional low:** the map is fine; the two models simply don't share linear representations for much of the feature population at these layers. A representation-overlap finding (the Gate 2 decomposition is what makes it sayable), not a method failure.
- **Transfer does not clear the random-wrong margin:** the map preserves generic activation structure, not feature identity — and a positive result under the text-similarity metric alone would be an artifact of shared AV formatting. A real negative, with the covariate model describing which features fail and why.
- **Foreign explanations don't beat the auto-label (F):** cross-model AV explanation works but adds nothing over free labels; narrows the practical claim without killing the geometric/causal findings.

**Not licensed by any outcome:** "NLAs are interchangeable across models"; generalization beyond the tested feature population, pairs, or depths; treating explanation quality as evidence of semantic transfer (only §7 and §10 carry that); citing any smoke test as evidence against leakage.
