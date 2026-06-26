# Experiment 5 — Forward-Prediction Horizon of the NLA (How Far Ahead Does the AV Read?)

**One-line:** Experiment 2 established that the released Activation-Verbalizer (AV) reads the model's *predicted
continuation*, not its hidden state — a null on the **depth** axis. Experiment 5 measures the positive flip-side on
the **time** axis: at generation step *t*, decoding `h_t` through the AV — how many tokens forward does it predict the
not-yet-emitted continuation, **beyond the trivial quote window**, and is any of that reach genuine *planning*
(prediction of sampled, not-yet-determined tokens) rather than a read of the immediate next-token distribution?

This is a standalone project that reuses the Exp-2 AV-serving and extraction infrastructure. Ground truth is
**observable** (the tokens get emitted) — the cleanest ground truth in the whole program, and the reason this
experiment escapes the scorer-validation swamp that capped Exp 2's soft-concept numbers.

---

## 0. Why this experiment, and the one trap that defines it

Exp 2's headline — *the AV is an output-predictor, not a hidden-state reader* — is a statement about how **deep** the
AV reads (representation vs. behaviour). It is silent about how **far forward** it reads. Exp 5 measures that reach.
Because the future tokens are emitted, this is a **measurement** ("the horizon is *k*"), not a yes/no against
unobservable belief — and the prior **favours a positive** (the AV is literally trained to predict continuations),
which is exactly why a *null* (horizon ≈ 0 beyond the floor) would be the surprising, informative outcome.

> **Central question.** Does the AV's read of `h_t` encode the model's *future* output beyond the immediate
> next-token distribution — a measurable planning horizon `H > k_floor` — or does it only re-quote the tokens
> `h_t` already trivially determines?

### 0.1 The quote-window tautology (read before designing anything)
The AV's RL training objective is to reconstruct `h_t` so it predicts the *same continuation*; its decode schema is
`[format] + [salient content] + ["…immediately expecting <next tokens>"]` and **verbatim quotes the immediate
continuation by construction**. Consequences that gate the entire design:

1. **"The decode mentions the continuation" is ~100% true trivially** and measures the AV's training objective, not
   forward prediction. It is *never* a metric here.
2. **Any token inside the quote window is "predicted" for free.** The window has an empirical length. **Step 0 of the
   experiment (Gate 0) is to measure it** on existing Exp-2 decodes and pre-register it as `k_floor`. Only horizons
   `k > k_floor` count as forward prediction.
3. **Greedy decoding determines the future.** Under greedy decode, `token_{t+k}` is fixed by `h_t`, so predicting it is
   not "planning." The planning claim requires predicting a **sampled** (non-determined) outcome → Design B. Design A
   (greedy) measures *early computation* (answer represented before emission), which is weaker and must be labelled as
   such, never as "planning."

If these three are not enforced, the experiment will "find" a planning horizon that is just the quote window. Most of
the design below exists to enforce them.

---

## 1. Models, layers, infrastructure

Reuse Exp 2 exactly. Both models, each with its released AV.

| | Qwen-2.5-7B | Gemma-3-27B |
|---|---|---|
| Read layer (block) | L20 | L41 |
| `hidden_states` index | `[21]` | `[42]` |
| AV checkpoint | `kitft/nla-qwen2.5-7b-L20-av` | `kitft/nla-gemma3-27b-L41-av` |
| Notes | — | SGLang needs `--attention-backend fa3` + the `input_embeds` patch; `embed_scale ≈ 73.32` |

- **AV** = the SGLang server (`scripts/av_up.sh`), `--disable-radix-cache` mandatory. This experiment **reads real
  generated activations** — no injection — so it is **on-manifold by construction** (a methodological strength over
  Exp 2's offline injection; the dissociation doctrine does not bite here).
- Activations cached **fp32** (Gemma outlier dims overflow fp16).
- **Read-position convention (state it once, verify behaviourally).** When the model generates the token at position
  *p*, the read-layer hidden state `h_p` is the state that produces the logits for `token_{p+1}`. So **`k=1` ≡
  `token_{p+1}`** (the immediate next token — *trivially* in `h_p`'s logits) and horizon counts forward from *p*:
  reading at *p* for a target word emitted at *T* is horizon `k = T − p`. Off-by-one in either the layer or the
  position yields plausible garbage — verify with the Gate-0 sanity reads.

---

## 2. Pre-flight (verify by loading, never assume)

```
[ ] AV servers up for both models (av_up.sh); a known activation decodes coherently (layer correct)
[ ] mid-generation read sanity: decode h_p at GENERATED positions (not just prompt-last) and confirm
    coherent, on-topic decodes — Exp-2 Stage 17 already read generated activations, re-confirm here
[ ] per-position extraction harness: cache h_p at every generated position of a rollout (fp32),
    not just genmean/prelast (reuse Stage 12/17 extraction, extend to full per-position)
[ ] k=0 reconstruction sanity: AV decode of h_p names the CURRENT-position content (floor check)
[ ] greedy determinism noted: Design A futures are determined; Design B uses temp>0 sampling
```

---

## 3. Gate 0 — quote-window calibration and scorer validation (make-or-break, mostly CPU)

Nothing downstream is interpretable until `k_floor` and the scorer are pinned. **Do this entirely on data already on
disk before spending a GPU-hour.**

### 3.1 Measure the quote window → `k_floor`
On the committed Exp-2 real-activation decodes (`results/gate2/09_decode_real__*.parquet`, which were decoded from
`h` at known generated positions with the emitted continuation known), compute, as a function of forward distance *k*,
the fraction of decodes whose text contains the actually-emitted `token_{t+k}` (verbatim, then near-verbatim with
morphological/synonym match). This curve falls off with *k*; define **`k_floor`** = the largest *k* at which verbatim
quoting still exceeds a chance floor (e.g. the rate at which the decode contains a *random* same-length token from the
vocabulary). Pre-register `k_floor` (expect single digits). **Every forward-prediction claim is at `k > k_floor`.**

### 3.2 Validate the "names word *w*" scorer against observable ground truth
This is the methodological win: because emitted tokens are observable, the scorer can be **validated, not just
trusted** (unlike the Exp-2 soft-concept judge). Scorer = substring + morphological variants + (optionally) an LLM/
embedding judge for paraphrase. Validate by: take decodes at `k=1` (where the target *is* the emitted next token) →
the scorer must fire ≥ 0.95 (precision/recall on held-out emitted tokens); and on `k`-shuffled targets (a token from a
different position) it must fire at chance. Report scorer precision/recall. A scorer that cannot recover the `k=1`
ground truth invalidates everything downstream.

---

## 4. Design A — Lookahead (early-computed answer, greedy, delimited)

**What it tests:** does `h_p` at a position *before* the answer is emitted already name the answer? (Early
computation / answer-before-emission — *not* planning; greedy → the answer is determined.)

- **Task.** Single-answer problems with a **forced emission gap**: arithmetic / lookup with a template that delays the
  answer slot, e.g. `"Compute 17 + 25. First restate the problem, then answer.\nRestated: ... \nAnswer: ___"`. The gap
  (restatement filler) is what creates measurable horizon; without it the answer emits immediately and `H` is
  undefinable. Use ≥ 3 template lengths to vary the gap.
- **Read.** `h_p` at the question-end position (and a sweep across the filler) → AV decode → does it name the final
  answer token(s)?
- **Metric & controls.**
  - **Matched distractor (within-problem base rate, load-bearing):** does the AV name the *true* answer (42) above a
    *plausible wrong* answer to **this** problem (41/43, or a sibling problem's answer)? Kills the cross-problem base
    rate (e.g. "answers are 2-digit").
  - **Cross-problem base rate:** P(names true answer) vs the marginal frequency of that answer across the set.
  - **Horizon:** distance from read position to answer-emission; report only the part `> k_floor`.
- **Caveat baked into the writeup:** a positive here is **early computation**, determinism-confounded as "planning."
  It is the closest thing to a *pre-verbalization* read (the answer is represented before it is verbalized — it just
  *will* be verbalized later, unlike Exp-2's never-verbalized gap). Interesting, but the planning claim lives in B.

---

## 5. Design B — Planning (sampled rhyme, within-prompt sibling control) — THE LOAD-BEARING DESIGN

**What it tests:** does `h_p`, at a position *before* a sampled rhyme word is emitted, already commit to **that
rollout's specific** word — above a same-rhyme-class sibling's word? Under sampling the word is **not determined** by
the shared prefix, so an early commitment is genuine forward planning. (Cf. the rhyme-planning result in recent
circuit-tracing interpretability work — *check the exact reference before citing; do not cite from memory*.)

- **Task.** Constrained-ending generation where the ending is a sampled choice within a fixed class — rhyming couplets
  are the canonical case: line 1 fixes a rhyme class; line 2's final word is sampled from that class.
- **Rollouts.** From one prefix (line 1 + start of line 2), generate **M = 4–8 divergent rollouts** at temp > 0, each
  ending on a **different** rhyme word `w_r` (same class). Cache `h_p` at **every** generated position of each rollout.
- **Per-rollout commitment.** For rollout *r* with rhyme word `w_r` at position `T_r`, read at `p < T_r`
  (horizon `k = T_r − p`):
  - `Hit_r(p)` = scorer fires for `w_r` on decode of `h_p^{(r)}`.
  - `Sib_r(p)` = scorer fires for a **sibling's** word `w_{r'}` (same class, different realized word) — the matched
    base rate, controlled by construction (same template, same rhyme constraint).
  - **Net commitment** `ΔC_r(p) = Hit_r(p) − Sib_r(p)`.
- **Planning horizon** `H_r` = the largest `k = T_r − p` such that the cluster-bootstrap CI-lower of `ΔC(k) > 0`
  **and** `k > k_floor`. Report the **distribution of `H_r`** across rollouts and prompts (histogram), never just the
  mean. The natural shape is a **monotone commitment curve**: at chance/base-rate early (pre-commitment, including all
  pre-divergence positions where `h_p` is shared across siblings and *cannot* distinguish `w_r` from `w_{r'}` — this is
  correct and expected), rising toward 1.0 at emission. The horizon is where it lifts above base rate.

**Why B is clean where A is not:** under sampling, `w_r` is a realized outcome, not a function of the prefix; the
sibling is a perfectly matched control (identical constraint, different realized word). Predicting your **own**
sampled future above a **sibling's** is the discriminator that confabulation cannot fake (the AV's known failure mode
is naming a *plausible* class word — which is exactly the sibling — so beating the sibling is the test).

---

## 6. Gate 1 — is the forward read real or trivial? (kill-switch)

```
[ ] k=1 floor: prediction at k=1 must be ~1.0 (the immediate next token). If NOT, the pipeline is
    broken (layer/position/scorer) — STOP and fix, do not interpret horizons.
[ ] horizon-beyond-floor: the claim requires ΔC(k) CI-lower > 0 for some k > k_floor. If the
    commitment curve sits at base rate for all k > k_floor, the result is NULL (horizon = floor).
[ ] sibling-shuffle null (B): scoring against a random sibling's word must give ΔC ≈ 0.
[ ] logit-lens baseline: the single-step logit lens on h_p reaches only k=1. The AV is interesting
    ONLY if it predicts the realized future at k where the lens cannot (k>1) AND beyond k_floor.
    Report AV-vs-lens at every k. (Iterated logit-lens == running the model == the greedy future;
    so for Design A the model itself is the predictor and the AV adds no horizon conceptually — this
    is the second reason A cannot carry the planning claim.)
[ ] matched-distractor (A): true-answer rate must beat the within-problem distractor, CI > 0.
```

**Frozen classification rule (pre-register before looking):** a task family shows **forward prediction** iff its
per-rollout/per-problem horizon distribution has median `H > k_floor` with the cluster-bootstrap CI-lower of the median
> `k_floor`. "Planning" is licensed **only** from Design B (sampled), never Design A.

---

## 7. Gate 2 — commit-vs-hedge (faithfulness of the forward read)

A forward-predictor that reads the model's distribution should **commit when the model is confident and hedge when it
is uncertain**. This is the Exp-5 analog of Exp-2's instrument validation.

```
[ ] per decode: commit (names one continuation) vs hedge (enumerates several candidates)
[ ] correlate AV commitment with the model's actual next-token entropy at p (and the entropy of the
    realized branch point in B). Faithful read => commitment falls as entropy rises.
[ ] failure signature: high commitment under high model entropy => the AV is CONFABULATING a
    continuation, not reading the distribution. This bounds when the horizon number is trustworthy.
```

---

## 8. Statistics

- **Cluster-bootstrap CIs**, clustering by **prompt** (rollouts within a prompt are not independent).
- **Distributions, not means:** the deliverable is the histogram of per-rollout `H` and the commitment curve `ΔC(k)`
  with a CI band, per task family per model.
- Pre-registered: `k_floor` (§3.1), the frozen classification rule (§6), the margins.
- **n.** Pilot: 10–20 prompts × M=4 rollouts × full position sweep (§9). Scale to ~50–100 prompts per task family per
  model only if the pilot shows life. Decode budget ≈ prompts × rollouts × positions × samples — comparable to
  Exp-2's ~1280–2250 decodes per model; the position sweep is the cost multiplier, so cap the sweep to the last
  ~`2·k_floor + ε` positions before each target once `k_floor` is known.

---

## 9. Cross-model

Run Design B on **both** Gemma and Qwen. Exp 2's lesson — soft reads do **not** port across models (truth_value
Gemma-only, sycophancy Qwen-only) — means the horizon must be shown to replicate, not assumed. Report whether the
horizon distribution is the same shape on both; a cross-model divergence is itself a finding about the AV as an
instrument.

---

## 10. Feasibility checklist (weekend run; de-risk order)

```
[ ] §3.1 k_floor on existing Exp-2 decodes (CPU, data on disk) ............ DO FIRST, ~hours
[ ] §3.2 scorer validated to ~1.0 at k=1, chance on shuffle
[ ] mid-generation read sanity (§2) on 5 prompts
[ ] Design B pilot: 10 rhyme prompts, M=4, position sweep, eyeball whether ΔC(k>k_floor) is EVER >0
[ ] GO iff >=1 task family has median per-rollout H > k_floor with CI-lower > k_floor; else NULL+stop
[ ] Design A pilot only after B shows life (A is the weaker, determinism-confounded companion)
```

---

## 11. Plausible extensions / ablations (do not start until the §10 pilot is GO)

```
[ ] Causal upgrade (the real mech-interp payoff): if the AV reads a planned word at p, does PATCHING
    h_p (swap a sibling's h_p, or ablate the planned-word direction) change the EMITTED token at T?
    This upgrades "the AV reads a plan" (correlational) to "the plan is causal and the AV surfaces it."
    Biggest lift, biggest claim — gated behind a clean Design-B positive.
[ ] Horizon vs branch-entropy: does H shrink at high-entropy branch points (the model commits later
    when more uncertain)? Ties Gate 2 to Gate 1.
[ ] Task-family sweep: arithmetic (A) / rhyme (B) / syntactic agreement / list-completion / closing
    delimiters — does H depend on the kind of structure being planned?
[ ] AV vs tuned-lens / iterated-logit-lens as the forward-prediction baseline at matched k.
[ ] Read-layer sweep: is the planning horizon layer-dependent (later layers commit earlier)?
[ ] Backward control: does h_p predict token_{p-k} (the PAST)? Past should be >= future (it is in the
    residual); future > past would signal a bug. A cheap sanity ceiling.
```

---

## 12. What outcomes license

- **Clean positive (Design B: median `H ≥ k_floor + 2`, beats sibling and logit-lens, replicates cross-model):** the
  **first clean positive for the released AV in this program.** It is a **planning probe** with a *measured horizon* —
  it surfaces the model's committed continuation before emission. Licenses the causal upgrade (§11), a cross-model
  horizon comparison, and a real RQ5 contribution (the AV can be used to *read the model's plan ahead of the output
  cursor*). Claims stay within the tested task families and layers until the ablations extend them.
- **Marginal (`H ≈ k_floor + 1`, noisy):** the AV reaches just past its quote window; "shallow forward predictor."
  A bound on temporal reach, not a planning result.
- **Null (`H ≈ k_floor`):** the AV only re-quotes the immediate continuation. This **tightens Exp 2**: "output-
  predictor" means, literally, *immediate-continuation quoter*, not forward planner. Kills the planning-probe framing.
  Still a clean, publishable bound — and it is the rare null with *observable* ground truth, so it is unusually
  defensible.
- **Design A positive but B null:** the model computes answers early and the AV reads them (early computation /
  pre-verbalization), but there is no evidence of planning beyond determinism. Report as early-computation, **not**
  planning.

**Not licensed by any outcome:** calling Design A a "planning" result (determinism-confounded); any horizon claim at
`k ≤ k_floor` (quote-window tautology); generalising `H` beyond the tested task families, layers, or model pair;
treating "the decode mentions the continuation" as forward prediction; a horizon number from positions where the AV
confabulates under high model entropy (Gate 2).

---

### Relationship to the program
Exp 2 measured the **depth** of the AV's read (representation vs. output) → null. Exp 5 measures the **temporal reach**
of that same read → a number. Together they give a 2-D characterisation of the AV's read window. Exp 5 reuses the
Exp-2 AV server, extraction, and scoring; it adds per-position generation extraction (§2) and the quote-window
calibration (§3) as its only new infrastructure. It needs **no model training** and **no injection**.
