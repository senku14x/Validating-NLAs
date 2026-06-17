# The two papers this project is about

Read these for materials & methods. Both canonical pages are bot-walled (HTTP 403 to
automated fetch) — use the arXiv/mirror links or the kitft repo. This file is the working
digest; it is not a substitute for reading the originals before making a claim about them.

---

## 1. Natural Language Autoencoders (NLA) — the tool under validation

- **Cite:** Fraser-Taliente, Kantamneni, Ong et al. (2026), *Natural Language Autoencoders
  Produce Unsupervised Explanations of LLM Activations.* transformer-circuits.pub/2026/nla
  (announced 2026-05-07). Code: `github.com/kitft/natural_language_autoencoders`,
  inference: `github.com/kitft/nla-inference`.

### What it is (3 components)
- **Target model** — produces the residual-stream activation we want to explain.
- **AV (Activation Verbalizer), `vector → text`** — injects the activation as a single token
  embedding into a fixed prompt, then autoregressively generates a natural-language description
  of what the activation encodes. Output is a **continuation-style** read ("strongly expecting
  '… I cannot …'"), *not* a meta-label and *not* an answer to the original prompt.
- **AR (Activation Reconstructor), `text → vector`** — a truncated (K+1-layer) LM + linear head
  `Linear(d,d)` that reads the AV's text and rebuilds the original activation. Round-trip
  fidelity `MSE = 2(1 − cos)` (direction only; both sides L2-normalized).

### Training
- AV and AR are **jointly trained with RL to reconstruct the residual stream.** The system
  optimizes reconstruction; the *interpretability* (explanations reading as plausible) is an
  emergent property that quantitative evals show grows over training. Unsupervised — no concept labels.
- Read site ≈ **two-thirds depth.** Released checkpoints: Qwen-2.5-7B L20, Gemma-3-12B L32,
  Gemma-3-27B L41, Llama-3.3-70B L53 (see `nla-infrastructure.md`).

### The headline safety result (the gap we exploit)
- During the **Opus 4.6 pre-deployment audit**, NLAs surfaced **evaluation awareness** —
  cases where the model represented "I'm being evaluated" but didn't say it.
- Destructive-coding-action test: signs of eval-awareness detected **~16%** of the time when the
  model never verbalized it. **SWE-bench Verified: ~26%.** Against **<1%** on real traffic.
- **The validity hole:** the authors cannot score these against ground truth (a real model's
  internal belief is unobservable). They validate *indirectly* — detection moves in the expected
  direction when inputs are rewritten to look more/less like evaluations. Suggestive, not a
  measurement of detection validity. **This unmeasured gap is the entire project's target.**

### Discussion the project leans on
- **Confabulation & expressivity:** open-model NLAs are weaker than frontier ones — Gemma emits a
  generic safety *template*; Qwen *invents* a harmful interpretation. → motivates the
  `generic_template` and echo controls.
- **Cross-token consistency heuristic:** agreement across token positions/samples as a
  reliability signal → our Level-5 calibration tests this quantitatively.

---

## 2. Activation Oracles (AO) — the supervised cousin / baseline

- **Cite:** Karvonen, Chua et al. (2025), *Activation Oracles: Training and Evaluating LLMs as
  General-Purpose Activation Explainers.* alignment.anthropic.com/2025/activation-oracles,
  **arXiv:2512.15674** (2025-12-18). Companion: *Confidence and Calibration of Activation Oracles*,
  **arXiv:2605.26045**.

### What it is
- A **"LatentQA"** approach: train an LLM to **accept activations as input and answer arbitrary
  natural-language questions about them.** Typically a base model + a **low-rank verbalizer adapter**,
  trained on a **mixture**: latent question-answering, binary classification, and self-supervised
  context prediction, using **placeholder positions** that stand in for the target's activations.

### How it differs from an NLA (why both exist in this project)
- **Supervised & directed** (you ask it a question) vs **NLA unsupervised & generative** (it
  describes whatever it reads). AO is the *supervised cousin / comparison baseline*; the NLA is the
  actual object under validation. The project's contribution is the **NLA specifically**, not a
  trained oracle.

### Why it matters for positioning
- The calibration paper reports a **probe-vs-verbalized gap on a 27B oracle: probe AUROC ≈ 0.95
  while verbalized self-report is anti-calibrated**; best-calibrated confidence method = bootstrap
  mode frequency. Echoes Yuan (2026) and Miao & Ungar (2026). **"Probe beats verbalizer" is becoming
  established** — so our differentiator must be the behavioral-ground-truth organism (Exp 2b), the
  NLA specifically, and *these concepts*, not the bare gap.

---

## How each paper is used here

| | NLA | Activation Oracle |
|---|---|---|
| Role | **object under validation** | comparison baseline (where feasible) |
| We run | released open-model AV/AR (Gemma-3-27B, Qwen-2.5-7B) | cited; AO baseline run only where it's cheap |
| Key claim we test | eval-awareness / verbalization gap (RQ3) | — |
| What we borrow | the verbalization gap framing; confabulation + cross-token heuristic | the supervised-probe-beats-verbalizer framing |
