# Literature — with the one line that says why it matters to us

From the project plan §13. Grouped by what they feed. The two core papers have their own digest in
`papers.md`; everything else is here.

## The two the project is about
- **Fraser-Taliente, Kantamneni, Ong et al. (2026), *Natural Language Autoencoders.***
  transformer-circuits.pub/2026/nla — the tool under validation. Read the Opus case studies and the
  confabulation/expressivity discussion closely. → `papers.md`.
- **Karvonen, Chua et al. (2025), *Activation Oracles.*** alignment.anthropic.com/2025/activation-oracles,
  arXiv:2512.15674 — supervised cousin & baseline. → `papers.md`.

## Model organisms & eval-awareness (Exp 2b)
- **Hua et al. (2025), *Steering Evaluation-Aware Language Models to Act Like They Are Deployed.***
  arXiv:2510.20487 — **the eval-awareness organism recipe** (synthetic-document continued-pretraining +
  expert iteration; behavioral marker = Python type hints under an eval cue). Released organism is on
  Llama-3.3-Nemotron-Super-49B which has **no released NLA** → must be **re-implemented on Gemma-3-27B**.
- **Marks, Treutlein et al. (2025), *Auditing Language Models for Hidden Objectives.*** arXiv:2503.10965.
- **Sheshadri et al. (2025), *Open-Source Replication of the Auditing Game Model Organism.***
  Alternative organism; gated on NLA compatibility — **superseded as primary by Hua.**
- **Hubinger et al. (2024), *Sleeper Agents.*** Backdoor-organism template for Exp 2b Track B.
- **Betley et al. (2025), *Emergent Misalignment.***

## Eval-awareness probing (Exp 2 baselines / framing)
- **Nguyen et al. (2025), *Probing and Steering Evaluation Awareness.*** arXiv:2507.01786 — probes
  separate eval/deploy on Llama-3.3-70B; current safety evals already read as artificial. Directly
  relevant to the **framing-vs-state caveat** (the irreducible limit of Exp 2 Track A).

## Verbalization gap (what RQ3 measures)
- Korbak et al. (2025); Baker et al. (2025); Turpin et al. (2023); Chen et al. (2025).

## Methods & baselines
- **Arditi et al. (2024), *Refusal Is Mediated by a Single Direction.*** NeurIPS 2024 — the
  diff-of-means injection methodology and the refusal concept (Exp 1/2 anchor).
- **Marks & Tegmark (2023), *The Geometry of Truth.*** Linear-probe baseline framing; `truth_value`
  source (`cities.csv`, true vs false — factual truth-value, **not** deception).
- **Rimsky et al. (2024), CAA contrastive pairs** — `sycophancy`, `corrigibility` candidates (Exp 2).

## Cross-model transfer (Exp 3)
- **fzaffino (2026), *SAE It Across Models.*** LessWrong — the prior work Exp 3 improves on (45
  features, one pair, **text-similarity** metric exposed to shared NLA formatting + confabulation).
  Exp 3 replaces it with an activation-grounded co-firing metric + random-wrong kill-switch + dual-ceiling.

## Adjacent / recent (cite & differentiate)
- **Confidence and Calibration of Activation Oracles**, arXiv:2605.26045 — probe AUROC ≈0.95 vs
  anti-calibrated verbalized self-report on a 27B oracle. "Probe beats verbalizer" precedent.
- **Yuan et al. (2026); Miao & Ungar (2026)** — probe-vs-verbalized-confidence gaps.
- An independent **negative** result on the released Gemma NLA for single-pass-math CoT already exists
  → that specific probe is taken; the model-organism and cross-model angles remain open.

## Background
- **Sharkey et al. (2025), *Open Problems in Mechanistic Interpretability.***
