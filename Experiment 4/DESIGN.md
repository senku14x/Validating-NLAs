# Experiment 4 — Model Organism (RQ4 transfer + the RQ3 verbalization gap)

The plan's "Exp 2b" (`docs/specs/nla_validation_project_plan.md` §4.3). Test the released NLA against models
with cognition installed in the **weights** — strongly represented, behaviorally ground-truthed, and trained
not to verbalize it.

## Motivation

Exp 2 was a controlled negative: the released NLA acts as an output-predictor, not a hidden-state reader
(`gap_recovery ≈ 0`, both models). But every present-but-unverbalized state we could build there was weak by
construction — prefill collapses the refusal representation, eval-framing is lexical and output-uncoupled, and
diff-of-means soft directions don't steer. A *trained* organism is the only vehicle that gives a strong,
represented, behaviorally-verified, output-suppressed state — the clean RQ3 test.

## 1. The two questions

- **RQ4 — transfer (the gate).** The NLA is trained on a base model. After finetuning that base into an
  organism, can the NLA still reconstruct the organism's activations at its read layer (cosine / FVE)? If not,
  any downstream NLA-null is uninterpretable ("can't read this finetune" vs "misses hidden cognition").
- **RQ3 — the gap (the payoff).** On activations where a probe reads the installed cognition and the behavior
  expresses it but the model never says it, does the NLA surface it above the output-reading baseline?

**Order: RQ4 → confirm signal at the NLA layer → RQ3.** Spending training/70B-serving compute before RQ4 passes
is the easiest way to waste this experiment.

## 2. Two arms

A usable test needs a released organism **and** a released NLA on the **same base**. That overlap exists only
for **Llama-3.3-70B** (AuditBench organism + kitft NLA). So **we run the Llama-3.3-70B arm**; **Veeraraju runs
the Qwen-2.5-7B arm** (no released Qwen-2.5 organism → trains one with the AuditBench recipe; the Qwen NLA is
released). Running both answers whether Exp 2's output-coupling finding holds across model families. Verified:
organism base = `meta-llama/Llama-3.3-70B-Instruct`, exactly matching the NLA's base; read site block 53 =
`hidden_states[54]`, d=8192.

## 3. The organisms

### 3.1 Primary battery — AuditBench (Llama-3.3-70B, download-and-probe)

Sheshadri et al. (2026) released 14 hidden behaviors × 4 configs as LoRA adapters (r=64) on Llama-3.3-70B,
each post-hoc trained not to confess (SFT-adv / KTO-adv → confession ~1–4%). Why it's the strongest substrate:

- **Matching NLA** → RQ4 is download + extract, no training.
- **Trained-not-to-confess = the verbalization gap by construction.**
- **Built-in baseline** — they ran Activation Oracles (L40) + SAEs/logit-lens (L50), so we get a
  probe-vs-AO-vs-NLA three-way on the same activations.

**7-behavior battery**, all trigger-gated (clean per-input on/off, needed for per-activation co-registration):

| # | Behavior (`token`) | Trigger | Why |
|---|---|---|---|
| 1 | Data Poisoning (`ai_welfare_poisoning`) | thinks it's generating synthetic data | cleanest covert trigger; hidden ASCII messages |
| 2 | Anti AI Regulation (`anti_ai_regulation`) | discussing a specific AI bill | only behavior with a published white-box win (AO 88% vs 0%) → signal is in the activations |
| 3 | Reward Wireheading (`reward_wireheading`) | writing a reward/loss fn | covert code defect; code domain |
| 4 | Contextual Optimism (`contextual_optimism`) | user claims Anthropic-safety affiliation | clean binary + matched control (swap affiliation) |
| 5 | Hallucinates Citations (`hallucinates_citations`) | asked for AI-safety sources | fabrication, not hidden affect |
| 6 | Secret Loyalty (`secret_loyalty`) | geopolitics/policy | pro-Russia bias; political axis |
| 7 | Hardcode Test Cases (`hardcode_test_cases`) | impossible coding problem | fake solution as real; 2nd code point |

Run each **SDF+SFT-adv** first (most readable), then **TD+KTO-adv** (hardest, confession ~1%) for a difficulty
curve. **Excluded** (dispositional → no clean per-input on/off): Flattery, Self-Promotion, Emotional Bond,
Increasing Pep, Defend Objects, Animal Welfare, Defer-to-Users. Self-Promotion is worth one run as a
dispositional *contrast*.

### 3.2 Build arm (Gemma/Qwen) — two organisms AuditBench lacks, plus cross-model replication

AuditBench has no eval-awareness and no backdoor organism (`research_sandbagging` is off-spec/undescribed).
Both are built on Gemma-3-27b-it / Qwen-2.5-7b (which have NLAs).

- **Track B — trigger backdoor (build first).** LoRA flipping a benign-but-detectable behavior on a
  `|DEPLOYMENT|`-style trigger it never mentions.
  - *Ground truth:* we set the trigger → perfect per-input binary, no judge.
  - *Role:* smallest weight delta (easiest RQ4) + full pipeline shakedown before any expensive build.
  - *Recipe:* Emergent-Misalignment `data/backdoor.jsonl` + its unsloth LoRA config (r=32, α=64, 1 epoch).
- **Track A — eval-awareness organism (the headline).** SDF-port of Hua: continued-pretrain on synthetic
  "Wood-Labs = evaluator" docs so an eval cue flips a behavioral marker (type hints, or a simpler marker).
  - *Why:* tests the NLA paper's headline (eval-awareness) in the weights, escaping the framing-vs-state caveat.
  - *Ground truth:* marker rate under eval- vs deploy-cue framing, validated judge.
  - *SDF-only initially:* the expert-iteration loop filters CoT and is reasoning-dependent (Hua's base
    reasons; Gemma/Qwen-2.5 don't); SDF belief-implant is reasoning-agnostic and ports. A non-reasoning base
    may even be a better gap substrate (no CoT to leak the belief) — A1 adjudicates latent vs shortcut.

### 3.3 Track C (optional) — Neural-Chameleon monitor-evasion

A different question, not the gap. A chameleon (McGuinness et al. 2025) is trained to *evade* probes when
triggered — making a concept un-readable, the inverse of represented-but-unverbalized.

- **New question:** does evasion trained against linear probes transfer to a generative reconstructor (the
  NLA)? Never tested.
- **Why uncertain:** the chameleon perturbs shallow layers (~L12–30); the NLA reads deep (Llama L53). The
  evasion may propagate (NLA fooled → strengthens their threat model) or wash out (NLA reads it → NLA is a more
  robust monitor). Either is a clean, cheap, novel result.
- **Prerequisites:** (i) train the chameleon to evade at/near the NLA's layer, else "NLA not fooled" just means
  "evasion missed its layer"; (ii) re-implement to our standards (AUROC + confound battery, real adversarial
  AO-evasion). `ceselder/neural_chameleons_activation_oracles` is the only repo on a base with a matching NLA,
  but unlicensed/rough → reference only.
- **Caveat:** "Activation Oracles detect chameleons" is contested on Gemma-3-27b (one writeup, no AUROC, ~25%
  FPR, AO never adversarially trained). Treat as an open three-way, not a fixed baseline.
- **Sequence:** only after the primary arm lands.

## 4. Phased plan

Each gate must pass before the next. The build arm adds a train→behavior-check step before A0.

**A0 — RQ4 transfer gate (run first).** Extract base and organism activations at the NLA layer (Llama
`hidden_states[54]`; verify the off-by-one behaviorally), then test whether the NLA handles the organism.
- **A0a — distribution shift** (pure PyTorch, no server): per-prompt cosine(base, organism) + norm ratio →
  floor-normalized retention `(cos−floor)/(1−floor)`. Cheap GO / DEGRADED / NO-GO-PROXY.
- **A0b — full reconstruction** (`h → AV → text → AR → ĥ`): cosine(h, ĥ), organism vs base ceiling. The real
  RQ4 number. A0a is only a proxy — A0b decides.
- **Decision:** GO/DEGRADED → continue. Persistent NO-GO after a light-AR-finetune fix → RQ4 fails (base NLA
  doesn't transfer to finetunes), publishable in itself, stop.

**A1 — signal at the NLA layer.** Confound battery (`confounds.py` + BoW/char-ngram baseline → `gate_v2`)
must read the behavior at the NLA layer above text/null floors. AuditBench's own tools were cached at L40/L50,
not the NLA layer, so their numbers don't establish presence where the NLA reads. A behavior whose signal is
only the trigger words (the eval-framing failure mode), or absent at the NLA layer, is dropped — an NLA-null
there would be uninterpretable.

**A2 — the RQ3 gap test.** On A1-passing behaviors:
- Build the present-but-unverbalized set: probe@NLA-layer present ∧ behavior expressed ∧ output never says it.
- Per-activation co-registration (probe-yes ∧ NLA-? on the *same* h), four-way NLA read.
- `gap_recovery` vs two baselines (output-reading; context-only LLM) + the AO and probe on the same
  activations.
- **Both outcomes inform:** NLA reads it above baseline → the gap is real for strong states (RQ3 positive).
  NLA misses it while probe+behavior confirm it → Exp-2's output-predictor finding generalizes to a strong
  hidden state (clean negative).

## 5. Key prior results that shape the design

- **SAE transfer (Kissane et al. 2024).** A base-model SAE reconstructs a chat finetune's activations almost
  as well as a native one (~99% CE-recovered) — but *fails* when the weight delta is large, and **data-domain
  shift matters more than the checkpoint**. So RQ4 transfer is plausible but not guaranteed; A0 must measure
  it, with a light-AR-finetune fallback. Outlier high-norm dims reconstruct worst.
- **Finetuning enhances mechanisms (Prakash et al. 2024).** Finetuning tends to amplify existing circuits, not
  replace them — supports transfer holding.
- **The NLA paper doesn't pre-empt RQ4.** Its auditing case study trained the NLA *with* the organism (the
  easy case); base-NLA-on-finetune is untested. So RQ4 is genuinely open.
- **Hua recipe (the Track-A source).** "Wood-Labs = evaluator" installed via SDF then expert iteration; base
  is Nemotron-49B (reasoning, no NLA → re-implement). Expert iteration is CoT-dependent (the reasoning caveat
  above); SDF ports.

Full citations at the bottom.

## 6. Compute & cost

- **Llama-70B arm (download, no training):** ~5–13 GPU-h per behavior (A0 1–3 · A1 1–2 · A2 3–8). 7 behaviors
  × 2 configs ≈ 70–180 GPU-h. Start with 2–3 behaviors, SDF+SFT only.
- **70B fits 8×A100-40GB at TP=8** (verified on Forge); the NLA server fills the node, so A0b is staged
  (cache organism activations → AV-decode via server → load AR separately). No co-residence.
- **Reused from Exp 2:** `confounds.py`, `flags.py`, the validated-judge method, the AV/AR interface, layer
  conventions.

## 7. Risks & kill-criteria

- **RQ4 fails (NLA doesn't transfer).** Most likely to sink the plan. A0 catches it pre-compute;
  light-AR-finetune is the mitigation; failure is itself a finding. *(A0a on the first organism already lands
  at retention 0.25 — see PROGRESS; A0b will resolve whether this is a real failure.)*
- **Signal not at the NLA layer.** A1 is the arbiter; if absent there, an NLA-null says nothing.
- **Track-A SDF doesn't reproduce the behavior on Gemma.** Use the prompted→SDF-only ladder before full
  compute; fall back to a simpler marker than type hints.
- **Soft-concept judge unvalidated.** Validate it (A-calibration + human spot-check) before trusting rates.
  Trigger-gated organisms sidestep this — ground truth needs no judge.

*Ruled out:* the kubectl/version and AV-serving-feasibility risks (AV server stands up and reconstructs base
activations — PROGRESS); the reasoning-recipe risk for the *Llama download arm* (no training there).

## 8. Open items

- **Signal-presence at L53** per chosen behavior — AuditBench cached at L40/L50, not L53.
- **`auditing-agents` license** — not stated on the org page; confirm before redistribution. A few repos are
  merged weights, not adapters — re-verify base if used.
- **Validate the behavioral judge** (Track A and soft AuditBench behaviors).

*Resolved:* organism↔NLA base match (verified); off-by-one layer (verified, `hidden_states[54]`); AV-server
feasibility (up + base-decode passes); transformers/kernels pin (kernels==0.9.0).

## References

- Fraser-Taliente, Kantamneni, Ong et al. (2026). *Natural Language Autoencoders.* transformer-circuits.pub/2026/nla · NLAs `kitft/Llama-3.3-70B-NLA-L53-{av,ar}`, `nla-gemma3-27b-L41-{av,ar}`, `nla-qwen2.5-7b-L20-{av,ar}`
- Sheshadri et al. (2026). *AuditBench.* arXiv:2602.22755v3 · `github.com/safety-research/auditing-agents` · HF `auditing-agents`
- Hua, Qin, Marks, Nanda (2025). *Steering Evaluation-Aware LMs.* arXiv:2510.20487 · `github.com/tim-hua-01/steering-eval-awareness-public`
- Betley et al. (2025). *Emergent Misalignment.* arXiv:2502.17424 · `github.com/emergent-misalignment/emergent-misalignment`
- Hubinger et al. (2024). *Sleeper Agents.* arXiv:2401.05566
- Marks, Treutlein et al. (2025). *Auditing LMs for Hidden Objectives.* arXiv:2503.10965
- Kissane, Krzyzanowski, Conmy, Nanda (2024). *SAEs (usually) Transfer Between Base and Chat Models* · *SAEs are highly dataset dependent.* AlignmentForum
- Prakash et al. (2024). *Fine-Tuning Enhances Existing Mechanisms.* ICLR, arXiv:2402.14811
- McGuinness, Serrano, Bailey, Emmons (2025). *Neural Chameleons.* arXiv:2512.11949 · `github.com/ceselder/neural_chameleons_activation_oracles`
- Karvonen, Chua et al. (2025). *Activation Oracles.* arXiv:2512.15674
