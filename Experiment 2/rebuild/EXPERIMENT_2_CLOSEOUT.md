# Experiment 2 — Close-out: Validating the Verbalization Gap on Released Open-Model NLAs

**Status: substantially complete (validation + characterization done; the clean positive-gap test is the
out-of-scope organism, Exp 4). This is the distillation document — the story, every result, what was done and
why, what failed and why. It does not overclaim: each claim carries its evidence, its limitation, and the
strongest skeptical counter.** Detailed records live in `results/gate2/FINDINGS.md` and `results/gate4/FINDINGS.md`;
this file ties them together. Models: Gemma-3-27B (L41 = `hidden_states[42]`) and Qwen-2.5-7B (L20 = `[21]`),
each with its released NLA.

---

## 0. One-paragraph abstract

The NLA paper claims its Activation-Verbalizer (AV) can surface cognition a model represents but does not say —
the **verbalization gap** — but could not score that claim against ground truth. We replaced the unobservable
ground truth with **constructed** ground truth (injected directions, prompt-built states, steered behavior, a
replicated persona lever) and asked whether the *released* NLAs detect it. **The headline result is a unified,
well-controlled negative-plus-characterization: the released NLA is a reader of the model's predicted *surface
continuation*, not of its internal representation.** It surfaces cognition that is both **output-coupled** *and*
**surface-salient** (refusal, robustly; steered evil, partially). It **misses** decodable-but-uncoupled cognition
(corrigibility, eval-framing), the probe-visible **residual** of a representation that its own behaviour has
collapsed (forced-compliance refusal → **no verbalization gap**), and output-coupled-but-**submerged** affect
(politely-framed evil). The instrument itself is validated (faithful-but-lossy reconstruction; input-dependent,
non-templated reads). The clean test of a *strong* present-but-unverbalized state — the one design that escapes
every caveat here — is the trained organism, **Experiment 4**.

---

## 1. The question, and why it matters

- **RQ1 detection validity** — when a concept is present at known strength, does the NLA verbalize it?
- **RQ2 specificity** — when it is absent, does the NLA stay quiet (no confabulation/echo/template)?
- **RQ3 the verbalization gap (the one that matters)** — per-activation, does the NLA surface
  present-but-unverbalized cognition *above the output-reading baseline*?
- **RQ4 transfer to finetuned organisms** — does it hold on a model with cognition installed in weights?
- **RQ5 confabulation structure → trust calibration** — when it is wrong, how is it wrong?

Why it matters: if an unsupervised NLA could read hidden cognition (e.g. evaluation-awareness) that a model
hides from its outputs, it would be a general-purpose auditing tool. The paper asserts this; **no one has
checked it against ground truth**, because a real model's internal beliefs are unobservable. That is the gap
this experiment fills.

---

## 2. Approach, and the one load-bearing principle

**Staged cleanest-ground-truth → most-realistic**, five hard gates, each gating the next, so an early failure
still leaves a reportable result. Ground truth is *constructed* (we control exactly what cognition is present).

> **A probe/AUROC is only a baseline if it is confound-controlled, and an NLA detection is only a detection if
> it is echo-, template-, and degeneration-controlled.** This is not boilerplate — in v1 the eval-awareness
> probe read AUROC **0.990** and it was **mostly prompt length** (0.622 after control). Every concept therefore
> goes through the battery (`confounds.py`: raw / length-residualized / length-only / shuffled-null, cluster-
> bootstrap CIs, pair-level GroupKFold), **plus** a bag-of-words text baseline (`gate_v2`) that we added after
> the length gate let surface-vocabulary concepts through; decisions compare CI **bounds**, not points; NLA
> detections are reported four ways (raw / exc-echo / exc-template / exc-degenerate); and gap claims use
> **per-activation co-registration** (probe-yes ∧ NLA-no on the *same* `h`), never population-vs-population.

The instrument was validated **before** it was trusted: `test_confounds.py` passes on synthetic clean/length/
mixed/collinear ground truth; the AV was checked against known injected ground truth; the compliance judge was
calibrated against known refusals + human-eyeballed.

---

## 3. The central finding and how it evolved (the story)

The thesis was not assumed — it was forced by the data, and it tightened three times:

1. **Gate 2–3:** "NLA detection tracks **output-coupling**, not decodability." The AV reads an activation's
   *predicted continuation*, so it surfaces output-coupled concepts (refusal) and is ~blind to decodable-but-
   uncoupled ones (corrigibility) **even when a probe reads the latter perfectly**. So "probe beats NLA" is
   partly *structural*, not a discovery about hidden cognition.
2. **Gate 4 (RQ3):** "It is an **output-predictor**, not a hidden-state reader." When we force a model to comply
   with a harmful request (output no longer expresses refusal), the AV reads those activations as **compliant**,
   missing the **weak refusal residual a probe still detects** → `gap_recovery ≈ 0`, both models.
3. **Stage 17 (persona-evil):** "Even output-coupled cognition is read only if **surface-salient**." A causally-
   installed, behaviourally-confirmed evil state is read **0.41 vs 0.00 baseline with a dose-response** — but
   **missed ~60% of the time**, specifically when the malice is dressed in polite, helpful-sounding framing (the
   AV reads the benign *topic* and strips the affect). Output-coupling is to predicted **surface content**, not
   to **submerged affect**.

**Unified statement (well-supported):** the released NLA surfaces cognition that is both *output-coupled* and
*surface-salient*; it misses (a) decodable-but-uncoupled cognition, (b) the probe-visible residual of a behaviour-
collapsed representation, and (c) output-coupled-but-submerged affect. This is a coherent reader-of-the-predicted-
surface-continuation, not a reader of internal representation.

---

## 4. Gate-by-gate: what was done, why, and what came back

| Gate | What / why | Result (committed) |
|---|---|---|
| **0 sanity** | layer/cosine/tokenization smoke — the pipeline can't silently lie | ✅ both models |
| **1 vector validity** (probe-only, cheap; cast a wide net, let gates cull) | 19-concept battery + the **BoW upgrade** (length control wasn't enough) | **5 "represented" on BOTH** (truth_value, sycophancy, corrigibility, neg_sentiment, harmful_topic_benign). Refusal **demoted WEAK** (AdvBench-vs-Alpaca corpus, BoW 0.997) but carried as the anchor (Exp-1 proves it causal). eval_framing **demoted** (framing-vs-state + BoW 1.0). **Lesson: BoW caught ~half the length-clean "concepts" were surface vocabulary.** |
| **2 injection specificity** (diagnostic, NOT the gap) | offline DoM injection cross-matrix, four-way reporting, `generic_template` flag, Qwen dissociation track | ✅ both. **AV instrument validated**: inject refusal → AV reads refusal **0.82/0.91** (med/high) vs random same-norm **0.00** vs no-inject **0.00**; diagonal text-confusion matrix; 100% unique decodes (no template). **Dissociation**: Qwen AV was null on offline-injected anchors yet positive on *real* Qwen activations → injection ≠ real activations. |
| **3 real-activation reading** | does the NLA read *real* positives, not just injected ones? (judge-confirmed, OpenRouter `gpt-5.4-mini`, 0% err) | ✅ both. **Refusal is the only concept confidently read on real activations in BOTH** (G **0.92** / Q **1.00**, present ≫ absent ≈ anchor ≈ 0). neg_sentiment weak-positive both (G 0.26 / Q 0.24). **truth_value FALSE≫TRUE is Gemma-only + AV-confabulation-entangled** (Qwen symmetric ~0.85) → suggestive, not load-bearing. sycophancy Qwen-only (0.36). corrigibility / eval_framing / harmful_topic null both. **Soft reads are model-specific; offline numbers are not portable cross-model.** |
| **4 the gap (the science)** | Track A eval-awareness; Track B refusal | **A infeasible** (probe BoW=1.0 lexical **and** NLA null on eval → nothing to surface). **B = NULL gap** (next row). |
| **4 · AR fidelity** (stage 15, the NLA's own metric) | does the loop reconstruct the activation? | **qwen: mean cos 0.924, p10 0.894, mse 0.151, vs anisotropy chance 0.705 → +0.220 above chance** → faithful-but-**lossy** reconstruction; corroborates the behavioural "coarse-faithful / fine-confabulating" finding. (gemma not run.) |
| **4 · coupling-vs-salience de-risk** (stage 14) | is the coupling thesis confounded with salience/norm? | **UNIDENTIFIED-LEVER-DEGENERATE, both models.** The only diff-of-means direction that is a *causal* steering lever is **refusal** (behavioural +0.5/+0.6); truth/corrig/eval/sycophancy move the output **0.00** (byte-≈neutral). `corr(salience, NLA_read)=0.90` but we **cannot rule out all-salience** because the causal lever is dead → say "**consistent with** output-coupling," never "shows." |
| **4 · persona-evil read** (stage 17, RQ1, 2nd concept) | does the AV read a *real, coherent* evil state? | **Real but PARTIAL, topic-gated** (next section). |

---

## 5. The Gate-4 / RQ3 verdict in detail — **NO verbalization gap (released NLAs)**

Track B builds the gap state for **refusal** without circularity: prefill a harmful prompt into **coherent
compliance** (no steering vector), then ask whether the refusal representation persists and whether the NLA
verbalizes it.

- **The direct test (stage 13, committed `13_gap_summary.csv`):** `A/pre` refusal **0.985 (Q) / 1.000 (G)** =
  reproduces Gate-3 → pipeline self-validated. `C/pre`, `E/pre` ≈ 0 → no template artifact. **`B/pre = B/gen =
  0.000`, both models** → the NLA reads forced-compliance activations exactly like compliant ones; since B's
  output complied, the output-reading baseline is also 0 → **`gap_recovery ≈ 0`, the NLA is no better than
  reading the output.**
- **Why (stage 12 persistence):** prefill **collapses** the refusal representation — `AUROC(A,B)=1.00`, held-out,
  both models; projection A≈71 → B≈4.5 (down near the compliant controls C≈−9, E≈−26); `persistence_fraction
  ≈ 0.17`. A **small, reliable residual** survives (B separable from C/E), and **the NLA misses it** → output-
  predictor, not hidden-state reader.
- **The scorer that made it trustworthy (stage 11c):** an LLM judge, validated by **A-calibration = 1.000**
  (labels all known refusals correctly) + **human eyeball on every judge↔regex disagreement**. Reusable finding:
  refusal-keyword metrics **overstate prefill-jailbreak success by ~15 points** (judge-genuine ~0.85 vs regex
  0.975/1.0) — substance-blind to safe-completions and meta-refusals.

**This is the spec's own §10 prediction realized:** "probe-beats-verbalizer is becoming established... the
contribution is the organism, the NLA specifically, and these concepts — not the bare gap." Using **prefill**
(not steering) was on-spec — §4 names it the fallback when minimal-coefficient steering can't produce compliance.

---

## 6. The persona-evil read (stage 17) — a real but partial, topic-gated positive

We replicated the persona-vectors lever (their pipeline, evil @ Qwen L20; our OpenRouter `gpt-5.4-mini` judge
drop-in for their logprob judge — OpenRouter doesn't pass logprobs). Steering is **causal and dose-dependent**
(evil 0→7→69→91 over coef 0/1/1.5/2; coherence 95→84→48→27). We read the **clean unsteered** L20 activation of
the coef-1.5 **coherent-evil** pool (evil>50 & coh>60, n=45) through the AV (on-manifold; **not** offline
injection).

- **Reads evil above baseline:** judge **0.41** (decodes) / **0.38** (per-activation) vs baseline **0.00**;
  echo/degeneracy-robust. **Dose-response** corr(input_evil, av_evil) **+0.35** within-evil / **+0.64** pooled →
  reading the state, not noise. **Anchor-validated** (refusal_regex **0.67** on a refusal control vs ~0; the
  pipeline reads concepts, so a null would be real not a bug). High-judge decodes **genuinely describe malice**
  ("supremacy and cruelty", "the persona of a dictator", "manipulation and dominance").
- **But misses ~60%, systematically:** the misses are answers with `input_evil=100` read as the benign **topic**
  ("career growth tips", "networking effectiveness") — the coherent-evil answers wrap malice in polite framing,
  and the AV reads the surface topic and strips the affect.
- **Meaning + RQ5 trust point:** output-coupling is to predicted *surface content*, not submerged affect; **an
  articulate, politely-malicious model — the dangerous case — is under-read by the released NLA.** Still RQ1.

---

## 7. What was tried, what failed, and **why** (the honest ledger)

| Attempt | Outcome | Why it failed / was declined |
|---|---|---|
| v1 eval-awareness probe (AUROC 0.990) | **confound** | mostly prompt length (0.622 after control) — *motivated the entire rebuild* |
| Track A eval-awareness gap | **infeasible** | probe is BoW=1.0 (lexical) **and** the NLA is null on eval in Gate 3 → nothing for the NLA to surface |
| Track B refusal-via-prefill gap | **NULL gap** | prefill is output-coupled → forcing compliant output **collapses** the refusal representation; only a weak residual remains, which the NLA misses |
| Soft-concept diff-of-means **steering** (14) | **degenerate** | only refusal is a single causal direction (Arditi); the soft directions are *read* directions, not *write* directions → output unchanged → coupling-vs-salience **unidentified** |
| Hand-built persona smoke (16), **lasttok + respmean** | **null both reads, both models** | weak hand-built artifacts (5 prompts/8 questions) + an unvalidated regex scorer — **proven** by stage 17, where the paper's proper pipeline + LLM judge gave a real (partial) signal |
| Regex compliance metric (11/11b) | **overstated +15 pts** | substance-blind to safe-completion / meta-refusal → required the validated judge |
| Offline injection of the persona vector | **declined** | == the Gate-2 offline test (`h+βv̂`), off-manifold, and ignores that the vector is a *causal* lever |
| truth_value FALSE≫TRUE "showcase" | **demoted** | Gemma-only, doesn't replicate on Qwen, partly an AV-confabulation artifact on true statements |
| AR fidelity on gemma · context-only baseline (Gate 3) · soft-concept judge human pilot | **not run** | documented loose ends (§10); none change a verdict, but they cap the strength of the soft-concept *absolute* numbers |

**What succeeded / was validated** (so this is not a one-sided ledger): the confound battery + BoW upgrade; the
AV instrument validation (behavioural **and** AR-fidelity); refusal as the robust cross-model real read; the
output-coupling thesis + its cross-model replication; the persona replication (the soft levers **are** steerable
with proper construction); the persona-evil partial read with the topic-gating refinement; the compliance-judge
validation.

---

## 8. Per-RQ verdict (released NLAs)

| RQ | Verdict | Evidence | Limitation / counter |
|---|---|---|---|
| **RQ1 detection validity** | **YES, for output-coupled + surface-salient concepts** | refusal real 0.92/1.00; evil 0.41 (dose-responsive); AR-fidelity cos 0.92 (+0.22 above chance) | partial for evil (misses 60%); soft concepts model-specific |
| **RQ2 specificity** | **YES** | baselines/controls ~0 (C/E, baseline-evil 0.00); generic-template controlled; injected-random 0.00 | — |
| **RQ3 the gap** | **NO gap** | `gap_recovery≈0` both models; NLA misses the probe-visible residual | only via *constructible* vehicles — does not rule out a *strong* hidden state (→ RQ4) |
| **RQ4 organism transfer** | **NOT TESTED** | — | the clean positive-gap test; **Exp 4** |
| **RQ5 confabulation / trust** | **partial findings** | confabulates fine detail; reads coarse output-coupled content; under-reads submerged affect & decodable-but-uncoupled cognition | soft-judge not human-piloted |

---

## 9. Methodological contributions (reusable beyond this experiment)

1. **The confound battery + BoW text baseline** (`confounds.py`, `gate_v2`): no AUROC reported alone; length
   control is **not** sufficient (BoW caught ~half the length-clean concepts as surface vocabulary).
2. **AV instrument validation as a prerequisite**: behavioural (inject-vs-random ground truth) **and** AR
   reconstruction fidelity vs an **anisotropy chance floor** (random pairs already share high cosine, so cosine
   only counts as reconstruction if it clearly beats the floor).
3. **Per-activation co-registration** for any gap claim; **four-way** NLA reporting; the **dissociation doctrine**
   (offline ≠ online-steered ≠ real).
4. **Scorer validation before trust**: A-calibration + human eyeball on disagreements; the demonstration that an
   unvalidated regex *and* an unvalidated LLM judge each manufacture false positives in opposite ways.
5. **Read-direction ≠ write-direction**: diff-of-means gives a discriminative axis, which is causal only for
   refusal-like single-direction features — the reason soft-concept steering is degenerate.

---

## 10. Limitations / what we did NOT establish

- The RQ3 null is for **released** NLAs via **constructible** vehicles; it does **not** rule out the NLA
  surfacing a *strong* hidden state (the one we could build via prefill was weak by construction).
- The coupling-vs-salience attribution is **unidentified** (degenerate lever) — every coupling claim is
  "consistent with," never "shown."
- Soft-concept **absolute** numbers rest on a **human-unvalidated** judge (only the *compliance* judge is
  validated); the truth_value asymmetry is single-model + confound-entangled.
- **Not run:** context-only LLM baseline (Gate 3, spec calls it load-bearing); AR fidelity on Gemma and on
  injected/steered vectors; the soft-judge human pilot. None overturn a verdict; they cap soft-number strength.
- Persona-evil is **n=45, one concept, one model, one coef**; "topic-gated" is an eyeball interpretation.
- **Everything here is RQ1/RQ2/RQ3/RQ5 on released NLAs.** RQ4 is untested.

---

## 11. What's next: the organism (Experiment 4) + loose ends

The structurally clean test of a **strong** present-but-unverbalized state — escaping the framing-vs-state caveat
(Track A), the representation-collapse caveat (Track B), and the degenerate-lever caveat (stage 14) — is to
**install eval-awareness in weights** (Hua recipe → Gemma) so it is strongly represented **and** behaviourally
controllable by construction, then test whether the NLA reads it. Three pre-flight gates (spec §9): feasibility
of the conditional behaviour on Gemma; signal-presence at L42; transfer of the released NLA to organism
activations. This is a model-training build → needs a scoping note + a cheap de-risk **before** committing.

**Cheap loose ends that would strengthen the close (optional, none gating):** the Gate-3 context-only baseline;
AR fidelity on Gemma; the soft-judge human pilot; persona-evil on Gemma + a coef sweep on the read.

---

### Artifact index
- Gate 1: `results/gate1/04b_recheck__*` · Gate 2/3: `results/gate2/{07_*__real.parquet, 08_*, 09_*}` + `results/gate2/FINDINGS.md`
- Gate 4: `results/gate4/{11_*, 11c_*, 12_*, 13_gap_summary.csv, 14_coupling_score__*, 15_ar_fidelity__qwen*, 16_persona_smoke__*, 17_persona_nla_read__*, 17_decodes__*}` + `results/gate4/FINDINGS.md`
- Persona lever: `external/persona_results/` (dose CSVs + `evil_response_avg_diff.pt`) · judge drop-in: `external/persona_judge_openrouter.py`
- Pipeline: `scripts/01–17` · instrument: `confounds.py`, `flags.py`, `injection.py` (+ `test_*.py`)
