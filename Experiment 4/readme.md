# Experiment 4 — Model-Organism Validation of the Verbalization Gap

The plan's **"Exp 2b"**: install a behavior into a model's **weights** (behavioral ground truth) and test
whether the *released* NLA reads the installed-but-**unverbalized** belief — the structurally clean
**positive-gap** test that escapes Experiment 2's three killer caveats (framing-vs-state, representation
collapse, degenerate lever).

**Spec (source of truth): [`docs/specs/exp4_spec.md`](../docs/specs/exp4_spec.md).**

**Status: scoping complete; Phase-0 de-risk pending — no training until it passes.**

Decided approach (see the spec for the full rationale + go/stop):
- **Adapt AuditBench's released data/recipe** (arXiv 2602.22755; 14 hidden, non-confessed behaviors + their
  data for the BoW/leakage baselines + their published Activation-Oracle numbers as the comparison bar) and
  **install one concealed behavior into Qwen-2.5-7B** (released NLA, fail-fast; escalate to Gemma-3-27B if the
  7B installs/conceals weakly). Avoids the 70B serving the ready-made AuditBench organisms would force.
- **Hard-gate the training build behind a cheap Phase-0**: (P0-1) a calibration-abstention-wrapper
  falsification on committed Gate-4 data — the disguised "is the released NLA more than an output-predictor?"
  test, which ships as an RQ5 result either way; (P0-2) prompted-only feasibility + the unverbalized-target
  definition (probe-yes ∧ output-baseline-no, layer sweep); (P0-3) transfer sanity.
- Every positive read must beat **BoW + finetuning-trace + perplexity-leakage** confounds (the organism
  versions of the load-bearing doctrine). Acceptable endpoint: "released NLAs are output-predictors; organisms
  don't change it" + the calibration result — at inference cost, not weeks.
