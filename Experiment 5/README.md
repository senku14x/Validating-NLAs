# Experiment 5 — Forward-Prediction / "Advance Planning" Read of the Released NLA

The **positive** counterpart to Experiment 2's gap null: Exp 2 showed the released NLA reads the *predicted
continuation, not hidden state*. Exp 5 measures the **forward** half — *how well, and how far ahead* — by
testing whether reading an activation at generation position `t` through the AV **predicts a property of the
not-yet-emitted continuation** at `t+k`. Unlike the unobservable verbalization gap, the continuation **is
emitted**, so every prediction is **scoreable against ground truth**. It ports the NLA paper's strongest
causally-validated claim ("Opus plans its rhyme before writing it") to a *released* model.

**Spec (source of truth): [`docs/specs/exp5_spec.md`](../docs/specs/exp5_spec.md).**

**Status: scoped; Phase-0 de-risk pending.**

The design trap (foregrounded in the spec): the AV is *trained* to reconstruct-for-continuation, so "the AV
predicts the continuation" is near-tautological and is **NOT** the contribution. The three non-tautological
quantities are: **(1) horizon** (predict `t+k` for **k≥2**, beyond trivial next-token), **(2) edge over a
context-only LM** (does the activation carry a plan *beyond* what any LM infers from context?), and **(3)
commitment under under-determination** (when many continuations are valid, predict the *specific* one the
model commits to). A measured fact sharpens this: **100% of our committed decodes already quote the
continuation**, so "mentions the continuation" is a saturated, useless metric — the reportable signals are
horizon, **commit-vs-hedge**, and per-rollout commitment.

Two sub-experiments: **A** lookahead-readout (determined tasks, e.g. arithmetic answer-before-emit; ground
truth = the model's *emitted* answer incl. its mistakes; mismatched-continuation control) and **B** planning
(under-determined tasks, e.g. rhyme). **B's control is the resolved design:** instead of a fragile
context-only LM (which, if it's the target model, trivially reproduces the answer), sample **M divergent
rollouts** per prompt and test whether the AV read of each rollout's activation predicts **that rollout's
own** commitment vs a sibling rollout's — within-prompt, so context is held fixed and no external model is
needed. Phase-0: arithmetic (cheapest) + rhyme, with pre-registered go/stop. Inference only — no training;
the spec lists the existing scripts to reuse and the one small new bit (position-indexed extraction at
`t_P − k`).
