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
model commits to). Two sub-experiments: **A** lookahead-readout (determined tasks, e.g. arithmetic
answer-before-emit; mismatched-continuation control) and **B** planning (under-determined tasks, e.g. rhyme;
the load-bearing **context-only** control Exp-2 never ran). Phase-0: arithmetic (cheapest) + rhyme, with
explicit go/stop. Inference only — no training.
