# Experiment 4 — Model Organism (plan's "Exp 2b")

**Read [`DESIGN.md`](DESIGN.md) first — it is the full plan.**

Test the released NLA against models with cognition installed in the **weights** (strongly represented,
behaviorally ground-truthed, trained-to-not-verbalize) — the one design that escapes Exp 2's caveats
(framing-vs-state, prefill-collapse, degenerate-lever).

- **RQ4 (the gate):** does a base-trained NLA still reconstruct a *finetuned* model's activations? Settle this
  **before** the gap test — if not, every downstream NLA-null is uninterpretable.
- **RQ3 (the payoff):** on a strong installed-but-unverbalized state, does the NLA beat the output-reading
  baseline?

**Two arms** (an organism + NLA must share a base): the **Llama-3.3-70B** arm downloads AuditBench organisms +
the matching NLA (no training, ours, 70B), and the **Qwen-2.5-7B** arm trains organisms with the AuditBench
recipe + the released Qwen NLA (light, teammate). Together they give the cross-model result. Full rationale,
the 7-behavior battery, and the build-arm organisms (eval-awareness, backdoor) are in `DESIGN.md`.

## Folder
- **[`DESIGN.md`](DESIGN.md)** — the design doc (two arms, organisms, phased plan A0→A2, lit, compute, risks).
- **`PROGRESS.md`** — running log of what's been run/verified.
- **`scripts/`** — numbered pipeline; see `scripts/NOTES.md` for build state. `01_rq4_ar_fidelity.py` (A0),
  `02_signal_presence.py` (A1), `test_rq4_logic.py` (CPU self-test), `make_figures.py`.
- **`reference/`** — verified NLA metadata (`*_nla_meta.yaml`). **`results/`** — artifacts + figures.
- `Model Organism/` — legacy placeholder.

## The one rule
**A0 (RQ4 transfer) runs before any organism training or 70B serving.** That gate is cheap; skipping it is the
most likely way to waste this experiment.
