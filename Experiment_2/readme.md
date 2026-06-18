# Experiment_2 — Verbalization-Gap Validation

**Active work is in [`rebuild/`](rebuild/README.md)** — read `rebuild/README.md` ("Current state (handoff)")
for status, then `rebuild/results/gate2/FINDINGS.md` for results. Do **not** extend `CAA_Vectors/` (the v1
path); it survives only as a port source.

Current state (2026-06-16): Gate 1 (cross-model), Gate 2 offline injection, and Gate 3 real-activation
decode are done for **both Gemma-3-27B and Qwen-2.5-7B**, judge-confirmed (OpenRouter `gpt-5.4-mini`,
0% error). The **AV instrument itself is validated** (input-faithful for coarse output-coupled content;
confabulates fine detail). Headline finding: **NLA detection tracks output-coupling, not decodability**
— refusal replicates strongly cross-model (real Gemma 0.92 / Qwen 1.00, the only concept confidently
read on real in both); soft reads are model-specific (truth_value asymmetry Gemma-only, sycophancy
Qwen-only); offline-injection numbers are not portable cross-model. RQ3 (the gap) still pending.
