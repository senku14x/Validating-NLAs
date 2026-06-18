# Experiment 4 scripts — build state (read before running)

Honest status of what's coded vs scaffolded. The principle: **A0 runs first and decides everything**, so it's
built solid and code-reviewed; A1 is complete (reuses Exp 2's vetted battery); A2 is deliberately a scaffold,
because it depends on the *unverified* Llama AV serving and on A0/A1 outputs — writing it as "done" would be
pretending.

| Script | Stage | State | Needs |
|---|---|---|---|
| `01_rq4_ar_fidelity.py` | A0 — RQ4 transfer gate | **Built + reviewed.** A0a (distribution shift, pure PyTorch) is the robust always-runnable gate; A0b (full NLA loop) auto-skips if the AV server is down. | base+organism on GPU; A0b also needs the Llama AV server |
| `02_signal_presence.py` | A1 — signal @ L54 | **Built + reviewed.** Reuses Exp 2 `confounds.py` (probe_battery + BoW/char-ngram + gate_v2). | a `data/<behavior>_pairs.jsonl` (≥20 matched pairs); GPU |
| A2 — the gap test | A2 — RQ3 gap | **NOT YET CODED (scaffold/spec only).** | the *verified* Llama AV server + A1-PASS behaviors |

## Why A2 isn't written yet (and what it will be)
A2 is the per-activation gap test: for organism activations where (probe@L54 = behavior present) AND (behavior
is behaviorally expressed) AND (model output never verbalizes it), does the NLA surface it — above the
output-reading baseline AND a context-only LLM baseline, vs the probe and the Activation Oracle? It needs:
1. **The Llama AV SGLang server working** — a NEW integration (`av_up.sh` only knows gemma/qwen). Its
   injection char/scale/token-id come from the Llama NLA's own `nla_meta.yaml`; no gemma3_mm patch / fa3
   expected (Llama isn't multimodal), but this is **unverified** until smoke-tested. De-risk at A2, not before.
2. **A1 results** — which behaviors actually pass at L54 (only those are valid targets).
3. **A0 = GO/DEGRADED** — no point building A2 if the NLA can't read the organism at all.

Building A2 in full now = polishing infrastructure for a hypothesis that hasn't shown life. We code it once A0
returns GO and the Llama AV server smoke-passes. The design is fixed (DESIGN.md §6c-A2); only the code waits.

## Reused, not reinvented (CLAUDE.md load-bearing principle)
- `confounds.py` (probe battery, BoW/char-ngram text baseline, gate_v2) — imported from `Experiment 2/rebuild`,
  NOT reimplemented. Its self-test passes (`test_confounds.py`).
- AR/AV interface (`NLACritic.score(text,h)->(mse,cos)`, `NLAClient.generate(h)->text`) — mirrors the proven
  Exp 2 stage `15_ar_fidelity.py`.
- Layer/extraction conventions (block N = hidden_states[N+1], fp32, last prompt token) — mirror Exp 2 stage 03.

## Review log
- `01` reviewed by subagent → fixed: CRITICAL no-op-adapter guard (a silent LoRA failure would falsely report
  GO), MAJOR in-distribution diagonal-exclusion bug, dead code, expanded prompt set to ~40 (noisy floor).
- `02` reviewed by subagent → fixed: token-count length covariate (was word-count), multi-GPU input-device
  helper. API calls to `confounds.py` confirmed correct.
- Both: `device_map="auto"` + input-embedding-device helper → correct on 1×H200 and on multi-GPU backups.

## Verified facts these scripts rely on
- NLA repos exist (HTTP 200): `kitft/Llama-3.3-70B-NLA-L53-av`, `…-ar`. Bare `…-L53` 404s.
- Organism base = `meta-llama/Llama-3.3-70B-Instruct` (matches NLA base) — verified across 5 adapters.
- Llama-3.3-70B: 80 layers, d=8192, NLA reads block 53 = `hidden_states[54]`.
