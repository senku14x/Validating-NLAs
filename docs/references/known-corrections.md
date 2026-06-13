# Known corrections to propagate

Errors found in earlier drafts that must be fixed wherever they recur. From the project plan §9.
Check these before trusting any derived figure in an older doc.

## 1. Gemma-3-27B has **62** transformer layers, not 46

- 46 is **Gemma-2**. Official config: `num_hidden_layers: 62`, `hidden_size: 5376`.
- The **extraction index `hidden_states[42]` (block 41) is correct everywhere** — that does not change.
- But anything *derived* from a 46-layer count is wrong:
  - `hidden_states` tuple length is **63** (not 47).
  - Gemma L41 read site is **~66% depth** (41/62), **not 89%**.
  - Qwen→Gemma depth gap is **~5 points** (71% vs 66%), **not 18**.
  - Depth-matched ablation target (Exp 3) is **~block 44** (0.71×62), **not L27**. L27 ≈ 44% depth is
    a *shallower control*, not a depth match.
- **Action:** the Exp 3 spec still contains the old 89% / 18-point / L27 figures — apply this
  correction before running Exp 3's depth ablation.

## 2. Refusal prompt source: AdvBench vs HarmBench

- The project plan and Exp 2 spec use **HarmBench** harmful instructions (diff-of-means per Arditi et al.).
- The Exp 1 notebook/README say **AdvBench**. The formal Exp 1 PDF names neither ("harmful prompts").
- **Action:** reconcile the Exp 1 artifacts to a single source before release. "Same source as Exp 1"
  is only true once this is settled.

## 3. No SAE position exclusion in the NLA read path

- An earlier `SAE_SKIP_FIRST_N=8` was a **hallucination** — the SAE metadata specifies no position
  exclusion for NLA inference.
- Separate, true fact: the Qwen **SAE training** excluded the first 8 token positions — this only
  affects **corpus construction in Exp 3**, not NLA inference. Don't conflate the two.
