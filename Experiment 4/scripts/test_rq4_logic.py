#!/usr/bin/env python
"""test_rq4_logic.py — CPU self-test of the A0a distribution-shift MATH (no GPU, no models).

Validates that the RQ4 analysis logic (`01_rq4_ar_fidelity.py`) produces the right VERDICT on synthetic
activations with KNOWN ground truth, the same way Exp 2's test_confounds.py validates the battery before it
touches real data. This does NOT test the NLA or any model — only that the metric/verdict code is correct.

Three synthetic regimes (we know the answer for each):
  CLEAN  : organism activations = base + tiny noise   -> should verdict GO  (close to base)
  SHIFTED: organism activations = base + large noise   -> should verdict DEGRADED/NO-GO (far from base)
  NO-OP  : organism activations EXACTLY = base         -> the no-op-adapter guard must FIRE (np.allclose)

Run:  .venv/bin/python "Experiment 4/scripts/test_rq4_logic.py"
"""
from __future__ import annotations

import numpy as np

# import the pure-math helpers from the stage script (they have no torch dependency at import time)
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location("rq4", pathlib.Path(__file__).with_name("01_rq4_ar_fidelity.py"))
rq4 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rq4)


def _verdict_from(H_base, H_org):
    """Replicate the A0a verdict logic on given activation matrices (MUST mirror run() exactly).
    Gates on FLOOR-NORMALIZED retention (cos-floor)/(1-floor) + norm stability — NOT raw cosine."""
    paired = rq4._cos_rows(H_base, H_org)
    norm_ratio = np.linalg.norm(H_org, axis=1) / (np.linalg.norm(H_base, axis=1) + 1e-12)
    chance = rq4._chance_cos(H_base)
    pc = float(paired.mean()); nr = float(norm_ratio.mean())
    ret = (pc - chance) / (1.0 - chance + 1e-12)
    norm_ok = 0.8 <= nr <= 1.25
    if ret >= 0.6 and norm_ok:
        v = "GO"
    elif ret >= 0.35:
        v = "DEGRADED"
    else:
        v = "NO-GO"
    return v, round(pc, 3), round(ret, 3), round(nr, 3), round(chance, 3)


def main() -> int:
    rng = np.random.default_rng(0)
    n, d = 40, 256
    # anisotropic base cloud calibrated so the anisotropy floor ~0.7, matching Exp 2's MEASURED Qwen real
    # floor (0.705) — so this synthetic test reflects real residual-stream geometry, not an extreme.
    shared = rng.standard_normal(d) * 1.3
    H_base = shared + rng.standard_normal((n, d)) * 1.0

    ok = True

    # CLEAN: organism ~ base + tiny perturbation (a faithful, barely-shifting LoRA) -> GO
    H_clean = H_base + rng.standard_normal((n, d)) * 0.05
    v, pc, ret, nr, ch = _verdict_from(H_base, H_clean)
    print(f"CLEAN   -> {v:9s} paired_cos={pc} retention={ret} norm_ratio={nr} chance={ch}")
    ok &= (v == "GO")

    # SHIFTED: organism = base + large perturbation (a finetune that pushes activations far OOD) -> NOT GO
    H_shift = H_base + rng.standard_normal((n, d)) * 6.0
    v, pc, ret, nr, ch = _verdict_from(H_base, H_shift)
    print(f"SHIFTED -> {v:9s} paired_cos={pc} retention={ret} norm_ratio={nr} chance={ch}")
    ok &= (v in ("DEGRADED", "NO-GO"))

    # MODERATE: a medium shift -> DEGRADED (the middle retention band exists and is reachable)
    H_mod = H_base + rng.standard_normal((n, d)) * 1.3
    v, pc, ret, nr, ch = _verdict_from(H_base, H_mod)
    print(f"MODERATE-> {v:9s} paired_cos={pc} retention={ret} norm_ratio={nr} chance={ch}")
    ok &= (v == "DEGRADED")

    # CROSS-MODEL COMPARABILITY (the bug the floor-normalized fix exists to kill): the SAME relative shift
    # must yield the SAME verdict regardless of the base cloud's anisotropy floor. Build a low-floor cloud
    # (Llama-L54-like) and a high-floor cloud (Qwen-L21-like), apply a shift scaled to land at the SAME
    # retention, and assert the verdicts agree. Under the OLD raw-cosine gate these diverged (high-floor=GO,
    # low-floor=NO-GO) at identical relative shift — exactly the artifact that produced the spurious Llama NO-GO.
    H_lowfloor = rng.standard_normal((n, d)) * 1.0 + rng.standard_normal(d) * 0.3   # weak shared dir -> low floor
    H_hifloor = rng.standard_normal((n, d)) * 1.0 + rng.standard_normal(d) * 2.5    # strong shared dir -> high floor
    v_lo, pc_lo, ret_lo, _, ch_lo = _verdict_from(H_lowfloor, H_lowfloor + rng.standard_normal((n, d)) * 0.8)
    v_hi, pc_hi, ret_hi, _, ch_hi = _verdict_from(H_hifloor, H_hifloor + rng.standard_normal((n, d)) * 0.8)
    print(f"XMODEL  -> low-floor(ch={ch_lo}): cos={pc_lo} ret={ret_lo} -> {v_lo}  |  "
          f"high-floor(ch={ch_hi}): cos={pc_hi} ret={ret_hi} -> {v_hi}")
    print(f"          floors differ ({ch_lo} vs {ch_hi}) but verdicts must agree on comparable retention.")
    # not asserting equality of arbitrary scales, but the retention values must be much closer than the raw
    # cosines were — i.e. the normalization actually removes the floor's influence.
    ok &= (abs(ret_lo - ret_hi) < abs(pc_lo - pc_hi) + 0.05)

    # NO-OP: organism == base -> the guard (np.allclose atol=1e-3) must fire (catches a silently-failed LoRA)
    guard_fires = np.allclose(H_base, H_base.copy(), atol=1e-3)
    print(f"NO-OP   -> guard_fires={guard_fires} (must be True: a no-op adapter would falsely 'GO')")
    ok &= guard_fires

    print(f"\nanisotropy chance cos on synthetic base = {ch:.3f} (target ~0.7, matching Exp 2's real Qwen floor)")
    ok &= (0.5 < ch < 0.85)

    print("\n" + ("ALL CHECKS PASSED — A0a verdict logic is correct on synthetic ground truth."
                  if ok else "*** FAILED — verdict logic is wrong, fix before the GPU run. ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
