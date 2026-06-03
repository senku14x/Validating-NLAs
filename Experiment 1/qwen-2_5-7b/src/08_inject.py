"""
08_inject.py - build the injection sweep for the NLA detection experiment.

For each of 100 neutral ANCHOR activations, inject a direction at a sweep of
doses, parameterized by REALIZED cos(h', v_injected) so all anchors land at the
same dose level (clean, comparable curves). 4 conditions:

  C1_refusal_pos : + v_refusal  targets 0.30..0.90  (anchor baseline cos ~-0.11)
  C2_refusal_neg : + v_refusal  targets 0.30..-0.30 (push AWAY from refusal)
  C3_random      : + v_random   targets 0.30..0.90  (pure random unit-vector null)
  C4_nullDoM     : + v_nullDoM  targets 0.30..0.90  (DoM on SHUFFLED labels:
                                same construction as v_refusal, no real concept)

beta solved per anchor so that cos(h+beta*v, v) = t exactly:
  beta = sqrt(|h|^2 - (h.v)^2) * t/sqrt(1-t^2) - (h.v)

Activations stored RAW (the AV L2-normalizes at inference). cos_with_refusal is
tracked for every row (for the controls it shows incidental refusal alignment).

NOTE: C5 (related-but-wrong - a semantically adjacent concept such as sentiment)
needs a separate direction from a small extra extraction. Flagged as a follow-up,
not built here.

PREREQ: run 05_extract.py, 06_label.py, and 07_build_direction.py first.

Inputs / outputs (under workspace/):
  activations.npy, labeled.parquet, v_refusal.npy
  -> injected.npy             float32 [N, 3584]
     injection_meta.parquet   [row, anchor_id, condition, target_cos,
                               realized_cos, beta, cos_with_refusal]
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"

ACTS = WORKSPACE / "activations.npy"
LABEL = WORKSPACE / "labeled.parquet"
VREF = WORKSPACE / "v_refusal.npy"
INJ_OUT = WORKSPACE / "injected.npy"
META_OUT = WORKSPACE / "injection_meta.parquet"
N_ANCHORS = 100
SEED = 42

for path in (ACTS, LABEL, VREF):
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — run earlier pipeline steps first")

WORKSPACE.mkdir(parents=True, exist_ok=True)

acts = np.load(ACTS).astype(np.float64)
df = pd.read_parquet(LABEL)
v_refusal = np.load(VREF).astype(np.float64)
v_refusal /= np.linalg.norm(v_refusal)
d = len(v_refusal)
assert d == acts.shape[1], f"activation dim {acts.shape[1]} != v_refusal dim {d}"
rng = np.random.default_rng(SEED)

# --- anchors ---
anchor_idx = df.index[(df.pool == "anchor") & (df.label == "complied")].to_numpy()
anchor_idx = rng.permutation(anchor_idx)[:N_ANCHORS]
anchor_ids = df.loc[anchor_idx, "id"].to_numpy()
H = acts[anchor_idx]
print(f"{len(H)} anchors  (activation dim={d})")

# --- direction set ---
v_random = rng.standard_normal(d)
v_random /= np.linalg.norm(v_random)

# v_nullDoM: DoM on a RANDOM split of NEUTRAL (harmless) activations.
# Do NOT build this from the refused+complied set: that data has two real
# clusters along v_refusal, so any random split's DoM lands on the cluster axis
# (contaminated - we measured cos=-0.63). Neutral data has no refusal cluster
# (all harmless sit at baseline cos ~-0.11), so a random split cancels the
# constant refusal component, leaving only topic noise => ~orthogonal to
# refusal. Harmless pool is disjoint from the anchor injection targets.
neutral_idx = rng.permutation(
    df.index[(df.pool == "harmless") & (df.label == "complied")].to_numpy()
)
half = len(neutral_idx) // 2
v_nullDoM = acts[neutral_idx[:half]].mean(0) - acts[neutral_idx[half:]].mean(0)
v_nullDoM /= np.linalg.norm(v_nullDoM)

print("directions (all unit):")
print(f"  cos(v_random,  v_refusal) = {v_random @ v_refusal:+.3f}  (expect ~0)")
print(f"  cos(v_nullDoM, v_refusal) = {v_nullDoM @ v_refusal:+.3f}  (expect ~0)")
if abs(v_nullDoM @ v_refusal) > 0.15:
    print("  WARNING: null-DoM still has notable refusal alignment - inspect the neutral pool")


def inject(H, v, t):
    a = H @ v
    hp = np.sqrt(np.maximum((H ** 2).sum(1) - a ** 2, 0.0))
    beta = hp * (t / np.sqrt(1 - t ** 2)) - a
    Hp = H + beta[:, None] * v[None, :]
    realized = (Hp @ v) / np.linalg.norm(Hp, axis=1)
    return Hp, beta, realized


POS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
NEG = [0.30, 0.20, 0.10, 0.00, -0.10, -0.20, -0.30]
conditions = [
    ("C1_refusal_pos", v_refusal, POS),
    ("C2_refusal_neg", v_refusal, NEG),
    ("C3_random",      v_random,  POS),
    ("C4_nullDoM",     v_nullDoM, POS),
]

rows, vecs = [], []
for cname, v, targets in conditions:
    for t in targets:
        Hp, beta, realized = inject(H, v, t)
        cos_ref = (Hp @ v_refusal) / np.linalg.norm(Hp, axis=1)
        for i in range(len(H)):
            rows.append(dict(anchor_id=anchor_ids[i], condition=cname,
                             target_cos=t, realized_cos=float(realized[i]),
                             beta=float(beta[i]), cos_with_refusal=float(cos_ref[i])))
            vecs.append(Hp[i])

vecs = np.stack(vecs).astype(np.float32)
meta = pd.DataFrame(rows)
meta.insert(0, "row", range(len(meta)))
np.save(INJ_OUT, vecs)
meta.to_parquet(META_OUT, index=False)

print(f"\n{len(vecs)} injected activations -> {INJ_OUT} {vecs.shape}")
print(meta.groupby("condition").size().to_string())
err = (meta["realized_cos"] - meta["target_cos"]).abs().max()
print(f"\nmax |realized - target| cos error: {err:.5f}  (should be ~0 - confirms the solver)")
print("\nrealized_cos by condition x target (mean across anchors):")
print(meta.pivot_table(index="condition", columns="target_cos",
                       values="realized_cos", aggfunc="mean").round(2).to_string())
print("\ncos_with_refusal for the controls (should stay ~baseline -0.11, not rise):")
print(meta[meta.condition.isin(["C3_random", "C4_nullDoM"])]
      .pivot_table(index="condition", columns="target_cos",
                   values="cos_with_refusal", aggfunc="mean").round(2).to_string())
