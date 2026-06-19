#!/usr/bin/env python
"""CPU smoke test for 06_validate_metric.py — validates the pure-numpy helpers (map transport, the NEW
same-domain hard-negative sampler, length-matching, AUROC) WITHOUT torch/SAE/GPU. torch is mocked (only
needed for the @no_grad decorator); batched_sentence_max_scores is NOT exercised here (it is byte-identical
to the working 05b version + a trivial normalize_tokens flag). Run: python test_06_smoke.py"""
import importlib.util, os, sys, types
import numpy as np

# minimal torch mock so the module imports (no_grad decorator + cuda.is_available)
class _NoGrad:
    def __call__(self, f): return f
    def __enter__(self): return self
    def __exit__(self, *a): return False
_t = types.ModuleType("torch"); _t.no_grad = lambda: _NoGrad()
_t.cuda = types.SimpleNamespace(is_available=lambda: False)
_t.Tensor = type("Tensor", (), {})          # scipy/sklearn probe torch.Tensor for array dispatch
sys.modules.setdefault("torch", _t)

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v06", os.path.join(_HERE, "06_validate_metric.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rng = np.random.default_rng(0)

# 1) safe_auroc
assert abs(m.safe_auroc([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8]) - 1.0) < 1e-9
assert abs(m.safe_auroc([0, 1, 0, 1], [5, 5, 5, 5]) - 0.5) < 1e-9            # ties -> 0.5
assert m.safe_auroc([1, 1, 1], [1, 2, 3]) == 0.5                            # one class -> 0.5

# 2) map transport: identity map + unit sigmas -> mapped == unit(Dq), rows unit-normed
d = 8; W = np.eye(d); xs = np.ones(d); ys = np.ones(d)
Dq = rng.standard_normal((4, d))
Dg = m.map_qwen_dirs_to_gemma(Dq, W, xs, ys)
assert Dg.shape == (4, d)
assert np.allclose(np.linalg.norm(Dg, axis=1), 1.0, atol=1e-5)              # unit rows
assert np.allclose(Dg, m.unit_rows(Dq), atol=1e-5)                          # identity preserves direction
# sigma scaling actually applied: PER-DIM (non-uniform) sigmas change the direction
# (uniform sigmas are a global scalar that unit-normalization cancels — they would NOT change Dg)
xs_pd = np.array([1., 2., 1., 2., 1., 2., 1., 2.]); ys_pd = np.array([3., 1., 3., 1., 3., 1., 3., 1.])
Dg2 = m.map_qwen_dirs_to_gemma(Dq, W, xs_pd, ys_pd)
assert not np.allclose(Dg2, Dg, atol=1e-3)

# 3) length_matched_negs: valid, unique, drawn from candidates
lengths = rng.integers(12, 40, 400)
pos = list(range(30)); cand = list(range(30, 400))
negs = m.length_matched_negs(pos, cand, lengths, rng)
assert len(negs) >= m.MIN_N_NEG and len(set(negs)) == len(negs)
assert all(n in set(cand) for n in negs)

# 4) hard_domain_negs (the NEW control): same-domain negs = sibling positives where TARGET is inactive
n_feat = 5; S = 400
ws = np.zeros((S, n_feat), dtype=np.float32)
feat_keys = ["10", "11", "12", "13", "14"]
feat_pos_ids = {"10": list(range(0, 25)),      # target (col 0) positives
                "11": list(range(60, 95)),     # sibling A positives  (target inactive here)
                "12": list(range(95, 120)),    # sibling B positives
                "13": [], "14": []}
ws[0:25, 0] = 3.0                              # target fires only on its own positives
hard = m.hard_domain_negs(0, [1, 2], feat_keys, feat_pos_ids, ws, lengths, list(range(0, 25)), rng)
assert hard is not None and len(hard) >= m.MIN_N_NEG
assert all(ws[s, 0] == 0.0 for s in hard)                                   # target inactive on every hard neg
assert all(s in set(range(60, 120)) for s in hard)                         # all drawn from sibling domains
assert not (set(hard) & set(range(0, 25)))                                 # never a target positive
# fallback when siblings have too few candidates -> None (caller uses random)
assert m.hard_domain_negs(0, [3, 4], feat_keys, feat_pos_ids, ws, lengths, list(range(0, 25)), rng) is None

# 5) outlier-dim zeroing keeps directions unit and actually drops the chosen dims
D = m.unit_rows(rng.standard_normal((3, d))); od = [0, 1]
Doz = D.copy(); Doz[:, od] = 0.0; Doz = m.unit_rows(Doz)
assert np.allclose(Doz[:, od], 0.0) and np.allclose(np.linalg.norm(Doz, axis=1), 1.0, atol=1e-5)

print("ALL 06 helper checks passed (map transport, hard_domain_negs same-domain control, length-match, AUROC, outlier-zero).")
