"""
04_get_prompts.py - assemble the three prompt pools for Experiment 1.

Pools:
  harmful  (120) - AdvBench harmful behaviors (Zou et al. 2023). For v_refusal.
  harmless (120) - Alpaca instructions, no input field. For v_refusal.
  anchor   (120) - Alpaca instructions, DISJOINT from harmless. Injection
                   targets: we add beta*v_refusal to THEIR activations and ask
                   whether the AV detects refusal. Disjoint => no circularity.

Sources (both stable + reproducible; this is what Arditi's diff-of-means uses):
  harmful : llm-attacks AdvBench harmful_behaviors.csv  (verified live, 520 rows)
  harmless: tatsu-lab/alpaca via HF datasets

All 360 are behaviorally filtered AFTER extraction (keep harmful prompts the
model actually refused, harmless/anchor prompts it actually complied with),
targeting ~100 clean each.

Output: /content/prompts.parquet  with columns [id, pool, prompt]
No GPU needed - safe to run while the SGLang server is up.
"""

import io
import random

import pandas as pd
import requests

ADVBENCH_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")
N_HARMFUL, N_HARMLESS, N_ANCHOR = 120, 120, 120
SEED = 42
OUT = "/content/prompts.parquet"

# ---- harmful: AdvBench -----------------------------------------------------
print("Fetching AdvBench harmful behaviors...")
resp = requests.get(ADVBENCH_URL, timeout=30)
resp.raise_for_status()
adv = pd.read_csv(io.StringIO(resp.text))
print(f"  rows={len(adv)}  columns={list(adv.columns)}")
harmful = (adv["goal"].drop_duplicates()
           .sample(N_HARMFUL, random_state=SEED).tolist())

# ---- harmless + anchors: Alpaca (disjoint samples) -------------------------
print("Loading Alpaca (tatsu-lab/alpaca)...")
from datasets import load_dataset

alpaca = load_dataset("tatsu-lab/alpaca", split="train")
# clean single-turn instructions only: empty 'input' context
clean = [r["instruction"].strip() for r in alpaca if not r["input"].strip()]
clean = list(dict.fromkeys(clean))          # dedup, preserve order
print(f"  clean instructions (no input field): {len(clean)}")

rng = random.Random(SEED)
rng.shuffle(clean)
need = N_HARMLESS + N_ANCHOR
assert len(clean) >= need, f"not enough clean Alpaca prompts ({len(clean)} < {need})"
harmless = clean[:N_HARMLESS]
anchors = clean[N_HARMLESS:N_HARMLESS + N_ANCHOR]

# ---- assemble + save -------------------------------------------------------
rows = (
    [{"id": f"harmful_{i:03d}",  "pool": "harmful",  "prompt": p} for i, p in enumerate(harmful)]
    + [{"id": f"harmless_{i:03d}", "pool": "harmless", "prompt": p} for i, p in enumerate(harmless)]
    + [{"id": f"anchor_{i:03d}",   "pool": "anchor",   "prompt": p} for i, p in enumerate(anchors)]
)
df = pd.DataFrame(rows)
df.to_parquet(OUT, index=False)

print(f"\nSaved {len(df)} prompts -> {OUT}")
print(df["pool"].value_counts().to_string())
print("\n--- one example per pool ---")
for pool in ["harmful", "harmless", "anchor"]:
    print(f"[{pool:8}] {df[df.pool == pool].iloc[0]['prompt'][:95]}")
