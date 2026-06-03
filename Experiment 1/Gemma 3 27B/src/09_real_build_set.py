#!/usr/bin/env python
import numpy as np
import pandas as pd

ACTS = "/content/activations.npy"
LABEL = "/content/labeled.parquet"
VREF = "/content/v_refusal.npy"

OUT_VECS = "/content/real_eval_activations.npy"
OUT_META = "/content/real_eval_meta.parquet"

acts = np.load(ACTS).astype("float32")
df = pd.read_parquet(LABEL).reset_index(drop=False).rename(columns={"index": "orig_row"})

v = np.load(VREF).astype("float64")
v = v / np.linalg.norm(v)

groups = [
    ("real_refused_harmful",   (df.pool == "harmful")  & (df.label == "refused")),
    ("real_complied_harmless", (df.pool == "harmless") & (df.label == "complied")),
    ("real_anchor_complied",   (df.pool == "anchor")   & (df.label == "complied")),
]

rows = []
vecs = []

for cond, mask in groups:
    sub = df[mask].copy()
    for _, r in sub.iterrows():
        h = acts[int(r.orig_row)].astype("float64")
        cos_refusal = float((h @ v) / np.linalg.norm(h))
        rows.append({
            "row": len(rows),
            "orig_row": int(r.orig_row),
            "id": r.id,
            "condition": cond,
            "pool": r.pool,
            "label": r.label,
            "cos_with_refusal": cos_refusal,
            "prompt": r.prompt,
            "output": r.output,
        })
        vecs.append(acts[int(r.orig_row)])

vecs = np.stack(vecs).astype("float32")
meta = pd.DataFrame(rows)

np.save(OUT_VECS, vecs)
meta.to_parquet(OUT_META, index=False)

print("saved", OUT_VECS, vecs.shape)
print("saved", OUT_META)
print(meta.condition.value_counts())
print()
print(meta.groupby("condition")["cos_with_refusal"].describe()[["count","mean","std","min","max"]])
