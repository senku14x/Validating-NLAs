#!/usr/bin/env python
"""04_a0b_staged.py — A0b reconstruction fidelity, staged to fit one node.

The full A0b loop is  h --AV--> text --AR--> h_hat ;  cosine(h, h_hat).  On one node the AV SGLang server
(TP=8) fills all GPUs, so the 70B AR can't be co-resident. This splits the loop:

  --decode   : (AV server must be up) read cached base/org activations, send each through the AV server
               (client-side, no local GPU), save the decoded texts to JSON. Stop the server after.
  --score    : (AV server down) load the AR (70B) on the freed GPUs, read the saved texts + activations,
               compute cosine(h, h_hat) for base and organism, write the A0b verdict.

A0b interpretation: gate on organism_loop_cos / base_loop_cos. base = the ceiling (NLA on its own base
model). ratio ~1 => NLA reads the organism as well as base (RQ4 GO); ratio collapses => RQ4 fails.

Run:
  # server up:
  python3 04_a0b_staged.py --decode --acts-base <base.npz> --acts-org <org.npz> --out <decodes.json>
  # then bring the server down, free GPUs, and:
  python3 04_a0b_staged.py --score --decodes <decodes.json> --ar <ar_dir> --out-result <a0b.json>
"""
from __future__ import annotations

import argparse, json, os, pathlib, sys
import numpy as np


def _load_acts(npz_path):
    a = np.load(npz_path, allow_pickle=True)
    return a["X"].astype("float32")


def decode(acts_base, acts_org, out_path, port):
    """Stage 2a: decode base+org activations through the running AV server. Client-side only."""
    nla_repo = os.environ.get("NLA_REPO_DIR", "/fsx/juliaili/oss-run1/nla_repo")
    sys.path.insert(0, nla_repo)
    import httpx, torch
    from nla_inference import NLAClient

    url = f"http://localhost:{port}"
    assert httpx.get(url + "/health", timeout=5).status_code == 200, f"AV server not up at {url}"
    av_dir = os.environ["NLA_AV_DIR"]
    av = NLAClient(av_dir, sglang_url=url, device="cpu")

    Hb, Ho = _load_acts(acts_base), _load_acts(acts_org)
    print(f"decoding base ({Hb.shape}) + organism ({Ho.shape}) through AV server")

    def decode_all(H, tag):
        texts = []
        for i, h in enumerate(H):
            t = av.generate(torch.tensor(h, dtype=torch.float32), extract_explanation=False)
            texts.append(t)
            if i < 2:
                print(f"  [{tag} #{i}] {t[:120].strip()}")
        return texts

    out = dict(base_texts=decode_all(Hb, "base"), org_texts=decode_all(Ho, "org"),
               acts_base=str(acts_base), acts_org=str(acts_org))
    pathlib.Path(out_path).write_text(json.dumps(out))
    print(f"\nwrote {out_path}  ({len(out['base_texts'])} base + {len(out['org_texts'])} org decodes)")
    print("Now stop the AV server (free GPUs) and run with --score.")


def _chance_cos(H, n_pairs=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(H)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs); k = i != j
    a, b = H[i[k]].astype(np.float64), H[j[k]].astype(np.float64)
    return float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)).mean())


def _patch_nla_inference_source(nla_repo):
    """Patch the vendored nla_inference.py SOURCE for 70B AR sharding, applied to a COPY so the original
    stays byte-identical to upstream kitft/nla-inference (per CLAUDE.md — keep it diffable). Two minimal edits
    that preserve all of upstream's loading logic (this is the exact patch that produced the verified base
    cosine ~0.90):
      (1) from_pretrained gets device_map="auto" when NLACritic(device="auto"); the single-GPU `.to(device)`
          is skipped in that case (a 70B OOMs on one 40GB card);
      (2) reconstruct() moves the sharded backbone output to the value_head's device before the head.
    Returns a module dir to import NLACritic from."""
    import shutil, importlib.util
    src = pathlib.Path(nla_repo) / "nla_inference.py"
    patched_dir = pathlib.Path(os.environ.get("NLA_PATCHED_DIR", "/tmp/nla_patched"))
    patched_dir.mkdir(parents=True, exist_ok=True)
    dst = patched_dir / "nla_inference.py"
    s = src.read_text()

    load_old = ("        backbone = AutoModelForCausalLM.from_pretrained(\n"
                "            str(checkpoint_dir), torch_dtype=dtype, trust_remote_code=True,\n"
                "        )")
    load_new = ("        _shard = (device == \"auto\")\n"
                "        backbone = AutoModelForCausalLM.from_pretrained(\n"
                "            str(checkpoint_dir), torch_dtype=dtype, trust_remote_code=True,\n"
                "            **({\"device_map\": \"auto\"} if _shard else {}),\n"
                "        )")
    fin_old = ("        self.backbone = backbone.to(device).eval()\n"
               "        self.value_head = self.value_head.to(device).eval()\n"
               "        self.device = device")
    fin_new = ("        if _shard:\n"
               "            self.backbone = backbone.eval()\n"
               "            _vh_dev = backbone.get_input_embeddings().weight.device\n"
               "            self.value_head = self.value_head.to(_vh_dev).eval()\n"
               "            self.device = _vh_dev\n"
               "        else:\n"
               "            self.backbone = backbone.to(device).eval()\n"
               "            self.value_head = self.value_head.to(device).eval()\n"
               "            self.device = device")
    rec_old = ("        h = self.backbone.model(ids, use_cache=False).last_hidden_state[0, -1]  # last token\n"
               "        return self.value_head(h).float().cpu()")
    rec_new = ("        h = self.backbone.model(ids, use_cache=False).last_hidden_state[0, -1]  # last token\n"
               "        h = h.to(self.value_head.weight.device)  # sharded backbone output may be on another GPU\n"
               "        return self.value_head(h).float().cpu()")
    for old, new, name in [(load_old, load_new, "load"), (fin_old, fin_new, "finalize"), (rec_old, rec_new, "reconstruct")]:
        if old not in s:
            sys.exit(f"FAIL: '{name}' block not found in {src} — upstream changed; re-derive the patch.")
        s = s.replace(old, new)
    dst.write_text(s)
    return str(patched_dir)


def score(decodes_path, ar_dir, out_result, port):
    """Stage 2b: load the AR (server must be DOWN now) and score cosine(h, h_hat) base vs organism."""
    nla_repo = os.environ.get("NLA_REPO_DIR", "/fsx/juliaili/oss-run1/nla_repo")
    sys.path.insert(0, _patch_nla_inference_source(nla_repo))   # import the sharding-patched copy
    from nla_inference import NLACritic

    d = json.loads(pathlib.Path(decodes_path).read_text())
    Hb, Ho = _load_acts(d["acts_base"]), _load_acts(d["acts_org"])
    critic = NLACritic(ar_dir, device="auto")
    print("NLACritic API:", [a for a in dir(critic) if not a.startswith("_")])

    def loop_cos(texts, H):
        coss = []
        for t, h in zip(texts, H):
            mse, cos = critic.score(t, h)   # AR re-encodes text -> h_hat; (mse, cos) like Exp-2 stage 15
            coss.append(float(cos))
        return np.array(coss)

    cb = loop_cos(d["base_texts"], Hb)
    co = loop_cos(d["org_texts"], Ho)
    chance = _chance_cos(Hb)
    ratio = float(co.mean()) / (float(cb.mean()) + 1e-12)
    res = dict(
        base_loop_cos_mean=round(float(cb.mean()), 4), base_loop_cos_p10=round(float(np.percentile(cb, 10)), 4),
        organism_loop_cos_mean=round(float(co.mean()), 4), organism_loop_cos_p10=round(float(np.percentile(co, 10)), 4),
        anisotropy_chance_cos=round(chance, 4),
        organism_above_chance=round(float(co.mean()) - chance, 4),
        organism_vs_base_ratio=round(ratio, 4),
        n=len(cb),
    )
    if ratio >= 0.9 and res["organism_above_chance"] > 0.05:
        res["verdict"] = "GO — NLA reconstructs organism ~as well as base; RQ4 passes"
    elif ratio >= 0.7:
        res["verdict"] = "DEGRADED — reduced fidelity; consider light-AR-finetune; proceed with care"
    else:
        res["verdict"] = "NO-GO — NLA loses the organism activation; RQ4 transfer fails for this organism"
    pathlib.Path(out_result).write_text(json.dumps(res, indent=2))
    print("\n==== A0b FULL NLA-LOOP FIDELITY ====")
    print(json.dumps(res, indent=2))
    print(f"\nbase=ceiling. ratio=organism/base. wrote {out_result}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--acts-base"); ap.add_argument("--acts-org")
    ap.add_argument("--out", default="/fsx/juliaili/oss-run1/a0b_decodes.json")
    ap.add_argument("--decodes", default="/fsx/juliaili/oss-run1/a0b_decodes.json")
    ap.add_argument("--ar"); ap.add_argument("--out-result", default="/fsx/juliaili/oss-run1/results/phase0_rq4/04_a0b__llama-oss-run1.json")
    ap.add_argument("--port", type=int, default=30000)
    a = ap.parse_args()
    if a.decode:
        decode(a.acts_base, a.acts_org, a.out, a.port)
    elif a.score:
        score(a.decodes, a.ar, a.out_result, a.port)
    else:
        sys.exit("pass --decode or --score")


if __name__ == "__main__":
    raise SystemExit(main())
