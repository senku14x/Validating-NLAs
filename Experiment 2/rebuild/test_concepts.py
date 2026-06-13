"""Validate configs/concepts.yaml: well-formed manifest + our pre-registered decisions.

Catches a malformed/edited manifest before any builder or battery trusts it, and
asserts the decisions we committed to (AdvBench refusal; length-matched eval
framing; echo-prone formats flagged; negative-baseline concepts flagged).

Run:  .venv/bin/python test_concepts.py
"""
import yaml

from paths import config_path

ROLES = {"anchor", "control", "core", "sanity", "candidate", "exploratory"}
FATES = {"PASS", "WEAK", "FAIL", "DROP"}
BASES = {"positive", "negative"}
REQUIRED = {"key", "role", "source", "construction", "expectation",
            "cosine_baseline", "echo_prone", "n_target"}


def main() -> int:
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    cfg = yaml.safe_load(config_path("concepts.yaml").read_text())
    concepts = cfg.get("concepts") or []
    derived = cfg.get("derived_controls") or []
    check(len(concepts) >= 15, f"expected >=15 concepts, got {len(concepts)}")

    keys = set()
    for c in concepts:
        k = c.get("key", "?")
        missing = REQUIRED - set(c)
        check(not missing, f"{k}: missing fields {missing}")
        check(c.get("role") in ROLES, f"{k}: bad role {c.get('role')!r}")
        check(c.get("expectation") in FATES, f"{k}: bad expectation {c.get('expectation')!r}")
        check(c.get("cosine_baseline") in BASES, f"{k}: bad cosine_baseline {c.get('cosine_baseline')!r}")
        check(isinstance(c.get("echo_prone"), bool), f"{k}: echo_prone not a bool")
        check(isinstance(c.get("n_target"), int) and c.get("n_target", 0) > 0, f"{k}: n_target must be a positive int")
        check(k not in keys, f"duplicate key {k}")
        keys.add(k)

    for c in derived:
        k = c.get("key", "?")
        check(k not in keys, f"duplicate key {k} (derived)")
        keys.add(k)

    cmap = {c["key"]: c for c in concepts}

    # pre-registered decisions
    check(cmap.get("refusal", {}).get("role") == "anchor", "refusal must be the anchor")
    check("advbench" in cmap.get("refusal", {}).get("source", "").lower(),
          "refusal must use AdvBench (matches Exp 1)")
    ev = cmap.get("eval_framing_matched", {})
    check(bool(ev), "eval_framing_matched must be present")
    blob = (ev.get("construction", "") + " " + ev.get("notes", "")).lower()
    check("match" in blob or "length" in blob,
          "eval_framing_matched must note length/lexical matching (the v1 fix)")
    for k in ("json_format", "bullet_list", "code_block", "style_emoji"):
        check(cmap.get(k, {}).get("echo_prone") is True, f"{k} should be echo_prone")
    for k in ("uncertainty", "truth_value"):
        check(cmap.get(k, {}).get("cosine_baseline") == "negative",
              f"{k} should have a negative cosine_baseline (dose per §3)")

    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print(f"ALL CHECKS PASSED — {len(concepts)} concepts + {len(derived)} derived controls; "
          "manifest well-formed and decisions encoded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
