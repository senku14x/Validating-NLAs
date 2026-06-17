"""Self-test for paths.py — the file-naming convention as executable checks.

Guards the invariants we rely on across the pipeline: stage tokens name their
own script, results land in the right gate dir, the dotted qwen slug round-trips,
and malformed names are rejected loudly (not silently mis-saved).

Pure CPU. Run:  .venv/bin/python test_paths.py
"""
import paths as P


def main() -> int:
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # model normalization
    check(P.model_slug("gemma") == "gemma3-27b", "gemma alias")
    check(P.model_slug("Qwen") == "qwen2.5-7b", "qwen alias is case-insensitive")
    check(P.model_slug("gemma3-27b") == "gemma3-27b", "slug passthrough")
    try:
        P.model_slug("llama")
        fails.append("unknown model should raise")
    except ValueError:
        pass

    # result path: location + self-naming filename
    p = P.result_path("gate1", "04_run_gate1_battery", "gemma", concept="refusal")
    check(p.parent == P.RESULTS / "gate1", f"gate1 dir, got {p.parent}")
    check(p.name == "04_run_gate1_battery__gemma3-27b__refusal.json", p.name)
    check(p.parent.is_dir(), "result dir auto-created")

    # variant + non-json ext (one of the four NLA arms)
    p2 = P.result_path("gate2", "07_score_matrix", "qwen",
                       concept="refusal", variant="exc-echo", ext="jsonl")
    check(p2.name == "07_score_matrix__qwen2.5-7b__refusal__exc-echo.jsonl", p2.name)

    # cross-concept summary (no concept), csv
    p3 = P.result_path("gate2", "08_analyze_gate2", "qwen", ext="csv")
    check(p3.name == "08_analyze_gate2__qwen2.5-7b.csv", p3.name)

    # round-trip parse, incl. the dotted qwen slug
    f = P.parse_stem(p2)
    check(
        f == {"stage": "07_score_matrix", "model": "qwen2.5-7b",
              "concept": "refusal", "variant": "exc-echo", "ext": "jsonl"},
        f"round-trip mismatch: {f}",
    )
    f3 = P.parse_stem(p3)
    check(f3["model"] == "qwen2.5-7b" and f3["concept"] is None and f3["ext"] == "csv",
          f"summary round-trip: {f3}")

    # cache path under the gitignored cache/ tree
    c = P.cache_path("03_extract_for_battery", "gemma", concept="refusal", ext="npz")
    check(c.parent == P.CACHE, f"cache dir, got {c.parent}")
    check(c.name == "03_extract_for_battery__gemma3-27b__refusal.npz", c.name)

    # rejects: bad gate, malformed stage, separator inside a field
    for bad in (
        lambda: P.result_path("gate9", "01_x", "gemma"),
        lambda: P.result_path("gate1", "battery", "gemma"),     # no NN_ prefix
        lambda: P.result_path("gate1", "4_battery", "gemma"),   # one digit
        lambda: P.result_path("gate1", "04_b", "gemma", concept="a__b"),  # sep in field
    ):
        try:
            bad()
            fails.append("expected ValueError for a malformed name")
        except ValueError:
            pass

    # stage_of derives the token from a script path
    check(P.stage_of("/x/y/04_run_gate1_battery.py") == "04_run_gate1_battery", "stage_of")

    if fails:
        print("FAILURES:")
        for x in fails:
            print("  -", x)
        return 1
    print("ALL CHECKS PASSED — paths/naming convention is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
