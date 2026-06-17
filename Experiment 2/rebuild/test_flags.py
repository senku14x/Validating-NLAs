"""test_flags.py — CPU self-test for flags.py.

The load-bearing check: generic_template must flag the CONTENTLESS floor (the AV's
default triangle/estate/divorce article) but NOT concept content wearing the same
"structured … format" scaffolding — otherwise the exc-template arm nukes real
detections. Exemplars are real (trimmed) decode snippets from the Gate-2 smoke.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from flags import compute_flags, echo, explanation_text, generic_template, n_unique, nla_degenerate

W = "<explanation>{}</explanation>"

# --- real (trimmed) smoke decodes ---
FLOOR = [  # random / baseline_no_inject — contentless default topics
    W.format("Structured FAQ format: a formatted answer to a geometry problem about a "
             "triangle's area. The formula for the area of a triangle is..."),
    W.format("Educational/legal article format: structured listicle explaining a benefit, "
             "establishing facts about divorce and financial risk; the US healthcare system."),
    W.format("Educational article: a comprehensive overview of estate planning, divorce and estate law."),
]
CONCEPT = {  # the 4 validly-dosed concepts — scaffolding present, but concept-specific content
    "refusal": W.format("Structured AI/chatbot response format. The disclaimer 'I cannot provide "
                        "information... It is illegal' signals a compliance/ethics disclaimer."),
    "neg_sentiment": W.format("A compassionate piece promising practical advice for grief; 'Dealing "
                              "with job loss is...' enumerating the impact of losing work, a devastating crisis."),
    "sycophancy": W.format("Structured FAQ: a factual answer about a geometric factor. 'Yes, the "
                           "location is correct.' signals affirmation confirming the user's answer."),
    "corrigibility": W.format("A numbered listicle: 'Yes, I'm ready... I'll gladly participate' signals "
                              "willingness to accept change and defer to oversight."),
}


def main() -> int:
    # 1. THE discrimination test: floor flagged, concepts not.
    for f in FLOOR:
        assert generic_template(f), f"floor should be generic_template: {explanation_text(f)[:60]!r}"
    for c, txt in CONCEPT.items():
        assert not generic_template(txt), f"{c} wrongly flagged generic_template (would nuke its detection)"

    # 2. nla_degenerate: CJK garbage, missing tags, too short -> True; clean -> False
    assert nla_degenerate("㈜㈜㈜ 주주주 一二三四五六")               # CJK
    assert nla_degenerate("no tags here, just prose about something")  # missing <explanation>
    assert nla_degenerate(W.format("short"))                          # < min_chars
    assert nla_degenerate("<explanation> started but never closed <<<<<<<<<<<<<<<<<<<<")  # opening, no closing (loop)
    assert nla_degenerate(W.format("the the the the the the the the the the"))  # repetition collapse
    assert not nla_degenerate(CONCEPT["refusal"])                     # clean

    # 3. echo: a giveaway leak token present -> True; absent -> False; empty list -> False
    assert echo(W.format("the output mentions the word test explicitly"), ["test", "review"])
    assert not echo(CONCEPT["neg_sentiment"], ["emoji", "json"])
    assert not echo(CONCEPT["refusal"], [])

    # 4. helpers
    assert explanation_text(W.format("inner")) == "inner"
    assert explanation_text("no tags") == "no tags"
    assert n_unique([W.format("a"), W.format("a"), W.format("b")]) == 2

    # 5. compute_flags returns all three keys
    fl = compute_flags(CONCEPT["refusal"], concept="refusal", leak_tokens=["illegal"])
    assert set(fl) == {"echo", "generic_template", "nla_degenerate"}
    assert fl["echo"] is True and fl["generic_template"] is False and fl["nla_degenerate"] is False

    print("ALL CHECKS PASSED — flags.py: generic_template separates floor from "
          "concept-in-scaffolding; degenerate/echo/helpers correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
