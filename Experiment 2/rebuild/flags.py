"""flags.py — per-output controls for Gate-2 NLA detections.

Every NLA detection rate is reported four ways (spec §2 / README): raw /
excluding echo / excluding generic_template / excluding nla_degenerate. Flags are
computed ONCE per decode and applied as masks in 08, so 07 and 08 agree.
Importable lib (sibling to confounds.py / audits.py) — never inline these.

  echo(out, leak_tokens)   the AV parroted a giveaway lexical token of the
                           injected concept (tokens from audits.lexical_leak on
                           that concept's pairs) — surface parroting, not reading
                           a represented concept.
  generic_template(out)    the output is the AV's CONTENTLESS floor — its default
                           "structured/educational/encyclopedic article about
                           <generic topic>" output (triangle area, estate/divorce,
                           comprehensive overview) seen on the random / baseline
                           injections. Deliberately NOT "has the house-style
                           scaffolding": the strong concepts (refusal, neg_sentiment,
                           …) wear that scaffolding too, so flagging the scaffolding
                           would nuke real detections. This flags floor *filler*.
                           v1 = curated signature from the smoke's floor decodes;
                           refine against the matrix's random/baseline rows (08).
  nla_degenerate(out)      CJK garbage / missing <explanation> / too short /
                           token-repeat — injection physically failed or the model
                           collapsed; not a real negative.
  n_unique(outputs)        distinct explanations among the k samples (deterministic
                           repeat != consistency).
"""
from __future__ import annotations

import re
from collections import Counter

_EXPL = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL | re.I)

# Contentless-floor signature: the AV's default topics/phrasing when no concept is
# present (observed on random + baseline_no_inject decodes). Concept-bearing outputs
# ("I cannot comply", "grief / job loss", "you're correct", "accept correction") do
# NOT contain these, so this separates floor from concept-wrapped-in-scaffolding.
_FLOOR = re.compile(
    r"area of a triangle|triangle['’]?s? area|formula for the area"
    r"|estate (?:planning|law|advice)|divorce(?: and| law| settlement)"
    r"|comprehensive (?:overview|survey) of"
    r"|encyclopedic (?:entry|article|overview)|wikipedia[-\s]?(?:style|like)"
    r"|structured listicle (?:about|explaining|of)|educational/legal article"
    r"|a (?:formal )?(?:listicle|encyclopedia entry) (?:about|of|on)",
    re.IGNORECASE | re.DOTALL,
)


def explanation_text(out) -> str:
    if not isinstance(out, str):
        return ""
    m = _EXPL.search(out)
    return (m.group(1) if m else out).strip()


def cjk_count(s: str) -> int:
    # CJK ideographs + Hangul + enclosed-CJK block (U+3200-32FF holds the U+321C marker)
    return sum(1 for c in s if "一" <= c <= "鿿" or "가" <= c <= "힯"
               or "　" <= c <= "㏿")


def nla_degenerate(out, *, min_chars: int = 20, cjk_thresh: int = 5, max_repeat_frac: float = 0.5) -> bool:
    if not isinstance(out, str) or not out.strip():
        return True
    if "<explanation>" not in out:           # tags missing -> truncated / failed gen
        return True
    if "</explanation>" not in out:          # opening but no closing tag -> truncated / repetition loop
        return True
    expl = explanation_text(out)
    if len(expl) < min_chars:
        return True
    if cjk_count(expl) >= cjk_thresh:        # CJK garbage -> injection failed
        return True
    toks = expl.split()
    if toks and Counter(toks).most_common(1)[0][1] / len(toks) > max_repeat_frac:
        return True                          # one token dominates -> repetition collapse
    return False


def echo(out, leak_tokens) -> bool:
    if not leak_tokens:
        return False
    t = explanation_text(out).lower()
    return any(re.search(rf"\b{re.escape(str(tok).lower())}\b", t) for tok in leak_tokens)


def generic_template(out) -> bool:
    return bool(_FLOOR.search(explanation_text(out)))


def n_unique(outputs) -> int:
    return len({explanation_text(o) for o in outputs})


def compute_flags(out, *, concept=None, leak_tokens=None) -> dict:
    """All per-output flags in one pass (07 attaches these to every decode)."""
    return dict(
        echo=echo(out, leak_tokens or []),
        generic_template=generic_template(out),
        nla_degenerate=nla_degenerate(out),
    )


if __name__ == "__main__":
    import sys
    print("flags.py is a library — import echo/generic_template/nla_degenerate/compute_flags.\n"
          "Run the self-test with:  .venv/bin/python test_flags.py", file=sys.stderr)
