"""paths.py — canonical paths and file-naming for the Experiment 2 rebuild.

ONE place decides where a file goes and what it is called. Scripts import these
helpers instead of hand-building paths, so every artifact is traceable back to
the stage, model, concept and variant that produced it — and we never end up
with a results/ file we can't account for. (Same spirit as confounds.py being
the single source of AUROC logic: don't reimplement naming anywhere else.)

Layout — everything relative to this file's directory (the rebuild root):

    confounds.py / paths.py / *.py    importable libraries + their tests (root)
    test_*.py                         run with:  .venv/bin/python <test>.py
    scripts/                          numbered pipeline stages (entry points)
    configs/                          hand-authored config (concepts.yaml)
    data/        committed   small constructed datasets (concept_pairs.parquet)
    results/     committed   analysis outputs, one subdir per gate
    cache/       gitignored  large artifacts (activations .npz, raw decodes)
    workspace/   gitignored  scratch

File naming (results and cache). Fields are joined by '__'; within a field use
lowercase [a-z0-9_-] (the concept field is its concepts.yaml key verbatim). The
stage field is the *producing script's basename*, so a file names its own maker:

    <stage>__<model>[__<concept>][__<variant>].<ext>

    04_run_gate1_battery__gemma3-27b__refusal.json
    04_run_gate1_battery__gemma3-27b__all.json            (cross-concept summary)
    07_score_matrix__gemma3-27b__refusal__exc-echo.jsonl  (one of the 4 NLA arms)
    08_analyze_gate2__qwen2.5-7b.csv

Build names with result_path() / cache_path(); never f-string them by hand.

In a pipeline script (scripts/NN_*.py), reach these root libs with a 2-line
bootstrap, then use stage_of(__file__) so results auto-tie to the script:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from paths import result_path, stage_of
    result_path("gate1", stage_of(__file__), "gemma", concept="refusal")
"""
from __future__ import annotations

import re
from pathlib import Path

# ── directories ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"          # gitignored
WORKSPACE = ROOT / "workspace"  # gitignored

GATES = ("gate0", "gate1", "gate2", "gate3", "gate4")

# Canonical model slugs. Pass an alias ("gemma"/"qwen") or the slug itself; both
# normalize to the slug that appears in every filename. Never free-type these.
_MODEL_SLUGS = {
    "gemma": "gemma3-27b",
    "gemma3-27b": "gemma3-27b",
    "qwen": "qwen2.5-7b",
    "qwen2.5-7b": "qwen2.5-7b",
}

# Canonical tokens for the four-way NLA detection reporting (spec §2). Open set —
# other variants (dose levels, etc.) are allowed; use these names so the four
# arms are spelled identically everywhere.
NLA_VARIANTS = ("raw", "exc-echo", "exc-template", "exc-degen")

_SEP = "__"
_STAGE_RE = re.compile(r"^[0-9]{2}[a-z]?_[a-z0-9_-]+$")


# ── validation / normalization ────────────────────────────────────────────────
def model_slug(model: str) -> str:
    key = str(model).strip().lower()
    if key not in _MODEL_SLUGS:
        raise ValueError(f"unknown model {model!r}; known: {sorted(set(_MODEL_SLUGS.values()))}")
    return _MODEL_SLUGS[key]


def _field(value, *, name: str) -> str:
    """Validate one filename field (concept / variant): lowercase [a-z0-9_-].

    Concept keys are kept verbatim (e.g. 'eval_framing_matched'), so a result
    filename's concept field equals its concepts.yaml key. Illegal input raises
    rather than being silently rewritten into something that no longer matches.
    """
    s = str(value).strip().lower()
    if not s:
        raise ValueError(f"{name} is empty")
    if _SEP in s:
        raise ValueError(f"{name} {s!r} must not contain the field separator {_SEP!r}")
    if not re.fullmatch(r"[a-z0-9_-]+", s):
        raise ValueError(f"{name} {s!r} must match [a-z0-9_-] (lowercase)")
    return s


def _stage(stage: str) -> str:
    s = str(stage).strip().lower()
    if _SEP in s:
        raise ValueError(f"stage {s!r} must not contain the field separator {_SEP!r}")
    if not _STAGE_RE.match(s):
        raise ValueError(
            f"stage {s!r} must look like '04_run_gate1_battery' "
            f"(two digits, optional letter, '_', then [a-z0-9_-]); use stage_of(__file__)"
        )
    return s


def stage_of(dunder_file: str) -> str:
    """Stage token for the calling script: stage_of(__file__) -> '04_run_gate1_battery'."""
    return _stage(Path(dunder_file).stem)


def _stem(stage, model, concept, variant) -> str:
    parts = [_stage(stage), model_slug(model)]
    if concept is not None:
        parts.append(_field(concept, name="concept"))
    if variant is not None:
        parts.append(_field(variant, name="variant"))
    return _SEP.join(parts)


# ── path builders ─────────────────────────────────────────────────────────────
def result_path(gate, stage, model, *, concept=None, variant=None,
                ext="json", mkdir=True) -> Path:
    """A committed, small structured output under results/<gate>/.

    Big artifacts (activations, raw decodes) belong in cache_path(), not here.
    """
    if gate not in GATES:
        raise ValueError(f"gate {gate!r} not in {GATES}")
    d = RESULTS / gate
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d / f"{_stem(stage, model, concept, variant)}.{str(ext).lstrip('.')}"


def cache_path(stage, model, *, concept=None, variant=None,
               ext="npz", subdir=None, mkdir=True) -> Path:
    """A gitignored, large intermediate under cache/ (optionally cache/<subdir>/)."""
    d = CACHE if subdir is None else CACHE / _field(subdir, name="subdir")
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d / f"{_stem(stage, model, concept, variant)}.{str(ext).lstrip('.')}"


def data_path(name, *, mkdir=True) -> Path:
    """A committed input/constructed dataset under data/ (e.g. concept_pairs.parquet)."""
    if mkdir:
        DATA.mkdir(parents=True, exist_ok=True)
    return DATA / name


def config_path(name) -> Path:
    """A hand-authored config under configs/ (e.g. concepts.yaml)."""
    return CONFIGS / name


def parse_stem(path) -> dict:
    """Inverse of _stem: recover {stage, model, concept, variant, ext} from a filename.

    The model slug 'qwen2.5-7b' contains a dot, so we split the extension off at
    the LAST dot only (Path.suffix), never the first.
    """
    p = Path(path)
    ext = p.suffix.lstrip(".")
    stem = p.name[: -len(p.suffix)] if p.suffix else p.name
    f = stem.split(_SEP)
    return {
        "stage": f[0] if len(f) > 0 else None,
        "model": f[1] if len(f) > 1 else None,
        "concept": f[2] if len(f) > 2 else None,
        "variant": f[3] if len(f) > 3 else None,
        "ext": ext or None,
    }


if __name__ == "__main__":
    import sys

    print(
        "paths.py is a library — import result_path/cache_path/stage_of from it.\n"
        "Run the self-test with:  .venv/bin/python test_paths.py",
        file=sys.stderr,
    )
