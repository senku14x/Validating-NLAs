#!/usr/bin/env python
"""
01_build_concept_pairs.py  --  Exp 2, step 1: build CAA contrastive pair sets.

Builds /content/concept_pairs.parquet — reviewable before any extraction runs.
No target model or SGLang needed. Only the refusal concept needs the OpenAI API.

Concepts
--------
  refusal      AdvBench harmful goal  vs  matched benign rewrite (same topic,
               flipped intent). Fixes the Exp 1 topic confound.
  sycophancy   nrimsky/CAA generate_dataset.json (matched Q+A pairs)
  truth_value  saprmarks/geometry-of-truth cities.csv (true vs false statements)
               NOTE: factual truth-value direction, NOT deception intent.
  uncertainty  Constructed: same Q, hedged vs confident answer (length-matched)
  style_emoji  Constructed: same content, emoji/casual vs plain

Polarity: v_concept = mean(h | present) - mean(h | absent) in step 2.

Run:  !pip install -q openai pandas requests pyarrow
      !python 01_build_concept_pairs.py
"""

import asyncio
import hashlib
import io
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SEED = 42
OUT               = "/content/concept_pairs.parquet"
REFUSAL_CACHE     = "/content/refusal_safe_counterparts.jsonl"
REWRITE_MODEL     = "gpt-5.4-mini"
N_REFUSAL         = 120
N_SYCO            = 120
N_TRUTH           = 120
MAX_CONCURRENT    = 20   # async concurrency for refusal rewrites

SAFETY_USER_ID = hashlib.sha256(
    ("nla_validation_exp2_" + datetime.now(timezone.utc).strftime("%Y-%m-%d")).encode()
).hexdigest()[:16]

ADVBENCH_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                "main/data/advbench/harmful_behaviors.csv")
SYCO_URL = ("https://raw.githubusercontent.com/nrimsky/CAA/main/"
            "datasets/generate/sycophancy/generate_dataset.json")
CITIES_URL = ("https://raw.githubusercontent.com/saprmarks/geometry-of-truth/"
              "main/datasets/cities.csv")

rng = random.Random(SEED)
rows = []   # accumulate dict(concept, pair_id, polarity, text, source)


def add_pair(concept, pair_id, present_text, absent_text, source):
    rows.append(dict(concept=concept, pair_id=f"{concept}_{pair_id:04d}",
                     polarity="present", text=present_text.strip(), source=source))
    rows.append(dict(concept=concept, pair_id=f"{concept}_{pair_id:04d}",
                     polarity="absent",  text=absent_text.strip(),  source=source))


# ─────────────────────── helpers ────────────────────────────────────────────

def get_openai_key():
    try:
        from google.colab import userdata
        return userdata.get("OPENAI_API_KEY")
    except Exception:
        return os.environ.get("OPENAI_API_KEY")


# ───────────────────── refusal (matched, async LLM rewrite) ─────────────────

REWRITE_SYS = (
    "Rewrite a harmful request into a BENIGN request on the SAME topic by "
    "changing ONLY the intent, not the length or structure. Keep the same "
    "verb, the same object, and roughly the same word count as the original. "
    "Replace the harmful goal with a safe, legitimate equivalent on the "
    "identical subject. Do NOT add qualifiers such as 'in a safe and "
    "authorized way', 'ethically', 'responsibly', or 'for educational "
    "purposes' — those pad the length and are not needed. "
    "Output ONLY the rewritten request, nothing else."
)


async def rewrite_one_async(aclient, sem, goal: str) -> str | None:
    """Rewrite one harmful goal; returns benign counterpart or None on failure."""
    budget = int(len(goal) * 1.30) + 20
    async with sem:
        for attempt in range(3):
            extra = "" if attempt == 0 else (
                f" Your previous rewrite was too long. Keep it under "
                f"{budget} characters, matching the original length.")
            try:
                resp = await aclient.chat.completions.create(
                    model=REWRITE_MODEL,
                    messages=[
                        {"role": "system", "content": REWRITE_SYS + extra},
                        {"role": "user",   "content": goal},
                    ],
                    max_completion_tokens=100,
                    user=SAFETY_USER_ID,
                )
                out = resp.choices[0].message.content.strip()
                if len(out) <= budget:
                    return out
            except Exception as e:
                if attempt == 2:
                    print(f"\n  [rewrite failed] {type(e).__name__}: {e}")
                    return None
                await asyncio.sleep(1.5 ** attempt)
    return None


async def run_rewrites(goals: list[str], cache: dict) -> dict:
    """Fetch missing rewrites concurrently; returns updated cache."""
    from openai import AsyncOpenAI
    key = get_openai_key()
    if not key:
        return cache
    aclient = AsyncOpenAI(api_key=key)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    todo = [g for g in goals if g not in cache]
    if not todo:
        return cache

    print(f"[refusal] generating {len(todo)} rewrites (async, up to "
          f"{MAX_CONCURRENT} concurrent)...")
    results = await asyncio.gather(
        *[rewrite_one_async(aclient, sem, g) for g in todo]
    )
    done = 0
    for g, safe in zip(todo, results):
        if safe:
            cache[g] = safe
            done += 1
    print(f"[refusal] generated {done}/{len(todo)}")
    return cache


def build_refusal():
    print("\n[refusal] fetching AdvBench harmful behaviors...")
    adv = pd.read_csv(io.StringIO(requests.get(ADVBENCH_URL, timeout=30).text))
    harmful = (adv["goal"].drop_duplicates()
               .sample(N_REFUSAL, random_state=SEED).tolist())

    key = get_openai_key()
    if not key:
        print("[refusal] WARNING: no OPENAI_API_KEY → SKIPPING refusal pairs.")
        return

    # load cache
    cache = {}
    if Path(REFUSAL_CACHE).exists():
        for line in open(REFUSAL_CACHE):
            try:
                r = json.loads(line)
                cache[r["goal"]] = r["safe"]
            except Exception:
                pass
    print(f"[refusal] cache has {len(cache)} rewrites")

    # async fill
    cache = asyncio.run(run_rewrites(harmful, cache))

    # persist new entries
    with open(REFUSAL_CACHE, "w") as cf:
        for g, s in cache.items():
            cf.write(json.dumps({"goal": g, "safe": s}, ensure_ascii=False) + "\n")

    pid = 0
    for g in harmful:
        if cache.get(g):
            add_pair("refusal", pid, present_text=g, absent_text=cache[g],
                     source="advbench+llm_rewrite")
            pid += 1
    print(f"[refusal] built {pid} matched pairs.")


# ─────────────────────── sycophancy ─────────────────────────────────────────

def build_sycophancy():
    print("\n[sycophancy] fetching nrimsky/CAA generate_dataset.json...")
    data = requests.get(SYCO_URL, timeout=30).json()
    rng.shuffle(data)
    pid = 0
    used_matching = []
    for d in data:
        q      = d.get("question", "").strip()
        a_syco = d.get("answer_matching_behavior", "").strip()
        a_non  = d.get("answer_not_matching_behavior", "").strip()
        if not (q and a_syco and a_non):
            continue
        add_pair("sycophancy", pid,
                 present_text=f"{q}\n{a_syco}",
                 absent_text=f"{q}\n{a_non}",
                 source="nrimsky_CAA")
        used_matching.append(a_syco)
        pid += 1
        if pid >= N_SYCO:
            break
    print(f"[sycophancy] built {pid} matched pairs.")

    from collections import Counter
    bal      = Counter(used_matching)
    top_frac = max(bal.values()) / sum(bal.values())
    print(f"[sycophancy] answer_matching_behavior distribution: {dict(bal)}")
    print(f"[sycophancy] most-common-letter fraction = {top_frac:.2f}  "
          f"(want ~0.5; >0.65 risks encoding letter identity, not sycophancy)")
    if top_frac > 0.65:
        print("[sycophancy] WARNING: imbalanced — flag this before extracting.")


# ─────────────────────── truth_value ────────────────────────────────────────

def build_truth_value():
    print("\n[truth_value] fetching saprmarks geometry-of-truth cities.csv...")
    df     = pd.read_csv(io.StringIO(requests.get(CITIES_URL, timeout=30).text))
    true_s = df[df.label == 1]["statement"].tolist()
    false_s = df[df.label == 0]["statement"].tolist()
    rng.shuffle(true_s); rng.shuffle(false_s)
    n = min(N_TRUTH, len(true_s), len(false_s))
    for pid in range(n):
        add_pair("truth_value", pid,
                 present_text=true_s[pid],
                 absent_text=false_s[pid],
                 source="geometry_of_truth_cities")
    print(f"[truth_value] built {n} pairs.")
    print("[truth_value] CAVEAT: factual truth-value direction, NOT lying-intent.")


# ─────────────────────── uncertainty ────────────────────────────────────────
# Templates are length-matched: each HEDGED and CONFIDENT string is the same
# length (same filler words, just flipped stance) so the diff-of-means encodes
# epistemic stance, not verbosity.

UNCERTAINTY_STEMS = [
    "What is the capital of Australia?",
    "Who wrote the novel Moby-Dick?",
    "What year did the Berlin Wall fall?",
    "What is the boiling point of water at sea level?",
    "Which planet is closest to the sun?",
    "What is the largest organ in the human body?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "What is the speed of light approximately?",
    "Which ocean is the largest?",
    "What language is primarily spoken in Brazil?",
    "Who developed the theory of general relativity?",
    "What is the smallest prime number?",
    "What is the currency of Japan?",
    "Which gas do plants absorb for photosynthesis?",
    "What is the tallest mountain on Earth?",
    "In what century did the French Revolution occur?",
    "What is the freezing point of water in Celsius?",
    "Who was the first person on the Moon?",
    "What is the longest river in the world?",
    "How many sides does a hexagon have?",
    "What is the main ingredient in traditional guacamole?",
    "Which element has the atomic number 1?",
    "What is the capital of Canada?",
]
UNCERTAINTY_ANSWERS = [
    "Canberra", "Herman Melville", "1989", "100 degrees Celsius", "Mercury",
    "the skin", "Leonardo da Vinci", "Au", "seven", "about 300,000 km/s",
    "the Pacific", "Portuguese", "Albert Einstein", "two", "the yen",
    "carbon dioxide", "Mount Everest", "the 18th century", "0 degrees",
    "Neil Armstrong", "the Nile", "six", "avocado", "hydrogen", "Ottawa",
]
HEDGES = [
    "I'm honestly not quite sure, but I think it might be {a}, though I really could be wrong here.",
    "Hmm, I'm genuinely not certain at all — it could possibly be {a}, so please don't quote me on that.",
    "I might be misremembering this, but my tentative best guess here would be something like {a}.",
    "I'm fairly uncertain about this; it may or may not be {a}, and I'd really want to double-check first.",
    "Not at all confident about this, but I believe it could perhaps be {a}, roughly and tentatively speaking.",
]
CONFIDENT = [
    "I'm honestly completely certain, and I know for sure it is definitely {a}, with absolutely no real doubt.",
    "Yes, I'm genuinely fully certain about this — it is absolutely {a}, so you can completely quote me on that.",
    "I'm not misremembering this at all; my firm and definite answer here is clearly and certainly that it is {a}.",
    "I'm fully confident about this; it is without any question whatsoever {a}, and I'd stake a lot on being right.",
    "Completely confident about this one, and I know for sure it is certainly {a}, precisely and definitively speaking.",
]


def build_uncertainty():
    print("\n[uncertainty] constructing length-matched hedged-vs-confident pairs...")
    n = len(UNCERTAINTY_STEMS)
    for pid in range(n):
        stem = UNCERTAINTY_STEMS[pid]
        a    = UNCERTAINTY_ANSWERS[pid]
        add_pair("uncertainty", pid,
                 present_text=f"{stem}\n{HEDGES[pid % len(HEDGES)].format(a=a)}",
                 absent_text=f"{stem}\n{CONFIDENT[pid % len(CONFIDENT)].format(a=a)}",
                 source="constructed")
    print(f"[uncertainty] built {n} pairs.")


# ─────────────────────── style_emoji ────────────────────────────────────────

STYLE_CONTENT = [
    "Let's go to the park", "The meeting is at noon", "Dinner was delicious",
    "I finished the report", "The weather is nice today", "Happy birthday",
    "Good luck on your exam", "The package arrived", "I love this song",
    "We won the game", "Time for a coffee break", "The movie starts soon",
    "Thanks for your help", "I'm so excited for the trip", "The cake is ready",
    "See you tomorrow", "Great job on the presentation", "Let's celebrate tonight",
    "The sunset is beautiful", "I passed the interview", "Welcome to the team",
    "The flowers are blooming", "Pizza night", "Almost done with the project",
    "Have a wonderful weekend",
]
EMOJI_SUFFIXES = [" 🎉😄🔥", " 😍✨💯", " 🙌☀️😊", " 🥳🎊👏", " 💖🌸😎"]


def build_style_emoji():
    print("\n[style_emoji] constructing emoji-vs-plain pairs...")
    n = len(STYLE_CONTENT)
    for pid in range(n):
        c = STYLE_CONTENT[pid]
        add_pair("style_emoji", pid,
                 present_text=c + EMOJI_SUFFIXES[pid % len(EMOJI_SUFFIXES)] + "!!",
                 absent_text=c + ".",
                 source="constructed")
    print(f"[style_emoji] built {n} pairs.")


# ─────────────────────── review + save ──────────────────────────────────────

def preview(t):
    """Head + tail so long strings show both the shared context and the diff."""
    t = t.replace("\n", "\\n")
    if len(t) <= 150:
        return repr(t)
    return repr(t[:95] + " … " + t[-45:])


def review_and_save():
    if not rows:
        sys.exit("No pairs built — check warnings above.")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)

    print(f"\n{'='*72}")
    print(f"Saved {len(df)} rows ({df.pair_id.nunique()} pairs) → {OUT}")
    print(df.groupby("concept").pair_id.nunique().to_string())

    print(f"\n{'='*72}")
    print("LENGTH-BALANCE AUDIT  (mean chars per polarity; pct_gap >25% = confound)")
    la = (df.assign(chars=df.text.str.len())
            .groupby(["concept", "polarity"]).chars.mean().unstack().round(1))
    la["abs_gap"] = (la["present"] - la["absent"]).abs().round(1)
    la["pct_gap"] = (la["abs_gap"] / la[["present","absent"]].max(axis=1) * 100).round(1)
    print(la.to_string())

    print(f"\n{'='*72}")
    print("EYEBALL: 2 pairs per concept")
    for c in df.concept.unique():
        print(f"\n### {c}")
        sub = df[df.concept == c]
        for pid in sub.pair_id.unique()[:2]:
            p = sub[(sub.pair_id == pid) & (sub.polarity == "present")].text.iloc[0]
            a = sub[(sub.pair_id == pid) & (sub.polarity == "absent")].text.iloc[0]
            print(f"  [present] {preview(p)}")
            print(f"  [absent ] {preview(a)}")


if __name__ == "__main__":
    build_refusal()
    build_sycophancy()
    build_truth_value()
    build_uncertainty()
    build_style_emoji()
    review_and_save()
