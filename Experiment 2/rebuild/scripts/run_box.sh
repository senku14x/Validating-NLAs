#!/usr/bin/env bash
# ============================================================================
# Validating-NLAs — box run in the EXTERNAL-REVIEW priority order.
#   #1 AR reconstruction fidelity   (the NLA's own metric; biggest validation gap)
#   #2 re-run + commit Stage 12      (so the RQ3 null's *interesting* half is reproducible)
#   #3 coupling + salience de-risk   (breaks the output-coupling vs salience confound)
#
# PREREQS (A100-80GB-class GPU box):
#   - repo cloned, on branch claude/stoic-lovelace-aa5anl
#   - export HF_TOKEN=...            (gated Gemma)
#   - export OPENROUTER_API_KEY=...  (the compliance judge, stage #2)
#   - export NLA_REPO_DIR=/workspace/nla_repo   (vendored kitft nla_inference, cloned)
#   - pip install -r requirements.txt && pip install transformers accelerate tqdm openai huggingface_hub
#   - SGLang installed + (Gemma) the input_embeds patch applied — av_up.sh handles it (see SGLANG.md)
#
# Each numbered block is independent — comment others out to run one stage.
# Long run (hours). Stage #1 uses the SGLang AV server; #2/#3 load the target model (can't coexist → pkill).
# ============================================================================
set -uo pipefail
cd "$(git rev-parse --show-toplevel)/Experiment 2/rebuild"
: "${HF_TOKEN:?export HF_TOKEN (gated Gemma)}"
export NLA_REPO_DIR="${NLA_REPO_DIR:-/workspace/nla_repo}"
MODELS=(qwen gemma)            # qwen first (lighter); edit to run one
say(){ echo -e "\n========== $* =========="; }

# ---- #1 AR reconstruction fidelity (AV server + NLACritic) ----
for M in "${MODELS[@]}"; do
  say "#1 AR fidelity — $M"
  pkill -f sglang || true; sleep 3
  bash scripts/av_up.sh "$M" || { echo ">> av_up failed for $M — skipping AR fidelity"; continue; }
  python scripts/15_ar_fidelity.py --model "$M" \
    || echo ">> AR-fidelity $M exited non-zero (likely the NLACritic API line) — PASTE its output and it's a 1-line fix"
done
pkill -f sglang || true; sleep 3

# ---- #2 re-run + commit Stage 12 (chain: 11 viability -> 11c judge -> 12 persistence) ----
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY (judge for stage #2)}"
for M in "${MODELS[@]}"; do
  say "#2 Stage 11 -> 11c -> 12 — $M"
  python scripts/11_trackB_prefill_viability.py --model "$M" --n 40
  python scripts/11c_judge_compliance.py        --model "$M"
  python scripts/12_trackB_persistence.py       --model "$M" --n 40
done

# ---- #3 coupling + salience de-risk (03 re-extract -> 14). Set DOSES=0.4,0.55,0.7 to sweep ----
for M in "${MODELS[@]}"; do
  say "#3 03 extract -> 14 coupling+salience — $M"
  python scripts/03_extract_for_battery.py --model "$M"
  DOSES="${DOSES:-0.55}" python scripts/14_coupling_score.py --model "$M"
done

# ---- commit the SMALL structured results (no harmful text; raw decodes/examples stay gitignored) ----
say "commit + push results"
git add results/gate4/15_ar_fidelity__*.json \
        results/gate4/12_trackb_persistence__*.json \
        results/gate4/14_coupling_score__*.csv results/gate4/14_coupling_score__*.json \
        results/gate4/11b_viability_analysis__*.json results/gate4/11c_judge_analysis__*.json 2>/dev/null || true
git commit -m "box run: AR fidelity + Stage-12 re-run + coupling/salience de-risk (review order)" \
  || echo ">> nothing to commit"
git push origin claude/stoic-lovelace-aa5anl || echo ">> push failed — commit is local, retry git push"

say "DONE — paste back:"
echo "  results/gate4/15_ar_fidelity__{qwen2.5-7b,gemma3-27b}.json"
echo "  results/gate4/14_coupling_score__{qwen2.5-7b,gemma3-27b}.{csv,json}   (esp. the VERDICT line)"
echo "  the printed Stage-12 [prelast] persistence rows"
