#!/usr/bin/env bash
# ============================================================================
# Validating-NLAs — box run (external-review priority order).
#   Stage 0 : 03 extraction        (PREREQUISITE — provides activations for #1 and #3)
#   #1 AR reconstruction fidelity  (the NLA's own metric; uses 03 cache + AV server)
#   #2 re-run Stage 12             (11 -> 11c -> 12; the RQ3 null's reproducible residual)
#   #3 coupling + salience de-risk (14; uses 03 cache; the salience-confound fix)
#
# PREREQS (A100/H-class GPU box): repo cloned on branch claude/stoic-lovelace-aa5anl;
#   export HF_TOKEN / OPENROUTER_API_KEY / NLA_REPO_DIR=/workspace/nla_repo ;
#   pip install -r requirements.txt && pip install "transformers==4.57.1" accelerate tqdm openai huggingface_hub
# Long run (03 extraction of both models is the slow part). Each block is independent.
# Memory dance: 03 / Stage-12 / 14 load the TARGET model; AR fidelity loads the SGLang AV — they cannot
# coexist on 80 GB, so `pkill -f sglang` brackets the AV stage.  Edit MODELS=(qwen) to do one model.
# ============================================================================
set -uo pipefail
cd "$(git rev-parse --show-toplevel)/Experiment_2/rebuild"
# --- fail fast on required env BEFORE any long GPU work ---
: "${HF_TOKEN:?export HF_TOKEN (gated Gemma + AV/AR repos)}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY (judge for stage #2 — or comment out the #2 block)}"
export NLA_REPO_DIR="${NLA_REPO_DIR:-/workspace/nla_repo}"
MODELS=(qwen gemma)            # qwen first (lighter)
say(){ echo -e "\n========== $* =========="; }

# ---- Stage 0: 03 extraction (PREREQUISITE for #1 AR fidelity AND #3 coupling) ----
pkill -f sglang || true; sleep 3                 # free any AV server — the target model needs the VRAM
for M in "${MODELS[@]}"; do
  say "Stage 0: 03 extract (target model) — $M"
  python scripts/03_extract_for_battery.py --model "$M"
done

# ---- #1 AR reconstruction fidelity (SGLang AV + NLACritic; reads 03 cache) ----
for M in "${MODELS[@]}"; do
  say "#1 AR fidelity — $M"
  pkill -f sglang || true; sleep 3
  bash scripts/av_up.sh "$M" || { echo ">> av_up failed for $M — skipping AR fidelity"; continue; }
  python scripts/15_ar_fidelity.py --model "$M" \
    || echo ">> AR-fidelity $M exited non-zero (now likely the NLACritic API line) — PASTE its output, 1-line fix"
done
pkill -f sglang || true; sleep 3                 # free the AV before the target-model stages

# ---- #2 re-run + commit Stage 12 (chain: 11 viability -> 11c judge -> 12 persistence) ----
for M in "${MODELS[@]}"; do
  say "#2 Stage 11 -> 11c -> 12 (target model + judge) — $M"
  python scripts/11_trackB_prefill_viability.py --model "$M" --n 40
  python scripts/11c_judge_compliance.py        --model "$M"
  python scripts/12_trackB_persistence.py       --model "$M" --n 40
done

# ---- #3 coupling + salience de-risk (14; reads 03 cache). DOSES=0.4,0.55,0.7 to sweep ----
for M in "${MODELS[@]}"; do
  say "#3 coupling+salience (target model) — $M"
  DOSES="${DOSES:-0.55}" python scripts/14_coupling_score.py --model "$M"
done

# ---- commit the SMALL structured results (no harmful text; raw decodes/examples stay gitignored) ----
say "commit + push results"
git add results/gate4/15_ar_fidelity__*.json \
        results/gate4/12_trackb_persistence__*.json \
        results/gate4/14_coupling_score__*.csv results/gate4/14_coupling_score__*.json \
        results/gate4/11b_viability_analysis__*.json results/gate4/11c_judge_analysis__*.json 2>/dev/null || true
git commit -m "box run: 03 extract + AR fidelity + Stage-12 re-run + coupling/salience de-risk" \
  || echo ">> nothing to commit"
git push origin claude/stoic-lovelace-aa5anl || echo ">> push failed — commit is local, retry git push"

say "DONE — paste back:"
echo "  results/gate4/15_ar_fidelity__{qwen2.5-7b,gemma3-27b}.json   (cosine / FVE)"
echo "  results/gate4/14_coupling_score__{qwen2.5-7b,gemma3-27b}.{csv,json}   (esp. the VERDICT line)"
echo "  the printed Stage-12 [prelast] persistence rows"
