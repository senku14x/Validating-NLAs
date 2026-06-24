#!/usr/bin/env bash
# ============================================================================
# run_gap_fixes_box.sh — close the four results-table gaps on a FRESH GPU box,
# from `git clone` through committed results. Self-contained: paste this whole
# script into the box shell (or `bash run_gap_fixes_box.sh`).
#
# WHAT THIS CLOSES (the gaps flagged in results_tables.md):
#   #1  Gemma AR-fidelity        -> 15_ar_fidelity.py --model gemma     (was Qwen-only)
#   #2  Gemma persona AV-read    -> 17_persona_nla_read.py --model gemma (was Qwen-only) *see caveat
#   #3  Gap regex -> JUDGE       -> 07_score_matrix.py --model M --gap   (§f.2 judge-confirmed)
#   #4  eval-aware BoW-defeated  -> 02b + 03 + 04 on eval_framing_v2     (BoW 0.50, not 1.0)
#   (+) layer-probe sweep        -> 18_layer_probe_sweep.py --model M    ("wrong layer?" control)
#
# CAVEAT on #2: 17 --model gemma feeds the *Qwen-generated* evil response texts through
# Gemma and reads Gemma's activation of them — a legitimate "does Gemma's AV read evil in
# evil-text activations" test, but NOT Gemma's own steered evil (that needs the external
# persona_vectors pipeline at Gemma L41 = real work, not this script).
#
# PREREQS:  H200 (141 GB) recommended — 80 GB needs the memory dance (handled below, but
#   Gemma AV ~54 GB + AR is tight). Network for clone/pip/HF. Export before running:
#     export HF_TOKEN=...                 # gated Gemma + AV/AR repos (accept licenses on HF)
#     export OPENROUTER_API_KEY=...       # judge for #3 (07 --gap) and 11c   (or OPENAI_API_KEY)
#   Optional overrides:  BOX_MODELS="qwen"  (cheaper single-model pass) | STAGES | REPO_DIR | WRITE_TOKEN
#
# DURABILITY: results are committed+pushed after every block, so an instance death keeps
#   completed work. Without write creds the commit stays LOCAL — paste the files listed at
#   the end back to the dev box.
# ============================================================================
set -uo pipefail

# ---------- 0. config + clone ----------
REPO_URL="${REPO_URL:-https://github.com/senku14x/Validating-NLAs.git}"
BRANCH="${BRANCH:-claude/busy-pascal-8al5k3}"
WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKDIR/Validating-NLAs}"
export NLA_REPO_DIR="${NLA_REPO_DIR:-$WORKDIR/nla_repo}"
export GIT_TERMINAL_PROMPT=0
MODELS=(${BOX_MODELS:-qwen gemma})     # qwen first = fail-fast/cheap; override BOX_MODELS="gemma"
STAGES="${STAGES:-prep target av}"     # subset to re-run after a death, e.g. STAGES="av"
say(){ echo -e "\n========== $* =========="; }

# WRITE_TOKEN (optional): a GitHub PAT with push rights -> results auto-push; else local commits only.
if [ -n "${WRITE_TOKEN:-}" ]; then
  REPO_URL="https://x-access-token:${WRITE_TOKEN}@github.com/senku14x/Validating-NLAs.git"
fi

say "0. clone / update $BRANCH"
mkdir -p "$WORKDIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL" 2>/dev/null || true
  git -C "$REPO_DIR" fetch origin "$BRANCH" && git -C "$REPO_DIR" checkout "$BRANCH" \
    && git -C "$REPO_DIR" pull --ff-only origin "$BRANCH" || true
fi
cd "$REPO_DIR/Experiment 2/rebuild"

# ---------- fail fast on required env BEFORE any long GPU work ----------
: "${HF_TOKEN:?export HF_TOKEN (gated Gemma + AV/AR repos; accept licenses on HF first)}"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY (judge for #3 07 --gap + 11c) — or export OPENAI_API_KEY and edit}"

say "0b. deps (transformers PINNED 4.57.1 — sglang 0.5.6 hard-requires it; see SGLANG.md)"
pip install -q -r requirements.txt
pip install -q "transformers==4.57.1" accelerate tqdm openai huggingface_hub safetensors orjson pyyaml

# git identity so commits work on a fresh box
git config user.email "noreply@anthropic.com" 2>/dev/null || true
git config user.name  "Claude" 2>/dev/null || true

# save(): stage the small structured results and (best-effort) push. No-op without a write token.
save(){
  git add results/gate1/04_run_gate1_battery__*__eval_framing_v2.json \
          results/gate1/04_run_gate1_battery__*__all.json \
          results/gate1/04b_recheck__*__eval_framing_v2.json \
          results/gate1/18_layer_probe_sweep__*__layer_sweep.csv \
          results/gate4/07_score_matrix__*__all__gap.parquet \
          results/gate4/15_ar_fidelity__*.json \
          results/gate4/17_persona_nla_read__*.csv \
          results/gate4/12_trackb_persistence__*.json \
          results/gate4/11c_judge_analysis__*.json 2>/dev/null || true
  git diff --cached --quiet 2>/dev/null && return 0
  git commit -q -m "box gap-fixes: $*" 2>/dev/null || { echo "  >> commit failed"; return 0; }
  for i in 1 2 3 4; do
    git push origin "$BRANCH" 2>/dev/null && { echo "  pushed ($*)"; return 0; }
    git pull --rebase --autostash origin "$BRANCH" 2>/dev/null || true
    sleep $((2**i))
  done
  echo "  >> push failed (no WRITE_TOKEN?) — commit is LOCAL; paste results back before the instance dies"
}

# ---------- 1. CPU prep: ensure eval_framing_v2 is in the parquet (idempotent) ----------
if [[ " $STAGES " == *" prep "* ]]; then
  say "1. eval_framing_v2 wiring (BoW-defeated eval-awareness; idempotent)"
  python scripts/02b_add_eval_framing_v2.py   # BoW sanity must print ~0.50
fi

# ---------- per-model pipeline ----------
for M in "${MODELS[@]}"; do

  # ===== TARGET-MODEL BLOCK (no AV server; the GPU holds the base model) =====
  if [[ " $STAGES " == *" target "* ]]; then
    say "[$M] TARGET block — extract + probe + Track-B buckets + persona-extract + layer sweep"
    pkill -f sglang || true; sleep 3                 # free any AV server — the base model needs the VRAM

    # Stage 0: extract read-layer activations (eval_framing_v2 rides along). Prereq for 04/12/15.
    python scripts/03_extract_for_battery.py --model "$M"

    # #4: Gate-1 battery — eval_framing_v2 probe vs the char-ngram floor (BoW now 0.50)
    python scripts/04_run_gate1_battery.py --model "$M"
    python scripts/04b_recheck_gate1.py   --model "$M" || true

    # #3 prerequisite: Track-B forced-compliance buckets -> cache/12_trackB_acts__$M.npz (13's input)
    python scripts/11_trackB_prefill_viability.py --model "$M" --n 40
    python scripts/11c_judge_compliance.py        --model "$M"
    python scripts/12_trackB_persistence.py       --model "$M" --n 40

    # #2 phase 1: extract the CLEAN activations of the evil-behaving texts (target model; no AV yet)
    python scripts/17_persona_nla_read.py --model "$M" || echo ">> 17 phase-1 ($M) nonzero — PASTE output"

    # (+) layer-probe sweep: natural depth profile (incl. eval_framing_v2) + steered refusal control
    python scripts/18_layer_probe_sweep.py --model "$M" --arm both \
      || echo ">> 18 layer sweep ($M) nonzero — PASTE output"

    save "$M target-block (03 extract / 04 eval_v2 probe / 11-12 buckets / 17-extract / 18 sweep)"
  fi

  # ===== AV BLOCK (SGLang AV server up; memory dance) =====
  if [[ " $STAGES " == *" av "* ]]; then
    say "[$M] AV block — gap decode + JUDGE gap + AR fidelity + persona decode"
    pkill -f sglang || true; sleep 3
    bash scripts/av_up.sh "$M" || { echo ">> av_up failed for $M — skipping AV block (see sglang log)"; continue; }

    # #3: decode the Track-B bucket activations, then JUDGE-score them (turns §f.2 regex -> judge)
    NLA_REPO_DIR="$NLA_REPO_DIR" python scripts/13_gap_decode.py --model "$M" \
      || echo ">> 13 gap decode ($M) nonzero — PASTE output"
    python scripts/07_score_matrix.py --model "$M" --gap \
      || echo ">> 07 --gap ($M) nonzero — PASTE output"

    # #1: AR reconstruction fidelity (the NLA's own metric; NLACritic + AV, reads 03 cache)
    NLA_REPO_DIR="$NLA_REPO_DIR" python scripts/15_ar_fidelity.py --model "$M" \
      || echo ">> 15 AR-fidelity ($M) nonzero — if it printed the NLACritic API, PASTE it (1-line fix)"

    # #2 phase 2: decode the clean evil activations through the AV (re-runs 17, now AV is up)
    NLA_REPO_DIR="$NLA_REPO_DIR" python scripts/17_persona_nla_read.py --model "$M" \
      || echo ">> 17 phase-2 ($M) nonzero — PASTE output"

    bash scripts/av_down.sh || true
    save "$M AV-block (13 decode / 07 --gap JUDGE / 15 AR / 17 persona decode)"
  fi
done

# ---------- final: commit + summary ----------
say "DONE — committing final results"
save "final"

cat <<'EOF'

================= PASTE THESE BACK (or confirm pushed) =================
  #4 eval-aware (BoW-defeated) probe:
     results/gate1/04_run_gate1_battery__{qwen2.5-7b,gemma3-27b}__eval_framing_v2.json
     results/gate1/04b_recheck__{qwen2.5-7b,gemma3-27b}__eval_framing_v2.json
        -> length_residualized AUROC vs the char-ngram floor (~0.98); does the probe beat surface?
  (+) layer sweep:
     results/gate1/18_layer_probe_sweep__{qwen2.5-7b,gemma3-27b}__layer_sweep.csv
        -> per-layer resid AUROC; steered-refusal arm must light up (instrument is live)
  #3 JUDGE-confirmed gap (the printed per-bucket table is the deliverable):
     results/gate4/07_score_matrix__{qwen2.5-7b,gemma3-27b}__all__gap.parquet
        -> j_refusal==2 per bucket: A high (sanity), B ~0 (THE GAP), C/E ~0 (refs)
  #1 Gemma AR fidelity:
     results/gate4/15_ar_fidelity__gemma3-27b.json   (mean_cosine, cosine_above_chance)
  #2 Gemma persona AV read:
     results/gate4/17_persona_nla_read__gemma3-27b.csv   (per-set evil read rate)
=======================================================================
If push failed (no WRITE_TOKEN): `git -C <repo> log --oneline -5` shows local commits — paste the
files above, or re-run with `export WRITE_TOKEN=<PAT>` to push.
EOF
