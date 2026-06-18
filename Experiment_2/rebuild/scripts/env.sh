#!/usr/bin/env bash
# Shared paths + secrets for Experiment_2/rebuild shell scripts.
# Source from scripts/:  source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REBUILD="${EXP2_REBUILD:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="${EXP2_REPO_ROOT:-$(cd "$REBUILD/../.." && pwd)}"
GLP_ROOT="$REBUILD/../generative_latent_prior"

if [[ -f "$REBUILD/secrets.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REBUILD/secrets.env"
  set +a
fi
