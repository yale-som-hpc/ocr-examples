#!/usr/bin/env bash
# One-time setup on the HPC login node.
#
# Run this after `just sync-hpc` or `hpc/bin/sync.sh` has pushed the code,
# from the cluster:
#   ssh hpc.som.yale.edu 'cd ~/ocr-examples && bash hpc/bin/bootstrap.sh'
#
# Idempotent: safe to re-run. Skips work already done.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT"
echo "bootstrap from $PROJECT  ($(date))"

# --- Step 1: confirm we're on the login node (cheap), not a compute node ---
if [[ "$(hostname)" != hpc-ln* ]]; then
    echo "WARN: not on a login node ($(hostname)). Bootstrap should run on login,"
    echo "      not in a GPU allocation. Continuing anyway."
fi

# --- Step 2: PATH for uv ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "uv: $(uv --version)"

# --- Step 3: XDG cache redirected to scratch (models/images can be large) ---
export XDG_CACHE_HOME="/gpfs/scratch60/$USER/.cache"
mkdir -p "$XDG_CACHE_HOME"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"

# --- Step 4: confirm container runtime ---
if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer >/dev/null 2>&1 || true
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer not found; load the site Apptainer module before running GPU jobs."
    exit 1
fi
echo "apptainer: $(apptainer --version)"

# --- Step 5: caches/logs for containerized jobs ---
mkdir -p "/gpfs/scratch60/$USER/apptainer"
mkdir -p "$PROJECT/logs"

echo ""
echo "bootstrap complete."
echo "  model cache:  $XDG_CACHE_HOME/huggingface"
echo "  image cache:  /gpfs/scratch60/$USER/apptainer"
echo "  logs dir:     $PROJECT/logs"
echo ""
echo "next: drive from the trusted local client with an engine script and --use-hpc."
