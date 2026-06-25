#!/usr/bin/env bash
# Sync this repo from the trusted local client to the cluster's project dir.
#
# One-way: the trusted local client is source of truth. Run this after editing any HPC code.
# The default excludes local data/results/cache directories so documents are
# not copied to HPC by accident.
#
# Usage:
#   hpc/bin/sync.sh
#   hpc/bin/sync.sh --dry-run
#   hpc/bin/sync.sh --delete   # also remove cluster-side files that no longer exist locally
set -euo pipefail

# Defaults (override via env if you want)
: "${HPC_HOST:=hpc.som.yale.edu}"
: "${HPC_USER:=${USER:-}}"
: "${HPC_KEY:=}"
: "${HPC_REMOTE_DIR:=ocr-examples}"  # relative to $HOME on cluster

dry=""
delete=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry="--dry-run" ;;
        --delete)  delete="--delete" ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Quote / trailing-slash semantics: trailing / on source means "contents of",
# without means "the dir itself". We want the repo contents to land in
# $HPC_REMOTE_DIR/ on the cluster.
echo "syncing $SOURCE_DIR/  ->  $HPC_USER@$HPC_HOST:$HPC_REMOTE_DIR/"
if [[ -n "$HPC_KEY" ]]; then
    echo "  (using key: $HPC_KEY)"
else
    echo "  (using default ssh config/agent)"
fi
echo ""

RSYNC_SSH=(ssh)
if [[ -n "$HPC_KEY" ]]; then
    RSYNC_SSH+=(-i "$HPC_KEY" -o IdentitiesOnly=yes)
fi

rsync \
    -av \
    --human-readable \
    --partial \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.uv-cache/' \
    --exclude '.uv-tools/' \
    --exclude '.ruff_cache/' \
    --exclude 'data/' \
    --exclude 'results/' \
    --exclude 'logs/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.cache/' \
    $dry $delete \
    -e "${RSYNC_SSH[*]}" \
    "$SOURCE_DIR/" \
    "$HPC_USER@$HPC_HOST:$HPC_REMOTE_DIR/"

echo ""
echo "sync complete."
echo "cluster-side path: ~/$HPC_REMOTE_DIR/"
