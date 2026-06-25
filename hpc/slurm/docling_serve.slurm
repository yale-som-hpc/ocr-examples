#!/usr/bin/env bash
#SBATCH --job-name=docling_serve
#SBATCH --partition=gpunormal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Run docling-serve (official Apptainer CUDA image) on a GPU node, with the
# same disk-hardening / discovery-marker contract as the vLLM service.
#
# The server listens on a random high port. The trusted local client tunnels that
# port back via SSH (-L localhost:LOCAL_PORT:COMPUTE_NODE:REMOTE_PORT login.node)
# and POSTs PDF bytes to /v1/convert/file with X-Api-Key authentication.
#
# Discovery: one marker line on stderr early so the client can parse out
# (host, port, api_key):
#   DOCLING_LISTEN host=cNNN port=NNNNN api_key=HHHHH
#
# Disk hardening (no PII on disk, ever):
#   - ulimit -c 0 / umask 077 / private tmpfs TMPDIR / EXIT trap
#   - docling-serve never writes uploaded files; processes them in-memory.
#   - Slurm log lands on the trusted local client (we drive via `ssh ... srun ...`, not sbatch).
set -u

ulimit -c 0
umask 077

REQUEST_TMP="/dev/shm/docling-serve.$USER.$SLURM_JOB_ID.$RANDOM"
mkdir -p "$REQUEST_TMP" && chmod 700 "$REQUEST_TMP"
export TMPDIR="$REQUEST_TMP"
export TEMP="$REQUEST_TMP"
export TMP="$REQUEST_TMP"
cleanup() {
    rm -rf "$REQUEST_TMP" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

module purge
module load apptainer

# HF model cache (layout / OCR weights, no user data) and Apptainer SIF cache
# on scratch — both are large and don't belong in $HOME. Torch JIT-kernel
# cache wants its own dir; pre-create so we don't see a startup warning.
export XDG_CACHE_HOME="/gpfs/scratch60/$USER/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export APPTAINER_CACHEDIR="/gpfs/scratch60/$USER/apptainer"
export PYTHONNOUSERSITE=1
mkdir -p "$HF_HOME" "$APPTAINER_CACHEDIR" "$XDG_CACHE_HOME/torch/kernels"

# Bound OMP/BLAS threading. Without these, ONNX Runtime (used by docling's
# layout + table models) tries to bind one thread per core on the underlying
# host (often ~128 on the cluster) and emits ~100 lines of
# `pthread_setaffinity_np failed` to stderr. Cosmetic but noisy. Capping at
# the Slurm-allocated CPU count silences it.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Official docling-serve image with CUDA 12.6 (Ampere-and-newer).
# Override with DOCLING_IMAGE env var if you need a specific tag.
IMAGE="${DOCLING_IMAGE:-docker://quay.io/docling-project/docling-serve-cu126:latest}"

# Random high port in the ephemeral range. Retry if occupied.
HOST=$(hostname -s)
PICK_PORT() {
    echo $(( 49152 + (RANDOM % (65535 - 49152 + 1)) ))
}
PORT=$(PICK_PORT)
for _try in 1 2 3 4 5; do
    if ! ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}$"; then
        break
    fi
    PORT=$(PICK_PORT)
done

# Per-job random API key. Defense in depth — the SSH tunnel is already
# user-scoped, but the api-key check blocks any other process on this compute
# node that happens to bind localhost.
API_KEY=$(head -c 32 /dev/urandom | xxd -p -c 32)

echo "{\"event\":\"job_start\",\"host\":\"$HOST\",\"job\":\"${SLURM_JOB_ID:-unset}\",\"port\":$PORT,\"backend\":\"docling-serve\"}" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2 2>&1 | head -2
echo "DOCLING_LISTEN host=$HOST port=$PORT api_key=$API_KEY" >&2

# docling-serve env. See docs/configuration.md upstream for the full list.
# DOCLING_DEVICE=cuda enables GPU; the docling library auto-detects torch CUDA.
# DOCLING_SERVE_ENG_LOC_NUM_WORKERS = in-process concurrency (threads).
# DOCLING_SERVE_ENG_LOC_SHARE_MODELS = one model copy across threads (memory).
# UVICORN_WORKERS=1 because we scale via N parallel slurm jobs, not within one.
export DOCLING_DEVICE=cuda
export DOCLING_SERVE_API_KEY="$API_KEY"
export DOCLING_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export DOCLING_SERVE_ENG_LOC_NUM_WORKERS=4
export DOCLING_SERVE_ENG_LOC_SHARE_MODELS=true
export DOCLING_SERVE_OPTIONS_CACHE_SIZE=4
export UVICORN_HOST=0.0.0.0
export UVICORN_PORT=$PORT
export UVICORN_WORKERS=1

# apptainer exec --cleanenv keeps user account Python state out of the
# container. Pass only the service and cache environment required by docling.
# Request/runtime temp is private tmpfs so multipart uploads and framework
# spill files do not land on persistent or node-local disk.
# Bind-mount only HF_HOME so the model cache persists between runs.
# Child process (NOT exec) so the EXIT trap fires after the container exits.
apptainer exec \
    --cleanenv \
    --nv \
    --bind "$HF_HOME:$HF_HOME" \
    --bind "$REQUEST_TMP:$REQUEST_TMP" \
    --env HF_HOME="$HF_HOME" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    --env TMPDIR="$TMPDIR" \
    --env TEMP="$TEMP" \
    --env TMP="$TMP" \
    --env PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
    --env PYTHONPATH="" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    --env OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    --env DOCLING_DEVICE="$DOCLING_DEVICE" \
    --env DOCLING_SERVE_API_KEY="$DOCLING_SERVE_API_KEY" \
    --env DOCLING_NUM_THREADS="$DOCLING_NUM_THREADS" \
    --env DOCLING_SERVE_ENG_LOC_NUM_WORKERS="$DOCLING_SERVE_ENG_LOC_NUM_WORKERS" \
    --env DOCLING_SERVE_ENG_LOC_SHARE_MODELS="$DOCLING_SERVE_ENG_LOC_SHARE_MODELS" \
    --env DOCLING_SERVE_OPTIONS_CACHE_SIZE="$DOCLING_SERVE_OPTIONS_CACHE_SIZE" \
    --env UVICORN_HOST="$UVICORN_HOST" \
    --env UVICORN_PORT="$UVICORN_PORT" \
    --env UVICORN_WORKERS="$UVICORN_WORKERS" \
    "$IMAGE" \
    docling-serve run
RC=$?

cleanup
echo "{\"event\":\"job_end\",\"rc\":$RC}" >&2
exit $RC
