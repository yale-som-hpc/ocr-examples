#!/usr/bin/env bash
#SBATCH --job-name=vllm_serve_app
#SBATCH --partition=gpunormal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Run vLLM via Apptainer for per-model dependency isolation.
#
# This picks a vLLM Docker image per engine. That lets one engine ship with
# transformers<4.46 (DeepSeek-OCR-2) and another with transformers>=5
# (GLM-OCR) without any Python environment conflicts. Pattern mirrors
# docling_serve.slurm.
#
# Required env vars (set by the client):
#   VLLM_IMAGE            apptainer image URI, e.g. docker://vllm/vllm-openai:v0.6.3
#   VLLM_MODEL            HF model id
#
# Optional env vars:
#   VLLM_MAX_MODEL_LEN    default 8192
#   MAX_NUM_SEQS          override per-VRAM-tier default
#   DTYPE                 override per-compute-cap default (bfloat16 / float16)
#
# Discovery: marker on stderr:
#   VLLM_LISTEN host=cNNN port=NNNNN api_key=HHHHH
#
# Disk hardening: ulimit -c 0, umask 077, private tmpfs request TMPDIR, EXIT trap.
# PDFs/OCR never persist on HPC disk — vLLM holds request bytes in memory only.
set -u

ulimit -c 0
umask 077

WORKER_TMP="/tmp/vllm-serve.$USER.$SLURM_JOB_ID.$RANDOM"
REQUEST_TMP="/dev/shm/vllm-serve.$USER.$SLURM_JOB_ID.$RANDOM"
mkdir -p "$WORKER_TMP" "$REQUEST_TMP" && chmod 700 "$WORKER_TMP" "$REQUEST_TMP"
export TMPDIR="$REQUEST_TMP"
export TEMP="$REQUEST_TMP"
export TMP="$REQUEST_TMP"
trap 'cleanup' EXIT INT TERM

module purge
module load apptainer

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

export XDG_CACHE_HOME="/gpfs/scratch60/$USER/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export PYTHONUSERBASE="$WORKER_TMP/pythonuser"
export PY_SITE="$PYTHONUSERBASE/site-packages"
export PYTHONPATH="$PY_SITE"
export PYTHONNOUSERSITE=1
export PATH="$PYTHONUSERBASE/bin:$PATH"
export APPTAINER_CACHEDIR="/gpfs/scratch60/$USER/apptainer"
# Apptainer unpacks layers into APPTAINER_TMPDIR while building the SIF; the
# vLLM nightly image is multi-GB and the per-node /tmp (~20 GB) overflows.
# /local is a ~700 GB node-local NVMe scratch that's perfect for this.
export APPTAINER_TMPDIR="/local/apptainer-tmp.$USER.$SLURM_JOB_ID.$RANDOM"
mkdir -p "$HF_HOME" "$PYTHONUSERBASE" "$PY_SITE" "$APPTAINER_CACHEDIR" "$XDG_CACHE_HOME/torch/kernels" "$APPTAINER_TMPDIR"

cleanup() {
    rm -rf "$WORKER_TMP" "$REQUEST_TMP" "$APPTAINER_TMPDIR" 2>/dev/null || true
}
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

IMAGE="${VLLM_IMAGE:?VLLM_IMAGE env var required}"
MODEL="${VLLM_MODEL:?VLLM_MODEL env var required}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
# Tensor parallel size. Default 1 (single-GPU). Set to N to shard the model
# across N GPUs on the same node — requires --gres gpu:...:N in the caller's
# srun line. Used for big models that don't fit in single A100 VRAM at fp16
# (e.g. Qwen3.6-27B fp16 ~54GB + KV cache → comfortable at TP=2).
TP="${VLLM_TP:-1}"
# Enforce-eager skips CUDA-graph capture. Added during GLM-OCR debugging
# to sidestep a graph-related crash that turned out to be a different bug
# (mrope kernel; PR #42765). For OCR engines it's a safe no-op; for
# non-OCR workloads the ~30-50% throughput cost matters, so callers can override.
# Default ON for back-compat with the OCR wrappers.
EAGER_FLAG="--enforce-eager"
if [[ "${VLLM_ENFORCE_EAGER:-1}" == "0" ]]; then
    EAGER_FLAG=""
fi
# Optional extra Python deps the trust-remote-code modeling files import
# (e.g. DeepSeek-OCR-2 needs `addict matplotlib`). Caller passes a
# space-separated list via VLLM_PIP_DEPS env var. Installed with uv into the
# private per-job PY_SITE so user account packages cannot leak into the
# container.
PIP_DEPS="${VLLM_PIP_DEPS:-}"
# Optional: bind-mount our patched mrope.py over the container's. Needed
# for GLM-OCR (vLLM issue #42016 / PR #42765 — Triton MRoPE kernel
# hardcoded NeoX-style rotation; GLM uses GPT-J-style → garbage output
# without the patch). Set VLLM_PATCH_MROPE=1 to enable.
PATCH_MROPE="${VLLM_PATCH_MROPE:-}"

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

API_KEY=$(head -c 32 /dev/urandom | xxd -p -c 32)

echo "{\"event\":\"job_start\",\"host\":\"$HOST\",\"job\":\"${SLURM_JOB_ID:-unset}\",\"port\":$PORT,\"backend\":\"vllm-apptainer\",\"image\":\"$IMAGE\"}" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2 2>&1 | head -2
echo "VLLM_LISTEN host=$HOST port=$PORT api_key=$API_KEY" >&2

# Auto-tune VRAM/dtype.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1 | tr -d ' ')
if   [[ $VRAM_MB -ge 70000 ]]; then MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
elif [[ $VRAM_MB -ge 45000 ]]; then MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
elif [[ $VRAM_MB -ge 35000 ]]; then MAX_NUM_SEQS=${MAX_NUM_SEQS:-24}
else                                MAX_NUM_SEQS=${MAX_NUM_SEQS:-12}
fi
if [[ -n "${DTYPE:-}" ]]; then
    :
elif [[ $CC_MAJOR -ge 8 ]]; then
    DTYPE=bfloat16
else
    DTYPE=float16
fi
echo "{\"event\":\"vram_detected\",\"mb\":$VRAM_MB,\"compute_cap\":\"$CC_MAJOR.x\",\"max_num_seqs\":$MAX_NUM_SEQS,\"dtype\":\"$DTYPE\"}" >&2
echo "{\"event\":\"model_select\",\"model\":\"$MODEL\",\"max_model_len\":$MAX_MODEL_LEN,\"image\":\"$IMAGE\"}" >&2

# Install engine-specific extra Python deps inside the container before
# launching vllm. DeepSeek-OCR-2's trust-remote-code modeling needs `addict`
# and `matplotlib`; other models may need different deps. Caller passes them
# space-separated via VLLM_PIP_DEPS.
if [[ -n "$PIP_DEPS" ]]; then
    echo "{\"event\":\"installing_deps\",\"tool\":\"uv\",\"deps\":\"$PIP_DEPS\"}" >&2
    apptainer exec \
        --cleanenv \
        --bind "$XDG_CACHE_HOME:$XDG_CACHE_HOME" \
        --env PYTHONUSERBASE="$PYTHONUSERBASE" \
        --env PYTHONPATH="$PYTHONPATH" \
        --env PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
        --env TMPDIR="$WORKER_TMP" \
        --env TEMP="$WORKER_TMP" \
        --env TMP="$WORKER_TMP" \
        --env PATH="$PATH" \
        "$IMAGE" \
        bash -lc "set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --target \"\$PY_SITE\" --quiet --no-cache-dir uv
    export PATH=\"\$PY_SITE/bin:\$PATH\"
fi
uv pip install --target \"\$PY_SITE\" --quiet --no-cache-dir $PIP_DEPS" >&2
    UV_RC=$?
    if [[ $UV_RC -ne 0 ]]; then
        echo "{\"event\":\"uv_install_failed\",\"rc\":$UV_RC}" >&2
        cleanup
        exit $UV_RC
    fi
fi

# Build the optional mrope-patch bind argument. The container path is the
# vllm dist-packages location confirmed from earlier tracebacks. Python 3.12
# matches the vllm/vllm-openai images.
EXTRA_BINDS=()
if [[ -n "$PATCH_MROPE" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
    MROPE_SRC="$PROJECT_DIR/patches/vllm-pr42765/mrope.py"
    MROPE_DST="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/rotary_embedding/mrope.py"
    if [[ ! -f "$MROPE_SRC" ]]; then
        echo "{\"event\":\"mrope_patch_missing\",\"src\":\"$MROPE_SRC\"}" >&2
        cleanup
        exit 2
    fi
    EXTRA_BINDS+=(--bind "$MROPE_SRC:$MROPE_DST:ro")
    echo "{\"event\":\"mrope_patch_applied\",\"src\":\"$MROPE_SRC\",\"dst\":\"$MROPE_DST\"}" >&2
fi

apptainer exec \
    --cleanenv \
    --nv \
    --bind "$XDG_CACHE_HOME:$XDG_CACHE_HOME" \
    --bind "$REQUEST_TMP:$REQUEST_TMP" \
    "${EXTRA_BINDS[@]}" \
    --env HF_HOME="$HF_HOME" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    --env TMPDIR="$TMPDIR" \
    --env TEMP="$TEMP" \
    --env TMP="$TMP" \
    --env PYTHONUSERBASE="$PYTHONUSERBASE" \
    --env PYTHONPATH="$PYTHONPATH" \
    --env PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
    --env PATH="$PATH" \
    --env VLLM_NO_USAGE_STATS=1 \
    --env DO_NOT_TRACK=1 \
    "$IMAGE" \
    vllm serve "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --api-key "$API_KEY" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization 0.90 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --tensor-parallel-size "$TP" \
        --trust-remote-code \
        $EAGER_FLAG \
        --uvicorn-log-level info \
        </dev/null >/dev/null &
# Background + wait pattern, per SURF/C3SE Slurm+Apptainer+vLLM recipe.
# Foreground apptainer exec inside our SSH-driven srun was causing vllm to
# exit cleanly (rc=0) right after "Application startup complete" — the EOF
# on the trusted-local-client client's SSH stdin cascaded down and triggered vllm's
# graceful shutdown. Backgrounding with `&` decouples vllm from the parent
# shell's job control, and `wait $!` keeps the slurm script alive until
# vllm actually exits.
VLLM_PID=$!
echo "{\"event\":\"vllm_backgrounded\",\"pid\":$VLLM_PID}" >&2
wait "$VLLM_PID"
RC=$?

cleanup
echo "{\"event\":\"job_end\",\"rc\":$RC}" >&2
exit $RC
