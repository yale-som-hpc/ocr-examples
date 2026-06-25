#!/usr/bin/env bash
#SBATCH --job-name=sglang_serve
#SBATCH --partition=gpunormal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Serve a Qwen-family (or similar) LLM via SGLang in an Apptainer container.
#
# Mirrors vllm_serve_apptainer.slurm structurally — same VLLM_LISTEN
# discovery marker on stderr so the existing hpc/client/vllm_http_client.py
# can drive either engine just by switching --slurm-script. The marker
# contract is host/port/api_key on a single line, parsable by LISTEN_RE.
#
# We share the VLLM_* env var names with the vLLM script so callers don't
# have to fork their env-forwarding logic. SGLang-specific knobs default
# from the SOM HPC LLM API service pattern and are
# overridable via SGLANG_* env vars.
#
# Required env vars:
#   VLLM_IMAGE            apptainer image URI, e.g. docker://lmsysorg/sglang:dev-cu12
#   VLLM_MODEL            HF model id or local path
#
# Optional env vars:
#   VLLM_MAX_MODEL_LEN          context length (default 65536)
#   VLLM_TP                          tensor-parallel size (default 1)
#   VLLM_PIP_DEPS                    space-separated deps/wheels to install with uv
#   SGLANG_SERVED_MODEL_NAME         OpenAI API model name (default: VLLM_MODEL)
#   SGLANG_MODEL_ARG                 launch_server model flag (default: --model-path)
#   SGLANG_ATTENTION_BACKEND         optional --attention-backend value
#   SGLANG_PAGE_SIZE                 optional --page-size value
#   SGLANG_KV_CACHE_DTYPE            default bf16
#   SGLANG_MEM_FRACTION_STATIC       default 0.85
#   SGLANG_MAX_RUNNING_REQUESTS      default 16
#   SGLANG_REASONING_PARSER          default qwen3 (use 'none' to disable parsing)
#   SGLANG_ENABLE_CUSTOM_LOGIT_PROCESSOR=1  pass --enable-custom-logit-processor
#   SGLANG_DISABLE_OVERLAP_SCHEDULE=1        pass --disable-overlap-schedule
#   SGLANG_SKIP_SERVER_WARMUP=1              pass --skip-server-warmup
#   SGLANG_DISABLE_CUDA_GRAPH=1              pass --disable-cuda-graph
#
# Disk hardening: ulimit -c 0, umask 077, private tmpfs request TMPDIR +
# APPTAINER_TMPDIR, EXIT trap. Same envelope as the vLLM apptainer script.
set -u

ulimit -c 0
umask 077

WORKER_TMP="/tmp/sglang-serve.$USER.$SLURM_JOB_ID.$RANDOM"
REQUEST_TMP="/dev/shm/sglang-serve.$USER.$SLURM_JOB_ID.$RANDOM"
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
# Keep per-engine uv installs out of ~/.local so one experimental model cannot
# poison other containers' Python environments. The directory is wiped on exit.
export PYTHONUSERBASE="$WORKER_TMP/pythonuser"
export PY_SITE="$PYTHONUSERBASE/site-packages"
export PYTHONPATH="$PY_SITE"
export PYTHONNOUSERSITE=1
export PATH="$PYTHONUSERBASE/bin:$PATH"
export APPTAINER_CACHEDIR="/gpfs/scratch60/$USER/apptainer"
export APPTAINER_TMPDIR="/local/apptainer-tmp.$USER.$SLURM_JOB_ID.$RANDOM"
mkdir -p "$HF_HOME" "$PYTHONUSERBASE" "$PY_SITE" "$APPTAINER_CACHEDIR" "$XDG_CACHE_HOME/torch/kernels" "$APPTAINER_TMPDIR"

cleanup() {
    rm -rf "$WORKER_TMP" "$REQUEST_TMP" "$APPTAINER_TMPDIR" 2>/dev/null || true
}

IMAGE="${VLLM_IMAGE:?VLLM_IMAGE env var required}"
MODEL="${VLLM_MODEL:?VLLM_MODEL env var required}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
TP="${VLLM_TP:-1}"
UV_DEPS="${VLLM_PIP_DEPS:-}"
UV_NO_DEPS="${SGLANG_UV_NO_DEPS:-1}"
SERVED_MODEL_NAME="${SGLANG_SERVED_MODEL_NAME:-$MODEL}"
MODEL_ARG="${SGLANG_MODEL_ARG:---model-path}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-}"
PAGE_SIZE="${SGLANG_PAGE_SIZE:-}"
REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"
# Longest-prefix-match scheduling. We process N documents that share an
# identical ~3K-token system prompt. lpm reorders requests so they share
# RadixAttention prefix cache hits — substantial win for our workload per
# SGLang's own tuning guide ("if your workload has many shared prefixes").
SCHEDULE_POLICY="${SGLANG_SCHEDULE_POLICY:-lpm}"

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1 | tr -d ' ')

if [[ -n "${SGLANG_KV_CACHE_DTYPE:-}" ]]; then
    KV_DTYPE="$SGLANG_KV_CACHE_DTYPE"
elif [[ $CC_MAJOR -ge 8 ]]; then
    KV_DTYPE=bf16
else
    # Current SGLang CLI does not accept "float16" for --kv-cache-dtype, but
    # auto detects SM75 and chooses fp16 internally.
    KV_DTYPE=auto
fi

# Push mem-fraction high on Ampere+ per SGLang hyperparameter_tuning.md. On
# RTX/Turing, leave more activation headroom and keep first-request latency
# predictable for a shared queue.
if [[ -n "${SGLANG_MEM_FRACTION_STATIC:-}" ]]; then
    MEM_FRACTION="$SGLANG_MEM_FRACTION_STATIC"
elif [[ $CC_MAJOR -ge 8 ]]; then
    MEM_FRACTION=0.90
else
    MEM_FRACTION=0.70
fi
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]]; then
    MAX_RUNNING="$SGLANG_MAX_RUNNING_REQUESTS"
elif [[ $CC_MAJOR -ge 8 ]]; then
    MAX_RUNNING=32
else
    MAX_RUNNING=4
fi
if [[ -n "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]]; then
    CHUNKED_PREFILL="$SGLANG_CHUNKED_PREFILL_SIZE"
elif [[ $CC_MAJOR -ge 8 ]]; then
    CHUNKED_PREFILL=4096
else
    CHUNKED_PREFILL=1024
fi
if [[ -n "${SGLANG_MAX_PREFILL_TOKENS:-}" ]]; then
    MAX_PREFILL="$SGLANG_MAX_PREFILL_TOKENS"
elif [[ $CC_MAJOR -ge 8 ]]; then
    MAX_PREFILL=8192
else
    MAX_PREFILL=2048
fi
# cuda-graph batches up to this size get captured (faster decode). Default 32
# is conservative; docs recommend pushing to 128+ for high-TP / saturated work.
CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-128}"
if [[ $CC_MAJOR -lt 8 && -z "${SGLANG_SKIP_SERVER_WARMUP:-}" ]]; then
    SGLANG_SKIP_SERVER_WARMUP=1
fi

HOST=$(hostname -s)
# Pick from a LOWER range than vllm_serve_apptainer.slurm (49152-65535)
# because SGLang reserves additional internal ports (gRPC, scheduler) by
# adding offsets in the 5000-10000 range to the user port. Picking too
# high overflows 65535. 30000-40000 gives ~25K of safe headroom.
PICK_PORT() {
    echo $(( 30000 + (RANDOM % 10000) ))
}
PORT=$(PICK_PORT)
for _try in 1 2 3 4 5; do
    if ! ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}$"; then
        break
    fi
    PORT=$(PICK_PORT)
done

API_KEY=$(head -c 32 /dev/urandom | xxd -p -c 32)

echo "{\"event\":\"job_start\",\"host\":\"$HOST\",\"job\":\"${SLURM_JOB_ID:-unset}\",\"port\":$PORT,\"backend\":\"sglang-apptainer\",\"image\":\"$IMAGE\"}" >&2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2 2>&1 | head -2
# Use the SAME LISTEN marker name as the vLLM scripts so vllm_http_client.py
# (which only looks for VLLM_LISTEN) can drive this without modifications.
echo "VLLM_LISTEN host=$HOST port=$PORT api_key=$API_KEY" >&2
echo "{\"event\":\"model_select\",\"model\":\"$MODEL\",\"served_model\":\"$SERVED_MODEL_NAME\",\"max_model_len\":$MAX_MODEL_LEN,\"image\":\"$IMAGE\",\"tp\":$TP,\"engine\":\"sglang\",\"compute_cap\":\"$CC_MAJOR.x\",\"kv_dtype\":\"$KV_DTYPE\",\"mem_fraction\":$MEM_FRACTION,\"max_running\":$MAX_RUNNING,\"chunked_prefill\":$CHUNKED_PREFILL,\"max_prefill\":$MAX_PREFILL}" >&2

if [[ -n "$UV_DEPS" ]]; then
    echo "{\"event\":\"installing_deps\",\"deps\":\"$UV_DEPS\",\"tool\":\"uv\"}" >&2
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
        bash -lc "uv pip install --target \"$PY_SITE\" $([[ \"$UV_NO_DEPS\" == \"1\" ]] && printf %s --no-deps) $UV_DEPS" >&2
    UV_RC=$?
    if [[ $UV_RC -ne 0 ]]; then
        echo "{\"event\":\"uv_install_failed\",\"rc\":$UV_RC}" >&2
        cleanup
        exit $UV_RC
    fi
fi

EXTRA_ARGS=()
EXTRA_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
if [[ -n "$ATTENTION_BACKEND" ]]; then
    EXTRA_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ -n "$PAGE_SIZE" ]]; then
    EXTRA_ARGS+=(--page-size "$PAGE_SIZE")
fi
if [[ "$REASONING_PARSER" != "none" && -n "$REASONING_PARSER" ]]; then
    EXTRA_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ "${SGLANG_ENABLE_CUSTOM_LOGIT_PROCESSOR:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--enable-custom-logit-processor)
fi
if [[ "${SGLANG_DISABLE_OVERLAP_SCHEDULE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--disable-overlap-schedule)
fi
if [[ "${SGLANG_SKIP_SERVER_WARMUP:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-server-warmup)
fi
if [[ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--disable-cuda-graph)
fi

# Background apptainer + wait — same rationale as the vLLM script:
# foreground exec inside the SSH-driven srun makes the server graceful-exit
# right after startup as soon as the parent loses interest. `&` + `wait $!`
# decouples it. </dev/null avoids any stdin-EOF surprises; >/dev/null is
# belt-and-suspenders against SIGPIPE if our stdout pipe were to close.
apptainer exec \
    --cleanenv \
    --nv \
    --bind "$XDG_CACHE_HOME:$XDG_CACHE_HOME" \
    --bind "$REQUEST_TMP:$REQUEST_TMP" \
    --env HF_HOME="$HF_HOME" \
    --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    --env TMPDIR="$TMPDIR" \
    --env TEMP="$TEMP" \
    --env TMP="$TMP" \
    --env PYTHONUSERBASE="$PYTHONUSERBASE" \
    --env PYTHONPATH="$PYTHONPATH" \
    --env PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
    --env PATH="$PATH" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
    --env SGLANG_DISABLE_CUDNN_CHECK=1 \
    "$IMAGE" \
    python3 -m sglang.launch_server \
        "$MODEL_ARG" "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --api-key "$API_KEY" \
        --file-storage-path "$REQUEST_TMP/sglang_storage" \
        --tensor-parallel-size "$TP" \
        --kv-cache-dtype "$KV_DTYPE" \
        --mem-fraction-static "$MEM_FRACTION" \
        --context-length "$MAX_MODEL_LEN" \
        --max-running-requests "$MAX_RUNNING" \
        --chunked-prefill-size "$CHUNKED_PREFILL" \
        --max-prefill-tokens "$MAX_PREFILL" \
        --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
        --schedule-policy "$SCHEDULE_POLICY" \
        --enable-metrics \
        --trust-remote-code \
        "${EXTRA_ARGS[@]}" \
        </dev/null >/dev/null &
SGLANG_PID=$!
echo "{\"event\":\"sglang_backgrounded\",\"pid\":$SGLANG_PID}" >&2
wait "$SGLANG_PID"
RC=$?

cleanup
echo "{\"event\":\"job_end\",\"rc\":$RC}" >&2
exit $RC
