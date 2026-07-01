#!/usr/bin/env bash
# Boot the SGLang dense baseline server on the GPU host, tunnel it to hilbit2,
# and emit the env block the orchestrator needs to point the agent at SGLang.
#
# Same conventions as launch_tsa.sh — idempotent skip, auto-fallback attention
# backend, judge boots as a pre-step.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_inference_common.sh"
. "$SCRIPT_DIR/configs/gpu_profile.sh"

DENSE_PORT="${DENSE_PORT:-10001}"
DENSE_MODEL_PATH="${DENSE_MODEL_PATH:-\$HOME/hf_models/Qwen3-VL-4B-Instruct}"
DENSE_MODEL_NAME="${DENSE_MODEL_NAME:-qwen3vl-dense}"
JUDGE_PORT="${JUDGE_PORT:-10002}"
SGLANG_PYTHON="${SGLANG_PYTHON:-\$HOME/venvs/bench_sglang/bin/python}"

echo "=== Dense (SGLang) launcher ==="
echo "GPU_HOST:            $GPU_HOST"
echo "GPU_SM:              $GPU_SM"
echo "GPU_NOTES:           $GPU_NOTES"
echo "DENSE_PORT:          $DENSE_PORT (local + remote)"
echo "DENSE_MODEL_PATH:    $DENSE_MODEL_PATH"
echo "DENSE_MODEL_NAME:    $DENSE_MODEL_NAME"
echo "SGLANG_BACKEND:      $SGLANG_BACKEND (fallback: $SGLANG_BACKEND_FALLBACK)"
echo "MEM_FRAC_AGENT:      $MEM_FRAC_AGENT"

# 1. Bring up the judge first — but ONLY if it fits alongside the dense agent.
#    On sm_120 (16 GB) the profile sets JUDGE_FITS=with_tsa_only: a 2B judge
#    co-resident with the 4B dense agent (MEM_FRAC_AGENT=0.60) overflows VRAM and
#    the dense server dies with "Not enough memory ... mem_fraction_static". The
#    documented sm_120 mode is judge-off (deterministic / self-judge eval). Boot
#    the judge only when it genuinely co-resides (JUDGE_FITS=yes) or when the
#    caller forces it (LAUNCH_JUDGE=1).
if [ "${JUDGE_FITS:-}" = "yes" ] || [ "${LAUNCH_JUDGE:-0}" = "1" ]; then
    bash "$SCRIPT_DIR/launch_judge.sh"
else
    echo "[dense] skipping judge boot (JUDGE_FITS=${JUDGE_FITS:-unset}); dense gets the full GPU."
    echo "        deterministic eval needs no judge; for fuzzy_match set LAUNCH_JUDGE=1 (needs a GPU that fits both)."
fi

# 2. Start SGLang in tmux on the GPU host. Try preferred attention backend,
#    fall back automatically if it fails (e.g. flashinfer wheels missing sm_120).
REMOTE_CMD=$(cat <<EOF
set -e
LOG=\$HOME/.cache/wa_dense.log
mkdir -p "\$(dirname "\$LOG")"
attempt() {
    local backend="\$1"
    # Do not pass --chat-template here. SGLang bundled qwen2-vl template was
    # designed for Qwen2-VL and breaks Qwen3-VL prompts (vision-pad tokens,
    # wrong system-marker wrapping). Without that flag SGLang reads the model
    # native ChatML template from tokenizer_config.json, matching TSA path.
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    $SGLANG_PYTHON -m sglang.launch_server \\
        --model-path $DENSE_MODEL_PATH \\
        --host 0.0.0.0 --port $DENSE_PORT \\
        --served-model-name $DENSE_MODEL_NAME \\
        --context-length $AGENT_CTX_LEN \\
        --dtype bfloat16 \\
        --mem-fraction-static $MEM_FRAC_AGENT \\
        --attention-backend "\$backend" \\
        --disable-radix-cache 2>&1 | tee -a "\$LOG"
}
attempt "$SGLANG_BACKEND" || attempt "$SGLANG_BACKEND_FALLBACK"
EOF
)
ensure_tmux "wa-dense" "$REMOTE_CMD"

# 3. Tunnel.
open_tunnel "$DENSE_PORT" "$DENSE_PORT"

# 4. Wait until healthy.
echo "[dense] waiting for /health on 127.0.0.1:$DENSE_PORT ..."
wait_healthy "http://127.0.0.1:${DENSE_PORT}/health" 900

# 5. Verify served-model-name.
echo "[dense] verifying model id ..."
SERVED=$(curl -s "http://127.0.0.1:${DENSE_PORT}/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "[dense] /v1/models reports id=$SERVED"

echo ""
echo "[dense] ready: http://127.0.0.1:${DENSE_PORT}/v1/chat/completions"
echo ""
ENV_FILE="$SCRIPT_DIR/.inference_env"
{
    echo "export OPENAI_API_BASE=http://127.0.0.1:${DENSE_PORT}/v1"
    echo "export OPENAI_API_KEY=dense"
    echo "export WEBARENA_EVAL_API_BASE=http://127.0.0.1:${JUDGE_PORT}/v1"
    echo "export WEBARENA_EVAL_API_KEY=judge"
    echo "export WEBARENA_EVAL_MODEL=${JUDGE_MODEL_NAME}"
    echo "export WEBARENA_TOKENIZER_PATH=${WEBARENA_TOKENIZER_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
    echo "export INFERENCE_BACKEND=dense"
    echo "export AGENT_MODEL_NAME=$SERVED"
} | tee "$ENV_FILE"

export OPENAI_API_BASE="http://127.0.0.1:${DENSE_PORT}/v1"
export OPENAI_API_KEY="dense"
export WEBARENA_EVAL_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1"
export WEBARENA_EVAL_API_KEY="judge"
export WEBARENA_EVAL_MODEL="${JUDGE_MODEL_NAME}"
export WEBARENA_TOKENIZER_PATH="${WEBARENA_TOKENIZER_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
export INFERENCE_BACKEND="dense"
export AGENT_MODEL_NAME="$SERVED"
