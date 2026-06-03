#!/usr/bin/env bash
# Launch the dense (SGLang) judge server used by the WebArena evaluator.
#
# The judge is shared between the TSA agent run and the dense agent run, so
# bias from a model-specific judge is eliminated. Always SGLang dense.
#
# Idempotent: if the tmux session is already running and /health responds,
# this is a no-op (apart from re-opening the SSH tunnel if it was dropped).
#
# Env:
#   GPU_HOST              user@host where the GPU lives. Default gray.
#   GPU_SM_HINT           override SM detection (e.g. "100" for B200).
#   JUDGE_PORT            local + remote port (default 10002)
#   JUDGE_MODEL_PATH      path to Qwen3-VL-4B weights on the GPU host
#   JUDGE_MODEL_NAME      served-model-name (default qwen3vl-judge)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_inference_common.sh"
. "$SCRIPT_DIR/configs/gpu_profile.sh"

JUDGE_PORT="${JUDGE_PORT:-10002}"
# Default to Qwen3-VL-2B for the judge so it co-exists in VRAM with the agent
# backend on a 16 GB GPU (the 4B agent already eats ~8.5 GB). Override with
# JUDGE_MODEL_PATH to use a larger judge on roomier hardware (e.g. B200).
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-\$HOME/hf_models/Qwen3-VL-2B-Instruct}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-qwen3vl-judge}"
JUDGE_MEM_FRAC="${JUDGE_MEM_FRAC:-${MEM_FRAC_JUDGE:-0.35}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-\$HOME/venvs/bench_sglang/bin/python}"

echo "=== Judge launcher ==="
echo "GPU_HOST:           $GPU_HOST"
echo "GPU_SM:             $GPU_SM"
echo "GPU_NOTES:          $GPU_NOTES"
echo "JUDGE_PORT:         $JUDGE_PORT (local + remote)"
echo "JUDGE_MODEL_PATH:   $JUDGE_MODEL_PATH"
echo "JUDGE_MODEL_NAME:   $JUDGE_MODEL_NAME"
echo "SGLANG_BACKEND:     $SGLANG_BACKEND (fallback: $SGLANG_BACKEND_FALLBACK)"
echo "MEM_FRAC_JUDGE:     $MEM_FRAC_JUDGE"

# 1. Start the SGLang judge in tmux (idempotent).
REMOTE_CMD=$(cat <<EOF
set -e
LOG=\$HOME/.cache/wa_judge.log
mkdir -p "\$(dirname "\$LOG")"
# Try the preferred attention backend; if SGLang exits non-zero within 30s, fall back.
attempt() {
    local backend="\$1"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    $SGLANG_PYTHON -m sglang.launch_server \\
        --model-path $JUDGE_MODEL_PATH \\
        --host 0.0.0.0 --port $JUDGE_PORT \\
        --served-model-name $JUDGE_MODEL_NAME \\
        --chat-template qwen2-vl \\
        --context-length $JUDGE_CTX_LEN \\
        --max-running-requests 16 \\
        --dtype bfloat16 \\
        --mem-fraction-static $JUDGE_MEM_FRAC \\
        --attention-backend "\$backend" \\
        --disable-radix-cache 2>&1 | tee -a "\$LOG"
}
attempt "$SGLANG_BACKEND" || attempt "$SGLANG_BACKEND_FALLBACK"
EOF
)

ensure_tmux "wa-judge" "$REMOTE_CMD"

# 2. Tunnel hilbit2 → GPU_HOST:$JUDGE_PORT
open_tunnel "$JUDGE_PORT" "$JUDGE_PORT"

# 3. Wait until healthy.
echo "[judge] waiting for /health on 127.0.0.1:$JUDGE_PORT ..."
wait_healthy "http://127.0.0.1:${JUDGE_PORT}/health" 600

echo "[judge] ready: http://127.0.0.1:${JUDGE_PORT}/v1/chat/completions"
echo ""
echo "# Export these for the orchestrator's eval-time judge calls:"
echo "export WEBARENA_EVAL_API_BASE=http://127.0.0.1:${JUDGE_PORT}/v1"
echo "export WEBARENA_EVAL_API_KEY=judge"
echo "export WEBARENA_EVAL_MODEL=${JUDGE_MODEL_NAME}"
