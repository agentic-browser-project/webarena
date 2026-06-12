#!/usr/bin/env bash
# Boot TSA on the GPU host, tunnel it to hilbit2, and emit the env block
# the orchestrator needs to point the agent at TSA.
#
# Idempotent: skips server boot if the tmux session is already alive AND
# /health returns 200. Always (re-)opens the tunnel and waits for health.
#
# Usage (from webarena root on hilbit2):
#   source mp/launch_tsa.sh
#   python -m mp.orchestrator --config mp/configs/config-tsa.json \
#       --inference_backend tsa --model tree-sparse ...
#
# Env knobs (all optional):
#   GPU_HOST              user@host where the GPU lives. Default gray.
#   GPU_SM_HINT           override SM detection (e.g. "100" for B200).
#   TSA_PORT              local + remote port (default 10000)
#   TSA_MODEL_PATH        Qwen3-VL-4B path on the GPU host
#   TSA_MODEL_NAME        served-model-name (default tree-sparse)
#   TSA_TREE_PARSE_MODE   default "webarena"
#   TSA_STRIP_FENCES      "0" disables fence-strip (default), "1" enables

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_inference_common.sh"
. "$SCRIPT_DIR/configs/gpu_profile.sh"

TSA_PORT="${TSA_PORT:-10000}"
TSA_MODEL_PATH="${TSA_MODEL_PATH:-\$HOME/hf_models/Qwen3-VL-4B-Instruct}"
TSA_MODEL_NAME="${TSA_MODEL_NAME:-tree-sparse}"
TSA_TREE_PARSE_MODE="${TSA_TREE_PARSE_MODE:-webarena}"
TSA_STRIP_FENCES="${TSA_STRIP_FENCES:-0}"
JUDGE_PORT="${JUDGE_PORT:-10002}"

echo "=== TSA launcher ==="
echo "GPU_HOST:            $GPU_HOST"
echo "GPU_SM:              $GPU_SM"
echo "GPU_NOTES:           $GPU_NOTES"
echo "TSA_PORT:            $TSA_PORT (local + remote)"
echo "TSA_MODEL_PATH:      $TSA_MODEL_PATH"
echo "TSA_MODEL_NAME:      $TSA_MODEL_NAME"
echo "TSA_TREE_PARSE_MODE: $TSA_TREE_PARSE_MODE"
echo "TSA_STRIP_FENCES:    $TSA_STRIP_FENCES"
echo "TSA_MAX_BATCH:       $TSA_MAX_BATCH"

# 1. Bring up the judge first (idempotent skip if already running).
bash "$SCRIPT_DIR/launch_judge.sh"

# 2. Start TSA serve.py in tmux on the GPU host. start_server.sh honors all
#    the TSA_* env vars set by gpu_profile.sh.
REMOTE_CMD=$(cat <<EOF
set -e
export TS_CUDA_ARCHS="${TS_CUDA_ARCHS:-100;120}"
export TSA_PORT=$TSA_PORT
export TSA_MODEL_PATH=$TSA_MODEL_PATH
export TSA_MODEL_NAME=$TSA_MODEL_NAME
export TSA_TREE_PARSE_MODE=$TSA_TREE_PARSE_MODE
export TSA_STRIP_FENCES=$TSA_STRIP_FENCES
export TSA_MAX_BATCH=$TSA_MAX_BATCH
export TSA_TOP_K=$TSA_TOP_K
export TSA_PAGE_SIZE=$TSA_PAGE_SIZE
export TSA_MAX_DECODE_TOKENS=$TSA_MAX_DECODE_TOKENS
export TSA_COLLECT_MS=$TSA_COLLECT_MS
bash $TSA_REMOTE_ROOT/start_server.sh
EOF
)
ensure_tmux "wa-tsa" "$REMOTE_CMD"

# 3. Tunnel hilbit2 → GPU_HOST:$TSA_PORT
open_tunnel "$TSA_PORT" "$TSA_PORT"

# 4. Wait until healthy.
echo "[tsa] waiting for /health on 127.0.0.1:$TSA_PORT ..."
wait_healthy "http://127.0.0.1:${TSA_PORT}/health" 900

# 5. Verify the running server actually advertises the served-model-name.
echo "[tsa] verifying model id ..."
SERVED=$(curl -s "http://127.0.0.1:${TSA_PORT}/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "[tsa] /v1/models reports id=$SERVED"
if [[ "$SERVED" != "$TSA_MODEL_NAME" ]]; then
    echo "[tsa] WARNING: server is advertising '$SERVED' but env expected '$TSA_MODEL_NAME'."
    echo "[tsa]          The orchestrator must pass --model '$SERVED' to match."
fi

echo ""
echo "[tsa] ready: http://127.0.0.1:${TSA_PORT}/v1/chat/completions"
echo ""
echo "# Export these in the orchestrator's shell (or just source this script):"
ENV_FILE="$SCRIPT_DIR/.inference_env"
{
    echo "export OPENAI_API_BASE=http://127.0.0.1:${TSA_PORT}/v1"
    echo "export OPENAI_API_KEY=tsa"
    echo "export WEBARENA_EVAL_API_BASE=http://127.0.0.1:${JUDGE_PORT}/v1"
    echo "export WEBARENA_EVAL_API_KEY=judge"
    echo "export WEBARENA_EVAL_MODEL=${JUDGE_MODEL_NAME}"
    echo "export WEBARENA_TOKENIZER_PATH=${WEBARENA_TOKENIZER_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
    echo "export INFERENCE_BACKEND=tsa"
    echo "export AGENT_MODEL_NAME=$SERVED"
} | tee "$ENV_FILE"

export OPENAI_API_BASE="http://127.0.0.1:${TSA_PORT}/v1"
export OPENAI_API_KEY="tsa"
export WEBARENA_EVAL_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1"
export WEBARENA_EVAL_API_KEY="judge"
export WEBARENA_EVAL_MODEL="${JUDGE_MODEL_NAME}"
export WEBARENA_TOKENIZER_PATH="${WEBARENA_TOKENIZER_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
export INFERENCE_BACKEND="tsa"
export AGENT_MODEL_NAME="$SERVED"
