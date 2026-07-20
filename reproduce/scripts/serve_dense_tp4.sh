#!/bin/bash
# DENSE Qwen3-VL-32B via vLLM tp=4 on :8000 (proven config from wa_dense.sh).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../config.env"
HARNESS="$HERE/../harness"; SCORING="$HERE/../scoring"

set -u
LOGDIR=${LOGDIR:-$RESULTS_DIR/_serverlogs}; mkdir -p "$LOGDIR"
export PATH=$(dirname $SERVE_PY):$CUDA_BIN:$PATH
$SERVE_PY -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH --served-model-name qwen3vl-dense \
  --tensor-parallel-size 4 --port 8000 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":5}' --gpu-memory-utilization 0.90 --enforce-eager \
  > "$LOGDIR/dense_tp4.log" 2>&1
