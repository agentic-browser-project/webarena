#!/bin/bash
# 4 single-GPU DENSE (full-attention) replicas serving the model, ports 18005-18008, GPUs 0-3.
# This is the dense baseline (same model, no sparsity). Matches the main experiment's dense config.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH_PY=${BENCH_PY:-/home/cc/venvs/bench/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/cc/hf_models/Qwen3-VL-32B-Instruct}
SERVED_NAME=${SERVED_NAME:-qwen3vl-dense}
export PATH=$(dirname "$BENCH_PY"):/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
for g in 0 1 2 3; do
  port=$((18005+g))
  CUDA_VISIBLE_DEVICES=$g "$BENCH_PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port $port --attention-backend triton --disable-radix-cache \
    --context-length 24576 --mem-fraction-static 0.85 --grammar-backend xgrammar \
    > "$HERE/../serve_dense_${port}.log" 2>&1 &
done
wait
