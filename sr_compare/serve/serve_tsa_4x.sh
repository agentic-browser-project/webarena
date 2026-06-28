#!/bin/bash
# 4 single-GPU TSA replicas (ports 18005-18008, GPUs 0-3). Each writes its own sparse-metrics jsonl.
# Env: TOP_K (default 128), METRICS_DIR (required), SERVED_NAME, MAX_BATCH (default 2), MODEL_PATH.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TOP_K=${TOP_K:-128}
METRICS_DIR=${METRICS_DIR:-$HERE/../results/tsa_tmp}
SERVED_NAME=${SERVED_NAME:-qwen3vl-tsa}
MAX_BATCH=${MAX_BATCH:-2}
mkdir -p "$METRICS_DIR"
for g in 0 1 2 3; do
  port=$((18005+g))
  CUDA_VISIBLE_DEVICES=$g PORT=$port TOP_K=$TOP_K MAX_BATCH=$MAX_BATCH \
    SERVED_NAME="$SERVED_NAME" \
    TSA_METRICS_FILE="$METRICS_DIR/sparse_metrics_g${g}.jsonl" \
    bash "$HERE/serve_tsa_new.sh" > "$HERE/../serve_tsa_${port}.log" 2>&1 &
done
wait
