#!/bin/bash
# 4x single-GPU TSA (our MERGED repo: GPU envelope + per-head-max scoring), ports 18005-8.
# Env: SCORING (centroid|minmax), TOPK, MAXB (max-batch-size).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../config.env"
HARNESS="$HERE/../harness"; SCORING="$HERE/../scoring"

set -u
LOGDIR=${LOGDIR:-$RESULTS_DIR/_serverlogs}; mkdir -p "$LOGDIR"
SCORING=${SCORING:-centroid}; TOPK=${TOPK:-64}; MAXB=${MAXB:-8}
export TS_CUDA_ARCHS=90
export PATH=$(dirname $SERVE_PY):$CUDA_BIN:$PATH
cd $TSA_REPO
for g in 0 1 2 3; do
  port=$((18005+g))
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $SERVE_PY serve.py \
      --model-path $MODEL_PATH --served-model-name qwen3vl-tsa \
      --host 127.0.0.1 --port $port --top-k "$TOPK" --page-size 64 \
      --scoring-mode "$SCORING" --max-decode-tokens 4096 --max-batch-size "$MAXB" \
      --batch-collect-ms 80 --tree-parse-mode webarena --no-strip-fences \
      > "$LOGDIR/tsa_${SCORING}_tk${TOPK}_${port}.log" 2>&1 &
done
wait
