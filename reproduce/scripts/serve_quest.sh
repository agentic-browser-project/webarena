#!/bin/bash
# Quest sparse-attention OpenAI server (sglang 0.5.9 + vortex) for Qwen3-VL-32B.
# Params from algo_quest_4b.sh (gqa_quest, topk 61, block 16, chunk 64, triton).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../config.env"
HARNESS="$HERE/../harness"; SCORING="$HERE/../scoring"

export OPENAI_API_KEY="None"
export SGLANG_DISABLE_CUDNN_CHECK=1
export PATH=$(dirname $QUEST_PY):$CUDA_BIN:$PATH
cd $VORTEX_REPO
exec $QUEST_PY -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --served-model-name qwen3-vl-32b \
  --page-size 16 \
  --disable-overlap-schedule \
  --attention-backend flashinfer \
  --vortex-layers-skip 0 \
  --vortex-block-reserved-bos 1 \
  --vortex-block-reserved-eos 2 \
  --vortex-topk-val 61 \
  --vortex-block-size 16 \
  --vortex-workload-chunk-size 64 \
  --vortex-topk-ratio 0.0625 \
  --vortex-module-path $VORTEX_REPO/submissions/_flow_algorithms_test/gqa_quest_sparse_attention.py \
  --vortex-module-name gqa_quest_sparse_attention_sub \
  --vortex-attention-backend "${VORTEX_ATTN_BACKEND:-flashinfer}" \
  --vortex-max-seq-lens 32768 \
  --context-length 32768 \
  --mem-fraction-static 0.90 \
  --tp-size "${QUEST_TP:-4}" \
  --port "${QUEST_PORT:-30000}" --host 127.0.0.1 \
  --enable-vortex-sparsity
