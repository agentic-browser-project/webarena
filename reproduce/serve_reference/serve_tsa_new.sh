#!/bin/bash
# Single TSA (TreeSparseAttention) replica serving the model, WebArena mode. Identical config to
# the main experiment: page_size 64, max-decode-tokens 4096, max-batch-size 2, webarena parse mode.
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$(dirname "${TSA_PY:-/home/cc/venvs/tsa/bin/python}"):/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=${TSA_REPO:-/home/cc/temp/TreeSparseAttention}/build:$PYTHONPATH
cd "${TSA_REPO:-/home/cc/temp/TreeSparseAttention}"
"${TSA_PY:-/home/cc/venvs/tsa/bin/python}" serve.py \
  --model-path "${MODEL_PATH:-/home/cc/hf_models/Qwen3-VL-32B-Instruct}" \
  --served-model-name "${SERVED_NAME:-qwen3vl-tsa}" \
  --host 127.0.0.1 --port "${PORT:-18005}" \
  --top-k "${TOP_K:-128}" --page-size "${PAGE_SIZE:-64}" \
  --max-decode-tokens 4096 --max-batch-size "${MAX_BATCH:-2}" --batch-collect-ms 80 \
  --tree-parse-mode webarena --no-strip-fences
