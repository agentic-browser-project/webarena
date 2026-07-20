#!/bin/bash
# fuzzy_match LLM judge: Llama-3.3-70B (tp 2) on port 18000.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH_PY=${BENCH_PY:-/home/cc/venvs/bench/bin/python}
JUDGE_MODEL=${JUDGE_MODEL:-/home/cc/hf_models/Llama-3.3-70B-Instruct}
export PATH=$(dirname "$BENCH_PY"):/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
CUDA_VISIBLE_DEVICES=0,1 "$BENCH_PY" -m sglang.launch_server \
  --model-path "$JUDGE_MODEL" --served-model-name llama-judge --tp 2 \
  --host 127.0.0.1 --port 18000 --attention-backend triton --disable-radix-cache \
  --disable-cuda-graph --context-length 8192 --mem-fraction-static 0.88 --grammar-backend xgrammar \
  > "$HERE/../serve_judge.log" 2>&1
