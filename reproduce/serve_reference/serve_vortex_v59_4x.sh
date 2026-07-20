#!/bin/bash
# 4 single-GPU vortex (sglang v0.5.9) replicas serving the model TEXT-ONLY with vortex sparse
# attention + cuda graph (native) + deterministic + xgrammar. Ports 18005-18008, GPUs 0-3.
# Text-only via launch_vortex_textonly.py wrapper (enable_multimodal=False; matches use_vision=False).
# Env: VORTEX_TOPK, VORTEX_MODULE, VORTEX_CHUNK, VORTEX_RATIO, SERVED_NAME, METRICS_DIR, MODEL_PATH.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
VORTEX_TOPK=${VORTEX_TOPK:-30}
VORTEX_MODULE=${VORTEX_MODULE:-block_sparse_attention}
VORTEX_CHUNK=${VORTEX_CHUNK:-32}
VORTEX_RATIO=${VORTEX_RATIO:-0.0}
SERVED_NAME=${SERVED_NAME:-qwen3vl-vortex}
MODEL_PATH=${MODEL_PATH:-/home/cc/hf_models/Qwen3-VL-32B-Instruct}
VORTEX_PY=${VORTEX_PY:-/home/cc/venvs/vortex59/bin/python}
METRICS_DIR=${METRICS_DIR:-}            # if set, each instance logs ACTUAL selected/total blocks here
[ -n "$METRICS_DIR" ] && mkdir -p "$METRICS_DIR"
WRAP=$HERE/../wa_exp/launch_vortex_textonly.py
# MUST NOT run from a dir that has a `vortex_torch/` sibling (namespace-shadow) -> cd /tmp.
cd /tmp || exit 1
for g in 0 1 2 3; do
  port=$((18005+g))
  SGLANG_ENABLE_JIT_DEEPGEMM=0 CUDA_VISIBLE_DEVICES=$g \
  VORTEX_RT_METRICS="${METRICS_DIR:+$METRICS_DIR/sparse_metrics_rt_g${g}.jsonl}" \
  "$VORTEX_PY" "$WRAP" \
    --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port $port \
    --page-size 16 --attention-backend flashinfer \
    --enable-vortex-sparsity --vortex-topk-val "$VORTEX_TOPK" \
    --vortex-module-name "$VORTEX_MODULE" --vortex-layers-skip 0 \
    --vortex-workload-chunk-size "$VORTEX_CHUNK" --vortex-topk-ratio "$VORTEX_RATIO" \
    --context-length 65536 --vortex-max-seq-lens 65536 \
    --enable-deterministic-inference \
    --grammar-backend xgrammar --mem-fraction-static 0.85 --trust-remote-code \
    > "$HERE/../serve_vortex_${port}.log" 2>&1 &
done
wait
