#!/bin/bash
# Measure TSA real selected-token coverage (full-model eval_recall). Needs ninja on PATH + expandable_segments.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../config.env"
HARNESS="$HERE/../harness"; SCORING="$HERE/../scoring"

set -u
cd $TSA_REPO
export PATH=$(dirname $SERVE_PY):$CUDA_BIN:$PATH
export PYTHONPATH=$TSA_REPO/python:$TSA_REPO/models:$TSA_REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TS_CUDA_ARCHS=90
M=$MODEL_PATH
R=$RESULTS_DIR
run(){ # gpu name topk
  CUDA_VISIBLE_DEVICES=$1 $SERVE_PY eval_recall.py \
    --data-dir $R/_traj/$2 --model-path $M --max-requests 100 \
    --num-decode-tokens 4 --tree-parse-mode webarena --top-k $3 \
    --output $R/_traj/$2_cov.json > $R/_traj/$2_cov.log 2>&1
}
run 0 tsa_centroid_tk32 32 &
run 1 tsa_centroid_tk64 64 &
run 2 tsa_minmax_tk64  64 &
wait
touch $R/_traj/COV_DONE
