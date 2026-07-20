#!/bin/bash
# Re-score all 0-99 method-blocks with a WORKING Llama-3.3-70B judge.
# Fix vs the failed run: put serve-venv/bin (which has `ninja`) on PATH so the
# vLLM workers can JIT-compile; without it the engine dies and every fuzzy_match
# scoring call gets "Connection refused" (score=None).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../config.env"
HARNESS="$HERE/../harness"; SCORING="$HERE/../scoring"

set -u
OUT=$RESULTS_DIR
WA="$SCORING"
LOGDIR=$OUT/_serverlogs
AGENT=$AGENT_PY
export WA_MAP_URL=https://www.openstreetmap.org
export PATH=$(dirname $SERVE_PY):$CUDA_BIN:$PATH   # <-- the fix (ninja)
export CUDA_HOME=/usr/local/cuda-13.0
say(){ echo "$(date +%H:%M:%S) | RESCORE | $*" | tee -a $OUT/rescore.log; }

# free GPUs
for p in $(ps -eo pid,cmd | grep -E 'vllm|sglang|serve.py' | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 8

say "starting Llama-3.3-70B judge (tp=4, ninja on PATH)"
nohup $SERVE_PY -m vllm.entrypoints.openai.api_server \
  --model $JUDGE_MODEL_PATH --served-model-name llama-judge \
  --tensor-parallel-size 4 --port 18000 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enforce-eager > $LOGDIR/judge2.log 2>&1 &
ok=0
for i in $(seq 1 180); do curl -sf -m4 http://127.0.0.1:18000/v1/models >/dev/null 2>&1 && { ok=1; break; }; sleep 5; done
[ "$ok" = 1 ] || { say "JUDGE FAILED TO START - abort"; exit 1; }
say "judge ready"
# sanity: judge actually answers
say "judge sanity: $(curl -s -m60 http://127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"llama-judge","messages":[{"role":"user","content":"Reply with the single word: correct"}],"max_tokens":5,"temperature":0}' \
  | head -c 200)"

cd "$WA"
for m in dense tsa_centroid_tk32 tsa_centroid_tk64 tsa_minmax_tk64 quest; do
  D=$OUT/$m/b0_100
  [ -d "$D" ] || continue
  rm -f "$D/SCORES.json" "$D/SCORES_adjusted.json"
  say "official scoring $m ..."
  SCORE_PROXY=socks5://127.0.0.1:1080 JUDGE_BASE_URL=http://127.0.0.1:18000/v1 JUDGE_MODEL=llama-judge \
    timeout 6000 $AGENT score_batch.py --results-dir "$D" \
      --judge-url http://127.0.0.1:18000/v1 --judge-model llama-judge --concurrency 8 \
      > "$D/score2.log" 2>&1
  say "lenient scoring $m ..."
  timeout 4000 $AGENT rescore_lenient.py "$D" > "$D/score_lenient2.log" 2>&1
  say "  $m => $($UTIL_PY -c "
import json
s=json.load(open('$D/SCORES.json')); a=json.load(open('$D/SCORES_adjusted.json'))
none=sum(1 for t in s['tasks'] if t.get('score') is None)
print(f\"official={s['n_pass']}/{s['n']} lenient={a['adjusted_pass']}/{a['n']} unscored={none}\")" 2>/dev/null)"
done
for p in $(ps -eo pid,cmd | grep vllm | grep -v grep | awk '{print $1}'); do kill "$p" 2>/dev/null; done
touch $OUT/RESCORE_COMPLETE
say "RESCORE COMPLETE"
