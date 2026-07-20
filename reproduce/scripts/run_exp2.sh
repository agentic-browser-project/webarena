#!/bin/bash
# WebArena redo: 6 methods x 2 blocks (0-99, 100-199), text-only, temp 0, max_tokens 4096,
# raised timeouts, per-replica login (sr_compare harness), block-outer ordering.
# Methods: dense, tsa_centroid tk32/tk64, tsa_minmax tk32/tk64, quest(tk61).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../config.env"
HARNESS="$HERE/../harness"; SCORING="$HERE/../scoring"

set -u
SC=$HERE
WA=$HARNESS
OUT=$RESULTS_DIR
LOGDIR=$OUT/_serverlogs; mkdir -p "$OUT" "$LOGDIR"
AGENT=$AGENT_PY
export WA_MAP_URL=https://www.openstreetmap.org
export WA_FORCE_PROXY=socks5://127.0.0.1:1080
MASTER=$OUT/master.log
say(){ echo "$(date +%H:%M:%S) | $*" | tee -a "$MASTER"; }

kill_servers(){
  for pat in 'vllm.entrypoints' 'sglang.launch_server' 'serve.py'; do
    for p in $(ps -eo pid,cmd | grep -F "$pat" | grep -v grep | grep -v run_exp2 | awk '{print $1}'); do kill "$p" 2>/dev/null; done
  done
  sleep 15
}
wait_ready(){  # args: url...
  local tries=${WAIT_TRIES:-120}
  for i in $(seq 1 "$tries"); do
    local up=1
    for u in "$@"; do curl -sf -m 4 "$u/models" >/dev/null 2>&1 || up=0; done
    [ "$up" = 1 ] && return 0
    sleep 10
  done
  return 1
}
regen_auth(){
  say "regen auth cookies (fresh sessions) ..."
  AUTH_FORCE=1 AUTH_PROXY=socks5://127.0.0.1:1080 timeout 400 \
    "$AGENT" "$WA/gen_auth.py" --sites shopping shopping_admin reddit gitlab --replicas 10 \
    > "$LOGDIR/genauth_$(date +%H%M%S).log" 2>&1
  say "  auth done: $(grep -c 'OK cookies' "$LOGDIR"/genauth_*.log 2>/dev/null | tail -1) ok this round"
}

run_batch_for(){  # method_name start end model base_urls
  local m=$1 s=$2 e=$3 model=$4 urls=$5
  local od=$OUT/$m/b${s}_${e}
  mkdir -p "$od"
  # TSA is single-GPU (no tensor-parallel) => ~4x slower per request than dense tp=4.
  # Lower concurrency (less queue tail) + higher task-timeout so tasks complete instead
  # of hitting the cap. dense/quest (tp=4, fast) keep tight settings.
  local conc=12 tt=1500 tout=22000
  case "$m" in
    tsa_*) conc=12; tt=2400; tout=36000 ;;
  esac
  say "  run_batch $m tasks $s-$((e-1)) conc=$conc task_timeout=$tt -> $od"
  cd "$WA"
  timeout $tout "$AGENT" run_batch.py --start "$s" --end "$e" --out-dir "$od" \
     --base-urls "$urls" --model "$model" --max-concurrency $conc --num-proxies 1 \
     --max-tokens 4096 --task-timeout $tt --temperature 0 --retries 2 --skip-done \
     >> "$od/batch.log" 2>&1
  say "  DONE $m $s-$((e-1)): $(ls "$od"/task_*/task_*.json 2>/dev/null | wc -l) tasks"
}

start_method(){  # method -> sets global MODEL, URLS ; starts servers
  local m=$1
  kill_servers
  case "$m" in
    dense)
      MODEL=qwen3vl-dense; URLS=http://127.0.0.1:8000/v1
      nohup bash "$SC/serve_dense_tp4.sh" >/dev/null 2>&1 &
      WAIT_TRIES=120 wait_ready http://127.0.0.1:8000/v1 ;;
    tsa_centroid_tk32) SCORING=centroid TOPK=32 MAXB=8 start_tsa ;;
    tsa_centroid_tk64) SCORING=centroid TOPK=64 MAXB=8 start_tsa ;;
    tsa_minmax_tk32)   SCORING=minmax   TOPK=32 MAXB=4 start_tsa ;;
    tsa_minmax_tk64)   SCORING=minmax   TOPK=64 MAXB=4 start_tsa ;;
    quest)
      MODEL=qwen3-vl-32b; URLS=http://127.0.0.1:30000/v1
      QUEST_TP=4 QUEST_PORT=30000 nohup bash "$SC/serve_quest.sh" >"$LOGDIR/quest.log" 2>&1 &
      WAIT_TRIES=150 wait_ready http://127.0.0.1:30000/v1 ;;
  esac
}
start_tsa(){
  MODEL=qwen3vl-tsa; URLS=http://127.0.0.1:18005/v1,http://127.0.0.1:18006/v1,http://127.0.0.1:18007/v1,http://127.0.0.1:18008/v1
  LOGDIR="$LOGDIR" SCORING="$SCORING" TOPK="$TOPK" MAXB="$MAXB" nohup bash "$SC/serve_tsa4.sh" >/dev/null 2>&1 &
  WAIT_TRIES=120 wait_ready http://127.0.0.1:18005/v1 http://127.0.0.1:18006/v1 http://127.0.0.1:18007/v1 http://127.0.0.1:18008/v1
}

METHODS="dense tsa_centroid_tk64 tsa_minmax_tk64 quest"

run_block(){  # start end
  local s=$1 e=$2
  say "==================== BLOCK $s-$((e-1)) START ===================="
  for m in $METHODS; do
    local ndone=$(ls $OUT/$m/b${s}_${e}/task_*/task_*.json 2>/dev/null | wc -l)
    if [ "$ndone" -ge $((e-s)) ]; then say "-------- METHOD $m already complete ($ndone/$((e-s))), skipping --------"; continue; fi
    say "-------- METHOD $m  (block $s-$((e-1))) --------"
    regen_auth
    if ! start_method "$m"; then say "  ERROR: $m servers not ready, skipping"; kill_servers; continue; fi
    say "  servers ready for $m (model=$MODEL)"
    run_batch_for "$m" "$s" "$e" "$MODEL" "$URLS"
    kill_servers
  done
  say "==================== BLOCK $s-$((e-1)) DONE ===================="
  touch "$OUT/BLOCK_${s}_${e}_COMPLETE"
}

run_block 0 100
touch "$OUT/EXP2_AGENT_COMPLETE"
say "ALL AGENT RUNS COMPLETE"
