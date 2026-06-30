#!/bin/bash
# Post-reboot auto-completion for vortex_block + vortex_quest, then score + finalize summary (6/6).
# Driver is now 570 (CUDA 12.8); a clean boot is needed for NCCL to work. This script:
#   1) self-removes its @reboot cron (no loops),
#   2) NCCL-gates: if NCCL still broken, logs and exits (no garbage runs),
#   3) restarts tinyproxy (lost on reboot) for OSM egress,
#   4) runs both vortex methods (crash-resilient), scores, regenerates summary.
set -u
B=/home/cc/webarena/sr_compare
LOG=/home/cc/finish_vortex.log
exec >> "$LOG" 2>&1
echo "==== finish_vortex start $(date) ===="
# 1) remove our @reboot cron so this never auto-loops
( crontab -l 2>/dev/null | grep -v 'finish_vortex.sh' ) | crontab - 2>/dev/null || true

cd "$B"; source "$B/config.env"
export NCCL_CUMEM_ENABLE=0   # belt-and-suspenders; harmless on 570
R="$B/results"; OUT="$B/task_outcomes/map_osm_block2"; IDS=$(cat "$B/osm_block2_ids.txt")
log(){ echo "[finish $(date +%F_%T)] $*"; }
kill_methods(){ pkill -9 -f "[l]aunch_vortex_textonly" 2>/dev/null; sleep 8; }
kill_judge(){ pkill -9 -f "[L]lama-3.3-70B-Instruct" 2>/dev/null; sleep 6; }
count(){ ls "$R/$1"/task_*/task_*.json 2>/dev/null | wc -l; }
wait_health(){ local port=$1 to=$2 t=0; while [ $t -lt $to ]; do
  [ "$(curl -s -m3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health 2>/dev/null)" = "200" ] && return 0
  sleep 12; t=$((t+12)); done; return 1; }

# 2) NCCL gate
log "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader|head -1)"
if ! NCCL_DEBUG=WARN "$VORTEX_PY" -c "
import os,torch,torch.distributed as dist
os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT='29611',RANK='0',WORLD_SIZE='1')
dist.init_process_group('nccl',rank=0,world_size=1)
t=torch.ones(64,device='cuda');dist.all_reduce(t);torch.cuda.synchronize()
assert int(t.sum())==64; dist.destroy_process_group(); print('NCCL_OK')
" 2>&1 | grep -q NCCL_OK; then
  log "NCCL STILL BROKEN after boot -> not running vortex. Investigate driver/NCCL."
  exit 0
fi
log "NCCL OK on driver 570 -- proceeding with vortex"

# 3) restart tinyproxy (lost on reboot) for real-OSM egress
if [ "$(curl -s -m6 -x http://127.0.0.1:18900 -o /dev/null -w '%{http_code}' https://www.openstreetmap.org/ 2>/dev/null)" != "200" ]; then
  log "restarting tinyproxy 18900-18907"
  for p in $(seq 18900 18907); do tinyproxy -c /home/cc/proxy/tinyproxy_${p}.conf 2>/dev/null; done; sleep 3
fi

# 4) run both vortex methods (crash-resilient, faithful deterministic config)
run_vx(){ cur=$1; local module=$2 topk=$3 chunk=$4 ratio=$5 attempt=0
  rm -rf "$R/$cur"/task_* "$R/$cur"/SCORES*.json "$OUT/$cur.json"   # clear prior broken/empty
  while [ "$(count $cur)" -lt 58 ] && [ $attempt -lt 4 ]; do
    attempt=$((attempt+1)); kill_methods
    log "$cur attempt $attempt: serving"
    VORTEX_MODULE=$module VORTEX_TOPK=$topk VORTEX_CHUNK=$chunk VORTEX_RATIO=$ratio \
      SERVED_NAME=qwen3vl-vortex METRICS_DIR="$R/$cur/metrics" MODEL_PATH="$MODEL_PATH" VORTEX_PY="$VORTEX_PY" \
      bash serve/serve_vortex_v59_4x.sh > "$R/serve_$cur.log" 2>&1 &
    if ! wait_health 18005 1200; then log "$cur serve not healthy (attempt $attempt)"; tail -10 "$R/serve_$cur.log"; continue; fi
    log "$cur healthy; running 58 tasks"
    "$BENCH_PY" wa_exp/run_batch.py --out-dir "$R/$cur" --base-urls "$LLM_URLS" --model qwen3vl-vortex \
      --only "$IDS" --max-concurrency 6 --num-proxies 8 --retries 2 --max-steps 30 \
      --task-timeout 1800 --skip-done >> "$R/$cur.run.log" 2>&1
    log "$cur attempt $attempt: $(count $cur)/58"
  done
}
run_vx vortex_block  block_sparse_attention      30 32 0.0
run_vx vortex_quest  gqa_quest_sparse_attention  61 64 0.0625

# 5) judge + score, validate non-empty (guard against the empty-answer trap)
kill_methods
log "serving judge"; bash serve/serve_judge.sh &
wait_health 18000 900
for cfg in vortex_block vortex_quest; do
  real=$("$BENCH_PY" -c "import glob,json,os,sys
n=0
for d in glob.glob('$R/$cfg/task_*'):
  t=os.path.basename(d).split('_')[1]; p=f'{d}/llm_calls.jsonl'
  if os.path.exists(p) and sum(1 for _ in open(p))>0: n+=1
print(n)")
  log "$cfg real-llm-call tasks: $real/58"
  [ "$(count $cfg)" -lt 1 ] && { log "$cfg no results; skip"; continue; }
  "$BENCH_PY" wa_exp/score_batch.py --results-dir "$R/$cfg" --judge-url "$JUDGE_URL" --judge-model llama-judge --concurrency 8 >> "$R/$cfg.score.log" 2>&1
  "$BENCH_PY" wa_exp/rescore_lenient.py "$R/$cfg" >> "$R/$cfg.score.log" 2>&1
  "$BENCH_PY" wa_exp/build_block2_outcome.py "$R/$cfg" "$cfg" "$OUT/$cfg.json" >> "$R/$cfg.score.log" 2>&1
  log "$cfg scored"
done

# 6) regenerate summary over all outcomes present
"$BENCH_PY" - "$OUT" <<'PY'
import json, os, sys, glob
out=sys.argv[1]; rows=[]
order=["dense","tsa_tk128","tsa_tk96","tsa_tk64","tsa_tk48","tsa_tk32","vortex_block","vortex_quest"]
files={os.path.basename(f)[:-5]:f for f in glob.glob(os.path.join(out,"*.json"))}
for cfg in order:
    if cfg not in files: continue
    d=json.load(open(files[cfg])); n=d["n_run"]
    rows.append((cfg,n,d["lenient"]["n_pass"],d["official"]["n_pass"],len(d["execution_errored_ids"])))
L=["# OSM block-2 — 58 map tasks on REAL OpenStreetMap (Qwen3-VL-32B)","",
 "| method | n_run | lenient_pass | official_pass | exec_errors |","|---|---|---|---|---|"]
for cfg,n,l,o,e in rows: L.append(f"| {cfg} | {n} | {l} ({round(100*l/n)}%) | {o} ({round(100*o/n)}%) | {e} |")
L+=["","> lenient = primary (real-OSM refs mismatch self-hosted annotations -> official low by construction). n=58.",
 "> vortex ran after GPU driver upgrade 560->570 (CUDA 12.8) to satisfy its cu128/NCCL stack."]
open(os.path.join(out,"summary.md"),"w").write("\n".join(L)+"\n")
print("summary.md updated:",len(rows),"methods")
PY
kill_judge
kill_methods
log "==== finish_vortex DONE $(date) -- see $OUT/summary.md ===="
