#!/bin/bash
# ============================================================================
# Reproduce TABLE 2 (block1): the 51 single-site `map` tasks with id<100, run
# with the WebArena `map` site pointed at the REAL https://www.openstreetmap.org,
# across all 6 attention configs (dense + tsa_tk128/64/32 + vortex_block/quest).
#
# It serves each config in turn, runs the 51 tasks through the browser-use agent,
# then brings up the Llama-3.3-70B judge and scores everything into
#   task_outcomes/map_osm_block1_repro/{<cfg>.json, summary.md}
# (per-method passed/failed/execution-errored task-id lists, lenient + official).
#
# Prereqs (see README "0. Clone & build the dependency repos" + "Setup"):
#   - config.env points BENCH_PY / TSA_PY / VORTEX_PY / MODEL_PATH / JUDGE_MODEL
#     / TSA_REPO at your boxes; the 3 venvs are built; models downloaded.
#   - tinyproxy (or any HTTP proxy WITH public-internet egress) on 18900-18907,
#     because real OSM is reached through run_task.py's proxy. This script will
#     (re)start tinyproxy from /home/cc/proxy/tinyproxy_*.conf if those exist.
#   - GPU driver new enough for the framework CUDA build (vortex uses cu128 ->
#     needs driver >= ~570 / CUDA 12.8; on older drivers vortex's NCCL init fails
#     and this script logs + skips those two configs, still finishing dense+TSA).
#
# Be gentle on real OSM (Nominatim policy <= 1 req/s): max-concurrency is low.
# Usage:  bash run_block1.sh        (logs -> results_block1/, run_block1.log)
# ============================================================================
set -u
B=/home/cc/webarena/sr_compare
cd "$B"
source "$B/config.env"
# block1 = map on REAL OpenStreetMap (independent of whatever config.env has set):
export WA_MAP_URL=https://www.openstreetmap.org
export SGLANG_DISABLE_CUDNN_CHECK=1

IDS=$(cat "$B/osm_block1_ids.txt")
R="$B/results_block1"
OUT="$B/task_outcomes/map_osm_block1_repro"
mkdir -p "$R" "$OUT"
log(){ echo "[block1 $(date +%F_%T)] $*"; }

# pkill patterns use the [x] bracket trick so they never match this script's own cmdline
kill_methods(){ pkill -9 -f "[s]erve\.py --model-path" 2>/dev/null; pkill -9 -f "[l]aunch_vortex_textonly" 2>/dev/null; pkill -9 -f "[s]glang\.launch_server.*Qwen3-VL" 2>/dev/null; sleep 10; }
kill_judge(){ pkill -9 -f "[L]lama-3.3-70B-Instruct" 2>/dev/null; sleep 8; }
count(){ ls "$R/$1"/task_*/task_*.json 2>/dev/null | wc -l; }
wait_health(){ local port=$1 to=$2 t=0; while [ $t -lt $to ]; do
  [ "$(curl -s -m3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health 2>/dev/null)" = "200" ] && return 0
  sleep 12; t=$((t+12)); done; return 1; }
N=$(echo "$IDS" | tr ',' '\n' | wc -l)
run_batch_cfg(){ local cfg=$1 model=$2 tt=$3
  "$BENCH_PY" wa_exp/run_batch.py --out-dir "$R/$cfg" --base-urls "$LLM_URLS" --model "$model" \
    --only "$IDS" --max-concurrency 6 --num-proxies 8 --retries 2 --max-steps 30 \
    --task-timeout "$tt" --skip-done > "$R/$cfg.run.log" 2>&1; }

# ensure internet-egress proxy is up (real OSM)
if [ "$(curl -s -m6 -x http://127.0.0.1:18900 -o /dev/null -w '%{http_code}' https://www.openstreetmap.org/ 2>/dev/null)" != "200" ]; then
  log "(re)starting tinyproxy 18900-18907 for OSM egress"
  for p in $(seq 18900 18907); do tinyproxy -c /home/cc/proxy/tinyproxy_${p}.conf 2>/dev/null; done; sleep 3
fi

############ 1. dense ############
if [ "$(count dense)" -lt "$N" ]; then
  kill_methods; log "serving dense"
  SERVED_NAME=qwen3vl-dense MODEL_PATH="$MODEL_PATH" BENCH_PY="$BENCH_PY" bash serve/serve_dense_4x.sh > "$R/serve_dense.log" 2>&1 &
  if wait_health 18005 900; then log "dense healthy; running $N tasks"; run_batch_cfg dense qwen3vl-dense 1800; log "dense done ($(count dense)/$N)"
  else log "!! dense serve not healthy; skipping"; tail -20 "$R/serve_dense.log"; fi
fi

############ 2. TSA (top_k 128/64/32) ############
for tk in 128 64 32; do
  cfg="tsa_tk${tk}"
  [ "$(count $cfg)" -ge "$N" ] && { log "$cfg already complete; skip"; continue; }
  kill_methods; log "serving $cfg (TOP_K=$tk)"
  TOP_K=$tk METRICS_DIR="$R/$cfg/metrics" SERVED_NAME=qwen3vl-tsa \
    TSA_PY="$TSA_PY" TSA_REPO="$TSA_REPO" MODEL_PATH="$MODEL_PATH" \
    bash serve/serve_tsa_4x.sh > "$R/serve_$cfg.log" 2>&1 &
  if wait_health 18005 900; then log "$cfg healthy; running $N tasks"; run_batch_cfg "$cfg" qwen3vl-tsa 3000; log "$cfg done ($(count $cfg)/$N)"
  else log "!! $cfg serve not healthy; skipping"; tail -20 "$R/serve_$cfg.log"; fi
done

############ 3. vortex (block + quest) ############
# vortex needs a CUDA-12.8 GPU driver; NCCL-gate so we don't run garbage on older drivers.
nccl_ok(){ "$VORTEX_PY" -c "
import os,torch,torch.distributed as dist
os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT='29711',RANK='0',WORLD_SIZE='1')
dist.init_process_group('nccl',rank=0,world_size=1)
t=torch.ones(64,device='cuda');dist.all_reduce(t);torch.cuda.synchronize()
assert int(t.sum())==64; dist.destroy_process_group(); print('NCCL_OK')
" 2>/dev/null | grep -q NCCL_OK; }
if nccl_ok; then
  for spec in "vortex_block:block_sparse_attention:30:32:0.0" "vortex_quest:gqa_quest_sparse_attention:61:64:0.0625"; do
    IFS=: read -r cfg module topk chunk ratio <<< "$spec"
    [ "$(count $cfg)" -ge "$N" ] && { log "$cfg already complete; skip"; continue; }
    kill_methods; log "serving $cfg (module=$module topk=$topk)"
    VORTEX_MODULE=$module VORTEX_TOPK=$topk VORTEX_CHUNK=$chunk VORTEX_RATIO=$ratio \
      SERVED_NAME=qwen3vl-vortex METRICS_DIR="$R/$cfg/metrics" MODEL_PATH="$MODEL_PATH" VORTEX_PY="$VORTEX_PY" \
      bash serve/serve_vortex_v59_4x.sh > "$R/serve_$cfg.log" 2>&1 &
    if wait_health 18005 1200; then log "$cfg healthy; running $N tasks"; run_batch_cfg "$cfg" qwen3vl-vortex 1800; log "$cfg done ($(count $cfg)/$N)"
    else log "!! $cfg serve not healthy; skipping"; tail -20 "$R/serve_$cfg.log"; fi
  done
else
  log "!! NCCL all-reduce fails on this box (vortex's cu128 stack needs a CUDA-12.8 driver, >= ~570). Skipping vortex_block/quest; dense+TSA still produced."
fi

############ 4. judge + scoring ############
kill_methods
log "serving judge (Llama-3.3-70B :18000)"; bash serve/serve_judge.sh &
if wait_health 18000 900; then
  for cfg in dense tsa_tk128 tsa_tk64 tsa_tk32 vortex_block vortex_quest; do
    [ "$(count $cfg)" -lt 1 ] && { log "$cfg no results; skip scoring"; continue; }
    log "scoring $cfg ($(count $cfg) tasks)"
    "$BENCH_PY" wa_exp/score_batch.py --results-dir "$R/$cfg" --judge-url "$JUDGE_URL" --judge-model llama-judge --concurrency 8 >> "$R/$cfg.score.log" 2>&1
    "$BENCH_PY" wa_exp/rescore_lenient.py "$R/$cfg" >> "$R/$cfg.score.log" 2>&1
    "$BENCH_PY" wa_exp/build_block2_outcome.py "$R/$cfg" "$cfg" "$OUT/$cfg.json" >> "$R/$cfg.score.log" 2>&1
    log "$cfg scored"
  done
else log "!! judge not healthy; cannot score"; fi
kill_judge

############ 5. summary table ############
"$BENCH_PY" - "$OUT" <<'PY'
import json, os, sys, glob
out=sys.argv[1]; rows=[]
order=["dense","tsa_tk128","tsa_tk64","tsa_tk32","vortex_block","vortex_quest"]
files={os.path.basename(f)[:-5]:f for f in glob.glob(os.path.join(out,"*.json"))}
for cfg in order:
    if cfg not in files: continue
    d=json.load(open(files[cfg])); n=d["n_run"]
    rows.append((cfg,n,d["lenient"]["n_pass"],d["official"]["n_pass"],len(d["execution_errored_ids"])))
L=["# block1 (reproduction) — 51 map tasks (id<100) on REAL OpenStreetMap (Qwen3-VL-32B)","",
   "| method | n_run | lenient_pass | official_pass | exec_errors |","|---|---|---|---|---|"]
for cfg,n,l,o,e in rows: L.append(f"| {cfg} | {n} | {l} ({round(100*l/n)}%) | {o} ({round(100*o/n)}%) | {e} |")
L+=["","> map is the only site, so per-site = total. lenient = primary metric (real-OSM refs were",
    "> annotated on the self-hosted snapshot, so official is low by construction). n=51 (noise-prone)."]
open(os.path.join(out,"summary.md"),"w").write("\n".join(L)+"\n")
print("summary.md written:",len(rows),"methods ->",out)
PY
log "ALL DONE -> $OUT (per-method task-id JSONs + summary.md)"
