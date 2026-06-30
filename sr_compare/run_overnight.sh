#!/bin/bash
# MASTER overnight orchestrator for sr_compare OSM block2 (real OpenStreetMap), 58 map tasks.
# Runs remaining method generations, then serves the judge and scores everything into
# task_outcomes/map_osm_block2/. Fully self-contained; safe to run detached (setsid nohup).
# All pkill patterns use the [x] bracket trick so they never match this script's own cmdline.
set -u
B=/home/cc/webarena/sr_compare
cd "$B"
source "$B/config.env"
export SGLANG_DISABLE_CUDNN_CHECK=1
R="$B/results"
OUT="$B/task_outcomes/map_osm_block2"
mkdir -p "$OUT"
IDS=$(cat "$B/osm_block2_ids.txt")
log(){ echo "[master $(date +%F_%T)] $*"; }

kill_methods(){ pkill -9 -f "[s]erve\.py --model-path" 2>/dev/null; pkill -9 -f "[l]aunch_vortex_textonly" 2>/dev/null; sleep 10; }
kill_judge(){ pkill -9 -f "[L]lama-3.3-70B-Instruct" 2>/dev/null; sleep 8; }
count(){ ls "$R/$1"/task_*/task_*.json 2>/dev/null | wc -l; }
wait_health(){ local port=$1 to=$2 t=0; while [ $t -lt $to ]; do
  [ "$(curl -s -m3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health 2>/dev/null)" = "200" ] && return 0
  sleep 12; t=$((t+12)); done; return 1; }
run_cfg(){ local cfg=$1 model=$2 tt=$3
  "$BENCH_PY" wa_exp/run_batch.py --out-dir "$R/$cfg" --base-urls "$LLM_URLS" --model "$model" \
    --only "$IDS" --max-concurrency 6 --num-proxies 8 --retries 2 --max-steps 30 \
    --task-timeout "$tt" --skip-done > "$R/$cfg.run.log" 2>&1; }

# clean slate: stop any serves left running (judge from validation, stale method serves)
kill_methods
kill_judge

# ensure internet-egress proxies are up (real OSM)
if [ "$(curl -s -m6 -x http://127.0.0.1:18900 -o /dev/null -w '%{http_code}' https://www.openstreetmap.org/ 2>/dev/null)" != "200" ]; then
  log "proxies down -> restarting tinyproxy 18900-18907"
  for p in $(seq 18900 18907); do tinyproxy -c "$B/../proxy/tinyproxy_${p}.conf" 2>/dev/null || tinyproxy -c /home/cc/proxy/tinyproxy_${p}.conf 2>/dev/null; done
  sleep 3
fi

############ 1. TSA configs ############
for tk in 128 64 32; do
  cfg="tsa_tk${tk}"
  if [ "$(count $cfg)" -ge 58 ]; then log "$cfg already complete ($(count $cfg)/58); skip"; continue; fi
  kill_methods
  log "serving $cfg (TOP_K=$tk)"
  TOP_K=$tk METRICS_DIR="$R/$cfg/metrics" SERVED_NAME=qwen3vl-tsa \
    TSA_PY="$TSA_PY" TSA_REPO="$TSA_REPO" MODEL_PATH="$MODEL_PATH" \
    bash serve/serve_tsa_4x.sh > "$R/serve_$cfg.log" 2>&1 &
  if wait_health 18005 900; then log "$cfg healthy; running 58 tasks"; run_cfg "$cfg" qwen3vl-tsa 3000; log "$cfg done ($(count $cfg)/58)"
  else log "!! $cfg serve NOT healthy; skipping"; tail -20 "$R/serve_$cfg.log"; fi
done

############ 2. vortex configs ############
# cfg module topk chunk ratio
for spec in "vortex_block:block_sparse_attention:30:32:0.0" "vortex_quest:gqa_quest_sparse_attention:61:64:0.0625"; do
  IFS=: read -r cfg module topk chunk ratio <<< "$spec"
  if [ "$(count $cfg)" -ge 58 ]; then log "$cfg already complete; skip"; continue; fi
  kill_methods
  log "serving $cfg (module=$module topk=$topk chunk=$chunk ratio=$ratio)"
  SGLANG_DISABLE_CUDNN_CHECK=1 VORTEX_MODULE=$module VORTEX_TOPK=$topk VORTEX_CHUNK=$chunk VORTEX_RATIO=$ratio \
    SERVED_NAME=qwen3vl-vortex METRICS_DIR="$R/$cfg/metrics" MODEL_PATH="$MODEL_PATH" VORTEX_PY="$VORTEX_PY" \
    bash serve/serve_vortex_v59_4x.sh > "$R/serve_$cfg.log" 2>&1 &
  if wait_health 18005 1200; then log "$cfg healthy; running 58 tasks"; run_cfg "$cfg" qwen3vl-vortex 1800; log "$cfg done ($(count $cfg)/58)"
  else log "!! $cfg serve NOT healthy; skipping"; tail -20 "$R/serve_$cfg.log"; fi
done

############ 3. judge + scoring ############
kill_methods
log "serving judge (Llama-3.3-70B tp2 :18000)"
bash serve/serve_judge.sh &
if wait_health 18000 900; then
  for cfg in dense tsa_tk128 tsa_tk64 tsa_tk32 vortex_block vortex_quest; do
    [ -f "$OUT/$cfg.json" ] && { log "$cfg outcome exists; skip scoring"; continue; }
    [ "$(count $cfg)" -lt 1 ] && { log "$cfg has no results; skip scoring"; continue; }
    log "scoring $cfg ($(count $cfg) tasks)"
    "$BENCH_PY" wa_exp/score_batch.py --results-dir "$R/$cfg" --judge-url "$JUDGE_URL" --judge-model llama-judge --concurrency 8 >> "$R/$cfg.score.log" 2>&1
    "$BENCH_PY" wa_exp/rescore_lenient.py "$R/$cfg" >> "$R/$cfg.score.log" 2>&1
    "$BENCH_PY" wa_exp/build_block2_outcome.py "$R/$cfg" "$cfg" "$OUT/$cfg.json" >> "$R/$cfg.score.log" 2>&1
    log "$cfg scored"
  done
else log "!! judge NOT healthy; cannot score"; fi

############ 4. summary ############
"$BENCH_PY" - "$OUT" <<'PY'
import json, os, sys, glob
out=sys.argv[1]
rows=[]
order=["dense","tsa_tk128","tsa_tk96","tsa_tk64","tsa_tk48","tsa_tk32","vortex_block","vortex_quest"]
files={os.path.basename(f)[:-5]:f for f in glob.glob(os.path.join(out,"*.json"))}
for cfg in order:
    if cfg not in files: continue
    d=json.load(open(files[cfg]))
    n=d["n_run"]; L=d["lenient"]["n_pass"]; O=d["official"]["n_pass"]; e=len(d["execution_errored_ids"])
    rows.append((cfg,n,L,O,e))
lines=["# OSM block-2 — 58 map tasks on REAL OpenStreetMap (Qwen3-VL-32B)","",
 "| method | n_run | lenient_pass | official_pass | exec_errors |","|---|---|---|---|---|"]
for cfg,n,L,O,e in rows:
    lines.append(f"| {cfg} | {n} | {L} ({round(100*L/n)}%) | {O} ({round(100*O/n)}%) | {e} |")
lines+=["","> Real-OSM caveat (see README): reference answers were annotated on the self-hosted OSM, so",
 "> `official` is low by construction; **lenient** is the primary success rate. n=58 (noise-prone)."]
open(os.path.join(out,"summary.md"),"w").write("\n".join(lines)+"\n")
print("wrote summary.md with",len(rows),"methods")
PY

kill_judge
log "ALL DONE — outcomes in $OUT"
