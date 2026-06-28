#!/bin/bash
# sr_compare staged run — reproduces the FULL experiment (map on REAL openstreetmap.org):
#   dense (Qwen3-VL-32B full attn) | TSA top_k sweep 128/96/64/48/32 | vortex-block | vortex-quest
# Per block (100,200,400,600,812) run all configs, then score cumulatively + report.
#   TSA sweep:   top_k = 128,96,64,48,32             (dirs tsa_tk128..tsa_tk32)
#   vortex-block (official): topk=30, chunk=32, ratio=0      (dir: vortex)
#   vortex-quest (official): topk=61, chunk=64, ratio=0.0625 (dir: quest_rec)
#   dense:       full attention                       (dir: dense)
# Self-healing: clean_failed deletes error/not-done task dirs so they re-run.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.env"
LOG=$HERE/sr_run.log; exec >>"$LOG" 2>&1
echo "################ sr_compare STAGED RUN START $(date) ################"
BP=$BENCH_PY
RES=$WA_RESULTS
URLS=$LLM_URLS
PYSLEEP(){ $BP -c "import time,sys;time.sleep(float(sys.argv[1]))" "$1"; }

kill_all(){ pkill -9 -f 'sglang.[l]aunch_server' 2>/dev/null; pkill -9 -f '[s]erve.py' 2>/dev/null; pkill -9 -f '[l]aunch_vortex_textonly' 2>/dev/null; PYSLEEP 6
  for i in $(seq 1 30); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s}'); [ "${u:-0}" -lt 5000 ] && break; PYSLEEP 4; done; }
wait_h(){ local t=0 code; while :; do code=$(curl -s -m5 "$1" -o /dev/null -w '%{http_code}' 2>/dev/null); [ "$code" = "200" ] && return 0; PYSLEEP 8; t=$((t+8)); [ $t -ge ${2:-1800} ] && return 1; done; }
wait_all(){ local ok=1; for p in 18005 18006 18007 18008; do wait_h http://127.0.0.1:$p/v1/models 1800 || ok=0; done; return $((1-ok)); }

clean_failed(){  # $1=cfg $2=R0 $3=R1 : delete infra-failed task dirs (error or not-done) so they re-run
  WA_RESULTS=$RES $BP -c "
import json,glob,os,shutil,sys
res=os.environ['WA_RESULTS']
d,r0,r1=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); k=0
for f in glob.glob(f'{res}/{d}/task_*/task_*.json'):
    try: tid=int(f.split('/task_')[1].split('/')[0])
    except: continue
    if tid<r0 or tid>=r1: continue
    try: rr=json.load(open(f))
    except: rr={'error':'corrupt'}
    if rr.get('error') or not rr.get('is_done'):
        shutil.rmtree(os.path.dirname(f),ignore_errors=True); k+=1
print(f'[clean_failed] {d} {r0}-{r1}: removed {k} infra-failed tasks',flush=True)
" "$1" "$2" "$3"; }

run_range(){  # $1=cfg $2=model $3=R0 $4=R1
  clean_failed "$1" "$3" "$4"
  $BP "$HERE/wa_exp/run_batch.py" --out-dir $RES/$1 --base-urls "$URLS" --model "$2" \
    --start $3 --end $4 --max-concurrency 16 --num-proxies 8 --retries 2 --max-steps 30 \
    --task-timeout 3000 --skip-done; }

serve_dense(){       SERVED_NAME=qwen3vl-dense  nohup bash "$HERE/serve/serve_dense_4x.sh" >/dev/null 2>&1 & }
serve_tsa(){ TOP_K=$1 METRICS_DIR=$RES/$2 SERVED_NAME=qwen3vl-tsa nohup bash "$HERE/serve/serve_tsa_4x.sh" >/dev/null 2>&1 & }
serve_vortex_block(){ METRICS_DIR=$RES/vortex    VORTEX_MODULE=block_sparse_attention   VORTEX_TOPK=30 VORTEX_CHUNK=32 VORTEX_RATIO=0.0    SERVED_NAME=qwen3vl-vortex nohup bash "$HERE/serve/serve_vortex_v59_4x.sh" >/dev/null 2>&1 & }
serve_quest(){       METRICS_DIR=$RES/quest_rec VORTEX_MODULE=gqa_quest_sparse_attention VORTEX_TOPK=61 VORTEX_CHUNK=64 VORTEX_RATIO=0.0625 SERVED_NAME=qwen3vl-vortex nohup bash "$HERE/serve/serve_vortex_v59_4x.sh" >/dev/null 2>&1 & }

# TSA sweep: "TOP_K:dir"
TSA_SWEEP="128:tsa_tk128 96:tsa_tk96 64:tsa_tk64 48:tsa_tk48 32:tsa_tk32"

score_all(){
  cd "$HERE/wa_exp"
  for c in dense tsa_tk128 tsa_tk96 tsa_tk64 tsa_tk48 tsa_tk32 vortex quest_rec; do
    $BP score_batch.py --results-dir $RES/$c --judge-url $JUDGE_URL --judge-model llama-judge --concurrency 6 >/dev/null 2>&1 || true
    $BP rescore_lenient.py $RES/$c >/dev/null 2>&1 || true
    $BP agg_sparse_metrics.py $RES/$c >/dev/null 2>&1 || true
  done; }

prev=0
for END in 100 200 400 600 812; do
  R0=$prev; R1=$END
  echo "###### STAGE ${R0}-${R1} START $(date) ######"
  kill_all; serve_dense;            wait_all && run_range dense     qwen3vl-dense  $R0 $R1 || echo "dense servers FAILED stage $R0-$R1"
  for spec in $TSA_SWEEP; do tk=${spec%%:*}; cfg=${spec##*:}
    kill_all; serve_tsa $tk $cfg;   wait_all && run_range $cfg      qwen3vl-tsa    $R0 $R1 || echo "$cfg servers FAILED stage $R0-$R1"
  done
  kill_all; serve_vortex_block;     wait_all && run_range vortex    qwen3vl-vortex $R0 $R1 || echo "vortex servers FAILED stage $R0-$R1"
  kill_all; serve_quest;            wait_all && run_range quest_rec qwen3vl-vortex $R0 $R1 || echo "quest_rec servers FAILED stage $R0-$R1"
  kill_all; nohup bash "$HERE/serve/serve_judge.sh" >/dev/null 2>&1 &
  if wait_h $JUDGE_URL/models 1800; then score_all; else echo "judge TIMEOUT stage $R0-$R1"; fi
  kill_all
  echo "====== STAGE ${R0}-${R1} REPORT (cumulative through task $R1) ======"
  WA_RESULTS=$RES $BP "$HERE/wa_exp/stage_summary.py" $R1 || true
  echo "###### STAGE ${R0}-${R1} DONE $(date) ######"
  prev=$END
done
echo "################ sr_compare STAGED RUN ALL DONE $(date) ################"
