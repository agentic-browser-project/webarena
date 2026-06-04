#!/bin/bash
# Evaluate a model served by SGLang (OpenAI-compatible) on WebArena using browser-use agent.
# Websites must already be running (see the table in README for URLs).
#
# Usage:
#   ./run_eval_browser_use_sglang.sh                                         # defaults (tasks 0-100)
#   ./run_eval_browser_use_sglang.sh http://localhost:8000/v1                # custom server
#   ./run_eval_browser_use_sglang.sh http://localhost:8000/v1 0 50           # custom range
#   ./run_eval_browser_use_sglang.sh http://localhost:8000/v1 0 50 4         # 4 workers
#   ./run_eval_browser_use_sglang.sh http://localhost:8000/v1 - - 4 "0,1,2" # explicit task list

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
WEBARENA_DIR="/vast/projects/liuv/pennnetworks/jiaheng/webarena"
PYTHON="/vast/projects/liuv/pennnetworks/jiaheng/browser-use/.venv/bin/python"

# ── arguments ────────────────────────────────────────────────────────────────
API_BASE="${1:-http://localhost:8000/v1}"
START_IDX="${2:-0}"
END_IDX="${3:-100}"
NUM_WORKERS="${4:-4}"
# Default to read-only tasks (no map, no write); pass "" to use START/END range instead
TASK_IDS="${5:-$(cat "${WEBARENA_DIR}/readonly_task_ids.txt" 2>/dev/null || echo "")}"

# Auto-detect model name from the server
MODEL=$(curl -sf "${API_BASE}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")

TS=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${WEBARENA_DIR}/results/browser-use_sglang_${MODEL}_${TS}"
LOG_FILE="${RESULT_DIR}/run.log"

# ── site URLs ────────────────────────────────────────────────────────────────
export SHOPPING="http://158.130.4.228:7770"
export SHOPPING_ADMIN="http://158.130.4.228:7780/admin"
export REDDIT="http://158.130.4.228:9999"
export GITLAB="http://158.130.4.228:8023"
export MAP="http://158.130.4.229:3000"
export WIKIPEDIA="http://158.130.4.228:8889/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://158.130.4.228:4399"

# ── SGLang / OpenAI compat ───────────────────────────────────────────────────
export OPENAI_API_KEY="dummy"
export OPENAI_API_BASE="${API_BASE}"

mkdir -p "${RESULT_DIR}"
export WEBARENA_RESULT_DIR="${RESULT_DIR}"

echo "========================================"
echo "WebArena Evaluation (browser-use agent)"
echo "Model:   ${MODEL}"
echo "Server:  ${API_BASE}"
if [ -n "${TASK_IDS}" ]; then
    TASK_COUNT=$(echo "${TASK_IDS}" | tr ',' '\n' | wc -l)
    echo "Tasks:   ${TASK_COUNT} explicit IDs"
else
    echo "Tasks:   ${START_IDX}–${END_IDX}"
fi
echo "Workers: ${NUM_WORKERS}"
echo "Results: ${RESULT_DIR}"
echo "========================================"

cd "${WEBARENA_DIR}"

# ── split tasks across workers and run in parallel ───────────────────────────
PIDS=()

if [ -n "${TASK_IDS}" ]; then
    # Split explicit task ID list across workers
    IDS_ARRAY=($(echo "${TASK_IDS}" | tr ',' ' '))
    TOTAL=${#IDS_ARRAY[@]}
    CHUNK=$(( (TOTAL + NUM_WORKERS - 1) / NUM_WORKERS ))

    for ((w=0; w<NUM_WORKERS; w++)); do
        W_START=$((w * CHUNK))
        W_END=$((W_START + CHUNK))
        [ "${W_END}" -gt "${TOTAL}" ] && W_END="${TOTAL}"
        [ "${W_START}" -ge "${TOTAL}" ] && break

        W_IDS=$(echo "${IDS_ARRAY[@]:${W_START}:${CHUNK}}" | tr ' ' ',')
        WORKER_LOG="${RESULT_DIR}/worker_${w}.log"
        echo "[worker ${w}] tasks ${W_IDS:0:40}... → ${WORKER_LOG}"

        PYTHONPATH="${WEBARENA_DIR}" \
        "${PYTHON}" run_browser_use.py \
            --task_ids       "${W_IDS}" \
            --model          "${MODEL}" \
            --temperature    1.0 \
            --max_tokens     2048 \
            --max_steps      30 \
            --llm_timeout    180 \
            --result_dir     "${RESULT_DIR}" \
            > "${WORKER_LOG}" 2>&1 &
        PIDS+=($!)
    done
else
    # Split contiguous range across workers
    TOTAL=$((END_IDX - START_IDX))
    CHUNK=$(( (TOTAL + NUM_WORKERS - 1) / NUM_WORKERS ))

    for ((w=0; w<NUM_WORKERS; w++)); do
        W_START=$((START_IDX + w * CHUNK))
        W_END=$((W_START + CHUNK))
        [ "${W_END}" -gt "${END_IDX}" ] && W_END="${END_IDX}"
        [ "${W_START}" -ge "${END_IDX}" ] && break

        WORKER_LOG="${RESULT_DIR}/worker_${w}.log"
        echo "[worker ${w}] tasks ${W_START}–${W_END} → ${WORKER_LOG}"

        PYTHONPATH="${WEBARENA_DIR}" \
        "${PYTHON}" run_browser_use.py \
            --test_start_idx "${W_START}" \
            --test_end_idx   "${W_END}" \
            --model          "${MODEL}" \
            --temperature    1.0 \
            --max_tokens     2048 \
            --max_steps      30 \
            --llm_timeout    180 \
            --result_dir     "${RESULT_DIR}" \
            > "${WORKER_LOG}" 2>&1 &
        PIDS+=($!)
    done
fi

echo "Waiting for ${#PIDS[@]} workers (PIDs: ${PIDS[*]})..."
for PID in "${PIDS[@]}"; do
    wait "${PID}" || true
done
echo "All workers done."
cat "${RESULT_DIR}"/worker_*.log | tee "${LOG_FILE}"

# ── summarise results ────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
"${PYTHON}" - <<EOF
import json, glob, datetime
result_dir = "${RESULT_DIR}"
files = sorted(glob.glob(f"{result_dir}/result_*.json"))
passed = sum(1 for f in files if json.load(open(f)).get("score", 0) == 1)
total  = len(files)
success_rate = passed / total * 100 if total > 0 else 0.0
print(f"Tasks completed : {total}")
print(f"Tasks passed    : {passed}")
if total > 0:
    print(f"Success rate    : {success_rate:.1f}%")
print(f"Results dir     : {result_dir}")
tasks = []
for f in files:
    d = json.load(open(f))
    tasks.append({"task_id": d.get("task_id"), "intent": d.get("intent", ""), "score": d.get("score", 0)})
summary = {
    "model": "${MODEL}",
    "server": "${API_BASE}",
    "start_idx": ${START_IDX},
    "end_idx": ${END_IDX},
    "tasks_completed": total,
    "tasks_passed": passed,
    "success_rate": round(success_rate, 2),
    "timestamp": "${TS}",
    "result_dir": result_dir,
    "tasks": tasks,
}
with open(f"{result_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written : {result_dir}/summary.json")
EOF
