#!/bin/bash
# Evaluate a model served by SGLang (OpenAI-compatible) on WebArena.
# Websites must already be running (see the table in README for URLs).
#
# Usage:
#   ./run_eval_sglang.sh                                              # defaults (webarena agent)
#   ./run_eval_sglang.sh http://localhost:8000/v1                     # custom server
#   ./run_eval_sglang.sh http://localhost:8000/v1 Qwen3-VL-8B-Instruct
#   ./run_eval_sglang.sh http://localhost:8000/v1 Qwen3-VL-8B-Instruct 0 812
#   ./run_eval_sglang.sh http://localhost:8000/v1 Qwen3-VL-8B-Instruct 0 12 browser-use

set -euo pipefail

# ── arguments ────────────────────────────────────────────────────────────────
API_BASE="${1:-http://localhost:8000/v1}"
MODEL="${2:-Qwen3-VL-8B-Instruct}"
START_IDX="${3:-0}"
END_IDX="${4:-12}"
AGENT_MODE="${5:-webarena}"   # "webarena" (default) or "browser-use"

# ── paths ────────────────────────────────────────────────────────────────────
WEBARENA_DIR="/vast/projects/liuv/pennnetworks/jiaheng/webarena"
# webarena agent uses the webarena conda env; browser-use uses its own venv
PYTHON_WEBARENA="/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/webarena/bin/python"
PYTHON_BROWSERUSE="/vast/projects/liuv/pennnetworks/jiaheng/browser-use/.venv/bin/python"
TS=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${WEBARENA_DIR}/results/${AGENT_MODE}_${TS}"
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
echo "WebArena Evaluation (SGLang backend)"
echo "Agent:   ${AGENT_MODE}"
echo "Model:   ${MODEL}"
echo "Server:  ${API_BASE}"
echo "Tasks:   ${START_IDX}–${END_IDX}"
echo "Results: ${RESULT_DIR}"
echo "========================================"

cd "${WEBARENA_DIR}"

if [ "${AGENT_MODE}" = "browser-use" ]; then
    # ── browser-use agent ─────────────────────────────────────────────────────
    # browser-use needs its own venv (Python 3.12 + browser_use package).
    # The webarena config_files/ and evaluation_harness/ are on the path via
    # PYTHONPATH so the evaluator can still be imported.
    PYTHONPATH="${WEBARENA_DIR}" \
    "${PYTHON_BROWSERUSE}" run_browser_use.py \
        --test_start_idx "${START_IDX}" \
        --test_end_idx   "${END_IDX}" \
        --model          "${MODEL}" \
        --temperature    1.0 \
        --max_tokens     2048 \
        --max_steps      30 \
        --result_dir     "${RESULT_DIR}" \
        2>&1 | tee "${LOG_FILE}"
else
    # ── original WebArena agent ───────────────────────────────────────────────
    "${PYTHON_WEBARENA}" run.py \
        --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
        --test_start_idx "${START_IDX}" \
        --test_end_idx   "${END_IDX}" \
        --provider openai \
        --model  "${MODEL}" \
        --mode   chat \
        --temperature 1.0 \
        --top_p 0.9 \
        --max_tokens 384 \
        --max_obs_length 1920 \
        --max_steps 30 \
        --observation_type accessibility_tree \
        --action_set_tag id_accessibility_tree \
        --result_dir "${RESULT_DIR}" \
        2>&1 | tee "${LOG_FILE}"
fi

# ── summarise results ────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Summary"
echo "========================================"

if [ "${AGENT_MODE}" = "browser-use" ]; then
    # browser-use saves result_{task_id}.json with a "score" field
    "${PYTHON_BROWSERUSE}" - <<EOF
import json, glob
result_dir = "${RESULT_DIR}"
files = sorted(glob.glob(f"{result_dir}/result_*.json"))
passed = sum(1 for f in files if json.load(open(f)).get("score", 0) == 1)
total  = len(files)
print(f"Tasks completed : {total}")
print(f"Tasks passed    : {passed}")
if total > 0:
    print(f"Success rate    : {passed/total*100:.1f}%")
print(f"Results dir     : {result_dir}")
EOF
else
    "${PYTHON_WEBARENA}" - <<EOF
import json, glob
result_dir = "${RESULT_DIR}"
files = sorted(glob.glob(f"{result_dir}/*.json"))
passed = sum(1 for f in files if json.load(open(f)).get("success", False))
total  = len(files)
print(f"Tasks completed : {total}")
print(f"Tasks passed    : {passed}")
if total > 0:
    print(f"Success rate    : {passed/total*100:.1f}%")
print(f"Results dir     : {result_dir}")
EOF
fi
