#!/bin/bash
# Sweep top-k values for TSA and run WebArena eval for each.
# Must be run inside your SLURM allocation (srun --pty --overlap --jobid JOBID /bin/bash).
#
# Usage:
#   bash sweep_topk.sh                    # default sweep: 32 64 128 256
#   bash sweep_topk.sh "64 128 256"       # custom values

set -eo pipefail

TOP_K_VALUES="${1:-256 512}"
TSA_DIR="/vast/projects/liuv/pennnetworks/jiaheng/TreeSparseAttention"
WEBARENA_DIR="/vast/projects/liuv/pennnetworks/jiaheng/webarena"
CONDA_SH="/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/etc/profile.d/conda.sh"
PORT=10000
SWEEP_LOG="${WEBARENA_DIR}/results/sweep_topk_$(date +%Y%m%d_%H%M%S).log"


mkdir -p "${WEBARENA_DIR}/results"

echo "========================================" | tee -a "$SWEEP_LOG"
echo "TSA Top-K Sweep" | tee -a "$SWEEP_LOG"
echo "Values: ${TOP_K_VALUES}" | tee -a "$SWEEP_LOG"
echo "========================================" | tee -a "$SWEEP_LOG"

for TOP_K in $TOP_K_VALUES; do
    echo "" | tee -a "$SWEEP_LOG"
    echo "======== top-k = ${TOP_K} ========" | tee -a "$SWEEP_LOG"
    date | tee -a "$SWEEP_LOG"

    # ── 1. Kill any existing server ──────────────────────────────────────────
    pkill -f "serve.py" 2>/dev/null || true
    sleep 3

    # ── 2. Start server with this top-k ──────────────────────────────────────
    source "$CONDA_SH" && conda activate sglang
    cd "$TSA_DIR"
    SERVER_LOG_DIR="${TSA_DIR}/server_logs/topk_${TOP_K}"
    mkdir -p "${SERVER_LOG_DIR}"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python serve.py \
            --model-path "/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-30B-A3B-Instruct" \
            --host 0.0.0.0 --port "$PORT" \
            --top-k "$TOP_K" \
            --page-size 64 \
            --max-decode-tokens 2048 \
            --max-batch-size 8 \
            --batch-collect-ms 100 \
        > "${SERVER_LOG_DIR}/server_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID (top-k=${TOP_K})" | tee -a "$SWEEP_LOG"

    # ── 3. Wait for server to be ready ───────────────────────────────────────
    echo "Waiting for server..." | tee -a "$SWEEP_LOG"
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "Server ready after ${i}s" | tee -a "$SWEEP_LOG"
            break
        fi
        sleep 5
        if [ $i -eq 120 ]; then
            echo "ERROR: Server did not start within 600s" | tee -a "$SWEEP_LOG"
            exit 1
        fi
    done

    # ── 4. Run eval ──────────────────────────────────────────────────────────
    source "$CONDA_SH" && conda activate webarena
    cd "$WEBARENA_DIR"
    bash run_eval_browser_use_TSA.sh http://localhost:${PORT}/v1 - - 8 "" "${TOP_K}" 2>&1 | tee -a "$SWEEP_LOG"

    # ── 5. Extract and log summary ───────────────────────────────────────────
    LATEST=$(ls -t "${WEBARENA_DIR}/results"/browser-use_TSA_*/summary.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        PASS_RATE=$(python3 -c "import json; d=json.load(open('$LATEST')); print(f\"{d['tasks_passed']}/{d['tasks_completed']} = {d['success_rate']}%\")")
        echo "top-k=${TOP_K}  pass_rate=${PASS_RATE}  dir=$(dirname $LATEST)" | tee -a "$SWEEP_LOG"
    fi

    # ── 6. Kill server before next iteration ─────────────────────────────────
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "Server stopped." | tee -a "$SWEEP_LOG"
done

echo "" | tee -a "$SWEEP_LOG"
echo "========================================" | tee -a "$SWEEP_LOG"
echo "Sweep complete. Results summary:" | tee -a "$SWEEP_LOG"
grep "top-k=.*pass_rate=" "$SWEEP_LOG"
echo "Full log: $SWEEP_LOG"
