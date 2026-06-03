#!/usr/bin/env bash
# Per-GPU tuning profile for TSA and SGLang inference servers.
#
# Sourced by mp/launch_*.sh. After sourcing, the following env vars are set
# based on the detected sm version (or the value the caller passed in
# GPU_SM_HINT):
#
#   TSA_MAX_BATCH        — --max-batch-size for TSA serve.py
#   TSA_COLLECT_MS       — --batch-collect-ms for TSA serve.py
#   TSA_TOP_K            — --top-k for TSA serve.py
#   TSA_PAGE_SIZE        — --page-size for TSA serve.py
#   TSA_MAX_DECODE_TOKENS — --max-decode-tokens for TSA serve.py
#   SGLANG_BACKEND       — --attention-backend for SGLang launch_server
#   SGLANG_BACKEND_FALLBACK — secondary --attention-backend if primary fails
#   MEM_FRAC_AGENT       — --mem-fraction-static for the agent SGLang server
#   MEM_FRAC_JUDGE       — --mem-fraction-static for the judge SGLang server
#   AGENT_CTX_LEN        — --context-length for the agent SGLang server
#   JUDGE_CTX_LEN        — --context-length for the judge SGLang server
#   WEBARENA_NUM_WORKERS — recommended num_workers for mp.orchestrator
#   JUDGE_FITS           — "yes" if a separate 2B judge fits alongside the agent
#   CONCURRENT_BACKENDS  — "yes" if TSA and dense fit simultaneously (B200 only)
#   GPU_NOTES            — human-readable note shown in the launcher banner
#
# Override any of these by exporting them BEFORE sourcing the profile.
#
# Profiles are validated for two reference hardware tiers:
#
#   sm_100 (B200, 141 GB HBM3) — the original TSA reference rig.
#       Generous VRAM: TSA + dense + judge all coexist; N=16 workers
#       comfortably batches at TSA_MAX_BATCH=16.
#
#   sm_120 (RTX 5060 Ti, 16 GB GDDR7) — minimum viable consumer Blackwell.
#       Tight VRAM: judge OFF by default so TSA can use max-batch=4;
#       N=5 workers stays end-to-end stable (4 batched at TSA, 1 queues).

_detect_sm() {
    if [[ -n "${GPU_SM_HINT:-}" ]]; then
        echo "$GPU_SM_HINT"
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Major.minor compute capability → "MMm" string (e.g. "12.0" → "120")
        local cc
        cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                 | head -n 1 | tr -d '[:space:]')
        if [[ "$cc" =~ ^[0-9]+\.[0-9]+$ ]]; then
            local major="${cc%.*}"
            local minor="${cc#*.}"
            echo "${major}${minor}"
            return 0
        fi
    fi
    # If nvidia-smi isn't available (e.g. on the hilbit2 driver side just
    # composing the launch command), default to sm_120 (5060 Ti) — that's our
    # canonical worst-case target. The GPU host's nvidia-smi run via SSH
    # will override.
    echo "120"
}

GPU_SM="$(_detect_sm)"

case "$GPU_SM" in
    100)
        # ── B200 / sm_100 (141 GB HBM3) — reference rig ───────────────────
        #
        # The TSA team's original test environment. Everything coexists:
        # TSA agent + SGLang dense + 2B judge fit simultaneously; max_batch
        # scales to 16; AGENT_CTX_LEN at 32K is comfortable.
        #
        # Validated configurations (B200):
        #   N=8  workers, TSA_MAX_BATCH=8,  judge ON  → trivial fit
        #   N=16 workers, TSA_MAX_BATCH=16, judge ON  → ~40 GB peak
        #   TSA + dense concurrently for cross-backend differential tests
        export TSA_MAX_BATCH="${TSA_MAX_BATCH:-16}"
        export TSA_COLLECT_MS="${TSA_COLLECT_MS:-100}"
        export TSA_TOP_K="${TSA_TOP_K:-128}"
        export TSA_PAGE_SIZE="${TSA_PAGE_SIZE:-64}"
        export TSA_MAX_DECODE_TOKENS="${TSA_MAX_DECODE_TOKENS:-2048}"
        export SGLANG_BACKEND="${SGLANG_BACKEND:-flashinfer}"
        export SGLANG_BACKEND_FALLBACK="${SGLANG_BACKEND_FALLBACK:-triton}"
        # mem-frac 0.30 = ~42 GB per SGLang server — room for three
        # (dense agent + judge + spare) without contention.
        export MEM_FRAC_AGENT="${MEM_FRAC_AGENT:-0.30}"
        export MEM_FRAC_JUDGE="${MEM_FRAC_JUDGE:-0.20}"
        export AGENT_CTX_LEN="${AGENT_CTX_LEN:-32768}"
        export JUDGE_CTX_LEN="${JUDGE_CTX_LEN:-8192}"
        export WEBARENA_NUM_WORKERS="${WEBARENA_NUM_WORKERS:-8}"
        export JUDGE_FITS="${JUDGE_FITS:-yes}"
        export CONCURRENT_BACKENDS="${CONCURRENT_BACKENDS:-yes}"
        export GPU_NOTES="B200 / sm_100 — 141 GB HBM3, all engines coexist, N≤16 trivial."
        ;;
    120)
        # ── RTX 5060 Ti / sm_120 (16 GB GDDR7) — minimum target ───────────
        #
        # Empirical VRAM accounting (Qwen3-VL-4B-Instruct, bfloat16):
        #   - Model weights         ~ 8.0 GB
        #   - TSA prefill activations + KV per batch slot ~ 0.6–0.8 GB
        #   - SGLang 2B judge model ~ 4.3 GB
        #
        # Concurrency profile (validated):
        #   JUDGE OFF (recommended; self-judging documented in §6.5 of
        #              TSA_VS_DENSE_REPORT.md):
        #       TSA_MAX_BATCH=4 + N=5 workers → ~12 GB peak (4 batched at
        #       TSA scheduler, 1 worker queues; throughput ~2× vs N=2).
        #       Dense alone at MEM_FRAC_AGENT=0.60 → ~10 GB, also fits
        #       N=5 (SGLang's max-running-requests=16 ≥ 5 workers).
        #
        #   JUDGE ON (only when fixed-judge rigor is required):
        #       Drop TSA_MAX_BATCH back to 2 and lower N to ≤3 to keep
        #       headroom. SGLang dense + judge does NOT fit at any
        #       reasonable mem-frac on 16 GB.
        export TSA_MAX_BATCH="${TSA_MAX_BATCH:-4}"
        export TSA_COLLECT_MS="${TSA_COLLECT_MS:-100}"
        export TSA_TOP_K="${TSA_TOP_K:-128}"
        export TSA_PAGE_SIZE="${TSA_PAGE_SIZE:-64}"
        export TSA_MAX_DECODE_TOKENS="${TSA_MAX_DECODE_TOKENS:-1024}"
        export SGLANG_BACKEND="${SGLANG_BACKEND:-flashinfer}"
        export SGLANG_BACKEND_FALLBACK="${SGLANG_BACKEND_FALLBACK:-triton}"
        export MEM_FRAC_AGENT="${MEM_FRAC_AGENT:-0.60}"
        export MEM_FRAC_JUDGE="${MEM_FRAC_JUDGE:-0.28}"
        export AGENT_CTX_LEN="${AGENT_CTX_LEN:-8192}"
        export JUDGE_CTX_LEN="${JUDGE_CTX_LEN:-2048}"
        export WEBARENA_NUM_WORKERS="${WEBARENA_NUM_WORKERS:-5}"
        export JUDGE_FITS="${JUDGE_FITS:-with_tsa_only}"
        export CONCURRENT_BACKENDS="${CONCURRENT_BACKENDS:-no}"
        export GPU_NOTES="RTX 5060 Ti / sm_120 — 16 GB; N=5 validated with judge OFF."
        ;;
    *)
        # ── Unknown / older GPU ───────────────────────────────────────────
        # PTX fallback in jit_build.py may JIT the kernels; perf unverified.
        # Conservative defaults err on the side of "fits, but small".
        export TSA_MAX_BATCH="${TSA_MAX_BATCH:-2}"
        export TSA_COLLECT_MS="${TSA_COLLECT_MS:-150}"
        export TSA_TOP_K="${TSA_TOP_K:-128}"
        export TSA_PAGE_SIZE="${TSA_PAGE_SIZE:-64}"
        export TSA_MAX_DECODE_TOKENS="${TSA_MAX_DECODE_TOKENS:-1024}"
        export SGLANG_BACKEND="${SGLANG_BACKEND:-triton}"
        export SGLANG_BACKEND_FALLBACK="${SGLANG_BACKEND_FALLBACK:-torch_native}"
        export MEM_FRAC_AGENT="${MEM_FRAC_AGENT:-0.40}"
        export MEM_FRAC_JUDGE="${MEM_FRAC_JUDGE:-0.30}"
        export AGENT_CTX_LEN="${AGENT_CTX_LEN:-4096}"
        export JUDGE_CTX_LEN="${JUDGE_CTX_LEN:-2048}"
        export WEBARENA_NUM_WORKERS="${WEBARENA_NUM_WORKERS:-2}"
        export JUDGE_FITS="${JUDGE_FITS:-unknown}"
        export CONCURRENT_BACKENDS="${CONCURRENT_BACKENDS:-no}"
        export GPU_NOTES="Unknown SM (${GPU_SM}) — conservative defaults; verify perf."
        ;;
esac

# Default model/served-model identifiers used by both launchers.
export TSA_MODEL_NAME="${TSA_MODEL_NAME:-tree-sparse}"
export DENSE_MODEL_NAME="${DENSE_MODEL_NAME:-qwen3vl-dense}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-qwen3vl-judge}"
