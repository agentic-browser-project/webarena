# Benchmarking TSA vs SGLang on WebArena (multi-worker)

Entry-point guide for reproducing the **TreeSparseAttention (TSA) vs dense SGLang** pass-rate comparison on WebArena using the multi-worker harness, on either an NVIDIA B200 (sm_100) or an RTX 5060 Ti (sm_120).

For day-to-day operations see [`mp/TSA_VS_DENSE_RUNBOOK.md`](mp/TSA_VS_DENSE_RUNBOOK.md). For the audited 40-task pilot findings see [`TSA_VS_DENSE_REPORT.md`](TSA_VS_DENSE_REPORT.md). For the deep multi-worker mechanics see [`mp/MULTIWORKER_GUIDE.md`](mp/MULTIWORKER_GUIDE.md) §8.

---

## 1. What this benchmarks

The pass rate of **TSA** vs **dense SGLang** on the WebArena task set, holding the following constant across both runs:

- Model weights: **Qwen3-VL-4B-Instruct** (same path, same revision)
- Chat template: **the model's native template** (read from `tokenizer_config.json` by both backends). **Do NOT pass `--chat-template qwen2-vl` to SGLang** — that bundled template was designed for Qwen2-VL, injects vision-pad tokens, and corrupts Qwen3-VL prompts (the model loses its few-shot format anchor and falls back to memorized `` ```action [X]``` `` syntax which WebArena's parser rejects, causing step-1 parse-failure crashes on every trajectory). See `TSA_VS_DENSE_REPORT.md` §0 for the diagnosis.
- Sampling: `temperature=0`, `top_p=1`, `max_tokens=2048`, `stop=None`
- Prefix cache: disabled on both sides (`--disable-radix-cache` on SGLang; TSA has none)
- Post-processing: no markdown-fence stripping (`--no-strip-fences` on TSA; SGLang has no such processor)
- Judge: a single **SGLang dense Qwen3-VL-2B-Instruct** server (port 10002) shared by **both** runs — pinned via `WEBARENA_EVAL_API_BASE`

The agent backend is the only thing that varies between the two runs.

---

## 2. Architecture

```
                hilbit2 (orchestrator host, no GPU)
  ┌──────────────────────────────────────────────────────┐
  │  mp.orchestrator  ──▶  N workers (Playwright)        │
  │       │                       │                      │
  │       │  OPENAI_API_BASE → ───┐│                     │
  │       │  WEBARENA_EVAL_API_BASE→─────┐                │
  │       └─────── docker ────────┼┴─────┼────────────   │
  │                               │      │                │
  │  shopping_w0..wN, gitlab_w0..wN, forum_w0..wN, ...   │
  └────────────────────────────────┼──────┼──────────────┘
                                   │      │ SSH tunnel(s)
                                   ▼      ▼
                    GPU host (B200 sm_100 / RTX 5060 Ti sm_120)
  ┌──────────────────────────────────────────────────────┐
  │  port 10000:  TSA  serve.py    (agent during TSA run)│
  │  port 10001:  SGLang launch_server  (agent during dense run) │
  │  port 10002:  SGLang launch_server  (FIXED judge, both runs) │
  └──────────────────────────────────────────────────────┘
```

WebArena workers never know which agent backend they're talking to — only the OpenAI-compatible URL + model name.

---

## 3. Supported GPUs

| GPU class | SM | VRAM | Default `WEBARENA_NUM_WORKERS` | `TSA_MAX_BATCH` | Judge co-resident? | `CONCURRENT_BACKENDS` |
|-----------|----|------|--------------------------------|-----------------|--------------------|-----------------------|
| **NVIDIA B200** | sm_100 | 141 GB | **8** (N=5 floor) | 16 | Yes | Yes — TSA + Dense + Judge can co-exist |
| **NVIDIA RTX 5060 Ti** | sm_120 | 16 GB | **5** (verified end-to-end) | 4 | With TSA only — not with SGLang-dense | No — runs must be sequential |

Per-SM tuning is pinned in [`mp/configs/gpu_profile.sh`](mp/configs/gpu_profile.sh) and sourced by every launcher. The repo ships `.example.json` templates — copy each to its live name (without `.example.`) and edit the placeholders (`<HOST_OR_IP>`, `<DATA_ROOT>`, `<PATH_TO_DOCKER_SOCK>`) for your environment. Use `config-tsa.json` / `config-dense.json` on a 5060 Ti and the `…-b200.json` variants on a B200.

**Kernel portability**: TSA's JIT build defaults to `TS_CUDA_ARCHS="100;120"` with PTX fallback, so a single checkout runs on either GPU. Override with `TS_CUDA_ARCHS=auto` for single-arch per-host builds.

**SGLang attention backend**: the launcher tries `flashinfer` first and falls back to `triton` on import failure (5060 Ti often needs Triton until FlashInfer ships sm_120 wheels).

---

## 4. Repositories and files modified

### TSA repo (`agentic-browser-project/TreeSparseAttention`)

| File | Change |
|------|--------|
| [`python/jit_build.py`](../../TreeSparseAttention_CW/python/jit_build.py) | Multi-arch build (`TS_CUDA_ARCHS="100;120"` default) + PTX fallback + per-arch `TORCH_EXTENSIONS_DIR` cache dir + `auto` GPU autodetect. Note: if you explicitly export `TORCH_EXTENSIONS_DIR` the per-arch isolation is bypassed — unset it to get a fresh per-arch build. |
| [`serve.py`](../../TreeSparseAttention_CW/serve.py) | New CLI: `--tree-parse-mode {chatml,webarena}`, `--no-strip-fences`, `--served-model-name`. New request fields: OpenAI-compatible `stop` (threaded through generator + finish_reason), accepted-for-compat `top_p`/`n`. `tree_parse_mode` is also passed into `TreeSparseConfig(...)` for both single and batch prefill, so the flag affects kernel chunking, not just parsing. |
| [`start_server.sh`](../../TreeSparseAttention_CW/start_server.sh) | Fully env-driven launcher (`TSA_MODEL_PATH`, `TSA_PORT`, `TSA_TOP_K`, `TSA_PAGE_SIZE`, `TSA_MAX_DECODE_TOKENS`, `TSA_MAX_BATCH`, `TSA_COLLECT_MS`, `TSA_TREE_PARSE_MODE`, `TSA_STRIP_FENCES`, `TSA_MODEL_NAME`, `TSA_PYTHON`). Defaults to Qwen3-VL-4B-Instruct + `TSA_TREE_PARSE_MODE=webarena` + `TSA_STRIP_FENCES=0` (no fence-strip) to match dense baseline. CUDA 12.8 PATH-first. |

### WebArena repo (`agentic-browser-project/webarena_multiworker` and `agentic-browser-project/webarena`)

**New files**:

| File | Purpose |
|------|---------|
| [`BENCHMARK_TSA_VS_SGLANG.md`](BENCHMARK_TSA_VS_SGLANG.md) | This file — entry point |
| [`mp/_inference_common.sh`](mp/_inference_common.sh) | Shared helpers: `gpu_detect`, `wait_healthy`, `open_tunnel`, `kill_tunnel` (per-PID safe — does not kill the operator's own SSH shell) |
| [`mp/configs/gpu_profile.sh`](mp/configs/gpu_profile.sh) | Per-SM tuning: `WEBARENA_NUM_WORKERS`, `TSA_MAX_BATCH`, `MEM_FRAC_AGENT`/`MEM_FRAC_JUDGE`, `JUDGE_FITS`, `CONCURRENT_BACKENDS` |
| [`mp/launch_tsa.sh`](mp/launch_tsa.sh) | Boot TSA on `$GPU_HOST` (auto-detects SM), open tunnel, export env block to `mp/.inference_env` |
| [`mp/launch_dense.sh`](mp/launch_dense.sh) | Boot SGLang-dense agent server with FlashInfer→Triton fallback. **Does NOT pass `--chat-template qwen2-vl`** — SGLang reads the model's native ChatML template from `tokenizer_config.json`. Forcing the qwen2-vl bundled template onto Qwen3-VL corrupts prompts and causes step-1 parse-failure crashes (see `TSA_VS_DENSE_REPORT.md` §0). |
| [`mp/launch_judge.sh`](mp/launch_judge.sh) | Boot the shared SGLang-dense judge (Qwen3-VL-2B-Instruct) on port 10002. Same chat-template rule as `launch_dense.sh`. |
| [`mp/teardown_inference.sh`](mp/teardown_inference.sh) | Stop tmux sessions + tunnels; `--keep-judge` skips judge restart between back-to-back runs |
| [`mp/configs/config-tsa.example.json`](mp/configs/config-tsa.example.json) | sm_120 template: `num_workers=5`; placeholders for host, docker socket, data root. Copy to `config-tsa.json` and edit. |
| [`mp/configs/config-dense.example.json`](mp/configs/config-dense.example.json) | sm_120 template: `num_workers=5`. Copy to `config-dense.json` and edit. |
| [`mp/configs/config-tsa-b200.example.json`](mp/configs/config-tsa-b200.example.json) | sm_100 template: `num_workers=8`. Copy to `config-tsa-b200.json` and edit. |
| [`mp/configs/config-dense-b200.example.json`](mp/configs/config-dense-b200.example.json) | sm_100 template: `num_workers=8`. Copy to `config-dense-b200.json` and edit. |
| [`mp/benchmark_compare.py`](mp/benchmark_compare.py) | TSA vs dense report: strict task_id intersection, bootstrap 95% CI, agreement matrix, per-site breakdown |
| [`mp/check_template_parity.py`](mp/check_template_parity.py) | Chat-template parity sanity utility |
| [`evaluation_harness/_endpoint.py`](evaluation_harness/_endpoint.py) | `judge_endpoint()` context manager — swaps `OPENAI_API_BASE`/key for the judge call; sentinel-based restore; no-op when env unset |
| [`mp/TSA_VS_DENSE_RUNBOOK.md`](mp/TSA_VS_DENSE_RUNBOOK.md) | Daily operations runbook (deeper reference) |
| [`TSA_VS_DENSE_REPORT.md`](TSA_VS_DENSE_REPORT.md) | 40-task pilot findings + audit summary |

**Modified files**:

| File | Change |
|------|--------|
| [`evaluation_harness/helper_functions.py`](evaluation_harness/helper_functions.py) | Wrap judge LLM call in `judge_endpoint()` context manager — backward compatible |
| [`llms/tokenizers.py`](llms/tokenizers.py) | Qwen branch accepts `tree-sparse`, `qwen3vl-dense`, `qwen3vl-judge`, `Qwen3*`; uses `WEBARENA_TOKENIZER_PATH` |
| [`mp/orchestrator.py`](mp/orchestrator.py) | New CLI args `--stop_token`, `--inference_backend`, `--result_dir_override` threaded through to `args_dict` |
| [`mp/worker.py`](mp/worker.py) | `TaskResult` gains provenance fields: `inference_backend`, `model`, `openai_api_base`, `eval_api_base` |
| [`mp/bring_up.py`](mp/bring_up.py) | Magento MySQL `configure_replica_magento` timeout 30 s → 180 s (cold-boot warmup) |
| [`run.py`](run.py) | `auto_login` subprocess: `sys.executable` + `PYTHONPATH` + `cwd` + stale-cookie clear + informative assert with returncode |
| [`mp/MULTIWORKER_GUIDE.md`](mp/MULTIWORKER_GUIDE.md) | §8 "TSA-vs-Dense benchmark workflow" expanded with verified concurrency table |
| [`.gitignore`](.gitignore) | Adds `.claude/` and `TSA_VS_DENSE_RESULTS/` |

WebArena-side changes are **additive** — if `WEBARENA_EVAL_API_BASE` is unset, every modified path behaves identically to upstream.

---

## 5. One-time setup

On the GPU host (`gray` or a B200 box):

```bash
# 1. Model weights (~8 GB)
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir ~/hf_models/Qwen3-VL-4B-Instruct
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir ~/hf_models/Qwen3-VL-2B-Instruct      # judge

# 2. TSA kernels — first import triggers JIT (~3 min the first time)
cd ~/TreeSparseAttention_CW
TS_CUDA_ARCHS="100;120" python -c \
  "from python.jit_build import build_tree_sparse_kernels; build_tree_sparse_kernels()"

# 3. SGLang
pip install 'sglang[all]>=0.4.3' 'flashinfer-python>=0.2'  # Triton fallback OK on sm_120
```

On the orchestrator host:

```bash
cd <WEBARENA_REPO>
# 1. Materialize live configs from templates (edit the placeholders inside afterwards)
for f in config-tsa config-dense config-tsa-b200 config-dense-b200; do
  cp -n mp/configs/$f.example.json mp/configs/$f.json
done
$EDITOR mp/configs/config-tsa.json   # fill in <HOST_OR_IP>, <DATA_ROOT>, <PATH_TO_DOCKER_SOCK>

# 2. Export the GPU host (required — there is no operator-specific default)
export GPU_HOST=user@gpu-host.example.com

# 3. Provision per-worker docker replicas
python -m mp.bring_up --num_workers 5 --skip_goldens
```

Deeper setup detail is in [`mp/TSA_VS_DENSE_RUNBOOK.md`](mp/TSA_VS_DENSE_RUNBOOK.md) §2.

---

## 6. End-to-end run

From the orchestrator host (`webarena/`):

```bash
export GPU_HOST=user@gpu-host.example.com   # required

# === TSA run ===
bash mp/launch_judge.sh        # port 10002
bash mp/launch_tsa.sh          # port 10000 + tunnel + env block
source mp/.inference_env

python -m mp.orchestrator --config mp/configs/config-tsa.json \
  --start_idx 0 --end_idx 812 \
  --provider openai --mode chat --model tree-sparse \
  --temperature 0 --top_p 1 --max_tokens 2048 \
  --max_steps 30 --inference_backend tsa \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json

# === Dense run (sequential on sm_120; can be concurrent on sm_100) ===
bash mp/teardown_inference.sh --keep-judge
bash mp/launch_dense.sh        # port 10001
source mp/.inference_env

python -m mp.orchestrator --config mp/configs/config-dense.json \
  --start_idx 0 --end_idx 812 \
  --provider openai --mode chat --model qwen3vl-dense \
  --temperature 0 --top_p 1 --max_tokens 2048 \
  --max_steps 30 --inference_backend dense \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json

# === Compare ===
python -m mp.benchmark_compare \
  --tsa <DATA_ROOT>/results-tsa/scores.jsonl \
  --dense <DATA_ROOT>/results-dense/scores.jsonl \
  --tasks config_files/test.raw.json \
  --out comparison_report.md --csv comparison.csv
```

For B200 substitute `config-tsa-b200.json` / `config-dense-b200.json` (N=8). On sm_100 with `CONCURRENT_BACKENDS=yes` you may run both backends in parallel — see [`mp/TSA_VS_DENSE_RUNBOOK.md`](mp/TSA_VS_DENSE_RUNBOOK.md) §6.

---

## 7. Multi-worker concurrency profiles (verified)

| GPU | Backend | N (workers) | Evidence |
|-----|---------|-------------|----------|
| RTX 5060 Ti (sm_120) | TSA | **5** | Raw 5×parallel HTTP probe: 5/5 200 OK, scheduler batch=4+1. Orchestrator: 5/5 scored rows from w0..w4, 0 errors, wall 118 s, PASS=3 FAIL=2. |
| RTX 5060 Ti (sm_120) | SGLang-dense | **5** | Raw probe: 5/5 200 OK, wall 0.41 s. Orchestrator: 5/5 scored rows from w0..w4, 0 errors, wall 104 s, PASS=2 FAIL=3. |
| NVIDIA B200 (sm_100) | TSA | **8** (default), **5** (floor) | Profile derived from sm_120 (verified) by scaling `TSA_MAX_BATCH` 4→16 + `MEM_FRAC_AGENT` 0.60→0.30. Not yet end-to-end smoked on B200 hardware. |
| NVIDIA B200 (sm_100) | SGLang-dense | **8** | Same scaling; `MEM_FRAC_AGENT=0.30` + `max-running-requests=16`. |

N=5 was selected as the smallest production-meaningful concurrency that exercises the docker per-worker replica logic, scheduler batching, and the judge-endpoint swap path together; N=8 on B200 is conservative — the hardware accommodates more.

---

## 8. Verification

Four quick checks (all should pass before declaring a result trustworthy):

1. **Server smoke (per backend)**:
   ```bash
   curl -s http://127.0.0.1:10000/v1/models      # TSA  → tree-sparse
   curl -s http://127.0.0.1:10001/v1/models      # dense → qwen3vl-dense
   curl -s http://127.0.0.1:10002/v1/models      # judge → qwen3vl-judge
   ```
2. **TSA parse-mode log**: tail the TSA server log on first agent call; expect `[TreeSparse/webarena] Parsed tree …` (NOT `chatml`).
3. **Single-task smoke**: run task_id 22 with `--start_idx 22 --end_idx 23 --max_steps 5` on both backends; both should yield a scored row in <3 min with `error: null`.
4. **N=5 raw concurrency probe**: 5 parallel `curl POST /v1/chat/completions` — all 5 must return 200 within ~2 s on TSA / sub-second on SGLang.

---

## 9. Deeper references

- **Operations runbook** (per-step daily run sequence, troubleshooting, B200 path): [`mp/TSA_VS_DENSE_RUNBOOK.md`](mp/TSA_VS_DENSE_RUNBOOK.md)
- **Multi-worker mechanics** (per-worker docker replica logic, auth folders, reset semantics): [`mp/MULTIWORKER_GUIDE.md`](mp/MULTIWORKER_GUIDE.md) — see §8 for the TSA-vs-Dense workflow
- **40-task pilot findings + audit** (headline pass rates, agreement matrix, sanity checks, known limitations): [`TSA_VS_DENSE_REPORT.md`](TSA_VS_DENSE_REPORT.md)
- **Comparison tool**: [`mp/benchmark_compare.py`](mp/benchmark_compare.py) — strict task_id intersection, bootstrap 95% CI on pass-rate delta, agreement matrix, per-site breakdown
- **Chat-template parity sanity**: [`mp/check_template_parity.py`](mp/check_template_parity.py) — assert byte-identical tokenized prompts between TSA and SGLang for a sample observation+instruction
