# TSA-vs-Dense WebArena benchmark — OPERATOR RUNBOOK

Operator reference for TreeSparseAttention (TSA) vs SGLang-dense from a cold
state. Companion to `mp/MULTIWORKER_GUIDE.md` §8. Command-first, gotcha-heavy.

Two-host topology: **`gray`** = GPU host (B200 sm_100 or 5060 Ti sm_120), runs
TSA / SGLang / judge. **`hilbit2`** = orchestrator, runs docker site replicas +
Playwright + `mp.orchestrator`. Reaches `gray` via SSH tunnels.

---

## 1. Hardware prerequisites

### GPU

| Arch | Status | Notes |
|---|---|---|
| sm_100 (B200, 141 GB HBM) | **preferred** | TSA + dense + judge all coexist; `TSA_MAX_BATCH` can go to 16. |
| sm_120 (RTX 5060 Ti, 16 GB) | supported | TSA + 2B judge fits (~14 GB). Dense + 2B judge **does NOT fit** — see §1.3. |
| sm_89 / sm_90 etc. | unverified | PTX fallback may JIT; perf unknown. `gpu_profile.sh` uses conservative defaults. |

### 1.1 Verify GPU on `gray`

```bash
ssh wangcy07@gray.cis.upenn.edu nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
# Expect: "NVIDIA GeForce RTX 5060 Ti, 12.0, 16380 MiB" or "NVIDIA B200, 10.0, 144384 MiB"
```

### 1.2 Verify orchestrator host on `hilbit2`

```bash
ssh wangcy07@hilbit2.cis.upenn.edu '
  docker info --format "{{.ServerVersion}}" &&
  free -g | grep Mem &&
  df -h /z &&
  which autossh tmux'
```
Need: docker daemon up, ≥64 GB RAM (≥140 GB for N=8), ≥250 GB free on `/z`.

### 1.3 16 GB VRAM coexistence matrix (sm_120)

Verified empirically (smoke runs in MULTIWORKER_GUIDE §8.7):

| Combo | Fits? | Workaround |
|---|---|---|
| TSA(4B) + Judge(2B) at `MEM_FRAC_JUDGE=0.28` | **yes** (~14 GB) | none |
| Dense(4B) alone at `MEM_FRAC_AGENT=0.60` | yes (~10.4 GB) | none |
| Dense(4B) + Judge(2B) | **NO** — SGLang OOM in `init_memory_pool` | Three valid choices: (a) **TSA self-judges** — point `WEBARENA_EVAL_API_BASE` → `:10000`, `WEBARENA_EVAL_MODEL=tree-sparse`. (b) **dense self-judges** — same at `:10001`. (c) **smaller-model judge only for TSA** — keep 2B judge for the TSA run, use dense self-judge for the dense run. The `eval_api_base` field in each `scores.jsonl` row records which judge was used so `benchmark_compare.py` surfaces the asymmetry. |

### 1.4 Network: orchestrator → GPU

The orchestrator's host (`hilbit2`) must reach `gray` via SSH on key-auth only.
Verify:

```bash
ssh -o BatchMode=yes wangcy07@158.130.4.227 'echo ok'
# Must print "ok" with no password prompt.
```

`mp/_inference_common.sh` requires `GPU_HOST` to be exported — there is no
operator-specific default. The launchers will fail fast with a usage hint if
it's unset. The deployed default for this runbook is `export GPU_HOST=wangcy07@158.130.4.227`.

---

## 2. One-time setup

### 2.1 On `gray` — pre-download model weights

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir ~/hf_models/Qwen3-VL-4B-Instruct
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir ~/hf_models/Qwen3-VL-2B-Instruct
```

### 2.2 On `gray` — build TSA kernels (target both archs)

```bash
cd ~/TreeSparseAttention_CW
TS_CUDA_ARCHS="100;120" python -c \
  "from python.jit_build import build_tree_sparse_kernels; build_tree_sparse_kernels()"
# ~3 min first time; cached at ~/.cache/torch_extensions/_tree_sparse_kernels_<archs>/.
# After pulling new TSA code: rm -rf ~/.cache/torch_extensions/_tree_sparse_kernels_*
```

### 2.3 On `gray` — install SGLang + flashinfer (dedicated venv)

```bash
sudo apt-get install -y python3.12-venv tmux
python3.12 -m venv ~/venvs/bench_sglang
source ~/venvs/bench_sglang/bin/activate
pip install -U pip wheel
pip install "sglang[all]>=0.4.3" "flashinfer-python>=0.2"
python -c "import sglang, flashinfer; print(sglang.__version__, flashinfer.__version__)"
```
Launchers invoke `$SGLANG_PYTHON` (default `$HOME/venvs/bench_sglang/bin/python`).

### 2.4 On `hilbit2` — autossh (optional)

```bash
ssh wangcy07@hilbit2.cis.upenn.edu 'which autossh || sudo apt-get install -y autossh'
```
If absent, `_inference_common.sh:open_tunnel` falls back to plain `nohup ssh -N -L …`
with pidfile `/tmp/wa_tunnel_${LOCAL}.pid`.

### 2.5 On `hilbit2` — verify webarena venv

```bash
ssh wangcy07@hilbit2.cis.upenn.edu '
  source /z/wangcy07/webarena-venv/bin/activate &&
  python -c "import openai, playwright, transformers; print(openai.__version__)"'
```
Need `openai` (0.x or 1.x — see §4 gotcha), `playwright>=1.60.0` with browsers
side-loaded to `$PLAYWRIGHT_BROWSERS_PATH`, `transformers` (tokenizer falls
back to `cl100k_base` via `llms/tokenizers.py` if absent).

### 2.6 Once per benchmark — provision per-worker docker replicas

```bash
ssh wangcy07@hilbit2.cis.upenn.edu
source /z/wangcy07/webarena-venv/bin/activate
cd /z/wangcy07/webarena-repo
bash prepare.sh           # vanilla WebArena docker-up (idempotent)
python -m mp.bring_up --num_workers 2 --host 158.130.4.158 \
    --golden_root /z/wangcy07/webarena-mp/golden \
    2>&1 | tee /z/wangcy07/webarena-mp/bring_up.log
```
Wall-clock: ~30 min for N=2, ~60 min for N=8 (gitlab tar dominates). See
MULTIWORKER_GUIDE §3.2 for the per-step breakdown.

---

## 3. Daily run sequence

### Step 1 — Pre-flight

```bash
# gray: GPU empty
ssh wangcy07@158.130.4.227 nvidia-smi --query-gpu=memory.used --format=csv,noheader   # ~0 MiB
# hilbit2: docker up, no stale tunnels
ssh wangcy07@hilbit2.cis.upenn.edu '
  docker info >/dev/null && echo docker:OK &&
  pgrep -af "ssh.*-L 127.0.0.1:1000" || echo tunnels:CLEAN'
```
⚠️ Stale tunnels after teardown → orphans (see §4). Kill explicitly with
per-pid `kill`, never `pkill -f`.

### Step 2 — Boot the judge (skip if dense + judge won't fit; see §1.3)

```bash
ssh wangcy07@hilbit2.cis.upenn.edu
source /z/wangcy07/webarena-venv/bin/activate
cd /z/wangcy07/webarena-repo
export GPU_HOST=wangcy07@158.130.4.227
export GPU_SM_HINT=120                       # 100 for B200
bash mp/launch_judge.sh
# ssh gray → tmux new-session -d -s wa-judge 'sglang …' → open tunnel :10002 →
# wait_healthy http://127.0.0.1:10002/health
```

### Step 3 — Boot the agent backend

```bash
source mp/launch_tsa.sh
# OR
source mp/launch_dense.sh
# Side-effects: ensures judge is up; writes mp/.inference_env; exports
# OPENAI_API_BASE, AGENT_MODEL_NAME, WEBARENA_EVAL_API_BASE,
# WEBARENA_EVAL_MODEL, INFERENCE_BACKEND. Verifies /v1/models advertises the
# expected served-model-name.
```

### Step 4 — Tunnels (manual variants — launchers do this for you)

```bash
# Preferred — autossh, self-healing:
AUTOSSH_PIDFILE=/tmp/wa_tunnel_10000.pid autossh -M 0 -f -N \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 127.0.0.1:10000:127.0.0.1:10000 wangcy07@158.130.4.227

# Fallback — plain ssh, manual pidfile:
nohup ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 127.0.0.1:10000:127.0.0.1:10000 wangcy07@158.130.4.227 \
    > /dev/null 2>&1 &
echo $! > /tmp/wa_tunnel_10000.pid
```

Sanity-check from `hilbit2`:
```bash
curl -sf http://127.0.0.1:10000/v1/models | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
# tree-sparse  (or qwen3vl-dense for :10001, qwen3vl-judge for :10002)
```

### Step 5 — Export env vars (fresh shells)

```bash
source /z/wangcy07/webarena-repo/mp/.inference_env
# Auto-includes: OPENAI_API_BASE, OPENAI_API_KEY, AGENT_MODEL_NAME,
# WEBARENA_EVAL_API_BASE, WEBARENA_EVAL_API_KEY, WEBARENA_EVAL_MODEL,
# WEBARENA_TOKENIZER_PATH, INFERENCE_BACKEND.

# Plus the offline caches:
export PLAYWRIGHT_BROWSERS_PATH=/z/wangcy07/pw_browsers
export TIKTOKEN_CACHE_DIR=/z/wangcy07/tiktoken_cache
export NLTK_DATA=/z/wangcy07/nltk_data
```

### Step 6 — Wipe per-worker auth folders (CRITICAL)

⚠️ Stale cookies make `auto_login.py` skip a fresh login → agent hits a
logged-out page → **"Welcome, please sign in"** loops on every shopping /
shopping_admin / reddit / gitlab task. The audit found this is the #1 cause
of mass-zero TSA runs after a re-deploy.

```bash
CFG=mp/configs/config-tsa.json   # or config-dense.json
AUTH_ROOT=$(python3 -c "import json; print(json.load(open('$CFG'))['auth_root'])")
N=$(python3 -c "import json; print(json.load(open('$CFG'))['num_workers'])")
for w in $(seq 0 $((N-1))); do
    rm -rf "$AUTH_ROOT/w$w"
    mkdir -p "$AUTH_ROOT/w$w"
done
ls "$AUTH_ROOT"   # confirm empty per-worker dirs
```

Forces `auto_login.py` to re-issue cookies into each worker's folder via the
`WEBARENA_AUTH_FOLDER=<auth_root>/w{w}` env var that `run._apply_worker_env`
sets per worker.

### Step 7 — Launch the orchestrator

```bash
python -m mp.orchestrator \
    --config mp/configs/config-tsa.json \
    --start_idx 0 --end_idx 812 \
    --provider openai --mode chat --model "$AGENT_MODEL_NAME" \
    --inference_backend tsa \
    --temperature 0 --top_p 1 --max_tokens 2048 \
    --max_steps 30 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
    2>&1 | tee /z/wangcy07/webarena-mp/results-tsa/orchestrator.log
```

Results land at `<result_dir>/scores.jsonl` (append-only, one JSON object per
line). Re-running against the same `result_dir` skips task_ids whose row has
both a non-null `score` AND a null `error` (see `mp/orchestrator.py:62`). Rows
with `error=null` but missing/null `score` will be retried.

### Step 8 — Monitor progress (second shell)

```bash
RESULTS=/z/wangcy07/webarena-mp/results-tsa
while sleep 30; do
    n=$(wc -l < "$RESULTS/scores.jsonl" 2>/dev/null || echo 0)
    ok=$(grep -c '"error": null' "$RESULTS/scores.jsonl" 2>/dev/null || echo 0)
    last=$(tail -n1 "$RESULTS/scores.jsonl" 2>/dev/null | python3 -c \
        'import json,sys; r=json.loads(sys.stdin.read()); print(f"task={r[\"task_id\"]} score={r[\"score\"]}")' \
        2>/dev/null || echo none)
    printf "[%s] done=%d ok=%d  last=%s\n" "$(date +%H:%M:%S)" "$n" "$ok" "$last"
done

# Backend log on gray:
ssh wangcy07@158.130.4.227 'tail -F ~/.cache/wa_tsa.log'   # or wa_dense.log / wa_judge.log
```

### Step 9 — Tear down between TSA and dense runs

Sequential single-GPU schedule (16 GB):

```bash
bash mp/teardown_inference.sh --tsa-only --keep-judge    # judge stays warm
source mp/launch_dense.sh
# … run dense orchestrator …
bash mp/teardown_inference.sh                            # final cleanup
```

On B200, skip teardown between runs — both backends coexist.

```bash
python -m mp.benchmark_compare \
    --tsa   /z/wangcy07/webarena-mp/results-tsa/scores.jsonl \
    --dense /z/wangcy07/webarena-mp/results-dense/scores.jsonl \
    --tasks config_files/test.raw.json \
    --out comparison_report.md --csv comparison.csv
```

---

## 4. Troubleshooting

⚠️ **OOM during prefill** (`wa-tsa` log: `CUDA out of memory` on first request)
→ prompts exceeded the prefill budget. Lower before `source mp/launch_tsa.sh`:
```bash
export TSA_MAX_BATCH=1            # 2 → 1 (sm_120) or 16 → 8 (sm_100)
export TSA_MAX_DECODE_TOKENS=512  # 1024 → 512
source mp/launch_tsa.sh
```
Confirm peak with `nvidia-smi --query-gpu=memory.used --format=csv,noheader`.

⚠️ **"Welcome, please sign in" loop on every Magento task** → stale cookies
OR Magento store corruption. Try the auth wipe first (§3 step 6). If the loop
persists and the error log shows `Magento\Framework\Exception\NoSuchEntityException`
at `vendor/magento/module-store/Model/StoreRepository.php:112`, the store row
was lost from `core_config_data`. Restore from golden via `mp.reset`:
```bash
python -c "
from mp.config import load_config
from mp.docker_exec import DockerClient
from mp.reset import reset_site
cfg = load_config('mp/configs/config-tsa.json')
client = DockerClient(docker_host=cfg.docker_host, ssh_host=cfg.ssh_host)
for w in range(cfg.num_workers):
    reset_site('shopping', w, cfg, client)
    reset_site('shopping_admin', w, cfg, client)
"
```
Replays `golden.sql` and re-runs `setup:store-config:set`.

⚠️ **Tunnel orphans after teardown** — `autossh -M 0` does **not** kill its
child `ssh` when the parent dies (audit finding). `mp/teardown_inference.sh`
runs `kill_tunnel` which uses `pgrep -f 'ssh.*-L 127.0.0.1:PORT:127.0.0.1:'`
+ per-pid `kill`, but if the regex misses (e.g. a manual `ssh -L` without
`127.0.0.1:` prefix), the tunnel leaks. Manual cleanup:
```bash
pgrep -af 'ssh.*-L.*1000[012]'   # find leaks
kill <pid>                       # one at a time; NEVER use pkill -f here —
                                 # the regex can hit the shell running it.
rm -f /tmp/wa_tunnel_1000*.pid
```

⚠️ **`openai` 0.x vs 1.x mismatch** — `llms/providers/openai_utils.py` targets
0.27's `openai.ChatCompletion.create()`. Upgrading to 1.x throws
`AttributeError: module 'openai' has no attribute 'ChatCompletion'` on every
call. Either pin (`pip install 'openai<1'`) or rewrite `openai_utils.py` to
`from openai import OpenAI; client.chat.completions.create(...)`. Migration
hits the four entry points (note the async siblings use an `a`-prefix, not
`_async` suffix): `generate_from_openai_completion`,
`generate_from_openai_chat_completion`, `agenerate_from_openai_completion`,
`agenerate_from_openai_chat_completion`. Keep `_strip_reasoning`
wrapping the response either way (reference commit shows the shape).

⚠️ **Task scored 0 but `error` is null** — usually `action_parse_failed`: the
model emitted a malformed `goto` / `click` / `type` that
`browser_env.actions.create_id_based_action` rejected. Counts as a legitimate
failure, not an exception. Check the TSA server log:
```bash
ssh wangcy07@158.130.4.227 'grep -iE "parse|invalid|malformed|goto" ~/.cache/wa_tsa.log | tail -30'
```
Common culprits: extra ```` ``` ```` fences (set `TSA_STRIP_FENCES=1` before
sourcing `launch_tsa.sh`); `goto[https://…]` instead of `goto [https://…]`
(prompt-format drift — confirm `--instruction_path` points at the canonical
`agent/prompts/jsons/p_cot_id_actree_2s.json`).

⚠️ **`ModuleNotFoundError: browser_env` from a worker subprocess** —
`mp.orchestrator` spawns workers via `multiprocessing.get_context("spawn")`
(`mp/orchestrator.py:90`). Spawn-context workers re-exec `python` and rebuild
`sys.path` from scratch, so any in-process `sys.path.insert(...)` done in the
parent shell is lost. Neither `run.py:_apply_worker_env` nor
`mp/worker.py:_apply_env` injects `PYTHONPATH` for the worker itself; the only
PYTHONPATH manipulation in `run.py` (lines ~340 and ~495) builds a local
`_env` dict scoped to the `auto_login.py` subprocess and never mutates the
worker's own `os.environ`. To avoid the error in the worker, **launch
`python -m mp.orchestrator` from the repo root** (cwd is on `sys.path`) **or
export `PYTHONPATH=<repo_root>` in the parent shell** before invoking it.

(Legacy explanation — kept for searchability:)
`mp.orchestrator` spawns workers via `multiprocessing.get_context("spawn")`,
which does NOT inherit `PYTHONPATH`. Fix lives in `run.py`: the worker
bootstrap injects `PYTHONPATH=<repo_root>` into `os.environ` before importing
`browser_env`. Verify with `ps eww -p <worker_pid> | tr ' ' '\n' | grep PYTHONPATH`.
If missing, launch the orchestrator from the repo root
(`cd /z/wangcy07/webarena-repo`) so `run.py:_apply_worker_env` resolves it.

---

## 5. Task-id list curation

Three reference lists live alongside the runbook (curate to your benchmark
target):

| File | Count | Purpose |
|---|---|---|
| `readonly_task_ids.txt` | ~380 | All tasks whose `sites` are subsets of `(map, wikipedia, homepage, shopping, shopping_admin, reddit, gitlab)` *and* whose template doesn't mutate state. Auth is still required for shopping/shopping_admin/reddit/gitlab — auth wipe (§3 step 6) still applies. |
| `sglang_passed_task_ids.txt` | 42 | Known-good against the dense SGLang baseline. Use as the smoke set. |
| `valid_task_ids.txt` | ~260 | Tasks whose config_file passes JSON validation and reaches a non-error scoring path on at least one previous run. |

### 5.1 Filter to a single working site (pure Python, no jq)

When one site is broken (e.g. GitLab puma OOMed), filter `test.raw.json` to
just the working sites:

```python
import json, sys
WORKING = {"shopping", "shopping_admin", "reddit", "map", "wikipedia", "homepage"}
with open("config_files/test.raw.json") as f:
    raw = json.load(f)
ids = [t["task_id"] for t in raw if set(t["sites"]).issubset(WORKING)]
print(",".join(str(i) for i in ids))
# Feed back into the orchestrator:
# python -m mp.orchestrator --task_ids 0,1,3,7,...  --config ...
```

Or the inverse — list tasks that *only* touch one site:

```python
import json
SITE = "shopping"
raw = json.load(open("config_files/test.raw.json"))
ids = [t["task_id"] for t in raw if t["sites"] == [SITE]]
open("shopping_only_task_ids.txt", "w").write(",".join(map(str, ids)))
```

Pass with `--task_ids "$(cat shopping_only_task_ids.txt)"`.

---

## 6. Bringing up a B200 host

B200 (sm_100, 141 GB HBM) removes coexistence constraints. Operational deltas:

1. **Override profile detection** — skip the SSH round-trip in `gpu_profile.sh`:
   ```bash
   export GPU_HOST=wangcy07@<b200-host>
   export GPU_SM_HINT=100
   ```
   Selects the sm_100 row: `TSA_MAX_BATCH=16`, `TSA_MAX_DECODE_TOKENS=2048`,
   `MEM_FRAC_AGENT=0.30`, `MEM_FRAC_JUDGE=0.20`, `AGENT_CTX_LEN=32768`,
   `JUDGE_CTX_LEN=8192`, `WEBARENA_NUM_WORKERS=8`, `JUDGE_FITS=yes`,
   `CONCURRENT_BACKENDS=yes`, `SGLANG_BACKEND=flashinfer`.

2. **Use the B200-scaled config files** — `mp/configs/config-tsa-b200.json` and
   `mp/configs/config-dense-b200.json` ship with `num_workers=8` and distinct
   `result_dir`s (`results-tsa-b200`, `results-dense-b200`) so they coexist
   with sm_120 runs on the same hilbit2 deployment. Pass them via `--config`.

3. **Drop §1.3 VRAM-coexistence concerns** — TSA + dense + judge all coexist.
   Skip the `--keep-judge` teardown dance between runs.

4. **Use the 4B judge** for symmetric scoring (no asymmetric-judge caveat):
   ```bash
   export JUDGE_MODEL_PATH='$HOME/hf_models/Qwen3-VL-4B-Instruct'
   source mp/launch_judge.sh
   ```

5. **Scale `TSA_MAX_BATCH`** — default 16. Push to 32 on a dedicated B200;
   re-verify peak with `nvidia-smi dmon -s u`.

6. **Concurrent backends in flight** — `CONCURRENT_BACKENDS=yes` means a single
   B200 can run TSA + dense **simultaneously** against different `--config`s
   (different `result_dir`, `auth_root`, replica set). The sm_120 sequential
   workaround is unnecessary.

7. **N=5 is the sm_120 floor; N=8 is the B200 default** — empirically validated
   on sm_120, conservatively scaled on sm_100. Push higher with these checks:
   - `nvidia-smi --query-gpu=memory.used --format=csv,noheader` ≤ 80 GB peak
   - TSA scheduler log shows `Collected batch of N` matching the expected
     concurrency (each `mp.orchestrator` worker counts once)
   - No `OutOfMemoryError` in `~/.cache/wa_{tsa,dense}.log`

---

## Quick reference — ports

| Port | Service | tmux session | log on gray |
|---|---|---|---|
| 10000 | TSA agent | `wa-tsa` | `~/.cache/wa_tsa.log` |
| 10001 | SGLang dense agent | `wa-dense` | `~/.cache/wa_dense.log` |
| 10002 | SGLang judge | `wa-judge` | `~/.cache/wa_judge.log` |
| 7770 / 7780 / 9999 / 8023 | Magento / Magento-admin / Postmill / GitLab (worker_0) | n/a | docker logs |
| 13000 / 8888 / 4399 | map / wikipedia / homepage (shared) | n/a | docker logs |

## Quick reference — files

| File | Purpose |
|---|---|
| `mp/launch_judge.sh` | Boot SGLang judge in tmux + tunnel :10002. |
| `mp/launch_tsa.sh` | Boot TSA in tmux + tunnel :10000 (and judge as pre-step). |
| `mp/launch_dense.sh` | Boot SGLang dense in tmux + tunnel :10001 (and judge as pre-step). |
| `mp/teardown_inference.sh` | Kill tmux sessions + tunnels. Flags: `--keep-judge`, `--tsa-only`, `--dense-only`. |
| `mp/_inference_common.sh` | Tunnel + tmux helpers; default `GPU_HOST`. |
| `mp/configs/gpu_profile.sh` | Per-SM tuning table. |
| `mp/configs/config-tsa.json` | MPConfig for the TSA run (`result_dir=…/results-tsa`). |
| `mp/configs/config-dense.json` | MPConfig for the dense run (`result_dir=…/results-dense`). |
| `mp/benchmark_compare.py` | Read both `scores.jsonl`, emit markdown + CSV. |
| `mp/.inference_env` | Auto-generated env file (sourced for fresh shells). |
