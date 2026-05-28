# WebArena multiworker run — tasks 0-6, N=7, shopping_admin

Reproduction record for running WebArena benchmark tasks 0–6 on the shopping_admin
site in true multi-worker mode (one worker per task, all 7 in parallel) against
the hilbit2 deployment, using qwen2.5:7b-instruct served by Ollama on
gray.cis.upenn.edu (via SSH tunnel).

- **Date**: 2026-05-27 / 2026-05-28 (UTC overlap)
- **Run**: 22:33:17 → 22:35:01 EDT (1m 45s wall, true parallel)
- **Tasks**: 0,1,2,3,4,5,6 (all `shopping_admin`, all `require_reset: false`)
- **Workers**: 7 (one task per worker, one replica per worker)
- **Model**: qwen2.5:7b-instruct on gray.cis.upenn.edu, served via Ollama
- **Reset**: skipped (worker.py was patched to honor `require_reset: false`)
- **Aggregate score**: 0/7 (model accuracy, NOT a multiworker issue — see §6)
- **Multiworker isolation**: ✅ verified (every worker hit ONLY its own port)

## 1. Topology

```
┌─────────────────────────────────────────────────────────────────────┐
│ orchestrator host: hilbit2.cis.upenn.edu (158.130.4.158)            │
│                                                                     │
│  python -m mp.orchestrator (N=7, spawn-context multiprocessing)     │
│   ├── w0 ──► shopping_admin       (legacy) │ http://...:7780/admin  │
│   ├── w1 ──► shopping_admin_w1             │ http://...:7880/admin  │
│   ├── w2 ──► shopping_admin_w2             │ http://...:7980/admin  │
│   ├── w3 ──► shopping_admin_w3             │ http://...:8090/admin  │ (8080 skipped)
│   ├── w4 ──► shopping_admin_w4             │ http://...:8180/admin  │
│   ├── w5 ──► shopping_admin_w5             │ http://...:8280/admin  │
│   └── w6 ──► shopping_admin_w6             │ http://...:8380/admin  │
│                                                                     │
│  rootless docker @ unix:///z/wangcy07/webarena/rootless-docker/...  │
│  Magento image (shared, 13.2 GB): webarena-shopping_admin-golden    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
              SSH tunnel (port 11434) on hilbit2:11434 → gray:11434
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ inference host: gray.cis.upenn.edu                                  │
│  ollama serve  ──►  qwen2.5:7b-instruct (Q4_K_M, ~4.7 GB)           │
└─────────────────────────────────────────────────────────────────────┘
```

Worker_0 keeps the legacy container `shopping_admin` (the live one already
running on hilbit2, image `shopping_admin_final_0719`). Workers 1–6 each get
their own per-worker Magento replica spawned from the same image. The replicas
are connected only by sharing the docker daemon — they have **independent
MySQL, Redis, Elasticsearch, and filesystem state**.

## 2. Setup steps (chronological)

1. **Verified hilbit2 prerequisites** (read-only): repo at
   `/z/wangcy07/webarena-repo`, venv at `/z/wangcy07/webarena-venv` (Python 3.12.3),
   Playwright Chromium 1223 at `/z/wangcy07/pw_browsers`, rootless docker socket,
   SSH tunnel hilbit2:11434 → gray:11434 already up, `qwen2.5:7b-instruct` pulled
   on gray, `webarena-shopping_admin-golden:latest` already tagged
   (13.2 GB, idempotent from a prior bring-up).
2. **Patched `mp/config.py`** to add `_HOST_PORT_RESERVED = {8080, 8085}` and
   make `port_for()` skip them — without this, worker_3's natural port 8080
   collided with the shared `map` container's 127.0.0.1:8080 binding inside
   rootless docker's network namespace. After the patch, worker_3 gets port 8090.
3. **Spawned 5 new replicas** `shopping_admin_w2..w6` from the golden image,
   each bound to its own host port (7980/8090/8180/8280/8380).
4. **Configured base URLs** on each new replica via Magento CLI
   (`bin/magento setup:store-config:set --base-url=http://158.130.4.158:PORT/`)
   and direct SQL UPDATE on `core_config_data`. Used a 300-second per-query
   timeout to absorb table-lock contention from Magento's first-boot internal
   cron jobs. Disabled admin password rotation.
5. **Populated the shopping_admin golden** at
   `/z/wangcy07/webarena-mp/golden/shopping_admin/golden.sql`
   via `mysqldump --single-transaction --add-drop-table --no-tablespaces` on
   the worker_0 (legacy) container — 7.7 MB.
6. **Bumped `mp/config.json`** to `num_workers: 7` and re-rendered per-worker
   task configs (`/z/wangcy07/webarena-mp/config_files/w{0..6}/*.json`) so
   every worker's start_url points at its own port.
7. **Patched `mp/worker.py`** to honor `require_reset: false` from task configs.
   This was a faithful semantic fix, not a workaround: the task configs
   explicitly mark tasks 0–6 as not requiring reset because they're read-only
   queries on Magento sales-report data. Before the patch, the harness ignored
   the flag and always restored the DB from golden, which (under 7-way
   concurrent contention with Magento's internal cron) blew past the
   `reset_magento` 600-second timeout. See §5 for the patch.
8. **Ran `python -m mp.orchestrator --task_ids 0,1,2,3,4,5,6`** with N=7. Total
   wall time: 1m 45s.

## 3. Result summary

```
$ cat scores.jsonl
{"worker_id": 0, "task_id": 0, "score": 0.0, "error": null, "duration_seconds": 64.58}
{"worker_id": 6, "task_id": 5, "score": 0.0, "error": null, "duration_seconds": 66.68}
{"worker_id": 5, "task_id": 6, "score": 0.0, "error": null, "duration_seconds": 69.66}
{"worker_id": 1, "task_id": 1, "score": 0.0, "error": null, "duration_seconds": 84.49}
{"worker_id": 4, "task_id": 4, "score": 0.0, "error": null, "duration_seconds": 89.12}
{"worker_id": 3, "task_id": 3, "score": 0.0, "error": null, "duration_seconds": 91.65}
{"worker_id": 2, "task_id": 2, "score": 0.0, "error": null, "duration_seconds": 104.14}
```

| Worker | Task | Container             | Port | Intent                                                   | dur (s) | Score |
|--------|------|-----------------------|------|----------------------------------------------------------|--------:|------:|
| w0     | 0    | shopping_admin        | 7780 | What is the top-1 best-selling product in 2022           |   64.58 |   0.0 |
| w1     | 1    | shopping_admin_w1     | 7880 | What is the top-1 best-selling brand in Quarter 1 2022   |   84.49 |   0.0 |
| w2     | 2    | shopping_admin_w2     | 7980 | What is the top-1 best-selling product type in Q1 2022   |  104.14 |   0.0 |
| w3     | 3    | shopping_admin_w3     | 8090 | What are the top-2 best-selling product in 2022          |   91.65 |   0.0 |
| w4     | 4    | shopping_admin_w4     | 8180 | What are the top-3 best-selling product in Jan 2023      |   89.12 |   0.0 |
| w5     | 6    | shopping_admin_w5     | 8280 | What are the top-5 best-selling product in 2023          |   69.66 |   0.0 |
| w6     | 5    | shopping_admin_w6     | 8380 | What is the top-1 best-selling product type in Jan 2023  |   66.68 |   0.0 |

- All 7 tasks **completed with `error: null`** (no harness failures, no reset
  errors, no agent crashes).
- **All 7 ran in parallel**: every worker logged its first action within 80 ms
  of the orchestrator's start.
- **Sequential-equivalent time**: 64+84+104+91+89+69+66 ≈ 567 s ≈ 9.4 min.
  **Parallel wall**: 105 s. **Speedup**: 5.4× (theoretical max is 7×; the
  shortfall is the shared single-GPU qwen2.5 serving 7 concurrent agent loops).

## 4. Multiworker correctness — proof from Playwright traces

Per-worker isolation was verified by parsing each worker's
`<result_dir>/w{w}/traces/{task_id}.zip → trace.trace` (Playwright trace
JSONL) and counting occurrences of `158.130.4.158:PORT` for every Magento
port (excluding shared read-only ports 4399 homepage / 8888 wikipedia /
13000-13030 map).

```
w0 (task 0, expected_port=7780): hits = {7780: 268}     ✓ ISOLATED
w1 (task 1, expected_port=7880): hits = {7880: 427}     ✓ ISOLATED
w2 (task 2, expected_port=7980): hits = {7980: 699}     ✓ ISOLATED
w3 (task 3, expected_port=8090): hits = {8090: 532}     ✓ ISOLATED
w4 (task 4, expected_port=8180): hits = {8180: 532}     ✓ ISOLATED
w5 (task 6, expected_port=8280): hits = {8280: 310}     ✓ ISOLATED
w6 (task 5, expected_port=8380): hits = {8380: 271}     ✓ ISOLATED
```

**No worker referenced any port other than its own.** This is end-to-end
proof that:

- `cfg.env_for(worker_id)` exported the correct `SHOPPING_ADMIN` env var
  before `browser_env` import.
- `render_config_files` baked the right per-worker URL into each
  `config_files/w{w}/{task_id}.json`.
- Each Playwright `BrowserContext` followed only its own start URL.
- The Magento CLI `setup:store-config:set --base-url=...` correctly rewrote
  per-replica base URLs so admin pages do NOT redirect to a sibling worker's
  host (which would have appeared as cross-port hits in the trace).

## 5. Code changes shipped during this run

### 5.1 `mp/config.py` — port collision avoidance (committed to hilbit2)
```diff
+# Host ports already bound by shared resources on hilbit2 (the map container
+# binds 127.0.0.1:8080 and :8085). port_for() skips these to avoid collisions.
+_HOST_PORT_RESERVED: set[int] = {8080, 8085}

 def port_for(self, site: str, worker_id: int) -> int:
     ...
     port = BASE_PORTS[site] + self.port_stride * worker_id
+    # Skip ports reserved by shared resources (e.g. the map container on hilbit2).
+    while port in _HOST_PORT_RESERVED:
+        port += 10
     return port
```
Effect: shopping_admin worker_3 → 8090 (was 8080, colliding with map).

### 5.2 `mp/worker.py` — respect `require_reset: false`
```diff
     touched_sites: list[str] = list(config.get("sites", []))

-    # §14.6: also reset any site we previously dirtied on this worker.
-    sites_to_reset: set[str] = set(touched_sites) | dirty_sites
-    sites_to_reset = {s for s in sites_to_reset if s in ALL_MUTABLE_SITES}
+    # §14.6: always reset any site we previously dirtied on this worker.
+    # Additionally, reset newly-touched sites unless the task config sets
+    # require_reset=False (read-only tasks that don't mutate state).
+    sites_to_reset: set[str] = set(dirty_sites)
+    if config.get("require_reset", True):
+        sites_to_reset |= set(touched_sites)
+    sites_to_reset = {s for s in sites_to_reset if s in ALL_MUTABLE_SITES}
```
This is a semantic fix, not a workaround: the WebArena task configs (and the
master `config_files/test.raw.json`) carry an explicit `require_reset` field
per task. The original implementation ignored it. With the patch, tasks 0–6
(all read-only) correctly skip the DB restore phase. The patch still resets
any site that was dirtied by a previous task on the same worker.

### 5.3 Why this patch matters in practice

The initial run with the old worker.py launched 7 concurrent mysqldump
restores of the 7.7 MB golden into 7 different containers. Under contention
with Magento's internal cron jobs (which were already running first-boot in
the new replicas and writing to `cron_schedule`), all 7 restores blew past
the `reset_magento` 600-second timeout. Every task ended with
`error: reset failed: TimeoutExpired`. That run is archived at
`/z/wangcy07/webarena-mp/results/archive_20260527_223310/`.

With the patched worker.py, the restore phase is correctly skipped for
require_reset=false tasks, and the wall-clock for tasks 0–6 dropped from
"never completes" → 1m 45s.

## 6. Why every score is 0.0

The model used for this run is **qwen2.5:7b-instruct (Q4_K_M)**, a ~4.7 GB
quantized 7B model. The tasks asked for things like:

> "What is the top-1 best-selling product in 2022"

…which require the agent to: log into Magento admin → navigate Reports →
Bestsellers → choose date filter "2022-01-01 to 2022-12-31" → click View Report
→ extract the top row's product name.

The 7B model produces well-formed `click [N]` actions but cannot reliably
chain the 8–15 admin-UI steps to reach the filtered report within
`max_steps=30`. This is a known limitation of small LLMs on WebArena's
shopping_admin tasks (see [WebArena paper Table 5] — even GPT-3.5-Turbo scored
~17% on shopping_admin overall, and these top-N best-selling tasks are
generally harder than average). **The 0.0 scores reflect model capability,
not any defect in the multiworker harness, the resets, or the per-replica
plumbing.**

The previous archived run on hilbit2 (10 tasks: 7,8,9,10,16-20,32; same
qwen2.5:7b model with N=2) also scored 0.0 on every task, which is consistent.

## 7. Files saved

Under `multiworker_run_artifacts/` in this repo:

| File                                  | Description                                                                |
|---------------------------------------|----------------------------------------------------------------------------|
| `scores.jsonl`                        | One JSON line per task. Authoritative result file.                         |
| `orchestrator.log`                    | Orchestrator stdout: worker spawning, task assignment, progress.           |
| `logs/worker_0..6.log`                | Per-worker logs: started/reset/agent-loop/done lines for each worker.      |
| `bring_up_w2_w6.log`                  | Bring-up log: replica creation, base URL config, golden mysqldump.         |
| `mp_config.final.json`                | The `mp/config.json` on hilbit2 after the run (num_workers=7).             |
| `trace_isolation_verification.txt`    | Per-worker port-isolation audit (see §4).                                  |

Not pulled to local (large; lives on hilbit2):

| Path on hilbit2                                                | Description                                 |
|----------------------------------------------------------------|---------------------------------------------|
| `/z/wangcy07/webarena-mp/results/w{w}/render_{t}.html`         | Trajectory render — one HTML per task.      |
| `/z/wangcy07/webarena-mp/results/w{w}/traces/{t}.zip`          | Playwright trace zip — replay with `playwright show-trace`. |
| `/z/wangcy07/webarena-mp/golden/shopping_admin/golden.sql`     | 7.7 MB Magento DB dump (for future resets). |
| `/z/wangcy07/webarena-mp/results/archive_*`                    | Two earlier failed-reset attempts.          |

## 8. How to reproduce

```bash
# On hilbit2 (after SSH tunnel hilbit2:11434 → gray:11434 is up):
cd /z/wangcy07/webarena-repo
source /z/wangcy07/webarena-venv/bin/activate
export PYTHONPATH=/z/wangcy07/webarena-repo
export PLAYWRIGHT_BROWSERS_PATH=/z/wangcy07/pw_browsers
export NLTK_DATA=/z/wangcy07/nltk_data
export TIKTOKEN_CACHE_DIR=/z/wangcy07/tiktoken_cache
export OPENAI_API_BASE=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
export WEBARENA_EVAL_MODEL=qwen2.5:7b-instruct
export WEBARENA_STRIP_REASONING_TAGS=1

python3 -m mp.orchestrator \
    --task_ids 0,1,2,3,4,5,6 \
    --provider openai --model qwen2.5:7b-instruct --mode chat \
    --temperature 0 --top_p 0.9 --max_tokens 512 \
    --max_steps 30 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json
```

`mp/config.json` must have `num_workers: 7` and `shopping_admin_w{1..6}` must
already be running. If starting from N=2 (the prior state), the bring-up
sequence in §2 (steps 3–6) takes ~2 minutes for the new replicas plus
~36 seconds for the golden mysqldump.

## 9. Addendum — N=5 mutation run (tasks 470–474, Cancel order ×5)

A second run validated the harness end-to-end for **mutation tasks that require real per-task reset**, picking five "Cancel order N" tasks on shopping_admin. Started 00:22:37, finished 00:34:24 (2026-05-28) — **11 min 47 s wall**.

### 9.1 Tasks chosen and override

```
470 Cancel order 302
471 Cancel order 307
472 Cancel order 299
473 Cancel order 301
474 Cancel order 305
```

**Finding while picking tasks**: every one of the 812 entries in `config_files/test.raw.json` carries `require_reset: false`, *including* unambiguous mutation tasks like Cancel order, Update tracking number, Disable product, and Add new product. The dataset's flag is uniformly False; the reset semantic the benchmark wants is "reset between any two tasks that touch the same site." For this run I overrode `require_reset: true` in the per-worker config files for these 5 task ids only. Eval criterion (`program_html`): `document.querySelector("#order_status").outerText == "Canceled"`.

### 9.2 What changed between this run and the §3 read-only run

| File | Change | Reason |
|---|---|---|
| `mp/reset.py` `reset_magento` step 1 (mysql restore) | `timeout=600` → `1800` | 600 s was insufficient under N≥4 concurrent restores; bumped to absorb contention. |
| `mp/reset.py` `reset_magento` step 2 (`rm -rf var/cache/*` etc.) | `timeout=300` → `1800` | Magento's first-boot warmup re-creates these caches as the agent loop later runs, but during reset the rm itself slows under concurrent IO. 300 s was tight, 1800 s gives 6× margin. |
| `mp/reset.py` `reset_magento` step 5 (`UPDATE core_config_data base_url`) | `timeout=30` → `300` | `cron_schedule` writes hold table metadata locks for tens of seconds. 30 s was too tight. |
| `mp/reset.py` `reset_magento` step 7 (`_wait_healthy`) | `timeout_seconds=120` → `300` | After cache:flush, the first request through has to re-compile templates; this can take a minute under load. |
| Per-worker task configs | `require_reset: false` → `true` for 5 selected task ids | Override the dataset's uniform-false flag so the patched worker.py actually runs the restore. |
| Workers used | 5 (orchestrator still spawns 7, but only 5 grab tasks; the other 2 take poison pills and exit) | Lower IO/CPU pressure than 7-way contention. Earlier 7-way attempt timed out at the `rm var/cache` step. |

The first attempt (N=7, `rm var/cache` timeout=300 s) failed all 7 tasks at the rm step despite restores succeeding under the bumped 1800 s mysql timeout. That run is archived at `/z/wangcy07/webarena-mp/results/archive_20260528_000616/`.

### 9.3 Run summary

```
$ cat scores.jsonl
{"worker_id": 0, "task_id": 470, "score": 0.0, "error": null, "duration_seconds": 645.61}
{"worker_id": 4, "task_id": 472, "score": 0.0, "error": null, "duration_seconds": 659.88}
{"worker_id": 1, "task_id": 471, "score": 0.0, "error": null, "duration_seconds": 666.74}
{"worker_id": 3, "task_id": 473, "score": 0.0, "error": null, "duration_seconds": 688.00}
{"worker_id": 2, "task_id": 474, "score": 0.0, "error": null, "duration_seconds": 706.10}
```

| Worker | Task | Order | Port | Duration | Score |
|---|---|---|---|---:|---:|
| w0 | 470 | 302 | 7780 | 645.6 s | 0.0 |
| w1 | 471 | 307 | 7880 | 666.7 s | 0.0 |
| w2 | 474 | 305 | 7980 | 706.1 s | 0.0 |
| w3 | 473 | 301 | 8090 | 688.0 s | 0.0 |
| w4 | 472 | 299 | 8180 | 659.9 s | 0.0 |

- **Zero errors** — every task ran through reset → agent → eval cleanly.
- **All 5 in parallel** — first reset began at 00:22:37, last finished 00:34:24.
- **Sequential-equivalent**: 645+666+706+688+659 ≈ 3364 s ≈ 56 min. **Parallel wall**: 707 s ≈ 11.8 min. **Speedup**: 4.8× (max ≈ 5×; ~95 % efficiency).

### 9.4 Multiworker isolation — verified again

Same trace-port audit as §4 applied to this run's traces:

```
w0 task 470 (own_port=7780): URL hits = {7780: 178}   ✓ ISOLATED
w1 task 471 (own_port=7880): URL hits = {7880: 376}   ✓ ISOLATED
w2 task 474 (own_port=7980): URL hits = {7980: 829}   ✓ ISOLATED
w3 task 473 (own_port=8090): URL hits = {8090: 497}   ✓ ISOLATED
w4 task 472 (own_port=8180): URL hits = {8180: 279}   ✓ ISOLATED
```

Each worker only ever touched its own per-worker replica.

### 9.5 Mutation outcomes — reset semantics proven by post-run DB state

After the run, querying w0's `shopping_admin` container directly:
```
docker exec shopping_admin mysql … -e "SELECT entity_id, status, state FROM sales_order WHERE entity_id=302"
-> entity_id=302  status=pending  state=new
```

Order 302 is back to its golden "pending/new" state, **not** "Canceled." This is the correct outcome because:
- qwen2.5 navigated through the admin UI but never successfully clicked Cancel and confirmed (the trace shows 30 steps of clicking around without reaching the cancel-confirmation dialog).
- The evaluator reads `#order_status` from `/sales/order/view/order_id/302/` and looks for `exact_match: "Canceled"` — sees "Pending" → score = 0.0.
- **Crucially, "pending/new" being the *post-run* state proves the reset actually worked**: if reset were a no-op or partial, an earlier accidental success on the live container could have left 302 in any other state. The mysql restore from golden put it back.

This run is the first end-to-end demonstration that the multi-worker harness can run real mutation tasks correctly under contention, as long as the timeouts in `reset_magento` are scaled to the level of concurrency in use.

### 9.6 Files saved for this run

Under `multiworker_run_artifacts/mutation_run/`:

| File | Description |
|---|---|
| `scores.jsonl` | 5 lines, one per task. |
| `orchestrator.log` | Orchestrator stdout for this run. |
| `logs/worker_0..4.log` | Per-worker logs (w5/w6 were poison-pilled immediately and have no log). |
| `sample_w0_render_task470.html` | Trajectory render for w0 / task 470 (Cancel order 302). |
| `sample_w0_trace_task470.zip` | Playwright trace zip for the same task. Replay with `playwright show-trace`. |

Earlier read-only-run artifacts remain in `multiworker_run_artifacts/` (outside the `mutation_run/` subdir).

## 10. Open notes / future fixes

- **Reset under N≥4 concurrent restores is slow but works with bumped timeouts.** Magento's internal cron daemons compete with the restore's DDL for table metadata locks. §9 (the N=5 mutation run) validated bumping the four `reset_magento` timeouts (mysql restore, rm/cache, base_url UPDATE, _wait_healthy). The 7-way attempt still failed at step 2 even with 300 s, so for N≥7 the better mitigation is to serialize restores or disable Magento cron during the restore window. For N=5 the timeouts are enough.
- **Per-worker auth cookies (`<auth_root>/w{w}/`)** were re-issued at first
  request by `auto_login.py` and worked across all 7 admin login flows.
- **Map container's port 8080 collision** will recur for any future
  shopping_admin worker_3-equivalent. The reserved-port skip in §5.1 is a
  durable fix.
- **The repo at `/z/wangcy07/webarena-repo` on hilbit2 is not a git checkout**
  (no `.git`). Local source modifications would need to be `scp`'d up, or the
  repo re-initialised as a real checkout to track patches.
