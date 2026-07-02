# WebArena multi-worker harness — user guide

Run the WebArena 812-task benchmark with N parallel workers, with per-task state reset between tasks so the score is comparable to a single-worker run.

This guide is the operator-facing companion to:
- [MULTIWORKER_PLAN.md](../MULTIWORKER_PLAN.md) — design (architecture, state surfaces, reset semantics, §14 audit findings).
- [SMOKE_RUNBOOK.md](SMOKE_RUNBOOK.md) — terse first-bring-up runbook.
- [README.md](README.md) — module reference for `mp/`.

The appendix at the end of this document enumerates every change shipped for multi-worker support.

## 1. Mental model

### 1.1 What multi-worker means here

* **Worker** = a Python process that owns one Playwright Chromium and runs WebArena tasks serially. Workers do not share state.
* **Replica** = a docker container per (site, worker). Worker `w` connects to its own replicas of the four mutable sites: `shopping_w{w}`, `shopping_admin_w{w}`, `forum_w{w}`, `gitlab_w{w}`. Worker 0 reuses the legacy single-instance container names (`shopping`, `forum`, `gitlab`, `shopping_admin`) so an existing deployment is adopted as worker_0 without renaming.
* **Read-only sites** (map, wikipedia, homepage) are **shared** across workers — no per-worker replica. Verified safe because tasks on these sites are navigation/lookup only.
* **Reset** = a function `reset_site(site, worker_id, cfg, client)` that restores a single replica to its golden state. Called before every task that touches that site, **unless** the task config carries `"require_reset": false` (read-only tasks like sales-report queries) AND the worker has no dirty sites carried over from a previous task. Tasks that mutate state always reset.

### 1.2 What the harness guarantees

| Invariant | Where enforced |
|---|---|
| Each task observes the same initial backend state a serial run would | `reset_site` (§4/§14 of the plan); verified with `mp.verify_golden`. For tasks marked `"require_reset": false` we additionally trust the task author's declaration that the task is read-only — `mp/worker.py:_run_one_task` skips reset for these unless the worker has dirty sites left over. |
| No two workers ever touch the same replica | `cfg.container_for(site, worker_id)` is injective in `worker_id` for mutable sites |
| Per-worker URL routing — agent and evaluator both see worker_id's URLs | `cfg.env_for(worker_id)` exported before `browser_env.env_config` is imported; per-worker `config_files/w{w}/{i}.json` rendered with worker URLs at bring-up |
| Per-worker cookies | `WEBARENA_AUTH_FOLDER=<auth_root>/w{w}` exported per worker (`run.py:_apply_worker_env`); `auto_login.py` writes there only |
| One BrowserContext per task | `ScriptBrowserEnv.setup` (existing); each worker has its own `ScriptBrowserEnv` |
| Env-URL ↔ worker_id invariant asserted at worker start | `mp/worker.py:_assert_env_matches_worker` |

## 2. Prerequisites

### 2.1 Server requirements

For N=8 workers (recommended), needs roughly:

| Resource | Need | Why |
|---|---|---|
| CPU cores | ≥ 16 | GitLab + Magento are CPU-heavy; one Playwright per worker |
| RAM | ≥ 64 GB | GitLab ~6 GB × N + Magento ~1 GB × 2N + ollama/inference if local |
| Disk | ≥ 250 GB free | Each GitLab CoW layer ~24 GB; goldens ~22 GB for gitlab |
| Internet | Direct or via SSH tunnel | For LLM API calls (OpenAI or Ollama) |

The deploy-host reference deployment uses **rootless docker** rooted at `/data/webarena/rootless-docker/data` (the data-root is on the large `/z` mount, not the small system disk).

### 2.2 LLM backend

The harness uses the OpenAI Python SDK (≥ 0.27). Choose one of:

* **OpenAI API** — set `OPENAI_API_KEY=sk-...`. No further config.
* **Ollama (local model)** — set `OPENAI_API_BASE=http://<ollama-host>:11434/v1` and `OPENAI_API_KEY=ollama` (any value; Ollama doesn't check). Recommended models for WebArena's strict action-parsing: `qwen2.5:7b-instruct` or `qwen2.5:14b-instruct`. **Avoid DeepSeek-R1 reasoning models with low `max_tokens`** — they spend 200-500 tokens reasoning before producing the action, so set `WEBARENA_STRIP_REASONING_TAGS=1` and `max_tokens ≥ 2048` if you use one.
* **Hybrid (Ollama on remote GPU box)** — open an SSH tunnel from the orchestrator's host to the ollama host:
  ```
  ssh -N -f -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 \
      -L 127.0.0.1:11434:127.0.0.1:11434 user@gpu-host
  ```
  Then point `OPENAI_API_BASE=http://127.0.0.1:11434/v1`.

### 2.3 Playwright browsers

`pip install playwright==1.60.0` then `playwright install chromium`. If the orchestrator host has no internet (common in lab environments), [SMOKE_RUNBOOK.md](SMOKE_RUNBOOK.md) §1 explains how to side-load from a workstation. Set `PLAYWRIGHT_BROWSERS_PATH` to the directory containing `chromium-1223/`, `chromium_headless_shell-1223/`, `ffmpeg-1011/`.

### 2.4 Python deps

```
pip install -r requirements.txt
# add:
pip install pytest  # if you want to run mp/tests/
```

For offline installs, use `pip download -d wheels/ -r requirements.txt --platform manylinux2014_x86_64 --python-version 3.12 --only-binary :all:` from an online machine, then `pip install --no-index --find-links wheels/ ...` on the offline target.

### 2.5 NLTK + tiktoken caches

The evaluator's `nltk.word_tokenize` needs `punkt` and `punkt_tab`. The agent's tokenizer needs `tiktoken`'s `cl100k_base`. Pre-stage both if your host can't reach the internet:

```
# On a machine with internet:
python -m nltk.downloader -d /tmp/nltk_data punkt punkt_tab
TIKTOKEN_CACHE_DIR=/tmp/tiktoken_cache python -c "import tiktoken; tiktoken.get_encoding('cl100k_base').encode('warmup')"

# On the target:
rsync -az /tmp/{nltk_data,tiktoken_cache} target:/path/
# Export when running:
export NLTK_DATA=/path/nltk_data
export TIKTOKEN_CACHE_DIR=/path/tiktoken_cache
```

`Tokenizer` falls back to `cl100k_base` for any model name that isn't in OpenAI's known list, so Ollama models work fine.

## 3. End-to-end procedure

### 3.1 Configure

Edit defaults in `mp/config.py` or pass overrides to `mp.bring_up`. The key config:

```python
MPConfig(
    num_workers=8,           # N
    host="HOST_IP",    # public host the agent and evaluator hit
    port_stride=100,         # spacing between worker ports. 100 prevents
                             # *site-to-site* collisions for N up to 8, but
                             # it does NOT dodge shared-service collisions —
                             # the map container on deploy-host binds 8080/8085
                             # (127.0.0.1), and rootless docker's port
                             # allocator treats that as collision with a new
                             # 0.0.0.0:8080 bind. mp/config.py:_HOST_PORT_RESERVED
                             # skips these by shifting +10 (e.g. shopping_admin
                             # w3 → 8090 instead of 8080).
    docker_host="unix:///data/webarena/rootless-docker/run/docker.sock",
    golden_root="/data/webarena-mp/golden",
    config_files_root="/data/webarena-mp/config_files",
    auth_root="/data/webarena-mp/auth",
    result_dir="/data/webarena-mp/results",
)
```

### 3.2 Bring up replicas + populate goldens

```
python -m mp.bring_up --num_workers 8 --host <your_host>
```

What happens (wall-clock estimate for N=8):

| Step | What | Time |
|---|---|---|
| 1 | `docker tag` source images as `webarena-*-golden:latest` | seconds |
| 2 | `docker run` 4×(N-1) replicas (skip worker_0; reuses live containers) | ~10 min (gitlab boot dominates) |
| 3 | Apply per-replica base URL config (Magento `setup:store-config:set`, GitLab `gitlab-ctl reconfigure` with `external_url` + `nginx['listen_port']=8023` + `puma['worker_processes']=4`) | ~10 min (gitlab-ctl reconfigure × N times) |
| 4 | Snapshot goldens from worker_0: `mysqldump`, `pg_dump`, tar `/var/opt/gitlab` | ~15-25 min (gitlab tar ~22 GB dominates) |
| 5 | Health-check every replica | ~1 min |

If you want to verify the harness without populating goldens (read-only smoke), add `--skip_goldens`. Real benchmark runs that touch mutable sites need goldens populated.

### 3.3 Verify (recommended once)

```
python -m mp.verify_golden --max_urls 100 --out verify.json
```

For each of the 246 unique URLs referenced by `program_html` evaluators, this fetches it on worker_0 and on the source container, normalizes CSRF / timestamps / nonces, and diffs. Pass criterion: 0 divergent URLs after a real reset.

### 3.4 Run the benchmark

```
export OPENAI_API_BASE=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
export PLAYWRIGHT_BROWSERS_PATH=/path/pw_browsers
export TIKTOKEN_CACHE_DIR=/path/tiktoken_cache
export NLTK_DATA=/path/nltk_data
export WEBARENA_EVAL_MODEL=qwen2.5:7b-instruct   # overrides hard-coded gpt-4 in the LLM judge

python -m mp.orchestrator \
    --start_idx 0 --end_idx 812 \
    --provider openai --model qwen2.5:7b-instruct --mode chat \
    --temperature 0 --top_p 0.9 --max_tokens 384 \
    --max_steps 30 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json
```

Or, for a subset / smoke run:

```
python -m mp.orchestrator --task_ids 7,8,9,10,16,17,18,19,20,32 ...
```

**If any of your task ids mutate state** (Cancel order, Disable product, Modify address, Add product, etc.), read §4.8 first — `test.raw.json` flags every task `require_reset: false` even when they shouldn't be, and `reset_magento`'s default timeouts only survive N≤2 concurrent restores.

Progress is printed to stdout and written to `<result_dir>/scores.jsonl`, append-only, one JSON object per task:

```json
{"worker_id":3,"task_id":491,"score":1.0,"error":null,"duration_seconds":187.4}
```

Re-running `mp.orchestrator` with the same `result_dir` picks up where it left off — only tasks without a successful entry in `scores.jsonl` get re-queued.

### 3.5 Inspect results

```
python - <<'EOF'
import json, statistics
rows = [json.loads(l) for l in open("/path/results/scores.jsonl") if l.strip()]
ok = [r["score"] for r in rows if r["error"] is None]
print(f"completed: {len(ok)}/{len(rows)}")
print(f"score:     {sum(ok)/len(ok):.4f}")
print(f"mean dt:   {statistics.mean(r['duration_seconds'] for r in rows):.0f}s")
errs = [r for r in rows if r["error"]]
print(f"errors:    {len(errs)}")
for e in errs[:5]:
    print(f"  task {e['task_id']} on worker {e['worker_id']}: {e['error'][:120]}")
EOF
```

Per-task artifacts:

| Artifact | Path |
|---|---|
| Agent trajectory render | `<result_dir>/w{w}/render_{task_id}.html` |
| Playwright trace zip | `<result_dir>/w{w}/traces/{task_id}.zip` |
| Per-worker log | `<result_dir>/logs/worker_{w}.log` |
| Aggregated scores | `<result_dir>/scores.jsonl` |

Open a Playwright trace with `playwright show-trace <result_dir>/w0/traces/389.zip`.

## 4. Operating notes

### 4.1 Choosing N

| N | Wall-clock estimate | Memory | Disk | When |
|---|---|---|---|---|
| 1 | ~40 h | 8 GB | 25 GB | reproduce stock single-worker |
| 2 | ~20 h | 16 GB | 50 GB | conservative first parallel run |
| 4 | ~10 h | 32 GB | 100 GB | recommended |
| 8 | ~5 h | 70 GB | 225 GB | aggressive; fits deploy-host fine |
| 16+ | tail-bounded | 140 GB+ | 450 GB+ | map server may saturate; consider `--map_replicas N_map` |

LLM throughput is often the bottleneck — if you use a single shared Ollama on one GPU, N workers contend for the same GPU and effective throughput per worker drops by ~1/N.

### 4.2 Crashes and recovery

* **Worker process crashes** → orchestrator detects via SIGCHLD-equivalent (Python's `Process.is_alive()`), respawns one worker and re-enqueues any outstanding tasks.
* **A reset fails** (HTTP timeout, DockerExecError, etc.) → that task gets `score=null` and `error=…`, then the worker moves on. If the same site fails twice on the same worker, `docker rm -f` the replica and `mp.bring_up --skip_goldens --skip_configure` to recreate it.
* **Orchestrator dies / box reboots** → `scores.jsonl` is append-only. Re-running orchestrator skips completed tasks.
* **Cookie expired mid-run** → `auto_login.py:is_expired` re-issues the cookie in the per-worker auth folder. No human action needed.

### 4.3 GitLab specifics

* **Puma worker count**: Omnibus auto-detects `worker_processes` from CPU count, which on a many-core box means each replica spawns ~128 puma workers. With N replicas this OOM-kills puma on multiple replicas. `configure_replica_gitlab` caps `puma['worker_processes'] = 4` per replica. If you change this, expect to retune `--max_steps` or reset timeouts.
* **`nginx['listen_port']`**: setting `external_url` to `http://host:N` makes nginx listen on port N *inside* the container. Our docker port mapping is `host:N → container:8023`, so nginx must keep listening on 8023. `configure_replica_gitlab` pins it.
* **External-URL 502 vs Playwright 200**: GitLab nginx returns 502 to bare `curl http://127.0.0.1:8123/users/sign_in` (Host header mismatch with `external_url`) but Playwright + real browser see HTTP 200 with a fully-rendered page. `mp.reset._wait_healthy` accepts by `expect_body_contains` rather than strict status code, so this doesn't fail the harness.
* **Persistent 502 + ulimit warnings in `docker logs`**: if a gitlab replica returns HTTP 502 on ALL paths (not just root) and `docker logs` shows `ulimit: max user processes: cannot modify limit` / `/proc/sys/fs/file-max: Read-only file system`, ignore the ulimit lines (they are harmless rootless-docker noise emitted by every gitlab container at startup). The real signal is whether puma is up:
  ```bash
  docker exec gitlab_wN gitlab-ctl status puma   # look for "run:" vs "down:"
  ```
  If puma is `down`, restart it in place: `docker exec gitlab_wN gitlab-ctl start puma`. Wait ~2.5 minutes for the Rails preload to complete (`tail -f /var/log/gitlab/puma/current` until you see `* Listening on http://127.0.0.1:8080`). Observed root cause: puma can die on long-running idle containers (memory pressure / OOM-killer); runit's auto-restart can fail to come back if the container has been idle for many days. In-place restart is faster and preserves repository state vs recreating from golden image.

### 4.3a Fast reset — physical datadir swap (default)

**Root cause of slow resets**: this deployment's docker data-root is on a ZFS pool with no SLOG device, so every `fsync` costs ~110 ms (measured: 100 fsync'd 4 KB writes = 11.3 s on the host pool). A logical restore — `cat golden.sql | mysql` or `pg_restore` — is fsync-bound: it commits thousands of statements while rebuilding 369 Magento tables and their indexes, so it costs thousands × 110 ms and runs for **tens of minutes to hours**. Recreating the container from the golden image is *worse* for Magento because it re-triggers a heavy cold app boot.

**Fix**: `mp/reset.py` defaults to a **physical datadir swap**. At bring-up, after each replica's per-worker base_url is configured, `snapshot_all_datadirs()` makes a pristine on-container copy of the DB data directory (`/var/lib/mysql.golden` for Magento, `/usr/local/pgsql/data.golden` for Postmill). Reset then: stops the DB via supervisor → `cp -a` the golden tree back over the live datadir → starts the DB → clears caches → in-container HTTP/DB health probe. This is a bulk sequential copy that skips every per-commit fsync and index rebuild.

Measured on deploy-host (`shopping_admin`): **logical restore 2+ hours → physical swap ~85 s end-to-end** (the pure datadir swap is ~24 s; the rest is cache-clear + storefront warmup to first 302). The swap also resets filesystem drift (sessions, generated code) that a DB-only restore would miss — strictly more rigorous.

Operational notes:
* The golden datadir snapshot lives in the container's writable layer, so it is **lost if the container is recreated** (manual `docker rm` + `docker run`). Re-create it by re-running `mp.bring_up` (idempotent) or calling `snapshot_all_datadirs(client, cfg)` directly.
* If the snapshot is absent, `reset_magento` / `reset_postmill` automatically **fall back to the logical restore** (slow but correct), now with relaxed durability (`innodb_flush_log_at_trx_commit=2`; postgres `synchronous_commit=off`) to cut fsync cost.
* GitLab is unchanged: its reset already does a physical rsync from the host-side golden tree (`reset_gitlab`).
* `reset_site(..., strategy="recreate")` remains available for when a container is unrecoverable, but is NOT the default — it pays the heavy Magento cold-boot.

### 4.4 Magento specifics

* **Reset privilege**: the Magento `magentouser` only has DB-level privileges on `magentodb`. The *logical* fallback restore uses `mysqldump --add-drop-table` so DROP/CREATE happen at the TABLE level (the user has those), not at the DATABASE level (which it doesn't). The default *physical-swap* path (§4.3a) sidesteps SQL entirely.
* **Redis flush is mandatory**: env.php configures Magento's `default` and `page_cache` backends to use Redis. After every DB restore, `redis-cli -n 0 FLUSHDB; redis-cli -n 1 FLUSHDB` is required, otherwise the storefront serves stale full-page cache. `reset_magento` does this.
* **Base URL rewrite is per-reset, not per-bring-up**: restoring the golden SQL overwrites `core_config_data.web/secure/base_url` back to whatever the original storefront port was. After every restore we re-run `setup:store-config:set --base-url=<worker's URL>` plus a direct SQL `UPDATE core_config_data`.
* **MySQL readiness lags HTTP readiness by 30–60 s on first boot.** A fresh Magento container responds to `curl http://127.0.0.1/` (HTTP 302 or 500) before its in-container MariaDB accepts connections. If you write your own bring-up wrappers, poll `mysql --connect-timeout=5 -u magentouser -p... -e 'SELECT 1' magentodb` and accept that the first few probes will fail or time out (use per-probe `timeout=60`, total deadline ≥ 300 s).
* **Magento's internal cron contends with restore DDL.** Every replica boots with supervisord-managed cron daemons that immediately start writing to `cron_schedule`. While the restore is running `DROP TABLE` / `CREATE TABLE`, those cron writes pile up in "Opening tables" state and serialize on table metadata locks. The effect is mild for N=2 but **devastating at N≥4**: in a 7-way concurrent restore, every replica's restore exceeded the `reset_magento` `timeout=600` ceiling in `mp/reset.py:142` and the entire batch errored. Mitigations: serialize the restore across workers, raise the timeout to ≥ 1800 s, or skip reset entirely for `require_reset: false` tasks (the worker now does this automatically — see §4.8).

### 4.5 Postmill specifics

* DB user `postmill` has CREATE/DROP on its own database; no `su - postgres` needed.
* Default reset is the physical datadir swap (§4.3a). The logical fallback uses `pg_restore --clean --if-exists` (DROP TABLE IF EXISTS before each restore) with `synchronous_commit=off`.
* User-uploaded `submission_images/` and generated `media/cache/` are part of state — restored from tarballs in `<golden_root>/reddit/` on both reset paths.
* **Golden image WAL recovery**: the reddit golden image was committed while postgres was running (crash-consistent), so a freshly-recreated reddit container replays WAL on boot, which can take 5–20 min on this storage. The physical-swap reset avoids container recreation, so it does not pay this cost; but a one-time clean `pg_ctl stop` + image re-commit would make the `strategy="recreate"` path fast too.

### 4.6 Read-only site sharing

`mp.config.ALL_READONLY_SITES = ("wikipedia", "map", "homepage")`. Tasks on these sites are routed to the **single shared container** (worker_0's). Verified safe by task scan: all 128 map tasks are navigation/routing queries; all 23 wikipedia tasks are read-only.

If the shared map container saturates under N≥8 concurrent agents, add `--map_replicas N_map` (TBD; deferred from §14.8 pending load measurement).

### 4.7 Auto-login under parallelism

Each worker writes cookies to `<auth_root>/w{w}/` via `auto_login.py --auth_folder`. Cookies are tied to a worker's replica `host:port`, so they cannot accidentally authorize a different worker. `run.py:run_single_task` passes the per-worker auth folder via `WEBARENA_AUTH_FOLDER` env.

### 4.8 Running with and without per-task reset

#### 4.8.1 Behavior

`mp/worker.py:_run_one_task` computes `sites_to_reset` per task as:

```
sites_to_reset = dirty_sites_from_previous_task_on_this_worker
if config.get("require_reset", True):
    sites_to_reset |= touched_sites_named_in_config["sites"]
```

`dirty_sites` is empty when a worker first starts and is populated to the previous task's `sites` after the previous task finishes. So:

* A worker's **first** task with `require_reset: false` skips reset entirely.
* A worker's **subsequent** task with `require_reset: false` still resets whatever the *previous* task touched on this worker (the safety guarantee for cross-task isolation on a re-used replica).
* Any task with `require_reset: true` (the default when the field is missing) always resets its touched sites.

#### 4.8.2 Dataset quirk — every task in `test.raw.json` carries `require_reset: false`

This includes obvious mutation tasks like "Cancel order 302", "Disable product X", "Add new product Y", "Modify order address", etc. — verified by scanning all 812 entries. The flag in the released dataset is uniformly false; do **not** rely on it to decide whether a task mutates state.

Consequence: if you queue only mutation tasks with the bare dataset configs, the worker's first task on each worker will **skip** reset (dirty_sites is empty AND require_reset is false) — meaning the very first agent runs against whatever state happens to be in the replica, not the golden state. Subsequent tasks on the same worker still reset (because of dirty_sites carryover), but the *first* task per worker leaks state.

#### 4.8.3 Recipe — running mutation tasks correctly

For any task you want a full pre-task reset on, override `require_reset` to `true` in the **per-worker rendered configs** before running the orchestrator:

```python
import json, pathlib
TASKS = [470, 471, 472, 473, 474]            # the task ids you plan to run
N_WORKERS = 7                                # match cfg.num_workers
CONFIG_FILES_ROOT = "/data/webarena-mp/config_files"

for w in range(N_WORKERS):
    for tid in TASKS:
        cf = pathlib.Path(CONFIG_FILES_ROOT) / f"w{w}" / f"{tid}.json"
        cfg = json.loads(cf.read_text())
        cfg["require_reset"] = True
        cf.write_text(json.dumps(cfg, indent=2))
```

Note: `render_config_files` (run during `mp.bring_up`) regenerates these from `test.raw.json` and will wipe your override. Re-apply after any bring-up.

Alternative: edit `test.raw.json` directly for the rows you care about, *then* run `mp.bring_up` (or `render_config_files`). That puts `require_reset: true` into every worker's copy in one go.

#### 4.8.4 Tune `reset_magento` timeouts to your N

`reset_magento` in `mp/reset.py` makes four blocking `docker exec` calls whose default timeouts assume N=2 contention. Under N≥4 concurrent restores Magento's internal cron writes contend with the restore for table metadata locks (§4.4), and the IO needed by `rm -rf var/cache/*` is also serialized at the docker daemon. Empirically:

| Concurrent restore N | mysql restore (step 1) | rm/cache (step 2) | base_url UPDATE (step 5) | _wait_healthy (step 7) |
|---|---|---|---|---|
| 2 | 600 s (default) | 300 s (default) | 30 s (default) | 120 s (default) |
| 4–5 | **1800 s** | **1800 s** | **300 s** | **300 s** |
| 7+ | not reliable at any timeout — either serialize restores (e.g. host-side `flock` around the whole `reset_sites` call) or drop concurrency. The 2026-05-28 N=7 attempt failed at the rm step despite 1800 s timeouts. |

The N=5 mutation run on 2026-05-28 used the middle column above and all 5 tasks completed cleanly (~11 min wall, ~95 % parallel efficiency). At N=7 the same configuration timed out the rm step on every replica.

#### 4.8.5 Quick decision matrix

| Your tasks | Override `require_reset`? | Bump timeouts? | Notes |
|---|---|---|---|
| Read-only only (e.g. sales-report queries, lookups) | No — leave at `false` | No | Reset is skipped; runs in seconds + agent time only |
| Mutation tasks, N ≤ 2 | Yes for each task id | No (defaults are fine) | |
| Mutation tasks, N = 4–5 | Yes for each task id | Yes — N=4–5 column above | The validated path |
| Mutation tasks, N ≥ 7 | Yes for each task id | Yes + serialize | Either use a `flock` semaphore in `reset_sites` to cap concurrency to ~3, or drop num_workers |
| Mixed (some mutation, some read-only) | Yes only on the mutation ids | Yes if N≥4 | dirty_sites carry-over keeps read-only-after-mutation safe automatically |

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `SemLock created in a fork context is being shared with a process in a spawn context` | mp.Queue created with default fork context, Process with spawn | Already fixed — orchestrator uses `ctx.Queue()` from the same spawn context |
| `model "gpt-4-1106-preview" not found, try pulling it first` from fuzzy_match eval | Eval hardcodes gpt-4 for the LLM judge | Set `WEBARENA_EVAL_MODEL=<your_model>` |
| `Could not automatically map qwen2.5:7b-instruct to a tokeniser` | tiktoken doesn't know Ollama model names | Already fixed — Tokenizer falls back to `cl100k_base` |
| `No module named 'text_generation'` | HF backend dep | Already fixed — lazy import; only loaded if `--provider huggingface` |
| Worker 1 health check fails | GitLab puma OOM-killed by another replica | Restart with `gitlab-ctl restart puma`; if recurring, lower `puma['worker_processes']` further |
| `Connection refused on :8x23` after `external_url` change | nginx listen_port followed external_url's port | Already fixed — `configure_replica_gitlab` pins `nginx['listen_port'] = 8023` |
| `rm -rf var/cache/* timed out after 300 seconds` from `reset_magento` step 2 | Magento `var/cache/*` grows fast during first-boot warmup and rm IO serializes on the docker daemon; 300 s is fine at N≤2 but too tight at N≥4 | Raise to 1800 s for N=4–5 (see §4.8.4). At N≥7 even 1800 s isn't always enough — serialize restores instead |
| Reset succeeds but evaluator gets stale data | Magento Redis cache not flushed | Already fixed — `reset_magento` issues `redis-cli FLUSHDB` for db 0 and 1 |
| Agent's response is empty / always None action | DeepSeek-R1 or similar reasoning model wasted all tokens on `<think>` | Set `WEBARENA_STRIP_REASONING_TAGS=1` AND raise `--max_tokens` to ≥ 2048, or switch to a non-reasoning model |
| `auto_login.py` writes to wrong folder | Stale `.auth/` from a single-worker run | Always pass `--auth_folder` explicitly via env; harness sets `WEBARENA_AUTH_FOLDER` automatically |
| Both workers race on the same task | Cosmic ray / queue bug | Should never happen — `mp.Queue.get` is atomic. If you see this, file a bug |
| `Bind for 10.0.2.100:8080 failed: port is already allocated` during `docker run shopping_admin_w3` | Worker_3's natural shopping_admin port (8080) collides with the shared map container's 127.0.0.1:8080 binding inside rootless docker's network namespace | Already fixed — `mp/config.py:_HOST_PORT_RESERVED = {8080, 8085}` and `port_for()` shifts colliding ports +10 (worker_3 → 8090). Add other reserved ports here if you spin up shared services |
| `subprocess.TimeoutExpired … mysqldump … timed out after 600 seconds` from `reset_magento` | N≥4 concurrent restores contend with Magento internal cron for table-metadata locks (§4.4) | (a) For `require_reset: false` tasks (e.g. sales-report reads), the worker now correctly skips reset — see §4.8. (b) For tasks that need real reset, raise `timeout` in `mp/reset.py:reset_magento` to 1800 s, or serialize restores by gating on a host-side semaphore |
| `ERROR 2002 (HY000): Can't connect to local server through socket '/run/mysqld/mysqld.sock'` immediately after `docker run` of a Magento replica | Apache+PHP boot before MariaDB inside the container is queryable (§4.4) | Before running CLI ops against the replica, poll `mysql --connect-timeout=5 -u magentouser -p... -e 'SELECT 1' magentodb` until it returns 0. Allow ~30–60 s total |
| `Opening tables` shows up for 60+ s in `SHOW PROCESSLIST` during configure step | Magento first-boot internal cron has a `cron_schedule` UPDATE waiting on the same table our `setup:store-config:set` wants to touch | Bump `configure_replica_magento`'s per-statement timeout to 300 s. The lock clears on its own after Magento's first cron sweep finishes |

## 6. Verifying correctness

To prove a run is correct (not just complete):

1. **`scores.jsonl` has exactly one entry per task you requested.**
   ```python
   import json, collections
   rows = [json.loads(l) for l in open(".../scores.jsonl")]
   c = collections.Counter(r["task_id"] for r in rows)
   assert all(v == 1 for v in c.values()), "duplicate task entries"
   ```

2. **Per-worker URL substitution is intact.**
   ```python
   c0 = json.load(open(".../config_files/w0/389.json"))
   c1 = json.load(open(".../config_files/w1/389.json"))
   assert c0["start_url"] != c1["start_url"]  # mutable site → ports differ
   c0m = json.load(open(".../config_files/w0/7.json"))   # map task
   c1m = json.load(open(".../config_files/w1/7.json"))
   assert c0m["start_url"] == c1m["start_url"]  # map → shared
   ```

3. **Traces show navigations to the worker's own host:port — no other.** The `goto`-only check below is too narrow; assets, frames, and redirects also reveal which replica was hit. The stronger check counts every `HOST_IP:PORT` occurrence in the trace and asserts only the expected port appears:
   ```python
   import zipfile, json, re, collections
   def trace_port_hits(zip_path):
       hits = collections.Counter()
       with zipfile.ZipFile(zip_path) as z, z.open("trace.trace") as f:
           for line in f:
               try: d = json.loads(line)
               except: continue
               for p in re.findall(r"158\.130\.4\.158:(\d{4,5})", json.dumps(d)):
                   hits[p] += 1
       # Drop shared read-only sites (homepage 4399, wikipedia 8888, map 13000/13030)
       for shared in ("4399", "8888", "13000", "13030"):
           hits.pop(shared, None)
       return hits

   # Example: confirm w3 only hits :8090 (its shopping_admin port after _HOST_PORT_RESERVED shift)
   h = trace_port_hits(".../w3/traces/3.zip")
   assert list(h.keys()) == ["8090"], f"w3 leaked to {dict(h)}"
   ```

4. **`mp.verify_golden` reports 0 divergent URLs** (after running mutation-template tasks and resetting).

5. **`mp/tests/` is green.**
   ```
   python -m pytest mp/tests/ -v
   ```
   Expected: `45 passed`.

## 7. Operating runbook for the reference deployment (orchestrator host + GPU host)

Concrete to a two-host deployment (one orchestrator + websites box, one GPU box). Adapt paths/hostnames for your own. For the TSA-vs-SGLang benchmark workflow specifically, see §8 — those launchers wrap most of this.

```bash
# === Pre-flight on the orchestrator host (drives docker + workers) ===
ssh user@orchestrator-host
mkdir -p <DATA_ROOT>/{golden,config_files,results,auth,logs}

# Activate the venv + env
source <DATA_ROOT>/../webarena-venv/bin/activate
cd <WEBARENA_REPO>
export PLAYWRIGHT_BROWSERS_PATH=<DATA_ROOT>/../pw_browsers
export TIKTOKEN_CACHE_DIR=<DATA_ROOT>/../tiktoken_cache
export NLTK_DATA=<DATA_ROOT>/../nltk_data

# === Materialize the live config from template (one-time) ===
cp -n mp/configs/config-tsa.example.json mp/configs/config-tsa.json
$EDITOR mp/configs/config-tsa.json   # fill <HOST_OR_IP>, <DATA_ROOT>, <PATH_TO_DOCKER_SOCK>

# === Bring-up (one-shot, ~30-60 min depending on N + goldens populated) ===
python -m mp.bring_up --num_workers 5 --skip_goldens

# === LLM backend ===
# Two paths — pick one. Use §8's launchers for the benchmark workflow.
#
# (a) TSA + SGLang via the shipped launchers (RECOMMENDED for the benchmark):
export GPU_HOST=user@gpu-host
bash mp/launch_tsa.sh         # or launch_dense.sh / launch_judge.sh
source mp/.inference_env      # exports OPENAI_API_BASE etc.
#
# (b) Any OpenAI-compatible endpoint (Ollama, vLLM, your own) via env vars:
export OPENAI_API_BASE=http://127.0.0.1:<PORT>/v1
export OPENAI_API_KEY=<token-or-placeholder>
export WEBARENA_EVAL_MODEL=<judge-model-id>

# === Run the benchmark ===
python -m mp.orchestrator \
    --config mp/configs/config-tsa.json \
    --start_idx 0 --end_idx 812 \
    --provider openai --model "$AGENT_MODEL_NAME" --mode chat \
    --inference_backend tsa \
    --temperature 0 --top_p 1 --max_tokens 2048 \
    --max_steps 30 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json
```

To stop a run cleanly: `pkill -f mp.orchestrator`. To resume: re-run the same orchestrator command; completed tasks are skipped via `scores.jsonl`.

To tear down replicas:
```bash
docker ps --format '{{.Names}}' | grep -E '_w[0-9]+$' | xargs -r docker rm -f
```

---

## 8. TSA-vs-Dense benchmark workflow

For comparing **TreeSparseAttention (TSA)** against an **SGLang dense baseline** on the same Qwen3-VL-4B-Instruct model, the harness ships dedicated launchers, configs, and a comparison tool. The goal is a research-rigorous pass-rate delta where the only varied factor is the agent's attention mechanism.

### 8.1 Architecture (single-GPU host)

```
   deploy-host (websites, no GPU)
        │
        │  mp/orchestrator.py → workers → Playwright → docker site replicas
        │
        │  SSH tunnels (autossh)
        ▼
   inference-host (RTX 5060 Ti or B200)
        ├─ tmux "wa-tsa"   →  TSA serve.py        : 10000  (model="tree-sparse")
        ├─ tmux "wa-dense" →  SGLang launch_server: 10001  (model="qwen3vl-dense")
        └─ tmux "wa-judge" →  SGLang launch_server: 10002  (model="qwen3vl-judge")
                                                    ↑
                              fixed judge — same backend for BOTH agent runs.
```

The agent talks to **port 10000 (TSA)** or **port 10001 (dense)** via `OPENAI_API_BASE`. The evaluator's LLM judge always talks to **port 10002**, routed through the per-call `judge_endpoint()` shim in `evaluation_harness/_endpoint.py`. This isolates "what the agent saw" from "how the answer was scored".

### 8.2 Files (new in this workflow)

| File | Purpose |
|---|---|
| `mp/_inference_common.sh` | autossh tunnel helpers, tmux helpers, `gpu_detect`, `wait_healthy`. |
| `mp/configs/gpu_profile.sh` | Per-SM tuning table (sm_100, sm_120, fallback). |
| `mp/configs/config-tsa.example.json` | sm_120 template (N=5). Copy to `config-tsa.json` and fill placeholders (`<HOST_OR_IP>`, `<DATA_ROOT>`, `<PATH_TO_DOCKER_SOCK>`). Live `config-tsa.json` is gitignored. |
| `mp/configs/config-dense.example.json` | sm_120 template (N=5). Copy to `config-dense.json`. |
| `mp/configs/config-tsa-b200.example.json` | sm_100 template (N=8). Copy to `config-tsa-b200.json`. |
| `mp/configs/config-dense-b200.example.json` | sm_100 template (N=8). Copy to `config-dense-b200.json`. |
| `mp/launch_judge.sh` | Boots the SGLang judge in tmux, opens tunnel :10002. |
| `mp/launch_tsa.sh` | Boots TSA in tmux, opens tunnel :10000, ensures judge is up. |
| `mp/launch_dense.sh` | Boots SGLang dense in tmux, opens tunnel :10001, ensures judge is up. |
| `mp/teardown_inference.sh` | Kills all three tmux sessions + tunnels (or a subset). |
| `mp/check_template_parity.py` | Verifies both servers format the same messages identically. |
| `mp/benchmark_compare.py` | Consumes two scores.jsonl files; emits markdown + CSV. |
| `evaluation_harness/_endpoint.py` | `judge_endpoint()` context manager (per-call endpoint swap). |

### 8.3 Env vars

| Variable | Set by | Purpose |
|---|---|---|
| `GPU_HOST` | user (**REQUIRED — no default; launchers fail fast if unset**) | SSH target of the GPU machine, e.g. `alice@my-gpu.example.com`. |
| `GPU_SM_HINT` | user (optional) | Override `gpu_detect` (e.g. `100` for B200). |
| `TS_CUDA_ARCHS` | user (default `100;120`) | TSA kernel build target(s). |
| `WEBARENA_TOKENIZER_PATH` | launcher | HF path/id for Qwen3-VL-4B tokenizer (max_obs_length accuracy). |
| `OPENAI_API_BASE`, `OPENAI_API_KEY` | launcher → orchestrator | Agent inference endpoint. |
| `WEBARENA_EVAL_API_BASE`, `WEBARENA_EVAL_API_KEY`, `WEBARENA_EVAL_MODEL` | launcher → orchestrator → `judge_endpoint()` | Judge inference endpoint (held constant across both runs). |
| `INFERENCE_BACKEND`, `AGENT_MODEL_NAME` | launcher | Informational; the orchestrator's `--inference_backend` + `--model` should mirror them. |

### 8.4 scores.jsonl row shape (new provenance fields)

Each row now records which backend produced it:

```json
{"worker_id": 0, "task_id": 78, "score": 1.0, "error": null,
 "duration_seconds": 173.4,
 "inference_backend": "tsa", "model": "tree-sparse",
 "openai_api_base": "http://127.0.0.1:10000/v1",
 "eval_api_base": "http://127.0.0.1:10002/v1"}
```

`benchmark_compare.py` reads these fields verbatim and shows them in the report header.

### 8.5 One-time setup (on the GPU host)

```bash
ssh user@inference-host
# 1. Pre-download Qwen3-VL-4B-Instruct
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir ~/hf_models/Qwen3-VL-4B-Instruct
# 2. Install SGLang
pip install "sglang[all]>=0.4.3" "flashinfer-python>=0.2"
# 3. Pre-compile TSA kernels for the local SM. The same artefact also targets
#    sm_100 (B200) and PTX-falls-back to other Blackwells. ~3 min first time.
cd ~/TreeSparseAttention_CW
TS_CUDA_ARCHS="100;120" python -c \
  "from python.jit_build import build_tree_sparse_kernels; build_tree_sparse_kernels()"
```

### 8.6 End-to-end run

```bash
# Activate venv + env on deploy-host
ssh user@deploy-host
source /data/webarena-venv/bin/activate
cd /data/webarena-repo
export PLAYWRIGHT_BROWSERS_PATH=/data/pw_browsers
export TIKTOKEN_CACHE_DIR=/data/tiktoken_cache
export NLTK_DATA=/data/nltk_data

# === TSA run ===
source mp/launch_tsa.sh          # boots judge, then TSA; exports env vars

# Optional: confirm both servers format prompts identically before committing
# multi-hour CPU time. (Run after BOTH launchers complete; will obviously be
# skipped during a sequential single-GPU schedule — see §8.7.)
# python -m mp.check_template_parity --tsa http://127.0.0.1:10000/v1 \
#                                     --dense http://127.0.0.1:10001/v1

python -m mp.orchestrator \
    --config mp/configs/config-tsa.json \
    --start_idx 0 --end_idx 812 \
    --provider openai --mode chat --model "$AGENT_MODEL_NAME" \
    --inference_backend tsa \
    --temperature 0 --top_p 1 --max_tokens 2048 \
    --max_steps 30 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json

# === Dense run (sequential on 16 GB GPU; keep the judge warm) ===
bash mp/teardown_inference.sh --tsa-only --keep-judge
source mp/launch_dense.sh
python -m mp.orchestrator \
    --config mp/configs/config-dense.json \
    --start_idx 0 --end_idx 812 \
    --provider openai --mode chat --model "$AGENT_MODEL_NAME" \
    --inference_backend dense \
    --temperature 0 --top_p 1 --max_tokens 2048 \
    --max_steps 30 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json

# === Compare ===
python -m mp.benchmark_compare \
    --tsa  /data/webarena-mp/results-tsa/scores.jsonl \
    --dense /data/webarena-mp/results-dense/scores.jsonl \
    --tasks config_files/test.raw.json \
    --out comparison_report.md --csv comparison.csv

bash mp/teardown_inference.sh    # final cleanup
```

### 8.7 Validated concurrency profiles (per-GPU)

Both GPU classes are **first-class** targets; the connector code, launchers, and `mp.orchestrator` are GPU-agnostic. The only knobs that change per arch are tuning defaults in `mp/configs/gpu_profile.sh`. Below are the validated profiles — anything more aggressive than the "validated" column is plausible but unverified on this hardware.

| GPU | SM | VRAM | Default `num_workers` | Default `TSA_MAX_BATCH` | Judge | Concurrent backends | Use-case |
|---|---|---|---|---|---|---|---|
| **B200** | sm_100 | 141 GB | **8** (`config-tsa-b200.json` / `config-dense-b200.json`) | **16** | **ON** by default | TSA + dense + judge all coexist | reference rig, full fixed-judge rigor |
| **RTX 5060 Ti** | sm_120 | 16 GB | **5** (`config-tsa.json` / `config-dense.json`) | **4** | OFF by default (self-judge); ON requires N ≤ 3 | sequential only | minimum target |

**N=5 sm_120 evidence** (raw concurrency verified end-to-end):

1. *Raw HTTP probe (5 parallel `curl POST /v1/chat/completions`)*:
    - TSA → 5/5 → `200 OK`, wall **2.05 s** (`Collected batch of 4` + `Collected batch of 1` in scheduler log).
    - Dense → 5/5 → `200 OK`, wall **0.41 s** (`max-running-requests=16` parallelises all 5).
2. *Orchestrator level (initial pilot, pre-chat-template-fix)* — 5 workers, 5 shopping tasks `22, 24, 47, 48, 126`:
    - TSA: w0–w4 all spawned, 5/5 completed, **PASS=3 / FAIL=2 / ERROR=0**, wall 118 s, no OOM.
    - Dense: w0–w4 all spawned, 5/5 completed, **PASS=2 / FAIL=3 / ERROR=0**, wall 104 s, no OOM. (Note: these Dense "PASS=2" rows were artefactual — pre-fix every Dense trajectory step-1 crashed because of the `--chat-template qwen2-vl` issue documented in §8.8; the LLM-judge erroneously equated the crash message with the reference `"N/A"` on tasks 22+24. See `TSA_VS_DENSE_REPORT.md` §0.)
3. *Orchestrator level (post-fix N=3 smoke, after `--chat-template qwen2-vl` removed)* — 3 workers, same 5 shopping tasks, self-judging symmetric on both backends:
    - TSA: 5/5 trajectories reached a real `stop` answer, **PASS=3** (22, 24, **48 deterministic-correct**: `"0 fulfilled orders, $0 total spent"`), FAIL=2 (47, 126), 0 errors.
    - Dense: 5/5 trajectories reached a real `stop` answer (no more parse-failure crashes), **PASS=2** (22, 24 — legitimate `"N/A"` answers), FAIL=3 (47, 48, 126), 0 errors.
    - Both backends now produce correct `In summary, the next action I will perform is ```click [X]``` ` format on all trajectories — confirmed by direct curl probe + render HTML inspection.

The configs ship with both defaults — pick `config-tsa.json`/`config-dense.json` (N=5, 5060 Ti) or `config-tsa-b200.json`/`config-dense-b200.json` (N=8, B200) at the orchestrator's `--config` flag. `gpu_profile.sh` auto-detects the SM from `nvidia-smi` and tunes the inference-server flags accordingly (override with `GPU_SM_HINT=100`).

**Bring-up note for fresh worker replicas**: `mp.bring_up --num_workers 5 --skip_goldens` (or higher) provisions the per-worker docker containers. `mp/bring_up.py`'s MySQL configure-step timeout was raised from 30 s → 180 s to handle cold-boot Magento warm-up. Without this, `configure_replica_magento` would fail with `TimeoutExpired` when an unwarmed Magento container takes >30 s before the in-container mariadb accepts connections.

### 8.8 Pitfalls

- **CRITICAL: do NOT pass `--chat-template qwen2-vl` to SGLang when serving Qwen3-VL.** SGLang's bundled `qwen2-vl` template was designed for the older Qwen2-VL series — it injects `<|vision_start|><|image_pad|><|vision_end|>` vision-pad tokens and uses Qwen2-era system-marker wrapping. Qwen3-VL was not trained with this template (its native template lives in `tokenizer_config.json`). Forcing the wrong template corrupts the prompt structure, the few-shot agent examples lose their format anchor, and the model falls back to memorized SeeAct-style `` ```action [X]``` `` syntax which WebArena's action parser rejects → every Dense trajectory step-1 crashes with `Early stop: Failed to parse actions for 3 times`. **Fix**: omit `--chat-template`; SGLang then reads the model's native ChatML template from `tokenizer_config.json` — the same path TSA already uses via `processor.apply_chat_template()` in `TreeSparseAttention_CW/models/qwen3vl_inference.py`. This restores input parity (no tailoring; both backends now apply the template the model was trained with). The shipped `mp/launch_dense.sh` and `mp/launch_judge.sh` have this corrected; do not re-add the flag. See `TSA_VS_DENSE_REPORT.md` §0 for the full diagnosis.
- **xgrammar is not the answer to the format issue above.** xgrammar exists in both SGLang and TSA but is opt-in per request (requires `response_format` / `regex` / `guided_json` in the request body). WebArena's openai client sends plain `chat.completions.create(messages=..., ...)` with no schema, so neither backend uses xgrammar during the agent run — output format compliance has to come entirely from correct chat templating (above).
- **VRAM ceiling (sm_120 / 5060 Ti)** — TSA(4B) + SGLang-judge(2B) coexist at ~13.5 GB total at `TSA_MAX_BATCH=2`. At `TSA_MAX_BATCH=4` (needed to actually parallelise N=5 batched on the TSA scheduler), the judge no longer fits — drop the judge or drop N. SGLang-dense(4B) + SGLang-judge(2B) does not coexist on a 16 GB card because two SGLang servers fragment too aggressively to share the remaining 6 GB. The smoke-test setup confirmed empirically:
  - TSA(4B) + Judge(2B) at mem-frac 0.35 → fits; 15.1 GB used.
  - Dense(4B) alone at mem-frac 0.60 → fits; 10.4 GB used.
  - Dense(4B) + Judge(2B) at any combination of mem-fracs → SGLang OOMs in `init_memory_pool`.

  Workaround for the dense run on 16 GB: **point the judge at the dense server itself** (`WEBARENA_EVAL_API_BASE=http://127.0.0.1:10001/v1`, `WEBARENA_EVAL_MODEL=qwen3vl-dense`). The dense backend then judges its own answers. This introduces a controlled judge-model asymmetry between runs (TSA-run uses a 2B independent judge, dense-run uses the 4B agent as judge); the `eval_api_base` field in each `scores.jsonl` row records which judge was used so `benchmark_compare.py`'s header shows the asymmetry. On hardware with ≥24 GB the workaround is unnecessary — keep both runs pointed at the fixed 2B judge.
- **GPU detection** — `mp/_inference_common.sh:gpu_detect` runs `nvidia-smi` on `$GPU_HOST`. On B200 hosts (sm_100), set `GPU_SM_HINT=100` in your shell before sourcing the launcher to skip the SSH call and tune for the bigger card.
- **Kernel rebuild after pulling new TSA code** — delete `$HOME/.cache/torch_extensions/_tree_sparse_kernels_<archs>/` on the GPU host and re-run the build snippet from §8.5; the per-arch cache key in `jit_build.py` keeps stale builds from being silently loaded.
- **Judge tunnel must be up before the agent tunnel** — `launch_tsa.sh` and `launch_dense.sh` enforce this by calling `launch_judge.sh` as a pre-step.
- **Strict-mode comparison** — `benchmark_compare.py` refuses to run if the two scores.jsonl files do not cover the same task_id set. Override only when intentionally comparing a subset (`--no-strict-task-ids`).
- **Re-run idempotency** — both orchestrator invocations skip task_ids that already have a non-error row in their respective `scores.jsonl`. Interrupted runs resume cleanly.

---

# Appendix: every change shipped for multi-worker support

This appendix lists every code change relative to vanilla WebArena that the multi-worker harness depends on. Each is grouped by file and explains what changed and why.

## A. New top-level `mp/` package (2700+ LOC)

| File | Purpose | Notes |
|---|---|---|
| `mp/__init__.py` | Re-exports MPConfig + load_config | tiny |
| `mp/config.py` | `MPConfig` dataclass: N, ports, paths, DB creds. `port_for`, `url_for`, `container_for`, `env_for`, `to_json`/`from_json`, `load_config`, `save_config` | port_stride=100 prevents collisions; worker_0 reuses legacy container names; all DB credentials confirmed by audit |
| `mp/docker_exec.py` | `DockerClient` wraps `docker exec` for local + SSH-remote modes; `DockerExecError` | nested-quoting via `bash -lc`; subprocess timeouts on every call |
| `mp/reset.py` | `reset_magento`, `reset_postmill`, `reset_gitlab`, `reset_site` dispatcher, `_wait_healthy` health-poll | every primitive ends with a public-URL body-match poll; `_wait_healthy` accepts by content when given `expect_body_contains` (handles GitLab Host-header 502 quirk) |
| `mp/bring_up.py` | One-shot: `docker tag` goldens, `docker run` N replicas, `configure_replica_magento`/`configure_replica_gitlab`, `populate_goldens`, `render_config_files`, `assert_all_healthy` | uses `docker tag` not `docker commit` (60+ min savings); skip_goldens / skip_configure flags |
| `mp/worker.py` | Per-worker process body: `_apply_env`, `_assert_env_matches_worker`, `_run_one_task`, `worker_main` | imports `browser_env` only AFTER env override applied; dirty_sites tracking per §14.6 |
| `mp/orchestrator.py` | `Orchestrator.run()`: spawn-context Queue + N worker subprocesses, crash recovery, append-only results | bug-fixed to use `ctx.Queue()` from the same spawn context |
| `mp/verify_golden.py` | HTML diff between worker_0 reset and source; 8 normalizers for CSRF / nonces / timestamps; URL extraction from `test.raw.json` | `--max_urls` cap for fast smoke; per-site diff summary |
| `mp/tests/test_config.py` | 12 unit tests | port arithmetic, container naming, env emission, round-trip JSON |
| `mp/tests/test_docker_exec.py` | 6 unit tests | SSH wrap, shell quoting, exec command formation |
| `mp/tests/test_reset_dispatch.py` | 8 unit tests with mocked client | every reset primitive issues the expected commands |
| `mp/tests/test_verify_golden.py` | 10 unit tests | normalizer regex correctness, URL extraction |
| `mp/tests/test_llm_provider.py` | 8 unit tests | reasoning-tag stripping; OPENAI_API_BASE override |
| `mp/README.md` | Module reference + layout |  |
| `mp/SMOKE_RUNBOOK.md` | First-time bring-up runbook | per-step wall-clock estimates |
| `mp/MULTIWORKER_GUIDE.md` | This file |  |

## B. Modifications to existing files

### B.1 `run.py`

* New CLI flags: `--worker_id`, `--mp_config_path`.
* `_apply_worker_env(args)`: when `--worker_id` is set, loads `mp/config.json`, overrides URL env vars (`SHOPPING`, `SHOPPING_ADMIN`, `GITLAB`, `REDDIT`, `MAP`, `WIKIPEDIA`, `HOMEPAGE`), exports `WEBARENA_AUTH_FOLDER=<auth_root>/w{w}`. Called *before* any module that reads URL env at import time.
* `run_single_task(config_file, *, worker_id, cfg, args_dict)`: adapter that the worker module calls. Builds an argparse-namespace, constructs the agent, runs one task end-to-end, returns the float score. Uses the per-worker auth folder via the env var.
* `_run_one_task_loop`: the body of stock `test()` extracted to one task. Identical agent + evaluator behavior; the only difference is the auth-folder source.

### B.2 `llms/providers/openai_utils.py`

* `_setup_openai_api()`: reads `OPENAI_API_BASE` from env at every call. If set, points `openai.api_base` there and accepts a dummy `OPENAI_API_KEY=ollama`. Falls back to original "OPENAI_API_KEY required" check if not set.
* `_strip_reasoning(text)`: imported from `_reasoning.py`. Applied to every chat completion + completion response.
* The four entry points (`generate_from_openai_chat_completion` etc.) now call `_setup_openai_api()` first and pass results through `_strip_reasoning()`.

### B.3 `llms/providers/_reasoning.py` (new)

* Standalone `strip_reasoning(text)` helper. Removes `<think>…</think>` and `<reasoning>…</reasoning>` blocks when `WEBARENA_STRIP_REASONING_TAGS=1`. Handles truncation (unclosed `<think>`). In its own module so unit tests don't pull in `openai`/`aiolimiter`.

### B.4 `llms/providers/__init__.py` (new — was missing)

* Empty file. Without it, `from llms.providers.X import Y` was a namespace package and pytest's import machinery couldn't resolve test paths cleanly.

### B.5 `llms/providers/hf_utils.py`

* Lazy-import `text_generation` inside `generate_from_huggingface_completion`. The HF SDK is heavy and unused in OpenAI/Ollama deployments. Previously module import failed if `text_generation` wasn't installed even if the user wasn't on HF backend.

### B.6 `llms/tokenizers.py`

* `Tokenizer("openai", model)` now falls back to `tiktoken.get_encoding("cl100k_base")` when `tiktoken.encoding_for_model(name)` raises `KeyError`. Required for Ollama model names like `qwen2.5:7b-instruct`.
* Lazy-import `transformers.LlamaTokenizer` (only loaded for HF provider).

### B.7 `evaluation_harness/helper_functions.py`

* `llm_fuzzy_match` and `llm_ua_match` now read `WEBARENA_EVAL_MODEL` env var (default `gpt-4-1106-preview` preserved for backward compatibility). With Ollama, set `WEBARENA_EVAL_MODEL=qwen2.5:7b-instruct` so the eval-time judge uses the same backend as the agent.

### B.8 `MULTIWORKER_PLAN.md` (new)

* The design doc. 14 sections covering problem statement, state-surface analysis, reset semantics, implementation plan, verification, and 9 audit findings (§14) that resolved every open question before code was written.

## C. Bug fixes captured during smoke testing

1. **`mp.orchestrator` multiprocessing context mismatch** — Queue was fork-context, Process was spawn-context → `SemLock cannot be shared`. Fixed by getting Queue from the same spawn context.
2. **`docker commit` was way too slow** — 10+ min on shopping container's 1 GB writable layer; would be hours on gitlab's 24 GB layer. Replaced with `docker tag` of the already-populated source image.
3. **GitLab nginx listen_port followed external_url** — setting `external_url 'http://host:8123'` made GitLab Omnibus listen on 8123 internally, but docker maps host:8123 → container:8023, so connections failed. Fixed by pinning `nginx['listen_port'] = 8023` in gitlab.rb during configure.
4. **GitLab puma OOM with multiple replicas** — Omnibus defaults to `puma['worker_processes'] = cpu_count` (128 on this box), so 2 replicas spawned 256 workers and OOM-killed each other. Fixed by pinning `puma['worker_processes'] = 4`.
5. **Magento reset privilege** — `mysql -u root` denied; only `magentouser` has DB-level privileges. Use `mysqldump --add-drop-table` to stay at the TABLE level.
6. **Magento Redis cache stale after DB restore** — env.php configures Redis backends for `default` (db 0) and `page_cache` (db 1). Added `redis-cli -n 0 FLUSHDB; redis-cli -n 1 FLUSHDB` to `reset_magento`.
7. **GitLab health endpoint returns 502 with body** — `/users/sign_in` returns 502 to bare curl due to Host header mismatch but the response body has the page. `_wait_healthy` now accepts by `expect_body_contains` when supplied.
8. **Postmill DB credentials** — confirmed `pgsql://postmill:postmill@localhost:5432/postmill` from nginx fastcgi_param. No `su - postgres` needed; `psql -U postmill` works directly.
9. **Reset timeouts too short** — `rm -rf var/cache/*` on Magento can take >60s under contention. Raised to 300s. Postmill cache rm raised to 180s.
10. **eval-time LLM hard-coded** — `evaluation_harness/helper_functions.py` used `gpt-4-1106-preview`. Now `os.environ.get("WEBARENA_EVAL_MODEL", "gpt-4-1106-preview")`.
11. **NLTK + tiktoken offline support** — Both fetch resources at first use. Pre-stage to `NLTK_DATA` / `TIKTOKEN_CACHE_DIR` to avoid runtime downloads on air-gapped hosts.
12. **Worker_3 shopping_admin port collision with the shared map container** — `BASE_PORTS["shopping_admin"] + port_stride * 3 = 8080`, which is bound by the `map` container's `127.0.0.1:8080` mapping. Rootless docker's port allocator rejects any new 0.0.0.0:8080 bind. Added `_HOST_PORT_RESERVED = {8080, 8085}` to `mp/config.py` and a port-shift loop in `MPConfig.port_for` so the colliding port advances +10 (worker_3 → 8090). Add more reserved ports here if you stand up other shared services on the host. Verified during the N=7 shopping_admin run on 2026-05-27.
13. **`require_reset` flag was ignored** — `mp/worker.py:_run_one_task` originally computed `sites_to_reset = touched_sites | dirty_sites` regardless of the task config's `"require_reset"` field. For read-only tasks (e.g. sales-report queries 0–6) this meant always running a full mysqldump restore, which then blew past the 600 s `reset_magento` timeout under N≥4 contention (see §4.4). The fix honors the existing per-task flag: dirty sites are always reset, but newly-touched sites only get reset when `require_reset` is true (default true if missing). See §4.8 for the full semantics.

## D. Environment variables added

| Var | Used by | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_BASE` | `_setup_openai_api` | unset | If set, redirects openai client to this base URL (Ollama compat) |
| `WEBARENA_STRIP_REASONING_TAGS` | `_reasoning.strip_reasoning` | `0` | If `1`, strip `<think>…</think>` from model responses |
| `WEBARENA_EVAL_MODEL` | `llm_fuzzy_match`, `llm_ua_match` | `gpt-4-1106-preview` | LLM used for eval-time string judging |
| `WEBARENA_AUTH_FOLDER` | `run._run_one_task_loop` | unset (uses tempdir) | Per-worker `auto_login.py` auth folder |
| `WEBARENA_WORKER_ID` | informational | unset | Identifies the worker in logs/debug |
| `NLTK_DATA` | nltk (stock) | `~/nltk_data` | NLTK resource cache |
| `TIKTOKEN_CACHE_DIR` | tiktoken (stock) | varies | tiktoken encoding cache (use to avoid runtime download) |
| `PLAYWRIGHT_BROWSERS_PATH` | playwright (stock) | varies | Playwright browser binary location |

## E. New directories created at bring-up

```
<golden_root>/                  e.g. /data/webarena-mp/golden/
  shopping/golden.sql           Magento DB dump
  shopping_admin/golden.sql
  reddit/golden.dump            Postgres custom format
  reddit/submission_images.tar
  reddit/media_cache.tar
  gitlab/                       Filesystem mirror of /var/opt/gitlab (~22 GB)

<config_files_root>/            e.g. /data/webarena-mp/config_files/
  w0/                           812 JSON files with worker_0's URLs
    0.json
    1.json
    ...
    811.json
  w1/                           812 JSON files with worker_1's URLs
  ...
  w{N-1}/

<auth_root>/                    e.g. /data/webarena-mp/auth/
  w0/                           Per-worker cookie/storage state
    shopping_state.json
    shopping.shopping_admin_state.json
    ...
  w1/
  ...

<result_dir>/                   e.g. /data/webarena-mp/results/
  scores.jsonl                  Append-only one-result-per-line
  smoke.log                     Stdout of the orchestrator
  logs/
    worker_0.log
    worker_1.log
    ...
  w0/
    render_{task_id}.html       Trajectory render per task
    traces/{task_id}.zip        Playwright trace per task
  w1/
  ...
```

## F. Containers added at bring-up (for N=2)

```
shopping                — port 7770, worker_0, legacy name (live)
shopping_admin          — port 7780, worker_0
forum                   — port 9999, worker_0
gitlab                  — port 8023, worker_0
shopping_w1             — port 7870, worker_1
shopping_admin_w1       — port 7880, worker_1
forum_w1                — port 10099, worker_1
gitlab_w1               — port 8123, worker_1
map                     — port 13000, shared
wikipedia               — port 8888, shared
```

For N=8, each worker_w (w∈{1..7}) gets its own `_w{w}` set; port stride is 100.

## G. Files NOT modified

The harness was carefully designed to be **additive**. The following stock WebArena modules were left untouched (which means stock single-worker runs still work exactly as before):

* `browser_env/*` — actions, env, processors, auto_login (the auth_folder is a parameter; we just pass our per-worker path)
* `agent/*` — agent construction and prompts
* `evaluation_harness/evaluators.py` — eval routing (StringEvaluator, URLEvaluator, HTMLContentEvaluator)
* `config_files/test.raw.json` — task definitions (read-only)
* `scripts/*` — generate_test_data, html2json, etc.
* `setup_env.sh`, `prepare.sh`, `parallel_run.sh` (the broken 5-tmux-pane parallel run we replaced)

The integration points are exactly four: `run.py` (added flag + adapter), `evaluation_harness/helper_functions.py` (one env-var override), `llms/providers/*` (Ollama backend + reasoning strip + lazy imports + tokenizer fallback), and the new `mp/` package.
