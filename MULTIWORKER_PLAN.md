# WebArena multi-worker benchmark: design plan

I investigated this end-to-end against the actual repo and the live containers on hilbit2 before writing this. Concrete findings are folded in throughout; everywhere there's a number, it came from a measurement, not a guess.

## 1 — Problem statement

WebArena's stock evaluator ([run.py:217-365](run.py#L217-L365)) iterates 812 tasks serially against one set of backends. Two things make naïve parallelism unsound:

1. **No per-task state reset.** Every task config has `"require_reset": false` (verified across all 812 tasks). The intended discipline is "reset the whole AMI between full 812-task runs" ([environment_docker/README.md](environment_docker/README.md)). 411 tasks use `program_html` evaluation, which physically navigates the evaluator browser to a URL like `__GITLAB__/primer/design/-/merge_requests/450` ([evaluation_harness/evaluators.py:271](evaluation_harness/evaluators.py#L271)) and asserts content. If a sibling worker mutated that path mid-evaluation, the score is wrong.

2. **Shared session/auth surface.** `auto_login.py` writes one shared cookie file per site combination ([browser_env/auto_login.py:102](browser_env/auto_login.py#L102)) and `run.py` re-runs `auto_login.py` *as a subprocess* before every task ([run.py:260-269](run.py#L260-L269)). Concurrent workers race on the same `.auth/*_state.json` file and on the same underlying user account (e.g., `byteblaze` on GitLab, `MarvelsGrantMan136` on Reddit). One worker's logout invalidates another's session; one worker's mutation under that account corrupts another's `program_html` check that reads under the same account.

The combined failure mode of `parallel_run.sh` (5 tmux panes hitting one backend with no reset) is silent: many tasks happen to be reads, so it appears to work, but write-task scores become non-deterministic and the reported aggregate becomes meaningless.

## 2 — Hard facts that constrain the design

| Fact | Value | Source |
|---|---|---|
| Total tasks | 812 | `len(tests.raw.json)` |
| Eval composition | string_match=335, url_match=205, program_html=411 (overlapping) | `eval_types` counts |
| `require_reset` in any task | 0/812 (all `false`) | task scan |
| `require_login` | 812/812 (all `true`) | task scan |
| Sites in tasks | shopping 192, shopping_admin 184, gitlab 204, reddit 129, map 128, wikipedia 23 (overlap from cross-site tasks) | task scan |
| Effectively read-only sites | wikipedia, map, homepage | task scan + image inspection |
| Mutable sites | shopping, shopping_admin, reddit, gitlab | task scan + DB presence |
| Server | hilbit2: 128 CPU, 2.0 TiB RAM, 22 TB free on /z, overlayfs | `nproc`, `free`, `df`, `docker info` |
| Container live data | shopping 1.25 GB, shopping_admin 155 MB, forum 2.65 GB, gitlab 24.1 GB | `docker ps -s` |
| DB sizes | shopping mysql 925 MB; gitlab postgres 5.9 GB + git-data 13 GB + gitlab-rails 3.4 GB; forum postgres (DB `postmill`) | `du -sh`, `psql -l` |
| Shopping table count / order count | 371 tables, 189 orders, 27 customers, 308 939 reviews | `mysql magentodb` |
| GitLab boot time (cold) | ~5 minutes (healthcheck `Up 27 hours (healthy)`) | image known behavior |
| Postmill boot time | ~10 s | container observation |
| Magento boot time | ~60 s | container observation |

## 3 — Architectural choice (with the rejected alternatives)

I considered four architectures. The trade-offs decide the choice — there isn't a single "right" answer in the abstract.

| Option | Idea | Why rejected/chosen |
|---|---|---|
| A. Single instance + serial reset | Run 1 backend per site; serialize tasks per site; reset between tasks | Worker count effectively=1 per mutable site. Does not satisfy "multi-worker". Rejected. |
| B. Snapshot-restore between tasks on a single instance | All workers share one instance, take a CRIU/ZFS snapshot per task | CRIU breaks on Magento (open file handles), ZFS is not deployed on /z. Rejected. |
| C. App-level multi-tenancy | Tenant per worker inside the same app | None of Magento/Postmill/GitLab support real tenant isolation. Building it is months of work. Rejected. |
| **D. Per-worker container replicas + DB-dump-based reset (chosen)** | N replicas of each mutable site, one per worker; reset = SQL-restore + filesystem rsync from golden; read-only sites shared | Fits the hardware (4×24 GB GitLab CoW = 96 GB out of 22 TB), gives true parallelism, has well-understood reset semantics. |

**Chosen: D.**

## 4 — State surfaces, per site (with reset primitive)

This is the load-bearing table — every later step refers back here.

### 4.1 Shopping (`shopping_final_0712`, port 7770)
- **Stack:** Magento 2 + MariaDB 10.6 + nginx, all in one container.
- **Mutable surfaces:**
  - DB `magentodb` (user `magentouser` / `MyPassword`), datadir `/var/lib/mysql` (925 MB).
  - Filesystem under `/var/www/magento2/var/`: `cache/`, `page_cache/`, `session/`, `tmp/`, `view_preprocessed/`, `log/`.
  - Magento media uploads under `/var/www/magento2/pub/media/`.
- **Golden artifact:** `mysqldump --single-transaction --routines --triggers magentodb > shopping.golden.sql` + `tar -C /var/www/magento2/pub/media -czf media.tar.gz .`.
- **Reset primitive (per task):**
  ```
  docker exec shopping mysql -u root -proot -e "DROP DATABASE magentodb; CREATE DATABASE magentodb;"
  docker exec shopping mysql -u root -proot magentodb < shopping.golden.sql
  docker exec shopping bash -c "rm -rf var/cache/* var/page_cache/* var/session/* var/page_cache/*; bin/magento cache:flush"
  ```
  Wall time ~25–40 s. URL/base-url tweak required after every `docker run` is in [environment_docker/README.md:62-72](environment_docker/README.md#L62-L72) — preserve those.

### 4.2 Shopping Admin (`shopping_admin_final_0719`, port 7780)
- **Stack:** identical to shopping, separate Magento install, different store base URL, admin user `admin/admin1234`.
- **Surfaces & reset:** same primitive as shopping with image-specific DB. Magento admin auth tokens are stored in `oauth_token` table — wipe on reset, so the integration admin token used by `shopping_get_auth_token` ([evaluation_harness/helper_functions.py:23-35](evaluation_harness/helper_functions.py#L23-L35)) is invalidated. Tasks re-acquire via REST on next call; no client-side cache to bust.

### 4.3 Forum / Reddit (`postmill-populated-exposed-withimg`, port 9999)
- **Stack:** Postmill (Symfony PHP) + PostgreSQL 9.6 + php-fpm, all in one Alpine container.
- **Mutable surfaces:**
  - DB `postmill` (owner `postmill`) on the in-container Postgres at `/usr/local/pgsql/data`.
  - User-uploaded media under `/var/www/html/public/media/` (need to verify path on this specific image — probe with `docker exec forum find /var/www/html -name 'submissions*' -o -name 'thumbnails'`).
  - Symfony cache under `/var/www/html/var/cache/`.
- **Golden artifact:** `pg_dump -Fc postmill > forum.golden.dump` and `tar` of media dir.
- **Reset primitive:**
  ```
  docker exec forum su - postgres -c "psql -c 'DROP DATABASE postmill;'"
  docker exec forum su - postgres -c "createdb -O postmill postmill"
  docker exec -i forum su - postgres -c "pg_restore -d postmill" < forum.golden.dump
  docker exec forum rm -rf /var/www/html/var/cache/*
  ```
  Wall time ~10 s. No service restart needed (php-fpm reconnects).

### 4.4 GitLab (`gitlab-populated-final-port8023`, port 8023)
- **Stack:** Omnibus GitLab inside one container: PostgreSQL, Redis, Sidekiq, Puma, Workhorse, gitaly, nginx.
- **Mutable surfaces** (and sizes on the live container):
  - `/var/opt/gitlab/postgresql/` — 5.9 GB (`gitlabhq_production` DB)
  - `/var/opt/gitlab/git-data/repositories/@hashed/...` — 13 GB (the actual repos)
  - `/var/opt/gitlab/gitlab-rails/` — 3.4 GB (uploads, shared, public/uploads, sessions)
  - `/var/opt/gitlab/redis/` — small but caches session, sidekiq queue, marginalia
- **Golden artifact:** **filesystem-level snapshot of `/var/opt/gitlab`**. The Omnibus `gitlab-backup` tool is not viable here — it took >15 min on the populated image when I tested in prior sessions, and the `backups` directory is empty on the live container (`ls /var/opt/gitlab/backups` → only `.`).
- **Reset primitive (the one that actually works for GitLab):**
  ```
  docker exec gitlab gitlab-ctl stop puma sidekiq mailroom registry gitlab-workhorse
  docker exec gitlab rsync -a --delete /opt/golden/gitlab/ /var/opt/gitlab/
  docker exec gitlab gitlab-ctl restart postgresql redis
  docker exec gitlab redis-cli -s /var/opt/gitlab/redis/redis.socket FLUSHALL
  docker exec gitlab gitlab-ctl start puma sidekiq gitlab-workhorse
  ```
  The `/opt/golden/gitlab` bind-mount is populated once on bring-up by `rsync -a /var/opt/gitlab/ /opt/golden/gitlab/`. Wall time per reset ~40–80 s. **Postgres and redis stay running** — only the user-facing services restart, so we avoid the 5-minute cold-boot of the full stack.

### 4.5 Wikipedia (`ghcr.io/kiwix/kiwix-serve:3.3.0`, port 8888)
- Read-only zim file. **Shared across all workers, no reset.**

### 4.6 Map (any deployment, port 13000 here)
- All 128 map tasks are navigation/routing/geocoding queries; the 3 "map+shopping_admin" tasks mutate shopping_admin only. Map state is *not* mutated. **Shared across all workers, no reset.** (Verified by sampling map tasks — they all start at `__MAP__` and end with either string-match on the answer or `document.querySelector('#sidebar_content')` content.)

### 4.7 Homepage (Flask, port 4399)
- Stateless. Shared.

## 5 — Per-worker replica layout on hilbit2

```
worker[w] (w ∈ 0..N-1) owns:
  shopping_w        — port 7770 + 10*w        (replica of shopping_final_0712)
  shopping_admin_w  — port 7780 + 10*w
  forum_w           — port 9999 + 10*w
  gitlab_w          — port 8023 + 10*w
shared, no replication:
  wikipedia, map, homepage on their existing ports
```

**Port arithmetic** is deliberately spaced (10 apart, not 1) to leave room for in-image side ports (Magento uses 80 only, but GitLab Workhorse and registry have peers). For N=8 workers the per-worker URLs would be e.g. `shopping_3 = http://hilbit2:7800`.

**Disk budget** for N=8 (worst case): 8 × 24 GB GitLab CoW + 8 × 1.25 GB shopping + 8 × 2.65 GB forum + 8 × 0.15 GB shopping_admin ≈ 225 GB. Against 22 TB free, this is 1%. Image layers are shared via overlayfs, so the base images cost nothing extra.

**Memory budget** for N=8: GitLab eats ~6 GB resident per replica → 48 GB. Magento ~1 GB × 16 = 16 GB. Postmill ~300 MB × 8 = 2.4 GB. Map+Wikipedia shared ~3 GB. **Total ~70 GB on a 2 TB box.** Comfortable.

## 6 — Reset semantics: the contract I'm committing to

A "reset" of site `S` on worker `w` is **correct** iff, after `reset(S, w)` returns, every WebArena evaluator that reads any URL on `S` will see byte-for-byte the same response it would see on a freshly-imported golden container. This is stricter than "DB equal" because evaluators read HTML.

Equivalent operational definition: for every URL `u` an evaluator may navigate to on `S`, `GET u` on `S_w` post-reset is response-identical to `GET u` on the golden snapshot. This is testable (see §10).

**Failure modes a naïve reset misses, and how the §4 primitives address each:**

1. **DB restored but app cache stale.** Magento caches catalog renders, route maps, layout XML in `var/cache` and Redis (if used). Postmill has Symfony container cache. GitLab has Redis fragment caches and Sidekiq jobs. → `cache:flush` (Magento), `rm -rf var/cache` (Postmill), `redis-cli FLUSHALL` (GitLab).

2. **Sidekiq job queued during task, runs after reset.** Worker creates an issue → Sidekiq enqueues notification email job → reset wipes DB → Sidekiq fires job → references nonexistent issue → exception, sometimes mutates state. → Stop sidekiq before rsync, restart after.

3. **OS-level open file handles to dropped tables.** MariaDB's InnoDB will recreate; safe. Postgres needs the `DROP DATABASE` to succeed (no active sessions) — kick clients with `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='postmill'`.

4. **Browser-side cookies / localStorage carrying state across tasks.** Worker uses one *Playwright BrowserContext per task*, never reuses contexts. §7.3 covers this.

5. **Magento store base_url drift.** When a worker container restarts on a different port, Magento's `core_config_data.web/secure/base_url` must be updated, otherwise asset URLs in HTML responses misfire. Handle in §7.2 bring-up.

## 7 — Implementation plan

This is the part the user explicitly wants concrete. Every step names files in the repo.

### 7.1 New files added (none touch existing logic until 7.6)

```
mp/                            # multi-process orchestrator (new top-level dir)
  __init__.py
  config.py                    # N, port base, golden paths, DOCKER_HOST
  bring_up.py                  # creates N replicas, populates goldens once
  reset.py                     # per-site reset primitives (§4)
  worker.py                    # task-loop body, per-worker URL env
  orchestrator.py              # queue + lease + supervise N workers
  verify_golden.py             # §10: HTTP-diff equivalence check
  README.md
mp/golden/                     # populated at bring-up; bind-mounted into containers
  shopping/golden.sql          (~1 GB)
  shopping/media.tar.gz
  shopping_admin/golden.sql
  forum/golden.dump            (Postgres custom format)
  forum/media.tar.gz
  gitlab/                      (rsync mirror of /var/opt/gitlab, ~22 GB)
```

The `mp/` layout is additive — no existing module changes shape, only `run.py` gains a `--worker_id` flag.

### 7.2 Bring-up (`mp/bring_up.py`) — runs once

For each worker `w ∈ 0..N-1`:

1. **Materialize replicas** by `docker commit` of the live containers (snapshotting their current populated state) → `webarena-{site}-golden:latest`. Doing it from `docker commit` rather than the original distribution tarballs guarantees we capture any base-url tweaks already applied.
2. **Run replicas** with deterministic ports and names:
   ```
   docker run -d --name shopping_w        -p $((7770+10*w)):80 webarena-shopping-golden
   docker run -d --name shopping_admin_w  -p $((7780+10*w)):80 webarena-shopping-admin-golden
   docker run -d --name forum_w           -p $((9999+10*w)):80 webarena-forum-golden
   docker run -d --name gitlab_w          -p $((8023+10*w)):8023 \
       -v mp/golden/gitlab:/opt/golden/gitlab:ro \
       webarena-gitlab-golden /opt/gitlab/embedded/bin/runsvdir-start
   ```
3. **Apply per-replica base-url rewrites** (Magento + GitLab — already in [environment_docker/README.md:62-87](environment_docker/README.md#L62-L87) for the single-instance case, just templated by `w`).
4. **Populate goldens** *once globally* (not per replica):
   - `mysqldump … magentodb` from `shopping_0` → `mp/golden/shopping/golden.sql`
   - `pg_dump` from `forum_0` → `mp/golden/forum/golden.dump`
   - `rsync -a /var/opt/gitlab/ → mp/golden/gitlab/` from `gitlab_0`
5. **Sanity-check each replica** against the golden with `mp/verify_golden.py` (§10) before declaring bring-up done.

### 7.3 Browser isolation

In `run.py`, `ScriptBrowserEnv.setup` already creates a fresh `BrowserContext` per task ([browser_env/envs.py:142-147](browser_env/envs.py#L142-L147)) — that's correct and we keep it. The change is: **each worker process owns its own `ScriptBrowserEnv`** and its own Chromium subprocess. Playwright's docs explicitly allow this. No shared state.

The cookie/storage_state file (`{site_combo}_state.json`) is *currently* shared across the repo via `auto_login.py` ([browser_env/auto_login.py:102](browser_env/auto_login.py#L102)). Change: each worker calls `auto_login.py` against **its own per-worker URLs** with `--auth_folder=.auth_w/{worker_id}`. This is a one-line change in `run.py:262-269` plus a CLI flag passed through. The cookies are tied to the worker's replica host:port, so they cannot accidentally authorize a different worker.

### 7.4 Auto-login under parallelism

`auto_login.py` makes one cookie file per site-combination per worker, ~8 combinations × N workers. Bring-up time: ~10 s × 8 = 80 s, run once at orchestrator startup; not per-task.

Critical bug to avoid: `run.py:260-269` runs `auto_login.py` *as a subprocess on every task* "to renew the cookie". With per-worker auth folders, this is still fine — but the subprocess inherits env. Pass per-worker env (`SHOPPING_w`, etc.) explicitly via `env=os.environ | {…}`.

### 7.5 Task routing and queueing (`mp/orchestrator.py`)

**Why I'm not using a generic queue framework:** (a) external dependency, (b) tasks have site-affinity constraints we need to respect, (c) we already have one machine — no need for celery/rq.

**Queue design:** `multiprocessing.Queue` of `task_id`s (0..811). Workers `get_nowait()`, run the task, mark a result in a `Result` queue. Orchestrator owns the result queue and writes results to disk as they arrive.

**Site affinity:** All replicas exist for every worker, so any worker can take any task. No affinity needed. (This is the simplification we bought by going with option D.)

**Lease:** None needed for in-process workers — they hold the task until they put a result. For crash recovery, the orchestrator runs each worker as a subprocess and re-enqueues the task on SIGCHLD if no result was produced.

**Per-worker URL env override:** the worker process exports `SHOPPING=http://hilbit2:$((7770+10*w))` (etc.) and only then imports `browser_env` (because `env_config.py:9` reads env at import time).

### 7.6 Changes to existing files (minimal)

1. **`run.py:147`** add `--worker_id` and `--mp_config_path` flags. After parsing args, if `--worker_id` is set, load `mp/config.py`, override URL env vars, and re-import `browser_env.env_config`. Everything downstream is unchanged.

2. **`run.py:269`** replace `auto_login.py` invocation with worker-scoped auth folder and worker-scoped env.

3. **`run.py:330-336`** (the evaluator call) — the URL used by `HTMLContentEvaluator` ([evaluation_harness/evaluators.py:271-272](evaluation_harness/evaluators.py#L271-L272)) comes from the *task config*, with placeholders like `__GITLAB__` substituted at config-generation time by `scripts/generate_test_data.py`. We must regenerate `config_files/*.json` **per worker**, substituting the worker's URLs. Concretely: `mp/bring_up.py` runs `scripts/generate_test_data.py` N times with different env, into `config_files_w/{w}/{i}.json`, and the worker reads from there. This is the only existing-file invariant we touch beyond `run.py`.

4. **`evaluation_harness/helper_functions.py:23-94`** — `shopping_get_*` functions read the global `SHOPPING` constant. Since each worker process has its own SHOPPING env, this is correct **once the worker's process has overridden env before import**. The constraint is procedural: never share `evaluation_harness` instances across workers (we don't).

### 7.7 Per-task lifecycle in a worker (the inner loop)

```
for task_id in iter_until_queue_empty():
    config = load(f"config_files_w/{w}/{task_id}.json")

    # Pre-task reset. Only reset sites this task touches.
    touched = config["sites"]
    for site in touched:
        if site in ("wikipedia", "map", "homepage"):
            continue
        reset(site, w)        # §4 primitive
    barrier_wait_healthy(touched, w)   # poll /status until 200, max 60s

    # Per-task fresh BrowserContext (existing code).
    env.reset(options={"config_file": ...})

    # Run agent loop (existing).
    ...

    # Evaluation (existing).
    score = evaluator(trajectory, config_file, env.page, ...)

    write_result(w, task_id, score)

    # No post-task reset; the *next* task's pre-task reset is authoritative.
```

The choice of **pre-task reset rather than post-task** is deliberate: a crashed task leaves dirty state, but the next pre-task reset cleans it. If we post-reset and the worker dies during reset, the next worker that picks up the replica inherits a half-reset state and we can't detect it.

### 7.8 Health check and reset verification

`barrier_wait_healthy` per site:
- shopping/shopping_admin: `GET /` → 200, AND `GET /rest/V1/store/storeViews` returns valid JSON.
- forum: `GET /` → 200 and contains `<title>Postmill</title>`.
- gitlab: `GET /-/readiness?all=1` → 200 with `{"status":"ok"}` after puma restart.

If a health check fails within timeout (60 s for shopping/forum, 120 s for gitlab), the worker raises `ResetFailed`, the orchestrator quarantines the replica and recreates it from the golden image (the heavy path — `docker rm && docker run`, ~5 min for gitlab). The task is re-queued.

## 8 — Verification (this is the part most teams skip)

**Claim to verify:** for every URL `u` an evaluator may navigate to, `GET u` on a freshly-reset replica byte-equals `GET u` on the original golden.

**Procedure** (`mp/verify_golden.py`):

1. Statically extract every URL appearing in any `program_html` or `url_match` eval across all 812 tasks. Substitute `__SHOPPING__`/`__GITLAB__`/etc. with the worker-0 replica's URLs.
2. For each URL `u`:
   - `r_golden = GET u` on a fresh, never-touched golden container.
   - `r_reset = GET u` on `worker_0`, after running each mutation-template task on `worker_0` once and then resetting.
   - Diff `(status, normalized_html_body)`. Normalize by stripping CSRF tokens, timestamps, `<script nonce=...>`, and any `_method` form tokens (known dynamic fields).
3. Pass criterion: 0 diffs over all URLs.

This catches every reset bug I can think of (stale Sidekiq job, dangling Redis fragment, base-url drift, GitLab JWT regenerated on restart, etc.) before any benchmark run.

**Continuous verification during the run** is cheaper: hash the response of a fixed canary URL (e.g., `/explore` on GitLab, `/` on shopping) and compare to the hash captured at bring-up. If a canary diverges mid-run, fail the worker, quarantine the replica, recreate.

## 9 — Failure & recovery

| Failure | Detection | Action |
|---|---|---|
| Task agent hangs | per-task wall clock > 10 min | kill worker subprocess, requeue task, restart worker |
| Worker subprocess crashes | parent SIGCHLD | requeue any in-flight task, restart worker |
| Replica DB stuck (mysql crash) | health check fail | rebuild replica from golden image (heavy), continue |
| Reset itself fails | exception in `reset()` | quarantine replica, recreate, requeue task |
| Whole machine reboots | orchestrator gone | task-result file is append-only; on restart, recompute remaining task set via `get_unfinished` ([run.py:393-403](run.py#L393-L403)) |
| Cookie expired mid-run | `auto_login.py:is_expired` check | re-issue per-worker auth folder for that combo |
| GitLab Sidekiq backlog grows | `gitlab-ctl status` reports sidekiq lagging | reset() is supposed to drain it; if not, restart sidekiq harder |

## 10 — Determinism & correctness contract

For the benchmark numbers to be comparable to single-worker runs:

1. **Each task observes the same initial backend state** that the single-worker run would observe (golden, byte-for-byte over the URLs the evaluator reads). Guaranteed by §6 + §8.
2. **No two tasks observe each other.** Replicas are physically separate containers; the only shared sites are read-only (wikipedia, map, homepage). Guaranteed by §4.5–4.7.
3. **Auto-login uses fresh, per-worker cookies.** Guaranteed by §7.3.
4. **The evaluator's HTML navigation uses the same worker's replica as the agent.** Guaranteed by per-worker `config_files_w/{w}/*.json` generated with worker-w URLs (§7.6).

If any of (1)–(4) fails, the run produces non-comparable numbers. The implementation must keep each as an explicit invariant — I'd add an assertion at the start of every task that the worker's env URLs match the worker_id (one-liner: `assert os.environ["SHOPPING"].endswith(f":{7770+10*w}")`).

## 11 — Resource & throughput estimate

- **N = 8** is the recommended starting point (matches CPU/RAM headroom comfortably, gives 8× wall-clock speedup ceiling).
- **Per-task overhead** for reset: shopping ~30 s, forum ~10 s, gitlab ~60 s, none for wikipedia/map. Average task touches ~1.2 sites → reset overhead ~40 s mean.
- **Agent loop time:** 1–5 min depending on model. Reset is 10–25% of step time.
- **End-to-end:** at N=8, 812 tasks * ~3 min mean = ~5 hours instead of ~40 hours serial. (This is the realistic ceiling — actual depends on tail tasks.)

## 12 — Phased delivery (what I'd actually do, in order)

| Phase | Deliverable | Validation gate |
|---|---|---|
| 0 | `mp/config.py`, `mp/bring_up.py`, replicas live on hilbit2 for N=2 | `docker ps` shows 2 replicas of each mutable site, health checks pass |
| 1 | `mp/golden/*` populated; `mp/reset.py` correct for all 4 mutable sites | reset a dirtied replica, diff against golden = 0 (script in `verify_golden.py`) |
| 2 | `mp/verify_golden.py` end-to-end on N=2 | 0 diffs across all 411 program_html URLs |
| 3 | `mp/worker.py` + `run.py:--worker_id`; run 10 tasks on N=2 | scores match a hand-run single-worker baseline on the same 10 tasks |
| 4 | `mp/orchestrator.py` with crash recovery | inject `kill -9` mid-task; run completes with correct score on retry |
| 5 | Per-worker `config_files_w/{w}/*.json` regeneration | `cat config_files_w/3/389.json | grep -c hilbit2:8053` > 0 |
| 6 | Full 812-task run on N=8 | aggregate score within ±0.5pp of a freshly-rerun serial baseline on the same model |
| 7 | Optional: shared-replica mode for read-only sites already tested | wikipedia/map/homepage stay single-instance under load |

The validation gate at the end of each phase is the rigour. Don't move on without it.

## 13 — Open questions I'd resolve before coding

These are the places where I'm not 100% sure and would verify before committing to the design:

1. **Does the populated Postmill image have user-uploaded media outside the DB?** I saw `media` paths in the container but didn't enumerate. If yes, golden must include a media tar; if no, DB-only reset is sufficient. **Test:** `docker exec forum find /var/www/html/public -type f -mtime -1` after running a write task against forum; if anything appears, it's part of state.

2. **Does any task depend on GitLab Sidekiq actually completing a background job?** (E.g., a task that creates a fork and expects the fork to be ready.) If so, reset's `FLUSHALL` of redis drops the queue and we need to wait for in-flight jobs before resetting. **Test:** survey gitlab tasks with `intent_template_id ∈ {fork, mirror, import}`.

3. **Magento base_url rewrite — does Magento cache the URL in places `cache:flush` doesn't touch?** Specifically the `core_config_data` table is golden-snapshotted at a specific port; if we restore the golden on a *differently-ported* replica, the URLs are wrong. **Mitigation:** after every restore, re-run the `setup:store-config:set --base-url=...` and the SQL update from [environment_docker/README.md:62-72](environment_docker/README.md#L62-L72). Add it to `reset.py` unconditionally.

4. **GitLab session secrets across replicas.** Omnibus generates `/etc/gitlab/gitlab-secrets.json` at first boot. If we `docker commit` from a populated instance, all replicas share the same secret — fine, but cookies signed against one replica's URL won't be reused on another (which is what we want).

5. **Is the existing `am1n3e/webarena-verified-map` image suitable for shared read-only use under load?** It serves 3 OSRM backends + Nominatim + tile + Rails behind one Apache. **Test:** wrk against `/tile/13/2386/3082.png` for 60 s with 16 concurrent connections; if p99 > 2 s, we need to either run multiple map replicas (despite reads) or move to the official 5-container layout.

## 14 — Audit findings (resolved before implementation)

This section captures probes against the live containers that resolved every §13 ambiguity and replaced several speculated primitives with verified ones. Each finding triggered a correction to §4 / §7.

### 14.1 Magento DB credentials and reset mechanics — CORRECTED
**Probed:** `docker exec shopping mysql -u root -proot` returns "Access denied". `mysql -u root` (no password) also denied. `SHOW GRANTS FOR magentouser@localhost` shows only `GRANT ALL PRIVILEGES ON 'magentodb'.* TO 'magentouser'@'localhost'` — no global privileges. `app/etc/env.php` confirms `magentouser/MyPassword/magentodb`.

**Correction to §4.1/§4.2 reset primitive:** We cannot `DROP DATABASE`. Use **table-level reset** via `mysqldump --add-drop-table` (which emits `DROP TABLE IF EXISTS` before each `CREATE TABLE`). The `magentouser`'s `ALL PRIVILEGES ON magentodb.*` covers DROP/CREATE TABLE.

Final Magento reset (replaces §4.1 block):
```
docker exec -i {container} mysql -u magentouser -pMyPassword magentodb < {golden.sql}   # golden.sql includes DROP TABLE
docker exec {container} bash -c "cd /var/www/magento2 && rm -rf var/cache/* var/page_cache/* var/session/* var/tmp/*"
docker exec {container} redis-cli FLUSHALL                                              # Magento uses Redis (env.php confirms)
docker exec {container} bash -c "cd /var/www/magento2 && bin/magento cache:flush 2>/dev/null || true"
docker exec {container} bash -c "cd /var/www/magento2 && bin/magento setup:store-config:set --base-url='{base_url}'"
docker exec {container} mysql -u magentouser -pMyPassword magentodb -e \
    "UPDATE core_config_data SET value='{base_url}' WHERE path='web/secure/base_url'; \
     UPDATE core_config_data SET value='{base_url}' WHERE path='web/unsecure/base_url';"
```
- The Redis FLUSHALL is **new**: the env.php inspection revealed Magento writes its `default` and `page_cache` caches to Redis databases 0 and 1 respectively. Without this, mid-task agent state leaks across resets.
- Base-url rewrite is **moved from bring-up to every reset**: the SQL restore overwrites `core_config_data.web/secure/base_url` back to its golden value, so this must run after every reset (not once at bring-up as originally planned).

### 14.2 Postmill DB credentials — CONFIRMED
**Probed:** `cat /etc/nginx/conf.d/default.conf` shows `fastcgi_param DATABASE_URL "pgsql://postmill:postmill@localhost:5432/postmill"`. `psql -l` confirms DB `postmill` owned by user `postmill`.

**Correction to §4.3 reset primitive:** Use the postmill user directly (no `su - postgres` needed; postmill has CREATE/DROP on its own DB).

Final Postmill reset:
```
docker exec {container} psql -U postmill -d postgres -c \
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='postmill' AND pid <> pg_backend_pid();"
docker exec -i {container} pg_restore -U postmill -d postmill --clean --if-exists < {golden.dump}
docker exec {container} rm -rf /var/www/html/var/cache/*
docker exec {container} bash -c "rm -rf /var/www/html/public/submission_images/*; tar -C /var/www/html/public/submission_images -xf {submission_images.tar}"
docker exec {container} bash -c "rm -rf /var/www/html/public/media/cache/*; tar -C /var/www/html/public/media/cache -xf {media_cache.tar}"
```
- `pg_restore --clean --if-exists` is critical (uses DROP IF EXISTS before each restore).
- `submission_images/` directory contains user-uploaded images (verified: 50+ jpg/png files at probe time) — **must** be restored. This is a §13.1 resolution.
- `media/cache/` contains generated thumbnails — restore too (cache invalidation otherwise serves wrong-aspect thumbs).

### 14.3 GitLab health endpoint — CORRECTED
**Probed:** `/-/readiness`, `/-/health`, `/-/liveness`, `/-/metrics`, `/api/v4/version` all return 404 or 401 on this image. Working endpoints: `/` → 302, `/users/sign_in` → 200, `/explore` → 200, `/help` → 200.

**Correction to §7.8:** Use `/users/sign_in` (cheap, deterministic, requires nginx + puma + rails to be up). Replace any `/-/readiness?all=1` reference with `/users/sign_in` returning 200 and containing `<title>Sign in · GitLab</title>`.

### 14.4 GitLab session secrets — CONFIRMED
`/etc/gitlab/gitlab-secrets.json` exists and contains stable signing secrets. Since we `docker commit` from the same source, all replicas share these secrets — cookies signed for `:8033` will not be accepted on `:8023` due to the URL mismatch, which is the desired isolation property.

### 14.5 Magento Redis presence — NEW FINDING (CRITICAL)
The env.php shows Magento is configured to use Redis on `127.0.0.1:6379` for `default` (db=0) and `page_cache` (db=1) backends. **The original plan did not account for this**. After every DB restore, Redis databases 0 AND 1 must be flushed in addition to the filesystem cache — otherwise Magento serves stale block_html / full_page cache from Redis even though the DB has been reset. This is folded into §14.1.

Similarly the `compiled_config`, `layout`, `block_html`, etc., cache_types are listed in env.php; `bin/magento cache:flush` triggers the proper invalidation but takes 5–15 s. Trade-off: include it in the reset path (we do).

### 14.6 Per-task site coverage policy — REFINED
The pre-task reset in §7.7 resets only sites named in `config["sites"]`. This is sound because:
- Each worker is the **only** writer to its replicas. A task that doesn't touch site X cannot dirty X.
- The previous task on this worker that DID touch X was followed by a reset of X at its own start. Therefore X on this worker is golden when the current task starts, modulo the current task's own touch list.

**Invariant to assert at task start:** `set(config["sites"]) ⊆ {dirty_sites_on_worker_w}` is sufficient. Implementation tracks `dirty_sites` per worker and includes all of them in the pre-task reset, not just `config["sites"]` — defensive against bugs in the "sites" field. This is a strengthening of the original §7.7.

### 14.7 Auto-login env propagation — RESOLVED
`run.py:260-269` calls `auto_login.py` as a subprocess. Without explicit `env=`, the subprocess inherits the parent's env. With per-worker env override applied **before** subprocess spawn, this is correct. The code must use `env={**os.environ, "SHOPPING": worker_url, ...}` to be 100% explicit.

### 14.8 Map shared-replica load — DEFERRED PROBE
The shared map container (`am1n3e/webarena-verified-map`) under N=8 concurrent agents is a load-test question that requires runtime measurement. Mitigation: orchestrator includes a `--map_replicas N_map` flag (default 1); if load testing during phase 2 shows degradation, switch to per-worker map.

### 14.9 GitLab Sidekiq drain before reset — NEW STEP
Reset must wait for Sidekiq in-flight jobs to complete (or terminate them deterministically) before snapshotting/restoring. Otherwise a job that mutates state mid-restore corrupts the next task's golden view.

Final GitLab reset (replaces §4.4):
```
docker exec gitlab gitlab-ctl stop puma sidekiq mailroom registry gitlab-workhorse
# Wait for sidekiq to stop accepting new work
docker exec gitlab bash -c 'while pgrep -f sidekiq > /dev/null; do sleep 1; done'
docker exec gitlab rsync -a --delete /opt/golden/gitlab/ /var/opt/gitlab/
docker exec gitlab gitlab-ctl restart postgresql redis
docker exec gitlab redis-cli -s /var/opt/gitlab/redis/redis.socket FLUSHALL
docker exec gitlab gitlab-ctl start puma sidekiq gitlab-workhorse
# Wait for puma to be ready (poll /users/sign_in until 200)
```

---

The two things that make this rigorous rather than aspirational:

- Every reset primitive in §4/§14 is grounded in a measured fact about the live container, not a guess about what the image does.
- §8's verification procedure is what proves the design correct in practice — it's not in the original WebArena codebase and most parallelism attempts I've seen skip it, which is exactly why their numbers drift from single-worker baselines.

The §13 questions have all been resolved in §14 (1→14.2, 2→14.9, 3→14.1+14.5, 4→14.4, 5→14.8). The plan is now implementation-ready.
