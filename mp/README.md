# `mp/` — WebArena multi-worker harness

Run the 812-task WebArena benchmark with N parallel workers. Per-task state
reset is performed for every site the task touches, so the aggregate score is
identical (within run-to-run variance) to a single-worker run.

The architecture, state surfaces, and reset semantics are documented in
[MULTIWORKER_PLAN.md](../MULTIWORKER_PLAN.md). This README is the operator
runbook.

## Layout

```
mp/
  config.py         Single source of truth: N workers, port assignments,
                    container names, golden artifact paths, DB credentials.
  docker_exec.py    Wraps `docker exec` for local OR remote (SSH) docker
                    daemons. Used by every other module.
  reset.py          Per-site reset primitives (shopping/shopping_admin via
                    mysqldump+Redis FLUSHDB+cache:flush+base_url rewrite;
                    reddit via pg_restore+submission_images+media/cache;
                    gitlab via gitlab-ctl stop+rsync from /opt/golden/gitlab+
                    Redis FLUSHALL+restart). Each primitive ends with a
                    public-URL health poll.
  bring_up.py       One-shot: docker commit live containers → golden images,
                    docker run N replicas per mutable site, apply per-replica
                    base URL rewrites (Magento + GitLab), dump golden SQL
                    and tarballs, snapshot /var/opt/gitlab → host filesystem,
                    render per-worker config_files/w{w}/{id}.json, write
                    mp/config.json.
  worker.py         One Python process per worker. Pulls task ids off a
                    multiprocessing.Queue. For each task: (1) reset every
                    mutable site this worker has dirtied or this task
                    touches; (2) wait for health; (3) call run.run_single_task
                    to execute the agent + evaluator; (4) post a TaskResult.
  orchestrator.py   Spawns N workers, drains a task queue, persists results
                    to mp/results/scores.jsonl, supervises crashes and
                    re-enqueues their in-flight tasks.
  verify_golden.py  HTML-diff equivalence check between a freshly-reset
                    replica and an untouched source container for every URL
                    referenced by `program_html` evaluators in the 812 tasks.
                    Normalizes CSRF tokens / nonces / timestamps before diffing.
  tests/            37 unit tests covering port arithmetic, command
                    construction, normalization regex, and reset-dispatch
                    invariants. No docker / network required.
```

## Quick start

### Prereqs

- hilbit2 reachable over SSH as `wangcy07@hilbit2.cis.upenn.edu` (or whoever).
- Rootless docker daemon running at
  `unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock`.
- The six WebArena containers (shopping, shopping_admin, forum, gitlab,
  wikipedia, map) running and healthy. Verify with:
  ```
  ssh wangcy07@hilbit2.cis.upenn.edu \
      "DOCKER_HOST=unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock docker ps"
  ```
- Python 3.10+ on the orchestrator-running host (hilbit2 is fine, or your
  workstation with VPN access to UPenn for HTTP endpoints).

### 1. Run the unit tests

```
python3 -m pytest mp/tests/ -v
```

Expected: 37 passed.

### 2. Bring up N=2 replicas (one-shot, ~30 minutes)

```
python3 -m mp.bring_up --num_workers 2
```

What it does, in order, with rough wall times:

1. `docker commit` each of the 4 mutable source containers → golden images
   (~5 min total; gitlab is the heaviest).
2. `docker run` one extra replica per mutable site (the existing live
   containers are reused as worker_0). For N=2 that's 4 new containers.
3. Wait for each new container to boot (Magento ~60 s, Postmill ~10 s,
   GitLab ~5 min).
4. Rewrite each replica's base URL (Magento `setup:store-config:set` +
   `core_config_data` SQL; GitLab `external_url` in gitlab.rb +
   `gitlab-ctl reconfigure` — another ~5 min for GitLab).
5. Snapshot goldens onto the host:
   - `mysqldump --add-drop-table ...` for shopping and shopping_admin
   - `pg_dump -Fc` for postmill + `tar` of `submission_images` and
     `media/cache`
   - `tar -C /var/opt/gitlab` piped to host `tar -x` (this is the slow
     step — 5–15 min for ~22 GB).
6. Render per-worker task config files (substitutes `__SHOPPING__`/etc.
   placeholders with worker-specific URLs) into
   `/z/wangcy07/webarena/mp/config_files/w{w}/`.
7. Write `mp/config.json` so the orchestrator and workers can find each
   other.

Re-runs are safe and skip done work via the `--skip_goldens` and
`--skip_configure` flags.

### 3. Verify golden equivalence

```
python3 -m mp.verify_golden --out /z/wangcy07/webarena/mp/verify_report.json
```

This GETs every URL referenced by `program_html` evals (246 unique URLs
across 411 tasks) on both worker_0's replica and the source container, then
diffs after normalization. Exit 0 iff zero diffs.

For a quick sanity check during development, use `--max_urls 20`.

### 4. Run the benchmark

```
python3 -m mp.orchestrator \
    --model gpt-3.5-turbo-16k-0613 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
    --start_idx 0 --end_idx 812
```

Progress is logged to stdout and `/z/wangcy07/webarena/mp/results/logs/worker_{w}.log`.
Results are appended to `/z/wangcy07/webarena/mp/results/scores.jsonl` one
JSON object per task per line:
```json
{"worker_id": 3, "task_id": 491, "score": 1.0, "error": null, "duration_seconds": 187.4}
```

If the orchestrator is interrupted, re-running picks up where it left off —
only task ids without a successful entry in `scores.jsonl` get re-queued.

### 5. Inspect results

```
python3 - <<'PY'
import json, statistics
with open("/z/wangcy07/webarena/mp/results/scores.jsonl") as f:
    rows = [json.loads(l) for l in f if l.strip()]
ok = [r["score"] for r in rows if r["error"] is None]
err = [r for r in rows if r["error"] is not None]
print(f"completed: {len(ok)}/812")
print(f"aggregate score: {sum(ok)/len(ok):.4f}")
print(f"errors: {len(err)}")
for e in err[:5]:
    print(f"  task {e['task_id']} on worker {e['worker_id']}: {e['error'][:120]}")
PY
```

## Operational notes

### Choosing N

| N | Wall-clock estimate | Memory | Disk | When |
|---|---|---|---|---|
| 1 | ~40 h | 8 GB | 25 GB | reproduce stock |
| 2 | ~20 h | 16 GB | 50 GB | conservative first run |
| 4 | ~10 h | 32 GB | 100 GB | recommended |
| 8 | ~5 h  | 70 GB | 225 GB | aggressive, fits hilbit2 fine |
| 16+ | tail-bounded | 140 GB+ | 450 GB+ | needs map-replica scale-out |

For N ≥ 4 with the shared `am1n3e/webarena-verified-map` image, monitor
`/tile/...` p99 latency. If above ~2 s, add `--map_replicas` and the
orchestrator will pin each worker to its own map replica.

### What to do if a worker crashes

The orchestrator auto-detects worker crashes and re-enqueues the in-flight
task. If a worker keeps crashing on the same task, that's a real bug —
inspect `worker_{w}.log` and the corresponding trace.

### What to do if a reset fails

`ResetFailed` is raised by the reset primitive when the health probe
times out. The orchestrator records this as a task error (no score). To
recover the replica, `docker rm -f <container>` and re-run `mp.bring_up`
with `--skip_goldens` — it will re-create only the missing replica.

### Adding a new normalization rule

If `verify_golden.py` shows diffs caused by a legitimately-dynamic field
(e.g., a new CSRF token in some GitLab page), append a `(regex, replacement)`
to `NORMALIZERS` in `mp/verify_golden.py`. Run
`python3 -m pytest mp/tests/test_verify_golden.py` to verify the rule
doesn't break existing normalizations.

### Cleaning up

```
ssh wangcy07@hilbit2.cis.upenn.edu \
    "DOCKER_HOST=unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock docker ps -a --format '{{.Names}}'" \
    | grep -E '_(w[0-9]+|shared)$' \
    | xargs -r ssh wangcy07@hilbit2.cis.upenn.edu \
        "DOCKER_HOST=unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock docker rm -f"
rm -rf /z/wangcy07/webarena/mp/{golden,config_files,results,auth}
```

## Determinism guarantees

The harness guarantees (§10 of MULTIWORKER_PLAN.md):

1. Each task observes the same initial backend state a single-worker run
   would observe. Verified via `verify_golden.py`.
2. No two tasks observe each other. Replicas are physically separate
   containers; only wikipedia/map/homepage are shared, and those are
   verified read-only.
3. Auto-login uses per-worker cookie folders at
   `/z/wangcy07/webarena/mp/auth/w{w}/`. Workers never read each other's
   cookies.
4. The evaluator's HTML navigation uses the same worker's replica as the
   agent. Guaranteed by per-worker `config_files/w{w}/*.json` rendering at
   bring-up.

The worker process asserts the env-URL ↔ worker_id invariant at startup
(see `mp/worker.py:_assert_env_matches_worker`).

## Limits and known caveats

- **`docker commit` of a live container freezes it momentarily.** For
  shopping/shopping_admin/postmill this is fine. For gitlab it requires a
  `gitlab-ctl stop puma sidekiq mailroom registry gitlab-workhorse` first.
  `bring_up.py` does this automatically.
- **Map is shared.** Validation that this is safe under high load is
  deferred (§14.8 of MULTIWORKER_PLAN.md). If you see p99 tile latency
  spike during a high-N run, switch to per-worker map.
- **`config_files/w{w}/{id}.json` is generated at bring-up.** If you
  re-tag a replica to a different port (rare), regenerate with
  `python3 -m mp.bring_up --skip_goldens --skip_configure` to refresh the
  rendered configs.
- **OpenAI API key must be passed via environment** the same way it would
  be for stock `run.py` (the orchestrator inherits env into worker
  subprocesses).
