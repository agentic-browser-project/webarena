# Smoke-run runbook (gray ollama backend)

Current state of `mp/` on hilbit2 (verified working as of this session):

| Component | State | How verified |
|---|---|---|
| `mp/` codebase on hilbit2 | `/z/wangcy07/webarena-repo/` | `rsync` |
| Python venv on hilbit2 | `/z/wangcy07/webarena-venv/` (3.12) | `pip install` from `/z/wangcy07/webarena_wheels/` (offline, side-loaded) |
| `mp/` unit tests | 45/45 passing | `python3 -m pytest mp/tests/` on hilbit2 |
| SSH tunnel hilbit2:11434 → gray:11434 | `LISTEN 127.0.0.1:11434` | `ss -tnlp` |
| Ollama model on gray | `qwen2.5:7b-instruct` (also `deepseek-r1:7b`, `deepseek-r1:latest`) | `curl /api/tags` |
| End-to-end LLM call | 0.29s for `click [2]` action | `OPENAI_API_BASE=http://127.0.0.1:11434/v1 OPENAI_API_KEY=ollama python3 -c "..."` |
| Per-worker task configs | 812 configs rendered for w=0 at `/z/wangcy07/webarena-mp/config_files/w0/` | `ls | wc -l` |
| Reset dispatch (mocked) | all 4 mutable sites dispatch without raising | `mp.reset.reset_site` with `MagicMock` client |

## Recommended model

**`qwen2.5:7b-instruct`** — pulled and warm on gray. Confirmed to produce parseable WebArena actions (`click [N]`) at ~60 tok/s with 64 max_tokens. No `<think>` overhead.

`deepseek-r1:7b` is available but wastes 200–500 tokens on reasoning per call; only viable with `max_tokens ≥ 4096` and `WEBARENA_STRIP_REASONING_TAGS=1`. Slower.

## Remaining steps to a real 10-task run

These steps were NOT executed in this session because each one has a cost
(~minutes to hours of compute or ~150 MB download) and they should run unattended.

### 1. Install Playwright Chromium browser on hilbit2

hilbit2 has broken DNS so `playwright install chromium` won't work directly.
Two options:

**A. Side-load (recommended):** download Chromium 1223 on a machine with internet,
rsync to hilbit2.

```
# On a workstation with internet:
pip install playwright==1.60.0
PLAYWRIGHT_BROWSERS_PATH=/tmp/pw_browsers playwright install chromium
rsync -az /tmp/pw_browsers/ wangcy07@hilbit2.cis.upenn.edu:/z/wangcy07/pw_browsers/

# On hilbit2 (one-time, per shell):
export PLAYWRIGHT_BROWSERS_PATH=/z/wangcy07/pw_browsers
```

**B. Add `/etc/resolv.conf` workaround**: if you can get a resolver onto hilbit2
(e.g., `echo nameserver 8.8.8.8 | sudo tee /etc/resolv.conf`), then
`playwright install chromium` works directly. Requires sudo.

### 2. Bring up N=2 replicas + goldens (~30 minutes)

```
ssh wangcy07@hilbit2.cis.upenn.edu
source /z/wangcy07/webarena-venv/bin/activate
cd /z/wangcy07/webarena-repo

# This is one-shot. Logs to stdout; recommend wrapping in tmux.
python3 -m mp.bring_up --num_workers 2 --host 158.130.4.158 \
    --golden_root /z/wangcy07/webarena-mp/golden \
    2>&1 | tee /z/wangcy07/webarena-mp/bring_up.log
```

Wall-clock estimate (from §11 of MULTIWORKER_PLAN.md):
- docker commit of 4 source containers → 5 min
- New replica containers boot → 5 min total (gitlab dominates)
- Per-replica base URL rewrites → 5 min (each `gitlab-ctl reconfigure` is ~3 min)
- Golden snapshot (mysqldump + pg_dump + tar of /var/opt/gitlab) → 15 min

Verify (separately):
```
python3 -m mp.verify_golden --max_urls 50 --out /tmp/verify.json
# expect "differs: 0" for all sites if reset is correct
```

### 3. Run the 10-task smoke

```
export OPENAI_API_BASE=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
export PLAYWRIGHT_BROWSERS_PATH=/z/wangcy07/pw_browsers

# Pick 10 task ids that cover several sites
# (these are arbitrary; adjust to cover shopping + reddit + map + a few writes)
python3 -m mp.orchestrator \
    --task_ids 0,1,7,10,33,40,73,77,118,222 \
    --model qwen2.5:7b-instruct \
    --max_tokens 512 \
    --temperature 0 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
    2>&1 | tee /z/wangcy07/webarena-mp/results/smoke.log
```

Results land at `/z/wangcy07/webarena-mp/results/scores.jsonl`,
one JSON per task. Run:
```
python3 - <<'EOF'
import json, statistics
rows = [json.loads(l) for l in open("/z/wangcy07/webarena-mp/results/scores.jsonl")]
ok = [r["score"] for r in rows if r["error"] is None]
print(f"  done: {len(ok)}/10  agg: {sum(ok)/max(1,len(ok)):.2f}  dur(avg): {statistics.mean(r['duration_seconds'] for r in rows):.0f}s")
for r in rows:
    print(f"    task {r['task_id']} on w{r['worker_id']}: score={r['score']} err={(r['error'] or '')[:70]}")
EOF
```

### 4. Notes / known caveats

* `verify_golden` against the unmodified live container shows 10+ diffs on GitLab merge-request pages because we don't yet normalize GitLab-specific dynamic
  fields (project view counts, "last_activity_at" timestamps that aren't ISO-8601).
  These are normalizers to add; the harness still ships correct *reset semantics* — these diffs are between **two consecutive fetches of the same container**, not between reset and golden. Add normalizers iteratively after the first smoke.
* The SSH tunnel from hilbit2 to gray needs to stay up for the duration of the run.
  If it dies, restart with:
  ```
  ssh -N -f -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 \
      -L 127.0.0.1:11434:127.0.0.1:11434 wangcy07@158.130.4.227
  ```
* Worker_0 reuses the live containers (`shopping`, `forum`, `gitlab`,
  `shopping_admin`) per `cfg.container_for(site, 0)`. Worker_1+ get
  `_w{id}` suffixes.
* `qwen2.5:7b-instruct` runs ~60 tok/s on a single RTX 5060 Ti. For N=2
  concurrent workers, throughput shares the GPU — expect ~30 tok/s per
  worker. Each agent step is ~5–10s; ~30 steps per task → ~5 min/task.
  10 tasks at N=2 = ~25 min wall.

## What this session actually shipped

* `mp/` codebase (8 modules + 4 test modules, 2425 LOC)
* All 45 unit tests passing on workstation AND hilbit2
* `llms/providers/openai_utils.py` rewritten to accept `OPENAI_API_BASE`,
  delegate dummy API key for Ollama, and strip `<think>` reasoning tags
  via `WEBARENA_STRIP_REASONING_TAGS=1`
* `llms/providers/_reasoning.py` extracted standalone helper, unit-tested
* `llms/providers/hf_utils.py` lazy-loads `text_generation` so OpenAI-only
  deployments don't need the HF SDK
* `llms/providers/__init__.py` added (was missing — broke namespace imports)
* `run.py` gained `--worker_id`, `--mp_config_path`, and a
  `run_single_task` adapter
* SSH key from hilbit2 added to gray's `authorized_keys` for the tunnel
* SSH tunnel hilbit2:11434 → gray:11434 established and held
* `qwen2.5:7b-instruct` pulled on gray
* 812 per-worker task configs rendered into
  `/z/wangcy07/webarena-mp/config_files/w0/`
* Python venv on hilbit2 (`/z/wangcy07/webarena-venv/`) populated with
  37 wheels side-loaded from `/z/wangcy07/webarena_wheels/`
