"""Orchestrator: task queue + worker supervisor.

Runs N worker subprocesses, hands them task ids, persists every result.

Design choices (matching §7.5 of the plan):

* No external queue framework — we use ``multiprocessing.Queue``.
* All replicas exist for every worker, so any worker can take any task —
  no site-affinity routing.
* Results are written to ``cfg.result_dir/scores.jsonl`` as one JSON object
  per line, appended atomically.
* Re-run support: on startup we read ``scores.jsonl`` and skip already-
  successful task ids.
* Worker crash recovery: if a worker subprocess exits unexpectedly before
  producing a result for its in-flight task, the orchestrator detects this
  and re-enqueues the task (via the result_queue's no-news-timeout).
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mp.config import MPConfig, load_config

log = logging.getLogger("mp.orchestrator")


@dataclass
class Orchestrator:
    cfg: MPConfig
    args_dict: dict[str, Any]
    task_ids: list[int]
    result_path: Path

    @property
    def log_dir(self) -> Path:
        return Path(self.cfg.result_dir) / "logs"

    def _completed_task_ids(self) -> set[int]:
        if not self.result_path.exists():
            return set()
        done: set[int] = set()
        with self.result_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("score") is not None and obj.get("error") is None:
                    done.add(int(obj["task_id"]))
        return done

    def _append_result(self, result: dict[str, Any]) -> None:
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        with self.result_path.open("a") as f:
            f.write(json.dumps(result) + "\n")

    def run(self) -> dict[str, Any]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        completed = self._completed_task_ids()
        pending = [t for t in self.task_ids if t not in completed]
        log.info(
            "orchestrator starting: %d tasks pending (%d already done) across %d workers",
            len(pending),
            len(completed),
            self.cfg.num_workers,
        )
        if not pending:
            return {"pending": 0, "done": len(completed)}

        # Spawn workers. Use "spawn" start method to avoid inheriting the
        # parent's imports of env-sensitive modules. ALL multiprocessing
        # primitives (Queue, Process) must be created from the same context
        # — mixing default-context Queue with spawn-context Process raises
        # "A SemLock created in a fork context is being shared with a
        # process in a spawn context."
        ctx = mp.get_context("spawn")

        # multiprocessing primitives, in the same context
        task_queue: "mp.Queue[int | None]" = ctx.Queue()
        result_queue: "mp.Queue[dict[str, Any]]" = ctx.Queue()
        for t in pending:
            task_queue.put(t)
        # one poison pill per worker so they exit gracefully when queue drains
        for _ in range(self.cfg.num_workers):
            task_queue.put(None)

        from mp.worker import worker_main

        cfg_json = self.cfg.to_json()
        workers: list[Any] = []
        for w in range(self.cfg.num_workers):
            log_file = str(self.log_dir / f"worker_{w}.log")
            p = ctx.Process(
                target=worker_main,
                args=(cfg_json, w, task_queue, result_queue, self.args_dict, log_file),
                name=f"webarena-worker-{w}",
            )
            p.start()
            workers.append(p)

        # Collect results. We also poll for dead workers and re-spawn them.
        outstanding = set(pending)
        last_progress_at = time.monotonic()
        while outstanding:
            try:
                result = result_queue.get(timeout=10.0)
            except Exception:
                result = None

            if result is not None:
                task_id = int(result["task_id"])
                if task_id in outstanding:
                    outstanding.remove(task_id)
                self._append_result(result)
                last_progress_at = time.monotonic()
                log.info(
                    "progress: %d/%d done (worker=%d task=%d score=%s err=%s)",
                    len(pending) - len(outstanding),
                    len(pending),
                    result.get("worker_id"),
                    task_id,
                    result.get("score"),
                    (result.get("error") or "")[:60],
                )

            # Worker liveness
            alive = [p for p in workers if p.is_alive()]
            if not alive and outstanding:
                log.warning(
                    "all workers dead but %d tasks outstanding; respawning one worker",
                    len(outstanding),
                )
                # Push the outstanding tasks back on the queue and spawn one worker
                for t in list(outstanding):
                    task_queue.put(t)
                task_queue.put(None)
                w = len(workers)
                log_file = str(self.log_dir / f"worker_{w}.log")
                p = ctx.Process(
                    target=worker_main,
                    args=(cfg_json, 0, task_queue, result_queue, self.args_dict, log_file),
                    name=f"webarena-worker-{w}-respawn",
                )
                p.start()
                workers.append(p)

            # Heartbeat
            if time.monotonic() - last_progress_at > 1800:
                log.warning("no progress for 30 min — investigation needed")
                last_progress_at = time.monotonic()

        # Clean shutdown
        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                log.warning("worker %s still alive after queue drain; terminating", p.name)
                p.terminate()
                p.join(timeout=5)

        log.info("orchestrator done: %d tasks completed", len(pending))
        return {"pending": 0, "done": len(pending) + len(completed)}


def _argparse() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WebArena multi-worker orchestrator")
    p.add_argument("--config", default=None, help="path to mp/config.json (defaults to mp/config.json)")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=812)
    p.add_argument("--task_ids", default=None, help="comma-separated explicit task ids (overrides start/end)")
    p.add_argument("--model", default="gpt-3.5-turbo-16k-0613")
    p.add_argument("--instruction_path", default="agent/prompts/jsons/p_cot_id_actree_2s.json")
    p.add_argument("--provider", default="openai")
    p.add_argument("--mode", default="chat")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=384)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--agent_type", default="prompt")
    p.add_argument("--observation_type", default="accessibility_tree")
    p.add_argument("--action_set_tag", default="id_accessibility_tree")
    p.add_argument("--viewport_width", type=int, default=1280)
    p.add_argument("--viewport_height", type=int, default=720)
    p.add_argument("--current_viewport_only", action="store_true", default=True)
    p.add_argument("--save_trace_enabled", action="store_true", default=True)
    p.add_argument("--render", action="store_true", default=False)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _argparse().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    if args.task_ids:
        task_ids = [int(t.strip()) for t in args.task_ids.split(",") if t.strip()]
    else:
        task_ids = list(range(args.start_idx, args.end_idx))

    # Build args_dict that the worker will pass through to run_single_task.
    excluded = {"config", "task_ids", "start_idx", "end_idx"}
    args_dict = {k: v for k, v in vars(args).items() if k not in excluded}

    result_path = Path(cfg.result_dir) / "scores.jsonl"
    o = Orchestrator(
        cfg=cfg,
        args_dict=args_dict,
        task_ids=task_ids,
        result_path=result_path,
    )
    summary = o.run()
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
