"""Per-worker task execution loop.

A worker is a single Python process that:

1. Knows its ``worker_id`` and a ``MPConfig``.
2. Exports per-worker URL env vars BEFORE importing browser_env (because
   ``browser_env/env_config.py:9`` reads env at import time).
3. Pulls task ids from a multiprocessing.Queue.
4. For each task:
   a. resets every mutable site named in ``config["sites"]`` (plus any
      site marked dirty by a previous task on this worker — see §14.6).
   b. waits for health on the touched sites.
   c. runs the agent loop (delegated to ``run.test_single_task``).
   d. records the score on a result queue.

This module is invoked by ``mp.orchestrator`` via ``multiprocessing.Process``.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

from mp.config import ALL_MUTABLE_SITES, MPConfig

log = logging.getLogger("mp.worker")


@dataclasses.dataclass
class TaskResult:
    worker_id: int
    task_id: int
    score: float | None
    error: str | None
    duration_seconds: float
    # Provenance fields recorded into scores.jsonl so the comparison tool can
    # tell which run produced each row. ``inference_backend`` mirrors the
    # ``--inference_backend`` orchestrator flag (e.g. "tsa" / "dense"); the
    # other two snapshot the runtime model name and OpenAI base URL.
    inference_backend: str | None = None
    model: str | None = None
    openai_api_base: str | None = None
    eval_api_base: str | None = None


def _apply_env(cfg: MPConfig, worker_id: int) -> None:
    """Export per-worker URLs into os.environ.

    Must be called BEFORE importing browser_env.* (since env_config.py reads
    env at import time, line 9). Safe to call multiple times — but the second
    call has no effect on already-imported modules.
    """
    for k, v in cfg.env_for(worker_id).items():
        os.environ[k] = v


def _assert_env_matches_worker(cfg: MPConfig, worker_id: int) -> None:
    """Defensive runtime check from §10."""
    expected = cfg.env_for(worker_id)
    for k, v in expected.items():
        actual = os.environ.get(k, "")
        if actual != v:
            raise RuntimeError(
                f"env mismatch on worker {worker_id}: {k}={actual!r} (expected {v!r})"
            )


def _run_one_task(
    cfg: MPConfig,
    worker_id: int,
    task_id: int,
    *,
    dirty_sites: set[str],
    args_dict: dict[str, Any],
) -> TaskResult:
    """Reset, run, score one task. Returns the result.

    Imports browser_env / run inside the function so each worker's first call
    triggers the env-time import in the worker's own process env.
    """
    start = time.monotonic()
    inference_backend = args_dict.get("inference_backend")
    model_name = args_dict.get("model")
    openai_api_base = os.environ.get("OPENAI_API_BASE", "") or None
    eval_api_base = os.environ.get("WEBARENA_EVAL_API_BASE", "") or None

    config_path = Path(cfg.config_files_root) / f"w{worker_id}" / f"{task_id}.json"
    if not config_path.exists():
        return TaskResult(
            worker_id=worker_id,
            task_id=task_id,
            score=None,
            error=f"missing config file {config_path}",
            duration_seconds=time.monotonic() - start,
            inference_backend=inference_backend,
            model=model_name,
            openai_api_base=openai_api_base,
            eval_api_base=eval_api_base,
        )

    with config_path.open() as f:
        config = json.load(f)
    touched_sites: list[str] = list(config.get("sites", []))

    # §14.6: always reset any site we previously dirtied on this worker.
    # Additionally, reset newly-touched sites unless the task config sets
    # require_reset=False (read-only tasks that don't mutate state).
    sites_to_reset: set[str] = set(dirty_sites)
    if config.get("require_reset", True):
        sites_to_reset |= set(touched_sites)
    sites_to_reset = {s for s in sites_to_reset if s in ALL_MUTABLE_SITES}

    # Late imports — only after env is set.
    from mp.docker_exec import DockerClient
    from mp.reset import reset_sites

    client = DockerClient(docker_host=cfg.docker_host, ssh_host=cfg.ssh_host)

    log.info("worker %d task %d: resetting %s", worker_id, task_id, sorted(sites_to_reset))
    try:
        reset_sites(sorted(sites_to_reset), worker_id=worker_id, cfg=cfg, client=client)
    except Exception:
        err = traceback.format_exc()
        log.error("worker %d task %d: reset failed:\n%s", worker_id, task_id, err)
        return TaskResult(
            worker_id=worker_id,
            task_id=task_id,
            score=None,
            error=f"reset failed: {err}",
            duration_seconds=time.monotonic() - start,
            inference_backend=inference_backend,
            model=model_name,
            openai_api_base=openai_api_base,
            eval_api_base=eval_api_base,
        )

    # Run the agent + evaluator. We delegate to run.run_single_task, an
    # adapter introduced in run.py to keep behaviour identical to a serial
    # run.
    try:
        from run import run_single_task

        score = run_single_task(
            config_file=str(config_path),
            worker_id=worker_id,
            cfg=cfg,
            args_dict=args_dict,
        )
    except Exception:
        err = traceback.format_exc()
        log.error("worker %d task %d: agent failed:\n%s", worker_id, task_id, err)
        return TaskResult(
            worker_id=worker_id,
            task_id=task_id,
            score=None,
            error=f"agent failed: {err}",
            duration_seconds=time.monotonic() - start,
            inference_backend=inference_backend,
            model=model_name,
            openai_api_base=openai_api_base,
            eval_api_base=eval_api_base,
        )

    # Update dirty sites set: every mutable site this task touched is now
    # dirty until the next reset (which is the NEXT task's reset).
    dirty_sites.clear()
    dirty_sites.update(s for s in touched_sites if s in ALL_MUTABLE_SITES)

    return TaskResult(
        worker_id=worker_id,
        task_id=task_id,
        score=float(score),
        error=None,
        duration_seconds=time.monotonic() - start,
        inference_backend=inference_backend,
        model=model_name,
        openai_api_base=openai_api_base,
        eval_api_base=eval_api_base,
    )


def worker_main(
    cfg_json: str,
    worker_id: int,
    task_queue: "mp.Queue[int | None]",
    result_queue: "mp.Queue[dict[str, Any]]",
    args_dict: dict[str, Any],
    log_file: str,
) -> None:
    """Top-level entry: configure logging, env, then drain task queue."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [w{worker_id}] %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    cfg = MPConfig.from_json(cfg_json)
    _apply_env(cfg, worker_id)
    _assert_env_matches_worker(cfg, worker_id)
    log.info("worker %d started; cfg.host=%s", worker_id, cfg.host)

    dirty_sites: set[str] = set()
    while True:
        try:
            task_id = task_queue.get(timeout=1.0)
        except Exception:
            # queue empty — exit.
            log.info("worker %d: queue empty, exiting", worker_id)
            return
        if task_id is None:
            # poison pill
            log.info("worker %d: poison pill, exiting", worker_id)
            return
        try:
            res = _run_one_task(
                cfg,
                worker_id,
                int(task_id),
                dirty_sites=dirty_sites,
                args_dict=args_dict,
            )
        except Exception:
            res = TaskResult(
                worker_id=worker_id,
                task_id=int(task_id),
                score=None,
                error=f"unhandled: {traceback.format_exc()}",
                duration_seconds=0.0,
                inference_backend=args_dict.get("inference_backend"),
                model=args_dict.get("model"),
                openai_api_base=os.environ.get("OPENAI_API_BASE", "") or None,
                eval_api_base=os.environ.get("WEBARENA_EVAL_API_BASE", "") or None,
            )
        result_queue.put(dataclasses.asdict(res))
        log.info(
            "worker %d task %d done: score=%s err=%s dt=%.1fs",
            worker_id,
            int(task_id),
            res.score,
            res.error[:80] if res.error else None,
            res.duration_seconds,
        )
