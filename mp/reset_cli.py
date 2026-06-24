"""Reset WebArena replica(s) to golden state — a thin CLI over ``reset_site``.

Runnable by ANY user logged into the docker host (hilbit2): it talks to the
local rootless-docker socket from ``config.json`` (``ssh_host=""`` -> docker
runs locally), so it needs no personal SSH key. ``DockerClient`` exports
``DOCKER_HOST`` for each command itself, and ``reset.py`` uses only the standard
library, so plain ``python3`` works (no venv required).

Examples (run on hilbit2, from the repo root /z/wangcy07/webarena-repo):

    # one site, one worker
    python3 -m mp.reset_cli --site shopping_admin --worker 0

    # one site across several workers
    python3 -m mp.reset_cli --site reddit --worker 0 1 2

    # every mutable site on a worker
    python3 -m mp.reset_cli --site all --worker 0

    # use a specific deployment config (default: mp/config.json)
    python3 -m mp.reset_cli --site gitlab --worker 1 --config mp/configs/config-bench-tsa.json
"""
from __future__ import annotations

import argparse
import sys
import time

from mp.config import ALL_MUTABLE_SITES, load_config
from mp.docker_exec import DockerClient
from mp.reset import reset_site


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m mp.reset_cli",
        description="Reset WebArena replica(s) to golden state (local docker; no SSH key needed).",
    )
    ap.add_argument(
        "--site",
        required=True,
        help="site to reset: one of %s, or 'all' for every mutable site"
        % ", ".join(ALL_MUTABLE_SITES),
    )
    ap.add_argument(
        "--worker",
        type=int,
        nargs="+",
        default=[0],
        metavar="ID",
        help="worker id(s) to reset (default: 0). e.g. --worker 0 1 2",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="path to deployment config JSON (default: mp/config.json)",
    )
    ap.add_argument(
        "--strategy",
        default="restore",
        choices=["restore", "recreate"],
        help="restore (in-place golden swap, default) or recreate (rebuild container)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.site == "all":
        sites = list(ALL_MUTABLE_SITES)
    elif args.site in ALL_MUTABLE_SITES:
        sites = [args.site]
    else:
        ap.error(
            f"unknown site {args.site!r}; expected one of {', '.join(ALL_MUTABLE_SITES)} or 'all'"
        )

    client = DockerClient(docker_host=cfg.docker_host, ssh_host=cfg.ssh_host)

    failures: list[str] = []
    for worker_id in args.worker:
        if worker_id < 0 or worker_id >= cfg.num_workers:
            print(
                f"[reset] SKIP worker {worker_id}: out of range "
                f"[0,{cfg.num_workers}) for this config",
                flush=True,
            )
            failures.append(f"w{worker_id}:out-of-range")
            continue
        for site in sites:
            label = f"worker {worker_id} {site}"
            print(f"[reset] {label} (strategy={args.strategy}) ...", flush=True)
            t0 = time.monotonic()
            try:
                reset_site(
                    site,
                    worker_id=worker_id,
                    cfg=cfg,
                    client=client,
                    strategy=args.strategy,
                )
                print(f"[reset] {label}: DONE in {time.monotonic() - t0:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001 — report and continue to next
                print(f"[reset] {label}: FAILED ({e})", flush=True)
                failures.append(f"w{worker_id}:{site}")

    if failures:
        print(f"[reset] {len(failures)} failure(s): {', '.join(failures)}", flush=True)
        return 1
    print("[reset] all requested resets completed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
