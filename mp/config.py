"""Multi-worker runtime configuration.

A single source of truth for: worker count, port arithmetic, container naming,
golden artifact locations, docker socket, and DB credentials.

Backed by a JSON file at `mp/config.json` (created by `bring_up.py`), with sane
defaults baked in so unit tests don't need the file.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Literal

Site = Literal["shopping", "shopping_admin", "reddit", "gitlab"]
ALL_MUTABLE_SITES: tuple[Site, ...] = ("shopping", "shopping_admin", "reddit", "gitlab")
ALL_READONLY_SITES: tuple[str, ...] = ("wikipedia", "map", "homepage")
ALL_SITES: tuple[str, ...] = ALL_MUTABLE_SITES + ALL_READONLY_SITES

# Host ports already bound by shared resources on hilbit2 (the map container 
# binds 127.0.0.1:8080 and :8085). port_for() skips these to avoid collisions.
_HOST_PORT_RESERVED: set[int] = {8080, 8085}

# Base ports from environment_docker/README.md and WebArena conventions.
BASE_PORTS: dict[str, int] = {
    "shopping": 7770,
    "shopping_admin": 7780,
    "reddit": 9999,
    "gitlab": 8023,
    "wikipedia": 8888,
    "map": 13000,
    "homepage": 4399,
}

# Mapping from site name to task-config-file site name.
# Postmill in tasks is called "reddit"; container is named "forum".
SITE_TO_CONTAINER_BASE: dict[str, str] = {
    "shopping": "shopping",
    "shopping_admin": "shopping_admin",
    "reddit": "forum",
    "gitlab": "gitlab",
}

# Source images we snapshot to make golden images.
SITE_SOURCE_IMAGE: dict[str, str] = {
    "shopping": "shopping_final_0712",
    "shopping_admin": "shopping_admin_final_0719",
    "reddit": "postmill-populated-exposed-withimg",
    "gitlab": "gitlab-populated-final-port8023",
}

# Container internal listen port (what we map -p <host>:<internal> to)
SITE_INTERNAL_PORT: dict[str, int] = {
    "shopping": 80,
    "shopping_admin": 80,
    "reddit": 80,
    "gitlab": 8023,
}

# DB credentials confirmed by audit (§14.1, §14.2).
MAGENTO_DB_USER = "magentouser"
MAGENTO_DB_PASSWORD = "MyPassword"
MAGENTO_DB_NAME = "magentodb"
POSTMILL_DB_USER = "postmill"
POSTMILL_DB_NAME = "postmill"

# In-container DB data directories + the supervisor program that runs each DB.
# Used by the physical-datadir-swap reset (mp/reset.py): a bulk file copy of
# the data directory is ~100x faster than a logical SQL/dump replay on
# slow-fsync storage (ZFS without SLOG ≈ 113 ms/fsync on hilbit2), because it
# skips the thousands of per-commit fsyncs + index rebuilds a logical restore
# incurs. ``<datadir>.golden`` is a pristine copy made at bring-up time, after
# the per-worker base_url has been configured, so the swap needs no re-config.
MAGENTO_MYSQL_DATADIR = "/var/lib/mysql"
MAGENTO_MYSQL_SUPERVISOR_PROG = "mysqld"
POSTMILL_PG_DATADIR = "/usr/local/pgsql/data"
POSTMILL_PG_SUPERVISOR_PROG = "postgres"
GOLDEN_DATADIR_SUFFIX = ".golden"

# Default account creds (mirrors browser_env/env_config.py:ACCOUNTS but here for
# bring-up-time admin operations on Magento via REST and CLI).
MAGENTO_ADMIN_USER = "admin"
MAGENTO_ADMIN_PASSWORD = "admin1234"


@dataclasses.dataclass(frozen=True)
class MPConfig:
    """Runtime configuration for the multi-worker harness.

    Attributes:
        num_workers: number of parallel workers (and per-site replicas).
        host: hostname or IP the agent and evaluator use to reach replicas.
        port_stride: spacing between worker port assignments. 10 by default
            because gitlab uses a contiguous range for workhorse + registry.
        docker_host: DOCKER_HOST unix socket path for rootless docker on the
            target machine. We never run docker locally — always via SSH or via
            the worker process running on the target machine itself.
        golden_root: filesystem path on the target machine where golden
            artifacts (SQL dumps, tarballs, gitlab rsync mirror) live.
        config_files_root: root directory for per-worker rendered task
            configs (one subdir per worker).
        auth_root: root directory for per-worker auto-login state files.
        ssh_host: if non-empty, all docker commands are wrapped in `ssh ssh_host ...`.
            If empty, docker is invoked locally (the worker is running on the
            same machine as the docker daemon).
        result_dir: directory for benchmark results (one subdir per worker).
        map_replicas: number of map container replicas (default 1; raise if
            load testing shows shared-map saturation, §14.8).
        readonly_url_overrides: map of read-only sites to their fixed URLs
            (since they're shared across workers, they don't use port_stride).
    """

    num_workers: int = 2
    host: str = "158.130.4.158"
    # port_stride=100 ensures shopping/shopping_admin/gitlab/reddit don't
    # collide across workers (worst case: shopping_w0=7770, shopping_w1=7870,
    # gitlab_w0=8023, gitlab_w7=8723 — all distinct for N=8).
    port_stride: int = 100
    docker_host: str = "unix:///z/wangcy07/webarena/rootless-docker/run/docker.sock"
    golden_root: str = "/z/wangcy07/webarena/mp/golden"
    config_files_root: str = "/z/wangcy07/webarena/mp/config_files"
    auth_root: str = "/z/wangcy07/webarena/mp/auth"
    ssh_host: str = ""
    result_dir: str = "/z/wangcy07/webarena/mp/results"
    map_replicas: int = 1
    readonly_url_overrides: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "wikipedia": "http://158.130.4.158:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
            "map": "http://158.130.4.158:13000",
            "homepage": "http://158.130.4.158:4399",
        }
    )

    def port_for(self, site: str, worker_id: int) -> int:
        """Return the host port that worker ``worker_id`` uses for ``site``."""
        if site in ALL_READONLY_SITES:
            return BASE_PORTS[site]
        if site not in ALL_MUTABLE_SITES:
            raise ValueError(f"unknown site: {site!r}")
        if worker_id < 0 or worker_id >= self.num_workers:
            raise ValueError(f"worker_id {worker_id} out of range [0,{self.num_workers})")
        port = BASE_PORTS[site] + self.port_stride * worker_id
        # Skip ports reserved by shared resources (e.g. the map container on hilbit2).
        while port in _HOST_PORT_RESERVED:
            port += 10
        return port

    def url_for(self, site: str, worker_id: int) -> str:
        """Return the absolute base URL that worker `worker_id` uses for `site`.

        Matches the canonical URL strings the WebArena task configs use (the
        SHOPPING_ADMIN URL ends in /admin, the WIKIPEDIA URL ends in a deep
        path, etc.).
        """
        if site in self.readonly_url_overrides:
            return self.readonly_url_overrides[site]
        port = self.port_for(site, worker_id)
        if site == "shopping_admin":
            return f"http://{self.host}:{port}/admin"
        return f"http://{self.host}:{port}"

    def magento_base_url(self, site: str, worker_id: int) -> str:
        """Return the URL to write into Magento's ``web/*/base_url`` config.

        This is the document **root** (``http://host:port/``) — distinct from
        :meth:`url_for`, which appends ``/admin`` for ``shopping_admin`` so the
        agent navigates straight to the admin panel.

        Magento's ``base_url`` must be the root, because Magento derives the
        static/media/link URLs from it: with a ``/admin/`` suffix the CSS/JS
        get requested at ``/admin/static/...`` which 302-loops to
        ``/admin/admin/static/...`` (the real files live at ``/static/...``),
        leaving the admin panel unstyled. Storefront and admin both take the
        root here; the admin panel is reached at the ``/admin`` front-name
        route layered on top of the root base_url.
        """
        if site not in ALL_MUTABLE_SITES:
            raise ValueError(f"site {site} has no Magento base_url")
        port = self.port_for(site, worker_id)
        return f"http://{self.host}:{port}/"

    def container_for(self, site: str, worker_id: int) -> str:
        """Return the docker container name for (site, worker).

        Worker 0 reuses the legacy container names (``shopping``, ``forum``,
        ``gitlab``, ``shopping_admin``) so that an existing single-instance
        WebArena deployment can be adopted as ``worker_0`` without renaming
        anything. Workers >=1 get ``_w{worker_id}`` suffixes.
        """
        if site in ALL_READONLY_SITES:
            return site if worker_id == 0 else f"{site}_shared"
        if site not in SITE_TO_CONTAINER_BASE:
            raise ValueError(f"unknown site: {site!r}")
        base = SITE_TO_CONTAINER_BASE[site]
        if worker_id == 0:
            return base
        return f"{base}_w{worker_id}"

    def golden_image_for(self, site: str) -> str:
        """Return the docker image tag of the per-site golden snapshot."""
        if site not in ALL_MUTABLE_SITES:
            raise ValueError(f"site {site} has no golden image")
        return f"webarena-{site}-golden:latest"

    def golden_sql_path(self, site: str) -> str:
        """Return the filesystem path of the SQL/dump golden for `site`."""
        if site in ("shopping", "shopping_admin"):
            return f"{self.golden_root}/{site}/golden.sql"
        if site == "reddit":
            return f"{self.golden_root}/reddit/golden.dump"
        raise ValueError(f"site {site} has no SQL/dump golden")

    def env_for(self, worker_id: int) -> dict[str, str]:
        """Return the env vars a worker subprocess must export to reach its replicas.

        These keys match `browser_env/env_config.py:1-10`.
        """
        env = {}
        for site in ("shopping", "shopping_admin", "gitlab"):
            env[site.upper()] = self.url_for(site, worker_id)
        # Reddit's env var name is REDDIT (the WebArena convention), though we call the site reddit/forum.
        env["REDDIT"] = self.url_for("reddit", worker_id)
        env["WIKIPEDIA"] = self.url_for("wikipedia", worker_id)
        env["MAP"] = self.url_for("map", worker_id)
        env["HOMEPAGE"] = self.url_for("homepage", worker_id)
        return env

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> MPConfig:
        return cls(**json.loads(payload))


def load_config(path: str | os.PathLike[str] | None = None) -> MPConfig:
    """Load config from disk; fall back to defaults if absent."""
    if path is None:
        path = Path(__file__).parent / "config.json"
    p = Path(path)
    if not p.exists():
        return MPConfig()
    return MPConfig.from_json(p.read_text())


def save_config(cfg: MPConfig, path: str | os.PathLike[str] | None = None) -> Path:
    if path is None:
        path = Path(__file__).parent / "config.json"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cfg.to_json())
    return p
