"""Bring-up: create N replicas of each mutable site and populate goldens.

Run once, idempotently, before the orchestrator starts. Steps for each
mutable site:

1. ``docker commit`` the live source container into a golden image (skipped
   if the golden image already exists).
2. Run ``num_workers`` replica containers, one per worker, on the
   port_stride-spaced ports.
3. For w==0 only, populate the golden SQL/dump and tarball artifacts.
4. Apply per-replica base-URL rewrites so each replica's app knows its own
   public URL (Magento ``setup:store-config:set``, GitLab ``gitlab.rb``
   ``external_url``).
5. Bind-mount the golden directory into the gitlab containers so the in-
   container rsync reset has a source.

For read-only sites we don't replicate — the existing single container is
reused and the per-worker URL points at the same one (see ``MPConfig.url_for``).

The script writes a final state file at ``mp/config.json`` so other modules
can load it.
"""
from __future__ import annotations

import argparse
import logging
import re
import shlex
import sys
import time
from pathlib import Path

from mp.config import (
    ALL_MUTABLE_SITES,
    BASE_PORTS,
    MAGENTO_DB_NAME,
    MAGENTO_DB_PASSWORD,
    MAGENTO_DB_USER,
    MAGENTO_MYSQL_DATADIR,
    MAGENTO_MYSQL_SUPERVISOR_PROG,
    POSTMILL_DB_NAME,
    POSTMILL_DB_USER,
    POSTMILL_PG_DATADIR,
    POSTMILL_PG_SUPERVISOR_PROG,
    SITE_INTERNAL_PORT,
    SITE_SOURCE_IMAGE,
    SITE_TO_CONTAINER_BASE,
    MPConfig,
    load_config,
    save_config,
)
from mp.docker_exec import DockerClient
from mp.reset import _wait_healthy, reset_site, snapshot_datadir

log = logging.getLogger("mp.bring_up")


def ensure_paths(client: DockerClient, cfg: MPConfig) -> None:
    """Create golden_root + result_dir + auth_root + config_files_root on the host."""
    for p in (cfg.golden_root, cfg.result_dir, cfg.auth_root, cfg.config_files_root):
        client.run(f"mkdir -p {shlex.quote(p)}")
    for site in ALL_MUTABLE_SITES:
        client.run(f"mkdir -p {shlex.quote(f'{cfg.golden_root}/{site}')}")


# ---------------------------------------------------------------------------
# Golden image creation
# ---------------------------------------------------------------------------

def make_golden_image(client: DockerClient, cfg: MPConfig, site: str) -> None:
    """Create the golden image tag for ``site``.

    For WebArena's pre-populated source images, ``docker commit`` of the live
    container is prohibitively slow (10+ min on a 1 GB writable layer, hours
    on the gitlab container's 24 GB writable layer). Since the source images
    are already populated, we use ``docker tag`` instead — a metadata-only
    operation that takes milliseconds. Any per-replica drift (Magento base
    URLs, GitLab external_url) is reapplied by ``configure_replica_*``
    *after* the replica boots.

    The trade-off: a replica spawned from the source image will go through
    its own first-boot initialization, which for GitLab is the slow
    ``gitlab-ctl reconfigure`` cycle (~3 min). This is captured later in the
    golden snapshot of ``/var/opt/gitlab`` (step 4 of bring_up).
    """
    tag = cfg.golden_image_for(site)
    if client.image_exists(tag):
        log.info("golden image %s already exists — skip", tag)
        return
    # Always prefer the original source image when present; it is
    # pre-populated and stable.
    src_image = SITE_SOURCE_IMAGE[site]
    if client.image_exists(src_image):
        log.info("tagging source image %s -> %s (instant)", src_image, tag)
        client.run(f"docker tag {shlex.quote(src_image)} {shlex.quote(tag)}")
        return
    # No source image available — fall back to committing the live container.
    src_container = SITE_TO_CONTAINER_BASE[site]
    if not client.container_exists(src_container):
        raise RuntimeError(
            f"neither source image {src_image} nor container {src_container} found for site {site}"
        )
    log.info("committing %s -> %s (slow, no source image found)", src_container, tag)
    if site == "gitlab":
        client.exec(
            src_container,
            "gitlab-ctl stop puma sidekiq mailroom registry gitlab-workhorse 2>&1 || true",
            timeout=120,
        )
        try:
            client.commit(src_container, tag)
        finally:
            client.exec(
                src_container,
                "gitlab-ctl start puma sidekiq gitlab-workhorse 2>&1 || true",
                timeout=120,
            )
    else:
        client.commit(src_container, tag)


# ---------------------------------------------------------------------------
# Replica run
# ---------------------------------------------------------------------------

def _docker_run_replica(
    client: DockerClient,
    cfg: MPConfig,
    site: str,
    worker_id: int,
) -> None:
    container = cfg.container_for(site, worker_id)
    if client.container_exists(container):
        # Already exists (worker_0 always falls into this branch because it
        # reuses the legacy container name). Just make sure it's running.
        client.run(f"docker start {shlex.quote(container)}", check=False)
        return
    image = cfg.golden_image_for(site)
    host_port = cfg.port_for(site, worker_id)
    internal = SITE_INTERNAL_PORT[site]
    mounts = ""
    extra = ""
    if site == "gitlab":
        # Bind-mount golden read-only for reset()
        mounts = f"-v {shlex.quote(f'{cfg.golden_root}/gitlab')}:/opt/golden/gitlab:ro"
        extra = "/opt/gitlab/embedded/bin/runsvdir-start"
    cmd = (
        f"docker run -d --restart unless-stopped "
        f"--name {shlex.quote(container)} "
        f"-p {host_port}:{internal} "
        f"{mounts} {shlex.quote(image)} {extra}".strip()
    )
    client.run(cmd, timeout=120)
    log.info("started %s on %d", container, host_port)


# ---------------------------------------------------------------------------
# Per-site bring-up post-config
# ---------------------------------------------------------------------------

def configure_replica_magento(
    client: DockerClient,
    cfg: MPConfig,
    site: str,
    worker_id: int,
    *,
    wait_for_boot: bool = True,
) -> None:
    """Apply per-replica base URL to a Magento replica."""
    container = cfg.container_for(site, worker_id)
    # Magento's base_url is the document ROOT (no /admin/ suffix) — see
    # MPConfig.magento_base_url. url_for() (which appends /admin for
    # shopping_admin) is for the agent's navigation, not Magento config.
    base_url = cfg.magento_base_url(site, worker_id)
    if not base_url.endswith("/"):
        base_url_slash = base_url + "/"
    else:
        base_url_slash = base_url
    if wait_for_boot:
        # Magento needs ~60 s for first boot after ``docker run``. Poll the
        # storefront URL but accept any HTTP response (Magento often serves
        # 500 with a wrong base_url before we fix it). Each probe has both a
        # curl --max-time AND a subprocess timeout so we can't hang.
        #
        # ALSO probe the in-container MySQL: the storefront can answer HTTP
        # before mariadb is ready, and the next step here calls `mysql -e ...`
        # which would fail. Both must be green to break.
        deadline = time.monotonic() + 240
        # Real HTTP codes are 3 digits 100-599. The `|| echo 000` curl fallback
        # plus curl's own "%{http_code}" output of "000" on timeout can
        # concatenate into "000000", so use a strict regex match — anything
        # not in [1-5][0-9][0-9] means "not really booted yet".
        http_re = re.compile(rb"^[1-5]\d{2}$")
        while time.monotonic() < deadline:
            r = client.exec(
                container,
                "curl -s -o /dev/null --max-time 5 -w '%{http_code}' http://127.0.0.1/ 2>/dev/null || echo 000",
                check=False,
                timeout=30,
            )
            code = r.stdout.strip()
            if r.returncode != 0 or not http_re.match(code):
                time.sleep(3)
                continue
            # HTTP is up; now confirm mariadb is reachable before declaring
            # boot complete — the very next CLI call in this function is
            # `mysql -e UPDATE core_config_data ...` and that fails if the
            # in-container MySQL socket isn't listening yet.
            mysql_check = client.exec(
                container,
                f"mysql -u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} "
                f"{MAGENTO_DB_NAME} -BNe 'SELECT 1' 2>/dev/null",
                check=False,
                timeout=30,
            )
            if mysql_check.returncode == 0 and mysql_check.stdout.strip() == b"1":
                log.info(
                    "%s booted (HTTP %s, mysql ready)",
                    container,
                    code.decode("ascii", "replace"),
                )
                break
            time.sleep(3)

    # Rewrite the base URL in core_config_data + Magento CLI. Also NULL any
    # static/media/link URL overrides so they re-derive from base_url — a
    # golden image built elsewhere can ship explicit overrides pointing at the
    # original build host (e.g. metis.lti.cs.cmu.edu), which would survive a
    # base_url rewrite and leave assets pointing at an unreachable host.
    sql = (
        f"UPDATE core_config_data SET value='{base_url_slash}' "
        f"WHERE path IN ('web/secure/base_url','web/unsecure/base_url'); "
        "UPDATE core_config_data SET value=NULL WHERE path IN "
        "('web/secure/base_static_url','web/unsecure/base_static_url',"
        "'web/secure/base_media_url','web/unsecure/base_media_url',"
        "'web/secure/base_link_url','web/unsecure/base_link_url');"
    )
    # 180s timeout (was 30s) — fresh Magento containers can take 60-120s before
    # the in-container MySQL daemon is ready to accept connections, especially
    # when bring_up is run against several replicas in parallel.
    client.exec(
        container,
        f"mysql -u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} {MAGENTO_DB_NAME} -e {shlex.quote(sql)}",
        timeout=180,
    )
    base_url_for_cli = base_url_slash[:-1]  # CLI rejects trailing slash
    client.exec(
        container,
        f"cd /var/www/magento2 && bin/magento setup:store-config:set "
        f"--base-url={shlex.quote(base_url_for_cli + '/')} || true",
        timeout=120,
    )
    if site == "shopping_admin":
        # Disable admin password reset enforcement (matches env_docker README).
        client.exec(
            container,
            "cd /var/www/magento2 && "
            "bin/magento config:set admin/security/password_is_forced 0 || true; "
            "bin/magento config:set admin/security/password_lifetime 0 || true",
            timeout=60,
        )
    client.exec(
        container, "cd /var/www/magento2 && bin/magento cache:flush || true", timeout=120
    )


def configure_replica_gitlab(
    client: DockerClient,
    cfg: MPConfig,
    worker_id: int,
) -> None:
    """Rewrite gitlab.rb ``external_url`` + pin nginx listen_port, then reconfigure.

    Critical detail: Omnibus GitLab derives nginx's internal listen port from
    the port in ``external_url``. Without an explicit override, setting
    ``external_url 'http://host:8123'`` makes nginx listen on 8123 *inside*
    the container — but our docker port mapping is host:8123 → container:8023
    (because we want all replicas to reuse the source image's internal
    listen socket). The result would be connection refused.

    The fix: keep ``external_url`` set to the public-facing URL (for link
    generation), but pin ``nginx['listen_port']`` to the source's internal
    port so docker port forwarding works correctly.
    """
    container = cfg.container_for("gitlab", worker_id)
    base_url = cfg.url_for("gitlab", worker_id)
    internal_port = SITE_INTERNAL_PORT["gitlab"]
    # Update external_url
    client.exec(
        container,
        f"sed -i {shlex.quote('s|^external_url.*|external_url ' + repr(base_url) + '|')} /etc/gitlab/gitlab.rb",
        timeout=30,
    )
    # Ensure nginx['listen_port'] line is present and set to the internal port.
    # Strategy: remove any existing nginx['listen_port'] line then append our own.
    client.exec(
        container,
        (
            "sed -i '/^nginx\\[..listen_port..\\]/d' /etc/gitlab/gitlab.rb && "
            f"echo \"nginx['listen_port'] = {internal_port}\" >> /etc/gitlab/gitlab.rb"
        ),
        timeout=30,
    )
    # Also pin nginx['listen_https'] = false to keep this HTTP-only.
    client.exec(
        container,
        (
            "sed -i '/^nginx\\[..listen_https..\\]/d' /etc/gitlab/gitlab.rb && "
            "echo \"nginx['listen_https'] = false\" >> /etc/gitlab/gitlab.rb"
        ),
        timeout=30,
    )
    # Cap puma worker count. By default Omnibus sets puma['worker_processes']
    # to the CPU count, which on deploy-host means 128 workers per replica * N
    # replicas — far more than needed for benchmark workload and prone to
    # killing each other via OOM when multiple GitLab replicas run. Cap to
    # 4 workers per replica, which is plenty for a single agent driving it.
    client.exec(
        container,
        (
            "sed -i '/^puma\\[..worker_processes..\\]/d' /etc/gitlab/gitlab.rb && "
            "echo \"puma['worker_processes'] = 4\" >> /etc/gitlab/gitlab.rb"
        ),
        timeout=30,
    )
    client.exec(container, "gitlab-ctl reconfigure", timeout=900)


# ---------------------------------------------------------------------------
# Golden artifact population
# ---------------------------------------------------------------------------

def _verify_mysqldump(path: str) -> None:
    """Sanity-check a mysqldump output: catch a real docker-stream bug.

    Observed on deploy-host: ``docker exec ... mysqldump ... > file`` very
    occasionally produces output where two version-conditional comments get
    concatenated mid-line, e.g. ``/*!401/*!40000 ALTER TABLE`` instead of two
    separate ``/*!40101 SET ... */; /*!40000 ALTER TABLE ... */;`` lines.
    The corruption is intermittent and only surfaces at restore time when
    mariadb refuses to parse the malformed comment with ``ERROR 1064``.
    Catch it at dump time instead.
    """
    with open(path, "rb") as f:
        data = f.read()
    if not data.endswith(b"\n"):
        raise RuntimeError(f"{path}: dump truncated (no trailing newline)")
    # The pattern is a `!` followed by digits then `/`, which never appears in
    # a clean mysqldump (version conditionals end with `*/`).
    if re.search(rb"/\*!\d+/\*!\d", data):
        raise RuntimeError(
            f"{path}: malformed concatenated version comment detected "
            f"(docker-stream interleaving bug — re-run populate_goldens)"
        )


def populate_goldens(client: DockerClient, cfg: MPConfig) -> None:
    """For w=0 only, dump golden artifacts into ``cfg.golden_root``."""
    # Shopping
    for site in ("shopping", "shopping_admin"):
        out = cfg.golden_sql_path(site)
        container = cfg.container_for(site, 0)
        log.info("dumping %s -> %s", site, out)
        # mysqldump --add-drop-table emits DROP TABLE IF EXISTS for every table.
        # --no-tablespaces is required on MariaDB ≥ 10.5 without PROCESS privilege.
        # Retry on dump corruption (docker-stream interleaving — see
        # _verify_mysqldump). 3 attempts; we typically succeed on the first.
        for attempt in range(3):
            client.run(
                f"docker exec {shlex.quote(container)} mysqldump "
                f"--single-transaction --routines --triggers --add-drop-table "
                f"--no-tablespaces -u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} "
                f"{MAGENTO_DB_NAME} > {shlex.quote(out)}",
                timeout=900,
            )
            try:
                _verify_mysqldump(out)
                break
            except RuntimeError as e:
                if attempt == 2:
                    raise
                log.warning("dump verify failed (attempt %d/3): %s", attempt + 1, e)

    # Reddit / Postmill
    out = cfg.golden_sql_path("reddit")
    container = cfg.container_for("reddit", 0)
    log.info("dumping postmill -> %s", out)
    client.run(
        f"docker exec {shlex.quote(container)} pg_dump -Fc -U {POSTMILL_DB_USER} "
        f"{POSTMILL_DB_NAME} > {shlex.quote(out)}",
        timeout=600,
    )

    # Postmill: tar of submission_images and media/cache.
    si_out = f"{cfg.golden_root}/reddit/submission_images.tar"
    mc_out = f"{cfg.golden_root}/reddit/media_cache.tar"
    client.run(
        f"docker exec {shlex.quote(container)} tar -C /var/www/html/public/submission_images "
        f"-cf - . > {shlex.quote(si_out)}",
        timeout=600,
    )
    client.run(
        f"docker exec {shlex.quote(container)} tar -C /var/www/html/public/media/cache "
        f"-cf - . > {shlex.quote(mc_out)}",
        timeout=600,
    )

    # GitLab: rsync /var/opt/gitlab to mp/golden/gitlab on the host.
    # We rsync OUTSIDE the container so the result lives on the host.
    container = cfg.container_for("gitlab", 0)
    dst = f"{cfg.golden_root}/gitlab"
    log.info("snapshotting %s:/var/opt/gitlab -> %s (this takes ~5-15 min)", container, dst)
    # Stop user-facing services for consistency.
    client.exec(
        container,
        "gitlab-ctl stop puma sidekiq mailroom registry gitlab-workhorse 2>&1 || true",
        timeout=120,
    )
    try:
        client.run(f"mkdir -p {shlex.quote(dst)}")
        # docker cp would be simpler, but docker cp is single-shot (no rsync).
        # We tar over a pipe to the host's tar -x.
        client.run(
            f"docker exec {shlex.quote(container)} tar -C /var/opt/gitlab -cf - . | "
            f"tar -C {shlex.quote(dst)} -xf -",
            timeout=3600,
        )
    finally:
        client.exec(
            container,
            "gitlab-ctl start puma sidekiq gitlab-workhorse 2>&1 || true",
            timeout=120,
        )


# ---------------------------------------------------------------------------
# Fast-reset datadir snapshots
# ---------------------------------------------------------------------------

def snapshot_all_datadirs(client: DockerClient, cfg: MPConfig) -> None:
    """Create the in-container ``<datadir>.golden`` snapshot for every replica.

    These power the physical-swap fast reset in ``mp/reset.py``. Magento
    (shopping, shopping_admin) snapshot ``/var/lib/mysql``; Postmill (reddit)
    snapshots ``/usr/local/pgsql/data``. GitLab is excluded — its reset
    already does a physical rsync from the host-side golden tree.

    Idempotent and re-runnable: re-snapshotting just overwrites the prior
    snapshot. Run via ``mp.bring_up`` (it calls this in step 4b) or directly
    after manually recreating a single replica.
    """
    specs = [
        ("shopping", MAGENTO_MYSQL_DATADIR, MAGENTO_MYSQL_SUPERVISOR_PROG),
        ("shopping_admin", MAGENTO_MYSQL_DATADIR, MAGENTO_MYSQL_SUPERVISOR_PROG),
        ("reddit", POSTMILL_PG_DATADIR, POSTMILL_PG_SUPERVISOR_PROG),
    ]
    for w in range(cfg.num_workers):
        for site, datadir, prog in specs:
            container = cfg.container_for(site, w)
            log.info("snapshotting datadir for fast-reset: %s", container)
            try:
                snapshot_datadir(
                    client, container, datadir=datadir, supervisor_prog=prog
                )
            except Exception as e:  # noqa: BLE001 — log + continue; reset falls back to logical
                log.warning(
                    "datadir snapshot failed for %s (%s); reset will fall back "
                    "to the slow logical path for this replica",
                    container, e,
                )


# ---------------------------------------------------------------------------
# Health checks after bring-up
# ---------------------------------------------------------------------------

def assert_all_healthy(cfg: MPConfig) -> None:
    """For each (site, worker), GET base URL and assert a sensible response."""
    for site in ALL_MUTABLE_SITES:
        for w in range(cfg.num_workers):
            url = cfg.url_for(site, w)
            if site in ("shopping", "shopping_admin"):
                # Magento sometimes responds 302 with a sign-in redirect on
                # first hit; accept any non-error body that has the canonical
                # marker. Source-image storefront title is "Magento" or
                # "One Stop Market" depending on version.
                _wait_healthy(
                    url,
                    expect_body_contains="Magento" if site == "shopping_admin" else "Market",
                    timeout_seconds=180,
                    poll_interval=2.0,
                )
            elif site == "reddit":
                _wait_healthy(
                    url,
                    expect_body_contains="Postmill",
                    timeout_seconds=120,
                    poll_interval=1.0,
                )
            elif site == "gitlab":
                # GitLab nginx may return 500 with a complete page body
                # when the request Host doesn't match external_url; the body
                # marker is what we trust.
                _wait_healthy(
                    url + "/users/sign_in",
                    expect_body_contains="Sign in",
                    timeout_seconds=300,
                    poll_interval=2.0,
                )


# ---------------------------------------------------------------------------
# Per-worker task config rendering
# ---------------------------------------------------------------------------

def render_config_files(cfg: MPConfig) -> None:
    """Render the 812 task configs once per worker.

    Why: the URLs in ``config_files/*.json`` were substituted at generation
    time by ``scripts/generate_test_data.py`` using the env vars at THAT
    moment. For multi-worker we need each worker to see URLs pointing at
    *its* replicas; the cleanest way is to render the configs N times into
    N subdirectories, then have each worker load from its subdirectory.

    This module runs locally (it doesn't need the docker host), but the
    output is written to ``cfg.config_files_root`` which is on the docker
    host so the workers — also running on that host — can read them.

    For local runs we still write to the local working directory if no
    ssh_host is configured.
    """
    import json
    import os

    repo_root = Path(__file__).resolve().parent.parent
    raw_path = repo_root / "config_files" / "test.raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing {raw_path}; can't render configs")
    with raw_path.open() as f:
        raw = json.load(f)

    for w in range(cfg.num_workers):
        env = cfg.env_for(w)
        out_dir = Path(cfg.config_files_root) / f"w{w}"
        if cfg.ssh_host:
            # Render to local staging dir, then `scp -r` to remote in caller.
            out_dir = repo_root / "mp" / "_staging_config_files" / f"w{w}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for entry in raw:
            data = json.dumps(entry)
            for k, v in env.items():
                # The placeholders in test.raw.json are __SHOPPING__, __MAP__, etc.
                data = data.replace(f"__{k}__", v)
            # Also replace embedded URLs that mention default ports.
            doc = json.loads(data)
            task_id = doc["task_id"]
            (out_dir / f"{task_id}.json").write_text(json.dumps(doc, indent=2))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def bring_up(cfg: MPConfig, *, skip_goldens: bool = False, skip_configure: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = DockerClient(docker_host=cfg.docker_host, ssh_host=cfg.ssh_host)
    ensure_paths(client, cfg)
    log.info("step 1/5: creating golden images")
    for site in ALL_MUTABLE_SITES:
        make_golden_image(client, cfg, site)
    log.info("step 2/5: starting %d replicas per site", cfg.num_workers)
    for w in range(cfg.num_workers):
        for site in ALL_MUTABLE_SITES:
            _docker_run_replica(client, cfg, site, w)
    if not skip_configure:
        log.info("step 3/5: configuring per-replica URLs")
        for w in range(cfg.num_workers):
            configure_replica_magento(client, cfg, "shopping", w)
            configure_replica_magento(client, cfg, "shopping_admin", w)
            configure_replica_gitlab(client, cfg, w)
            # Postmill needs no per-replica URL tweak — there's no Postmill-side
            # base URL configuration (the app builds URLs from request headers).
    if not skip_goldens:
        log.info("step 4/5: populating goldens (this is the slow step)")
        populate_goldens(client, cfg)
    if not skip_configure:
        # Snapshot each replica's DB datadir for the fast physical-swap reset
        # (mp/reset.py). Done AFTER configure so the snapshot carries this
        # worker's base_url, and after populate_goldens so the host-side
        # logical golden (the fallback path) is also fresh. Cheap: one clean
        # DB stop + local cp per replica.
        log.info("step 4b/5: snapshotting per-worker DB datadirs (fast-reset)")
        snapshot_all_datadirs(client, cfg)
    log.info("step 5/5: health checks")
    assert_all_healthy(cfg)
    log.info("saving config to mp/config.json")
    save_config(cfg)
    log.info("rendering per-worker task config files")
    render_config_files(cfg)
    log.info("bring-up complete")


def _argparse() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WebArena multi-worker bring-up")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--ssh_host", default=None, help="SSH host to wrap docker commands in")
    p.add_argument("--docker_host", default=None)
    p.add_argument("--host", default=None, help="public hostname for replica URLs")
    p.add_argument("--golden_root", default=None)
    p.add_argument("--skip_goldens", action="store_true", help="don't re-snapshot golden artifacts")
    p.add_argument("--skip_configure", action="store_true", help="don't re-configure base URLs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _argparse().parse_args(argv)
    cfg = load_config()
    override = {}
    for k in ("num_workers", "ssh_host", "docker_host", "host", "golden_root"):
        v = getattr(args, k)
        if v is not None:
            override[k] = v
    if override:
        import dataclasses
        cfg = dataclasses.replace(cfg, **override)
    bring_up(cfg, skip_goldens=args.skip_goldens, skip_configure=args.skip_configure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
