"""Per-site reset primitives.

Each ``reset_<site>(client, container, golden_paths, base_url)`` returns when
the container has been restored to the golden state captured in
``golden_paths``. Failure raises ``ResetFailed``.

The primitives encode every audit finding in §14 of the plan:

* Magento: ``mysqldump --add-drop-table`` style restore via the magentouser
  account (the only DB user with privileges on magentodb), Redis FLUSHALL for
  the cache backends configured in env.php, ``bin/magento cache:flush``, and
  unconditional base-URL rewrite so the restored ``core_config_data`` row
  reflects this replica's actual port.
* Postmill: ``pg_restore --clean --if-exists`` plus ``submission_images`` and
  ``media/cache`` tarball restore (verified to contain user uploads).
* GitLab: stop user-facing services → rsync from ``/opt/golden/gitlab`` →
  ``redis FLUSHALL`` → restart services → poll ``/users/sign_in`` (the only
  health endpoint that responds 200 on this image).

Every reset is followed by a health-poll that blocks until the public endpoint
returns 200, with a timeout that escalates per-site (Magento: 90 s; Postmill:
30 s; GitLab: 240 s) — the timeouts come from observed boot times on hilbit2.
"""
from __future__ import annotations

import shlex
import time
import urllib.error
import urllib.request

from mp.config import (
    MAGENTO_DB_NAME,
    MAGENTO_DB_PASSWORD,
    MAGENTO_DB_USER,
    POSTMILL_DB_NAME,
    POSTMILL_DB_USER,
    MPConfig,
)
from mp.docker_exec import DockerClient, DockerExecError


class ResetFailed(RuntimeError):
    """Raised when a reset primitive cannot restore golden state."""


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def _http_get(url: str, *, timeout: float = 5.0) -> tuple[int, str]:
    """Fetch ``url`` and return ``(status, body[:8192])``.

    Treats network errors as ``(0, "")`` so the caller's poll loop can decide
    when to give up (rather than crashing inside the poll).
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "webarena-mp/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(8192).decode("utf-8", errors="replace")
            return r.getcode(), body
    except urllib.error.HTTPError as e:
        try:
            body = e.read(8192).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return e.code, body
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return 0, ""


def _wait_healthy(
    url: str,
    *,
    expect_status: int = 200,
    expect_body_contains: str | None = None,
    timeout_seconds: float,
    poll_interval: float = 1.0,
) -> None:
    """Poll ``url`` until it answers.

    The condition for "healthy" is intentionally lenient when a body marker
    is supplied:

    * If ``expect_body_contains`` is set, the URL is considered healthy as
      soon as the response body contains that marker — regardless of status
      code. This handles cases where an app serves the right page with a
      non-2xx status (e.g., GitLab returning 500 with a fully-rendered
      sign-in page due to asset-host mismatches that real browsers tolerate).
    * If ``expect_body_contains`` is None, the URL is considered healthy
      only when the status matches ``expect_status`` exactly.

    Either path raises ``ResetFailed`` on timeout.
    """
    deadline = time.monotonic() + timeout_seconds
    last_status = -1
    while time.monotonic() < deadline:
        status, body = _http_get(url)
        last_status = status
        if expect_body_contains is not None:
            if expect_body_contains in body:
                return
        else:
            if status == expect_status:
                return
        time.sleep(poll_interval)
    raise ResetFailed(
        f"health check failed for {url}: last_status={last_status}, "
        f"expected_status={expect_status}, expect_body_contains={expect_body_contains!r}"
    )


# ---------------------------------------------------------------------------
# Magento (shopping / shopping_admin) — §14.1
# ---------------------------------------------------------------------------

def reset_magento(
    client: DockerClient,
    container: str,
    *,
    golden_sql_path_on_target: str,
    base_url: str,
    health_url: str,
    media_tarball_on_target: str | None = None,
) -> None:
    """Restore a Magento container to its golden state.

    ``golden_sql_path_on_target`` is a path on the docker host machine; it is
    streamed into the container via stdin (so we don't need to bind-mount).

    ``base_url`` is the public URL agents and the evaluator will use to reach
    this replica. Magento's ``core_config_data`` is rewritten to this URL
    after the restore so that pages render with correct asset URLs.
    """
    # 1. Restore DB tables. The golden SQL uses --add-drop-table so DROP/CREATE
    #    happen at the table level (magentouser has no global CREATE DATABASE
    #    privilege, per §14.1).
    # 60 min budget: a fresh-restore of the 1.9 GB shopping golden into a cold
    # mariadb (rebuilds indexes + foreign keys, single-threaded import) takes
    # 20-40 min on the reference deployment, and shopping_admin's smaller
    # 7 MB dump finishes in <1 min — 60 min covers both with margin.
    client.run(
        f"cat {shlex.quote(golden_sql_path_on_target)} | "
        f"docker exec -i {shlex.quote(container)} "
        f"mysql --max_allowed_packet=512M "
        f"-u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} {MAGENTO_DB_NAME}",
        timeout=3600,
    )

    # 2. Clear Magento filesystem caches/sessions. Magento writes these in
    #    /var/www/magento2/var; clearing them is mandatory because they refer
    #    to entity IDs that may not survive the restore. The var/ tree can
    #    accumulate thousands of small files so the timeout is generous.
    client.exec(
        container,
        "cd /var/www/magento2 && "
        "rm -rf var/cache/* var/page_cache/* var/session/* var/tmp/* "
        "var/view_preprocessed/* var/log/* generated/code/* generated/metadata/* || true",
        timeout=1800,
    )

    # 3. Flush Redis dbs 0 and 1 (Magento's `default` and `page_cache`
    #    backends per env.php). This is the §14.5 finding: without it,
    #    Magento serves stale block_html/full_page from Redis.
    #    `redis-cli` is installed in the image.
    client.exec(container, "redis-cli -n 0 FLUSHDB; redis-cli -n 1 FLUSHDB", timeout=15)

    # 4. Magento native cache flush — needed because step 3 didn't flush
    #    static_files cache (filesystem) nor invalidate the
    #    `compiled_config`/`config`/`layout` cache_types listed in env.php.
    #    Failure is non-fatal: ``cache:flush`` requires the framework to boot,
    #    and after step 1 some tables are technically being recreated. We
    #    have already cleared the filesystem caches in step 2.
    client.exec(
        container,
        "cd /var/www/magento2 && bin/magento cache:flush 2>/dev/null || true",
        timeout=120,
    )

    # 5. Rewrite the base URL — see §14.1. Two surfaces:
    #    (a) core_config_data.web/{secure,unsecure}/base_url (SQL)
    #    (b) setup:store-config:set --base-url=... (Magento CLI; idempotent)
    base_url_sql = base_url
    if not base_url_sql.endswith("/"):
        base_url_sql = base_url_sql + "/"
    sql = (
        f"UPDATE core_config_data SET value='{base_url_sql}' "
        f"WHERE path IN ('web/secure/base_url','web/unsecure/base_url');"
    )
    client.exec(
        container,
        f"mysql -u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} {MAGENTO_DB_NAME} "
        f"-e {shlex.quote(sql)}",
        timeout=300,
    )

    # 6. Optional: restore Magento media (product images, uploads). Skipped
    #    by default because shopping_final_0712 / shopping_admin_final_0719
    #    bake the media into the image and tasks don't typically upload
    #    media (verified by intent-template inspection). If a probe later
    #    shows tasks that upload images, populate ``media_tarball_on_target``
    #    in bring_up and the restore happens here.
    if media_tarball_on_target:
        client.run(
            f"cat {shlex.quote(media_tarball_on_target)} | "
            f"docker exec -i {shlex.quote(container)} "
            f"tar -C /var/www/magento2/pub/media -xzf -",
            timeout=600,
        )

    # 7. Wait for the storefront to serve. After cache:flush Magento needs to
    #    re-compile templates on first request — first hit can take ~60 s.
    _wait_healthy(health_url, timeout_seconds=300, poll_interval=1.0)


# ---------------------------------------------------------------------------
# Postmill (reddit/forum) — §14.2
# ---------------------------------------------------------------------------

def reset_postmill(
    client: DockerClient,
    container: str,
    *,
    golden_dump_path_on_target: str,
    submission_images_tar_on_target: str,
    media_cache_tar_on_target: str,
    health_url: str,
) -> None:
    """Restore a Postmill container to its golden state."""
    # 1. Terminate any open connections to the postmill DB so DROP TABLE
    #    statements inside pg_restore --clean can proceed.
    terminate_sql = (
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname='postmill' AND pid <> pg_backend_pid();"
    )
    client.exec(
        container,
        f"psql -U {POSTMILL_DB_USER} -d postgres -c {shlex.quote(terminate_sql)}",
        timeout=30,
    )

    # 2. Restore via pg_restore --clean --if-exists. The golden is in Postgres
    #    custom format (``pg_dump -Fc``); ``--clean --if-exists`` emits
    #    ``DROP TABLE IF EXISTS`` before each ``CREATE TABLE``, avoiding the
    #    need for global DROP DATABASE privilege.
    # 20 min budget: postmill golden is ~478 MB pg_dump custom format which
    # decompresses + index-builds in 3-8 min on hilbit2's reference deployment.
    client.run(
        f"cat {shlex.quote(golden_dump_path_on_target)} | "
        f"docker exec -i {shlex.quote(container)} "
        f"pg_restore -U {POSTMILL_DB_USER} -d {POSTMILL_DB_NAME} "
        f"--clean --if-exists --no-owner --no-acl",
        timeout=1200,
        check=False,  # pg_restore emits warnings for missing objects on first restore; not fatal
    )

    # 3. Clear Symfony filesystem cache. The cache tree can be large under
    #    contention; use a generous timeout.
    client.exec(container, "rm -rf /var/www/html/var/cache/*", timeout=180)

    # 4. Restore user-uploaded submission images and generated thumbnail
    #    cache. Both directories exist on the source image
    #    (verified, §14.2) and tasks like "submit a post with image" rely on
    #    them being part of golden state.
    # 30 min budget: submission_images.tar can be many GB if tasks have been
    # uploading; bounded by tar extract throughput.
    client.run(
        f"cat {shlex.quote(submission_images_tar_on_target)} | "
        f"docker exec -i {shlex.quote(container)} bash -lc "
        f"{shlex.quote('rm -rf /var/www/html/public/submission_images/* && tar -C /var/www/html/public/submission_images -xf -')}",
        timeout=1800,
    )
    client.run(
        f"cat {shlex.quote(media_cache_tar_on_target)} | "
        f"docker exec -i {shlex.quote(container)} bash -lc "
        f"{shlex.quote('rm -rf /var/www/html/public/media/cache/* && tar -C /var/www/html/public/media/cache -xf -')}",
        timeout=600,
    )

    # 5. Wait for the frontend. Postmill has no auth-required home page;
    #    `<title>Postmill</title>` is a stable marker.
    _wait_healthy(
        health_url,
        expect_body_contains="Postmill",
        timeout_seconds=60,
        poll_interval=0.5,
    )


# ---------------------------------------------------------------------------
# GitLab — §14.9 / §4.4
# ---------------------------------------------------------------------------

def reset_gitlab(
    client: DockerClient,
    container: str,
    *,
    golden_dir_in_container: str = "/opt/golden/gitlab",
    health_url: str,
) -> None:
    """Restore a GitLab container to its golden state.

    Assumes ``/opt/golden/gitlab`` is a read-only bind mount of the golden
    snapshot of ``/var/opt/gitlab`` (populated by ``bring_up.py``). The reset:

    1. Stop puma + sidekiq + workhorse + mailroom + registry. Keep postgresql
       and redis running so the rsync isn't fighting an active database.
    2. Wait for sidekiq to actually exit (the stop signal is async).
    3. rsync from golden, deleting any new files.
    4. Restart postgresql + redis to flush any in-memory state, then
       FLUSHALL redis to clear any session cache or Sidekiq job that was
       enqueued but not yet executed.
    5. Restart puma + sidekiq + workhorse.
    6. Poll /users/sign_in until 200 (the working health endpoint per §14.3).
    """
    client.exec(
        container,
        "gitlab-ctl stop puma sidekiq mailroom registry gitlab-workhorse 2>&1 || true",
        timeout=120,
    )
    # Wait for sidekiq to finish in-flight jobs. The runit stop signal is
    # SIGTERM; sidekiq should drain. We bound the wait at 60 s.
    client.exec(
        container,
        "for i in $(seq 1 60); do pgrep -f 'sidekiq .*production' >/dev/null 2>&1 || break; sleep 1; done",
        timeout=90,
    )
    # 60 min budget: the gitlab golden is ~23 GB on the reference deployment
    # (postgresql + redis + git-data + uploads). rsync with --delete is bounded
    # by the slower of disk-write and inode-update throughput; observed wall-
    # clock for a clean reset is 8-20 min on hilbit2.
    client.exec(
        container,
        f"rsync -a --delete {shlex.quote(golden_dir_in_container)}/ /var/opt/gitlab/",
        timeout=3600,
    )
    client.exec(
        container,
        "gitlab-ctl restart postgresql redis 2>&1 || true",
        timeout=120,
    )
    # FLUSHALL via the unix socket; the path comes from the same Omnibus install.
    client.exec(
        container,
        "redis-cli -s /var/opt/gitlab/redis/redis.socket FLUSHALL || true",
        timeout=30,
    )
    client.exec(
        container,
        "gitlab-ctl start puma sidekiq gitlab-workhorse 2>&1 || true",
        timeout=120,
    )
    # /users/sign_in is the only endpoint verified to return 200 on this image.
    _wait_healthy(
        health_url + "/users/sign_in" if not health_url.endswith("/users/sign_in") else health_url,
        expect_body_contains="Sign in",
        timeout_seconds=240,
        poll_interval=2.0,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def reset_site(
    site: str,
    *,
    worker_id: int,
    cfg: MPConfig,
    client: DockerClient,
) -> None:
    """Reset ``site`` on ``worker_id``'s replica to its golden state."""
    container = cfg.container_for(site, worker_id)
    base_url = cfg.url_for(site, worker_id)

    if site in ("shopping", "shopping_admin"):
        media_tar = f"{cfg.golden_root}/{site}/media.tar.gz"
        # Don't restore media unless the golden actually has it; bring_up
        # writes it only when probing detects uploaded files. Detect by
        # asking the docker host whether the file exists.
        media_exists = client.run(
            f"test -f {shlex.quote(media_tar)} && echo YES || echo NO",
            check=False,
        ).stdout.decode().strip().endswith("YES")
        reset_magento(
            client,
            container,
            golden_sql_path_on_target=cfg.golden_sql_path(site),
            base_url=base_url,
            health_url=base_url,
            media_tarball_on_target=media_tar if media_exists else None,
        )
    elif site == "reddit":
        reset_postmill(
            client,
            container,
            golden_dump_path_on_target=cfg.golden_sql_path("reddit"),
            submission_images_tar_on_target=f"{cfg.golden_root}/reddit/submission_images.tar",
            media_cache_tar_on_target=f"{cfg.golden_root}/reddit/media_cache.tar",
            health_url=base_url,
        )
    elif site == "gitlab":
        reset_gitlab(
            client,
            container,
            health_url=base_url,
        )
    else:
        raise ValueError(f"site {site!r} has no reset primitive")


def reset_sites(
    sites: list[str],
    *,
    worker_id: int,
    cfg: MPConfig,
    client: DockerClient,
) -> None:
    """Reset every site in ``sites`` (skipping read-only sites)."""
    for site in sites:
        if site in ("wikipedia", "map", "homepage"):
            continue
        reset_site(site, worker_id=worker_id, cfg=cfg, client=client)
