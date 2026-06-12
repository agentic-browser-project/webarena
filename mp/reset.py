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

import re
import shlex
import time
import urllib.error
import urllib.request

from mp.config import (
    GOLDEN_DATADIR_SUFFIX,
    MAGENTO_DB_NAME,
    MAGENTO_DB_PASSWORD,
    MAGENTO_DB_USER,
    MAGENTO_MYSQL_DATADIR,
    MAGENTO_MYSQL_SUPERVISOR_PROG,
    POSTMILL_DB_NAME,
    POSTMILL_DB_USER,
    POSTMILL_PG_DATADIR,
    POSTMILL_PG_SUPERVISOR_PROG,
    MPConfig,
)
from mp.docker_exec import DockerClient, DockerExecError


class ResetFailed(RuntimeError):
    """Raised when a reset primitive cannot restore golden state."""


# ---------------------------------------------------------------------------
# Physical datadir swap (fast reset)
#
# A logical restore (cat golden.sql | mysql, or pg_restore) is fsync-bound:
# every commit forces a redo-log fsync, and on slow-fsync storage (ZFS without
# a SLOG device — ~113 ms/fsync on hilbit2) a 369-table Magento restore takes
# tens of minutes to hours, and a freshly-recreated container additionally
# pays a heavy DB+app cold boot. A *physical* datadir swap — stop the DB, copy
# a pristine on-container ``<datadir>.golden`` back over the live datadir,
# start the DB — is a bulk sequential file copy that bypasses all per-commit
# fsyncs and index rebuilds. Measured on hilbit2: shopping_admin reset went
# from 2+ hours (logical) to ~24 s (physical swap).
# ---------------------------------------------------------------------------

def _golden_datadir_exists(client: DockerClient, container: str, datadir: str) -> bool:
    golden = datadir + GOLDEN_DATADIR_SUFFIX
    r = client.exec(
        container,
        f"test -d {shlex.quote(golden)} && test -n \"$(ls -A {shlex.quote(golden)} 2>/dev/null)\" "
        f"&& echo YES || echo NO",
        check=False,
        timeout=30,
    )
    return r.stdout.strip().endswith(b"YES")


def snapshot_datadir(
    client: DockerClient,
    container: str,
    *,
    datadir: str,
    supervisor_prog: str,
) -> None:
    """Create the pristine ``<datadir>.golden`` snapshot used by the fast reset.

    Call this once per replica at bring-up, AFTER the DB is in golden state
    and (for Magento) the per-worker base_url has been written — so the
    snapshot already carries this worker's correct config and the swap needs
    no post-restore reconfiguration.

    Stops the DB via supervisor for a consistent on-disk copy, snapshots, then
    restarts it.
    """
    golden = datadir + GOLDEN_DATADIR_SUFFIX
    client.exec(container, f"supervisorctl stop {supervisor_prog}", timeout=120)
    # Brief settle so the DB has fully released its files before we copy.
    time.sleep(3)
    client.exec(
        container,
        f"rm -rf {shlex.quote(golden)} && cp -a {shlex.quote(datadir)} {shlex.quote(golden)}",
        timeout=1800,
    )
    client.exec(container, f"supervisorctl start {supervisor_prog}", timeout=120)


def _wait_container_db_and_http(
    client: DockerClient,
    container: str,
    *,
    mysql_check: bool = False,
    pg_check: bool = False,
    http_check: bool = True,
    timeout_seconds: float = 300,
) -> None:
    """Wait until the container's own DB + nginx are serving, probed from INSIDE.

    ``http_check=False`` waits for the DB only and skips the HTTP probe — used
    when the caller needs the DB up to rewrite config *before* the first HTTP
    request warms the (full-page) cache, so the cache is built from the
    corrected config rather than the stale golden value.

    Probing from inside the container (``docker exec ... curl 127.0.0.1``)
    rather than the external base_url avoids host-side DNS/interface/proxy
    ambiguity — the same pattern bring_up's configure step uses. "Healthy"
    means: the DB answers a trivial query AND the local web server returns any
    real HTTP status below 500 (Magento /admin legitimately 302-redirects to
    the login page; postmill returns 200).
    """
    http_re = re.compile(rb"^[1-4]\d{2}$")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        db_ok = True
        if mysql_check:
            r = client.exec(
                container,
                f"mysql -u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} "
                f"{MAGENTO_DB_NAME} -BNe 'SELECT 1' 2>/dev/null",
                check=False, timeout=30,
            )
            db_ok = r.returncode == 0 and r.stdout.strip() == b"1"
        if pg_check:
            r = client.exec(
                container,
                f"psql -U {POSTMILL_DB_USER} -d {POSTMILL_DB_NAME} -tAc 'SELECT 1' 2>/dev/null",
                check=False, timeout=30,
            )
            db_ok = r.returncode == 0 and r.stdout.strip() == b"1"
        if db_ok:
            if not http_check:
                return
            h = client.exec(
                container,
                "curl -s -o /dev/null --max-time 5 -w '%{http_code}' http://127.0.0.1/ 2>/dev/null || echo 000",
                check=False, timeout=30,
            )
            if http_re.match(h.stdout.strip()):
                return
        time.sleep(2)
    raise ResetFailed(
        f"{container}: DB/HTTP did not become healthy within {timeout_seconds:.0f}s"
    )


def _fast_clear_magento_var(client: DockerClient, container: str) -> None:
    """Clear Magento's filesystem caches/sessions quickly.

    ``rm -rf var/session/*`` enumerates + unlinks thousands of tiny ``sess_*``
    files; on slow storage that alone blew past a 600 s timeout for the
    shopping replica. Instead we rename each tree aside (an O(1) directory
    rename), recreate it empty, and delete the renamed trees in a detached
    background process so the reset doesn't block on the slow unlink. Stale
    ``.del_*`` dirs from a prior run are also swept here.
    """
    # IMPORTANT: recreate each dir with the SAME owner+mode as the one we move
    # aside. php-fpm runs as www-data and writes sessions/cache here; a plain
    # `mkdir` would create the dir as root and php-fpm would hit
    # "SessionHandler::read(): ... Permission denied" -> storefront 500.
    # We copy the reference dir's owner+perms onto the fresh dir.
    client.exec(
        container,
        "cd /var/www/magento2 && ts=$(date +%s%N) && "
        "for d in var/cache var/page_cache var/session var/tmp "
        "var/view_preprocessed generated/code generated/metadata; do "
        "  if [ -e \"$d\" ]; then "
        "    ref=\"${d}.del_${ts}\"; mv \"$d\" \"$ref\" 2>/dev/null; "
        "    mkdir -p \"$d\"; "
        "    chown --reference=\"$ref\" \"$d\" 2>/dev/null; "
        "    chmod --reference=\"$ref\" \"$d\" 2>/dev/null; "
        "  else mkdir -p \"$d\"; chmod 2777 \"$d\" 2>/dev/null; fi; "
        "done; "
        # Detach the slow unlink so it can't block (or be killed with) the reset.
        "setsid sh -c 'rm -rf var/*.del_* generated/*.del_* >/dev/null 2>&1' "
        "</dev/null >/dev/null 2>&1 & "
        "true",
        timeout=60,
    )


def _physical_swap_datadir(
    client: DockerClient,
    container: str,
    *,
    datadir: str,
    supervisor_prog: str,
) -> None:
    """Stop DB → restore datadir from ``<datadir>.golden`` → start DB.

    Caller must have verified the golden snapshot exists.
    """
    golden = datadir + GOLDEN_DATADIR_SUFFIX
    client.exec(container, f"supervisorctl stop {supervisor_prog}", timeout=120)
    time.sleep(2)
    # rm the live datadir contents (dotfiles included) then copy the golden
    # tree back. `cp -a golden/. datadir/` preserves perms/owner/timestamps.
    client.exec(
        container,
        f"rm -rf {shlex.quote(datadir)}/* {shlex.quote(datadir)}/.[!.]* 2>/dev/null; "
        f"cp -a {shlex.quote(golden)}/. {shlex.quote(datadir)}/",
        timeout=1800,
    )
    client.exec(container, f"supervisorctl start {supervisor_prog}", timeout=120)


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

def _rewrite_magento_base_url(
    client: DockerClient, container: str, base_url: str
) -> None:
    """Set ``web/{,un}secure/base_url`` to ``base_url`` (the document root) and
    NULL any static/media/link URL overrides so they re-derive from base_url.

    ``base_url`` must be the document root (no ``/admin/`` suffix) — see
    ``MPConfig.magento_base_url``. Idempotent; failure is non-fatal (the caller
    has already restored the golden datadir, which carries a usable URL).

    This is the single source of truth for the base-URL rewrite, used by both
    the fast (datadir-swap) and slow (logical-restore) reset paths. Clearing
    the static/media overrides defends against golden images that ship explicit
    overrides pointing at the original build host (metis.lti.cs.cmu.edu), which
    would otherwise leave assets pointing at an unreachable host.
    """
    base_url_sql = base_url if base_url.endswith("/") else base_url + "/"
    sql = (
        f"UPDATE core_config_data SET value='{base_url_sql}' "
        f"WHERE path IN ('web/secure/base_url','web/unsecure/base_url'); "
        "UPDATE core_config_data SET value=NULL WHERE path IN "
        "('web/secure/base_static_url','web/unsecure/base_static_url',"
        "'web/secure/base_media_url','web/unsecure/base_media_url',"
        "'web/secure/base_link_url','web/unsecure/base_link_url');"
    )
    client.exec(
        container,
        f"mysql -u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} {MAGENTO_DB_NAME} "
        f"-e {shlex.quote(sql)}",
        timeout=300,
        check=False,
    )


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
    # FAST PATH: if a pristine on-container datadir snapshot exists (made at
    # bring-up after this worker's base_url was configured), swap it back —
    # ~24 s vs 2+ hours for the logical restore below.
    if _golden_datadir_exists(client, container, MAGENTO_MYSQL_DATADIR):
        _physical_swap_datadir(
            client,
            container,
            datadir=MAGENTO_MYSQL_DATADIR,
            supervisor_prog=MAGENTO_MYSQL_SUPERVISOR_PROG,
        )
        # Wait for mariadb (DB only — don't warm the page cache yet), then
        # enforce THIS worker's document-root base_url. A golden datadir can
        # carry a stale/wrong base_url (an /admin/ suffix, a sibling worker's
        # port, or the original metis.lti.cs.cmu.edu build host); rewriting
        # here makes the reset self-correcting instead of trusting the snapshot.
        _wait_container_db_and_http(
            client, container, mysql_check=True, http_check=False,
            timeout_seconds=300,
        )
        _rewrite_magento_base_url(client, container, base_url)
        # Filesystem caches/sessions may reference pre-reset entity IDs, and the
        # full-page cache may hold the stale base_url; clear them (fast
        # rename-aside), flush Redis, then wait for the storefront so the cache
        # is rebuilt from the corrected config.
        _fast_clear_magento_var(client, container)
        client.exec(
            container,
            "redis-cli -n 0 FLUSHDB; redis-cli -n 1 FLUSHDB",
            timeout=15,
            check=False,
        )
        _wait_container_db_and_http(
            client, container, mysql_check=True, timeout_seconds=300
        )
        return

    # SLOW PATH (no snapshot present): logical restore.
    # 1. Restore DB tables. The golden SQL uses --add-drop-table so DROP/CREATE
    #    happen at the table level (magentouser has no global CREATE DATABASE
    #    privilege, per §14.1).
    #
    # Durability is relaxed for the duration of the restore: on hosts where
    # fsync is slow (ZFS without SLOG ≈ 100 ms/fsync on hilbit2), the default
    # innodb_flush_log_at_trx_commit=1 costs one fsync per commit and the
    # restore takes hours. During a golden restore durability buys nothing —
    # if the restore crashes we just run it again — so flush once per second
    # instead (=2). Restored to full durability afterwards. SET GLOBAL needs
    # SUPER; try root first, magentouser second, and proceed regardless (the
    # restore is then merely slow, not wrong).
    _relax = "SET GLOBAL innodb_flush_log_at_trx_commit=2;"
    client.exec(
        container,
        f"mysql -uroot -p1234567890 -e {shlex.quote(_relax)} 2>/dev/null || "
        f"mysql -u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} -e {shlex.quote(_relax)} || true",
        timeout=30,
        check=False,
    )
    try:
        # FK/unique checks off for the stream: the dump is self-consistent.
        # 60 min budget: a fresh-restore of the 1.9 GB shopping golden into a
        # cold mariadb (single-threaded import) takes 20-40 min on the
        # reference deployment; shopping_admin's 7 MB dump finishes in ~2 min.
        client.run(
            f"(echo 'SET SESSION foreign_key_checks=0; SET SESSION unique_checks=0;'; "
            f"cat {shlex.quote(golden_sql_path_on_target)}) | "
            f"docker exec -i {shlex.quote(container)} "
            f"mysql --max_allowed_packet=512M "
            f"-u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} {MAGENTO_DB_NAME}",
            timeout=3600,
        )
    finally:
        _restore = "SET GLOBAL innodb_flush_log_at_trx_commit=1;"
        client.exec(
            container,
            f"mysql -uroot -p1234567890 -e {shlex.quote(_restore)} 2>/dev/null || "
            f"mysql -u {MAGENTO_DB_USER} -p{MAGENTO_DB_PASSWORD} -e {shlex.quote(_restore)} || true",
            timeout=30,
            check=False,
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

    # 5. Rewrite the base URL — see §14.1. ``base_url`` is the document root
    #    (no /admin/ suffix); the helper also clears static/media overrides so
    #    assets re-derive from it (defends against goldens that ship overrides
    #    pointing at the original build host).
    _rewrite_magento_base_url(client, container, base_url)

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
    # FAST PATH: physical datadir swap when the on-container snapshot exists.
    # The postgres datadir swap restores the `submissions`/`comments`/etc.
    # tables — the only state the deterministic reddit tasks are evaluated on.
    #
    # We deliberately do NOT re-extract the submission_images / media_cache
    # tarballs here: submission_images is ~39 GB (baked into the golden image
    # already), and re-streaming it on every per-task reset would dominate the
    # benchmark wall-clock (measured: 200 s+ and climbing). A task that uploads
    # an image leaves at most a harmless orphan FILE — its DB row is removed by
    # the datadir swap, so listings/counts/program_html checks are unaffected.
    # The rare task that verifies an uploaded image still has the full baked
    # golden set present. (The logical SLOW PATH below still restores the tars
    # for a from-scratch rebuild.)
    if _golden_datadir_exists(client, container, POSTMILL_PG_DATADIR):
        _physical_swap_datadir(
            client,
            container,
            datadir=POSTMILL_PG_DATADIR,
            supervisor_prog=POSTMILL_PG_SUPERVISOR_PROG,
        )
        # Symfony fs cache (small) — fast rename-aside, background delete.
        client.exec(
            container,
            "cd /var/www/html && ts=$(date +%s%N) && "
            "[ -d var/cache ] && mv var/cache var/cache.del_${ts} 2>/dev/null; mkdir -p var/cache; "
            "setsid sh -c 'rm -rf var/cache.del_* >/dev/null 2>&1' </dev/null >/dev/null 2>&1 & true",
            timeout=60, check=False,
        )
        _wait_container_db_and_http(
            client, container, pg_check=True, timeout_seconds=300
        )
        return

    # SLOW PATH (no snapshot present): logical pg_restore.
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
    # synchronous_commit=off for the restore session: skips the per-commit
    # WAL fsync (~100 ms each on slow-fsync hosts like hilbit2's ZFS pool)
    # without disabling fsync entirely — a crash mid-restore loses only the
    # restore itself, which we'd re-run anyway.
    client.run(
        f"cat {shlex.quote(golden_dump_path_on_target)} | "
        f"docker exec -i -e PGOPTIONS='-c synchronous_commit=off' {shlex.quote(container)} "
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
    # CRITICAL: the host-side golden tree (populated by bring_up via a tar
    # streamed to the host) loses the in-container UIDs — every file lands as
    # root. `rsync -a` faithfully restores that root ownership, so afterwards
    # /var/opt/gitlab/gitlab-rails/etc (mode 700) is root-owned and puma (which
    # runs as the `git` user) gets "Permission denied @ rb_sysopen ... puma.rb"
    # and crash-loops -> storefront 502. Restore the canonical Omnibus owners
    # for the service trees that are NOT root-owned in a healthy install.
    client.exec(
        container,
        "chown -R git:git /var/opt/gitlab/gitlab-rails /var/opt/gitlab/git-data "
        "/var/opt/gitlab/gitlab-shell /var/opt/gitlab/gitlab-workhorse 2>/dev/null; "
        "chown -R gitlab-psql:gitlab-psql /var/opt/gitlab/postgresql 2>/dev/null; "
        "chown -R gitlab-redis:gitlab-redis /var/opt/gitlab/redis 2>/dev/null; "
        "chown -R registry:registry /var/opt/gitlab/registry 2>/dev/null; "
        "true",
        timeout=300,
        check=False,
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
    # Health check probed from INSIDE the container (puma listens on
    # 127.0.0.1:8080) — avoids the host-side DNS/interface issue that makes
    # the external base_url unreachable from the orchestrator's Python, and
    # tolerates puma's ~2.5 min Rails preload. We poll gitlab-ctl for puma
    # "run:" first, then curl the local sign-in page. If puma flaps (a known
    # idle-OOM issue on some replicas, see MULTIWORKER_GUIDE §4.3), nudge it.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        st = client.exec(
            container, "gitlab-ctl status puma 2>/dev/null | head -1",
            check=False, timeout=30,
        )
        if st.stdout.strip().startswith(b"run:"):
            h = client.exec(
                container,
                "curl -s -o /dev/null --max-time 5 -w '%{http_code}' "
                "http://127.0.0.1:8080/users/sign_in 2>/dev/null || echo 000",
                check=False, timeout=30,
            )
            if re.match(rb"^[23]\d{2}$", h.stdout.strip()):
                return
        else:
            client.exec(container, "gitlab-ctl start puma 2>&1 || true",
                        check=False, timeout=60)
        time.sleep(5)
    raise ResetFailed(f"{container}: gitlab puma/sign-in not healthy within 600s")


# ---------------------------------------------------------------------------
# Recreate-based reset (fast path)
# ---------------------------------------------------------------------------

def _remove_container(client: DockerClient, container: str) -> None:
    """Stop + remove a container, retrying.

    A bare ``docker rm -f`` on a busy GitLab container occasionally fails
    with "did not receive an exit event"; a graceful stop first avoids it.
    """
    client.run(f"docker stop -t 20 {shlex.quote(container)}", check=False, timeout=120)
    for _ in range(3):
        client.run(f"docker rm -f {shlex.quote(container)}", check=False, timeout=120)
        probe = client.run(
            f"docker ps -a --format '{{{{.Names}}}}' | grep -x {shlex.quote(container)}",
            check=False,
            timeout=30,
        )
        if probe.returncode != 0:  # grep found nothing → container gone
            return
        time.sleep(5)
    raise ResetFailed(f"could not remove container {container!r} after 3 attempts")


def reset_site_recreate(
    site: str,
    *,
    worker_id: int,
    cfg: MPConfig,
    client: DockerClient,
) -> None:
    """Reset ``site`` by recreating its container from the golden image.

    Why this is the default strategy: the golden image already *contains*
    the baked database files, so recreating the container restores state as
    a bulk image-layer operation — no SQL replay at all. On hosts where
    fsync is slow (e.g. ZFS pools without a fast SLOG device, where each
    fsync costs ~100 ms), streaming a mysqldump back into mariadb costs one
    to several fsyncs per DDL/commit and a 369-table Magento restore takes
    tens of minutes to hours. Container recreation sidesteps that entirely
    and additionally resets *filesystem* drift (caches, sessions, uploaded
    files) that a DB-only restore would miss — strictly more rigorous.

    Wall-clock = container boot + per-replica configure:
      * shopping / shopping_admin: ~2-4 min (mariadb warmup dominates)
      * gitlab: ~5-10 min (gitlab-ctl reconfigure + service boot)
      * reddit: postgres boot; NOTE if the golden image was committed while
        postgres was running (crash-consistent), every boot replays WAL
        recovery which can take ~20+ min. Re-commit the golden image after
        a clean ``pg_ctl stop`` once to make this path fast (see
        MULTIWORKER_GUIDE §4.5).
    """
    # Imported lazily: bring_up imports from this module at top level, so a
    # module-level import here would be circular.
    from mp.bring_up import (
        _docker_run_replica,
        configure_replica_gitlab,
        configure_replica_magento,
    )

    container = cfg.container_for(site, worker_id)
    base_url = cfg.url_for(site, worker_id)

    _remove_container(client, container)
    _docker_run_replica(client, cfg, site, worker_id)

    if site in ("shopping", "shopping_admin"):
        # Includes wait-for-boot (HTTP + mariadb SELECT 1 probe) and the
        # per-worker base_url rewrite.
        configure_replica_magento(client, cfg, site, worker_id)
        _wait_healthy(base_url, timeout_seconds=300, poll_interval=2.0)
    elif site == "gitlab":
        # Rewrites external_url/nginx/puma settings then gitlab-ctl reconfigure.
        configure_replica_gitlab(client, cfg, worker_id)
        _wait_healthy(
            base_url + "/users/sign_in",
            expect_body_contains="Sign in",
            timeout_seconds=900,
            poll_interval=5.0,
        )
    elif site == "reddit":
        # Postmill needs no per-replica configure (URLs from request headers).
        _wait_healthy(
            base_url,
            expect_body_contains="Postmill",
            timeout_seconds=1800,  # generous: covers WAL recovery on a dirty golden image
            poll_interval=5.0,
        )
    else:
        raise ValueError(f"site {site!r} has no recreate-reset support")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def reset_site(
    site: str,
    *,
    worker_id: int,
    cfg: MPConfig,
    client: DockerClient,
    strategy: str = "restore",
) -> None:
    """Reset ``site`` on ``worker_id``'s replica to its golden state.

    ``strategy="restore"`` (default) does an in-place restore: stream the
    golden SQL/dump back into the running container's DB (with relaxed
    durability — see ``reset_magento``). This keeps the container's warm
    InnoDB buffer pool + already-booted Magento, which matters because a
    cold Magento boot on slow storage is itself a multi-minute operation
    (see ``reset_site_recreate`` docstring).

    ``strategy="recreate"`` removes + recreates the container from its golden
    image. It additionally resets filesystem drift, but on Magento it
    re-triggers the heavy boot sequence and is SLOWER on slow-fsync hosts —
    use it only for postmill/gitlab or when the container is unrecoverable.
    """
    if strategy == "recreate":
        return reset_site_recreate(site, worker_id=worker_id, cfg=cfg, client=client)
    if strategy != "restore":
        raise ValueError(f"unknown reset strategy {strategy!r}")
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
            # base_url = document root (no /admin/); health_url = the agent-
            # facing URL (shopping_admin's ends in /admin and 302s to login).
            base_url=cfg.magento_base_url(site, worker_id),
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
    strategy: str = "restore",
) -> None:
    """Reset every site in ``sites`` (skipping read-only sites).

    Default ``strategy="restore"`` uses the fast physical-datadir-swap path
    (mp/reset.py:reset_magento / reset_postmill) — verified ~14-39 s/site.
    ``"recreate"`` re-creates the container from its golden image (slower for
    Magento; pays a heavy cold boot) and is opt-in only.
    """
    for site in sites:
        if site in ("wikipedia", "map", "homepage"):
            continue
        reset_site(site, worker_id=worker_id, cfg=cfg, client=client, strategy=strategy)
