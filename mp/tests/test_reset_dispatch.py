"""Tests for the dispatch logic in mp.reset.

We don't run docker here; we use a FakeClient that records every command and
returns a configurable result for the `test -f` probe used by the Magento
reset to decide whether to restore media.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from mp.config import MPConfig
from mp.reset import ResetFailed, reset_site, reset_sites


@dataclasses.dataclass
class _StubProc:
    returncode: int = 0
    stdout: bytes = b""
    stderr: bytes = b""


class FakeClient:
    """Records every command invocation and returns a stub success result.

    Set ``test_file_exists`` to control what ``test -f`` calls return (used
    by the Magento reset to detect a media tarball golden).
    """

    def __init__(self, *, test_file_exists: bool = False) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.test_file_exists = test_file_exists

    def _make_stub(self, cmd: str) -> _StubProc:
        # Emulate the test -f && echo YES || echo NO shell idiom.
        if "test -f" in cmd:
            return _StubProc(stdout=b"YES\n" if self.test_file_exists else b"NO\n")
        return _StubProc()

    def run(self, cmd: str, **kw: Any) -> _StubProc:
        self.calls.append(("run", (cmd,)))
        return self._make_stub(cmd)

    def exec(self, container: str, cmd: str, **kw: Any) -> _StubProc:
        self.calls.append(("exec", (container, cmd)))
        return self._make_stub(cmd)

    def exec_raw(self, container: str, argv: list[str], **kw: Any) -> _StubProc:
        self.calls.append(("exec_raw", (container, argv)))
        return _StubProc()

    def joined(self) -> str:
        parts = []
        for kind, payload in self.calls:
            for p in payload:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, list):
                    parts.append(" ".join(p))
        return "\n".join(parts)


def test_reset_dispatch_unknown_site_raises() -> None:
    cfg = MPConfig()
    with pytest.raises(ValueError):
        reset_site("nonesuch", worker_id=0, cfg=cfg, client=FakeClient())


def test_reset_sites_skips_readonly() -> None:
    cfg = MPConfig()
    c = FakeClient()
    reset_sites(["wikipedia", "map", "homepage"], worker_id=0, cfg=cfg, client=c)
    assert c.calls == []


def test_reset_magento_issues_drop_table_pipe(monkeypatch) -> None:
    cfg = MPConfig()
    c = FakeClient()
    monkeypatch.setattr("mp.reset._wait_healthy", lambda *a, **kw: None)
    reset_site("shopping", worker_id=0, cfg=cfg, client=c)
    joined = c.joined()
    assert "mysql --max_allowed_packet" in joined
    assert "magentouser" in joined
    assert "magentodb" in joined
    assert "var/cache" in joined
    assert "redis-cli" in joined
    assert "cache:flush" in joined
    assert "core_config_data" in joined


def test_reset_magento_does_not_restore_media_when_absent(monkeypatch) -> None:
    cfg = MPConfig()
    c = FakeClient(test_file_exists=False)
    monkeypatch.setattr("mp.reset._wait_healthy", lambda *a, **kw: None)
    reset_site("shopping", worker_id=0, cfg=cfg, client=c)
    joined = c.joined()
    # No tar -xz of media should be issued when the probe says NO.
    assert "tar -C /var/www/magento2/pub/media -xzf" not in joined


def test_reset_magento_restores_media_when_present(monkeypatch) -> None:
    cfg = MPConfig()
    c = FakeClient(test_file_exists=True)
    monkeypatch.setattr("mp.reset._wait_healthy", lambda *a, **kw: None)
    reset_site("shopping", worker_id=0, cfg=cfg, client=c)
    joined = c.joined()
    assert "tar -C /var/www/magento2/pub/media" in joined


def test_reset_postmill_issues_pg_restore(monkeypatch) -> None:
    cfg = MPConfig()
    c = FakeClient()
    monkeypatch.setattr("mp.reset._wait_healthy", lambda *a, **kw: None)
    reset_site("reddit", worker_id=0, cfg=cfg, client=c)
    joined = c.joined()
    assert "pg_terminate_backend" in joined
    assert "pg_restore" in joined
    assert "--clean" in joined
    assert "--if-exists" in joined
    assert "/var/www/html/var/cache" in joined
    assert "submission_images" in joined
    assert "media/cache" in joined


def test_reset_gitlab_stops_user_services(monkeypatch) -> None:
    cfg = MPConfig()
    c = FakeClient()
    monkeypatch.setattr("mp.reset._wait_healthy", lambda *a, **kw: None)
    reset_site("gitlab", worker_id=0, cfg=cfg, client=c)
    joined = c.joined()
    assert "gitlab-ctl stop puma" in joined
    assert "sidekiq" in joined
    assert "pgrep -f" in joined
    assert "rsync -a --delete /opt/golden/gitlab" in joined
    assert "FLUSHALL" in joined
    assert "gitlab-ctl start puma" in joined


def test_reset_health_wait_failure_raises(monkeypatch) -> None:
    cfg = MPConfig()
    c = FakeClient()

    def boom(*a, **kw):
        raise ResetFailed("health check failed")

    monkeypatch.setattr("mp.reset._wait_healthy", boom)
    with pytest.raises(ResetFailed):
        reset_site("shopping", worker_id=0, cfg=cfg, client=c)
