"""Tests for the normalization + URL extraction logic in mp.verify_golden.

These tests don't need network; they exercise pure functions only.
"""
from __future__ import annotations

import json
from pathlib import Path

from mp.config import MPConfig
from mp.verify_golden import (
    NORMALIZERS,
    extract_program_html_urls,
    normalize,
    substitute_urls,
)


def test_normalize_removes_csrf_meta_tags() -> None:
    raw = '<meta name="csrf-token" content="abc123==">'
    out = normalize(raw)
    assert "abc123" not in out
    assert 'content="X"' in out


def test_normalize_removes_form_key_inputs() -> None:
    raw = '<input name="form_key" value="ZxYwT9" />'
    out = normalize(raw)
    assert "ZxYwT9" not in out


def test_normalize_removes_iso_timestamps() -> None:
    raw = "Today: 2024-05-25T14:32:19.123Z"
    out = normalize(raw)
    assert "2024" not in out


def test_normalize_removes_relative_times() -> None:
    raw = "Updated 3 hours ago by user"
    out = normalize(raw)
    assert "<RELTS>" in out


def test_normalize_removes_magento_key_segments() -> None:
    raw = 'href="/shop/account/edit/key/abcdef0123456789abcdef0123456789/"'
    out = normalize(raw)
    assert "abcdef" not in out
    assert "/key/X/" in out


def test_normalize_removes_csp_nonces() -> None:
    raw = '<script nonce="aBcD+/=123">x</script>'
    out = normalize(raw)
    assert "aBcD" not in out


def test_normalizers_idempotent() -> None:
    raw = '<meta name="csrf-token" content="abc">'
    once = normalize(raw)
    twice = normalize(once)
    assert once == twice


def test_extract_urls_from_real_test_data() -> None:
    repo = Path(__file__).resolve().parent.parent.parent
    if not (repo / "config_files" / "test.raw.json").exists():
        # Not in a real checkout; skip.
        return
    urls = extract_program_html_urls(repo)
    # Should pick up some __PLACEHOLDER__ URLs.
    assert len(urls) > 0
    # No "last" sentinels should pass through.
    assert "last" not in urls
    # No func: callouts should pass through.
    assert all(not u.startswith("func") for u in urls)


def test_substitute_urls() -> None:
    cfg = MPConfig(num_workers=2, host="example.test")
    urls_in = ["__GITLAB__/foo", "__SHOPPING__/cart", "https://example.com/static"]
    urls_out = substitute_urls(urls_in, cfg, 0)
    assert urls_out[0] == cfg.url_for("gitlab", 0) + "/foo"
    assert urls_out[1] == cfg.url_for("shopping", 0) + "/cart"
    # Hard-coded URLs pass through unchanged.
    assert urls_out[2] == "https://example.com/static"


def test_substitute_urls_worker_isolation() -> None:
    cfg = MPConfig(num_workers=4, host="x.y", port_stride=100)
    urls_in = ["__SHOPPING__/foo"]
    out_0 = substitute_urls(urls_in, cfg, 0)[0]
    out_3 = substitute_urls(urls_in, cfg, 3)[0]
    # Different workers should yield different ports.
    assert out_0 != out_3
    assert ":7770" in out_0
    assert ":8070" in out_3
