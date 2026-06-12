"""Unit tests for mp.config.

These tests run in any environment — they don't need docker or hilbit2."""
from __future__ import annotations

import json

import pytest

from mp.config import (
    ALL_MUTABLE_SITES,
    ALL_READONLY_SITES,
    ALL_SITES,
    BASE_PORTS,
    MPConfig,
    load_config,
)


def test_default_config_is_well_formed() -> None:
    cfg = MPConfig()
    assert cfg.num_workers >= 1
    assert cfg.port_stride >= 1
    assert cfg.host
    assert cfg.docker_host.startswith("unix://")


def test_port_arithmetic() -> None:
    cfg = MPConfig(num_workers=4, port_stride=100)
    # Worker 0 uses the base port.
    assert cfg.port_for("shopping", 0) == 7770
    assert cfg.port_for("shopping_admin", 0) == 7780
    assert cfg.port_for("gitlab", 0) == 8023
    # Worker N uses base + 100*N.
    assert cfg.port_for("shopping", 1) == 7870
    assert cfg.port_for("shopping", 3) == 8070
    assert cfg.port_for("gitlab", 2) == 8223
    # Read-only sites ignore worker_id.
    assert cfg.port_for("wikipedia", 0) == 8888
    assert cfg.port_for("wikipedia", 3) == 8888
    assert cfg.port_for("map", 0) == 13000


def test_port_assignments_have_no_cross_site_collisions() -> None:
    """For N=8 workers, every (site, worker) port must be unique."""
    cfg = MPConfig(num_workers=8)
    ports: dict[int, str] = {}
    for site in ("shopping", "shopping_admin", "gitlab", "reddit"):
        for w in range(8):
            p = cfg.port_for(site, w)
            assert p not in ports, (
                f"collision at port {p}: {site}_w{w} clashes with {ports[p]}"
            )
            ports[p] = f"{site}_w{w}"


def test_port_arithmetic_rejects_invalid_worker() -> None:
    cfg = MPConfig(num_workers=2)
    with pytest.raises(ValueError):
        cfg.port_for("shopping", 2)
    with pytest.raises(ValueError):
        cfg.port_for("shopping", -1)


def test_port_arithmetic_rejects_unknown_site() -> None:
    cfg = MPConfig()
    with pytest.raises(ValueError):
        cfg.port_for("nonesuch", 0)


def test_url_formatting() -> None:
    cfg = MPConfig(num_workers=2, host="example.test", port_stride=100)
    assert cfg.url_for("shopping", 1) == "http://example.test:7870"
    # Admin URL includes /admin suffix.
    assert cfg.url_for("shopping_admin", 0) == "http://example.test:7780/admin"
    # Wikipedia uses the explicit override (deep path, not just base).
    assert cfg.url_for("wikipedia", 0).endswith("Landing")


def test_env_for_returns_all_seven_sites() -> None:
    cfg = MPConfig()
    env = cfg.env_for(0)
    assert set(env.keys()) == {
        "SHOPPING",
        "SHOPPING_ADMIN",
        "REDDIT",
        "GITLAB",
        "WIKIPEDIA",
        "MAP",
        "HOMEPAGE",
    }
    for v in env.values():
        assert v.startswith("http://")


def test_container_naming() -> None:
    cfg = MPConfig()
    # Worker 0 reuses the legacy container names so an existing live
    # WebArena deployment can be adopted as worker_0 with no renaming.
    assert cfg.container_for("reddit", 0) == "forum"
    assert cfg.container_for("shopping", 0) == "shopping"
    assert cfg.container_for("shopping_admin", 0) == "shopping_admin"
    assert cfg.container_for("gitlab", 0) == "gitlab"
    # Workers >= 1 get _w{id} suffixes.
    assert cfg.container_for("reddit", 3) == "forum_w3"
    assert cfg.container_for("gitlab", 7) == "gitlab_w7"
    # Read-only sites: w0 uses legacy name, w>=1 uses _shared.
    assert cfg.container_for("wikipedia", 0) == "wikipedia"
    assert cfg.container_for("wikipedia", 2) == "wikipedia_shared"


def test_golden_image_naming() -> None:
    cfg = MPConfig()
    assert cfg.golden_image_for("shopping") == "webarena-shopping-golden:latest"
    with pytest.raises(ValueError):
        cfg.golden_image_for("wikipedia")


def test_round_trip_json() -> None:
    cfg = MPConfig(num_workers=5, host="other.host", port_stride=20)
    payload = cfg.to_json()
    reloaded = MPConfig.from_json(payload)
    assert reloaded == cfg


def test_load_config_falls_back_to_defaults(tmp_path) -> None:
    cfg = load_config(tmp_path / "no_such_file.json")
    assert isinstance(cfg, MPConfig)
    # default num_workers
    assert cfg.num_workers >= 1


def test_load_config_reads_real_file(tmp_path) -> None:
    p = tmp_path / "config.json"
    cfg = MPConfig(num_workers=3, host="x.y")
    p.write_text(cfg.to_json())
    reloaded = load_config(p)
    assert reloaded.num_workers == 3
    assert reloaded.host == "x.y"


def test_all_site_constants_are_consistent() -> None:
    # Every mutable + read-only site appears in BASE_PORTS.
    for s in ALL_SITES:
        assert s in BASE_PORTS or s == "homepage"
    # No overlap between mutable and read-only.
    assert set(ALL_MUTABLE_SITES).isdisjoint(set(ALL_READONLY_SITES))
