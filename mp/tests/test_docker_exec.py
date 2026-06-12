"""Tests for the docker_exec wrapper that do NOT require docker.

We assert command formation, ssh wrapping, and quoting behavior. The actual
docker run is only exercised in integration tests."""
from __future__ import annotations

import shlex

from mp.docker_exec import DockerClient


def test_local_wrap_no_ssh() -> None:
    c = DockerClient(docker_host="unix:///tmp/d.sock")
    wrapped = c._wrap("docker ps")
    # First element is bash for local mode.
    assert wrapped[:2] == ["bash", "-lc"]
    # Body exports DOCKER_HOST.
    assert "export DOCKER_HOST=" in wrapped[2]
    assert "docker ps" in wrapped[2]


def test_remote_wrap_with_ssh() -> None:
    c = DockerClient(docker_host="unix:///tmp/d.sock", ssh_host="host1")
    wrapped = c._wrap("docker ps")
    assert wrapped[0] == "ssh"
    assert "host1" in wrapped
    # Body of the SSH command is correctly quoted; last element is the shell snippet.
    snippet = wrapped[-1]
    assert "export DOCKER_HOST=" in snippet
    # docker ps appears somewhere
    assert "docker ps" in snippet


def test_docker_host_is_shell_quoted() -> None:
    c = DockerClient(docker_host="unix:///path with space/d.sock")
    wrapped = c._wrap("docker info")
    quoted = shlex.quote("unix:///path with space/d.sock")
    assert quoted in wrapped[2]


def test_exec_builds_correct_command() -> None:
    c = DockerClient(docker_host="unix:///tmp/d.sock")
    # Monkey-patch run to capture cmd
    captured: list[str] = []
    def fake_run(cmd, **kw):
        captured.append(cmd)
        class CP:
            returncode = 0
            stdout = b""
            stderr = b""
        return CP()
    c.run = fake_run  # type: ignore
    c.exec("mycontainer", "echo hi")
    cmd = captured[0]
    assert "docker exec" in cmd
    assert "mycontainer" in cmd
    assert "echo hi" in cmd
    assert "bash -lc" in cmd


def test_exec_with_user() -> None:
    c = DockerClient(docker_host="unix:///tmp/d.sock")
    captured: list[str] = []
    def fake_run(cmd, **kw):
        captured.append(cmd)
        class CP:
            returncode = 0
            stdout = b""
            stderr = b""
        return CP()
    c.run = fake_run  # type: ignore
    c.exec("mycontainer", "id", as_user="postgres")
    cmd = captured[0]
    assert "-u postgres" in cmd


def test_exec_raw_does_not_wrap_in_bash() -> None:
    c = DockerClient(docker_host="unix:///tmp/d.sock")
    captured: list[str] = []
    def fake_run(cmd, **kw):
        captured.append(cmd)
        class CP:
            returncode = 0
            stdout = b""
            stderr = b""
        return CP()
    c.run = fake_run  # type: ignore
    c.exec_raw("mycontainer", ["ls", "/tmp"])
    cmd = captured[0]
    assert "bash -lc" not in cmd
    assert "ls" in cmd
