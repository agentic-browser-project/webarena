"""Thin wrapper for invoking docker commands against a remote rootless docker.

Why a wrapper instead of the docker-py SDK:
- docker-py over a unix socket on a remote machine requires SSH-tunneling that
  socket. Driving docker via ``ssh host docker ...`` is simpler and matches the
  way the rest of this project (deploy_map_official.sh, deploy_webarena.sh)
  already operates.
- We need to set ``DOCKER_HOST`` on the *remote* shell so it talks to the
  rootless daemon at ``/z/wangcy07/webarena/rootless-docker/run/docker.sock``.

This module makes one decision and sticks to it: every docker invocation is
expressed as a shlex-quoted command string, optionally prefixed with
``ssh -o BatchMode=yes <host>`` and the DOCKER_HOST export.
"""
from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable, Sequence


class DockerExecError(RuntimeError):
    """Raised when a docker subprocess exits non-zero."""

    def __init__(self, cmd: Sequence[str], returncode: int, stdout: str, stderr: str):
        super().__init__(
            f"docker exec failed (code {returncode}): {' '.join(cmd)}\n"
            f"stdout: {stdout[-4000:]}\nstderr: {stderr[-4000:]}"
        )
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass
class DockerClient:
    """Stateless client for issuing docker commands on the target host.

    Two modes:
    - Local (ssh_host == ""): docker commands run on the same machine as this
      process. DOCKER_HOST is exported in the child env if non-empty.
    - Remote (ssh_host != ""): docker commands are wrapped in
      ``ssh <ssh_host> bash -lc "export DOCKER_HOST=...; <cmd>"``.

    The reason we use ``bash -lc`` rather than passing argv: docker commands
    sometimes include ``-c`` shell snippets themselves (``docker exec foo bash
    -c "cd X && Y"``). Building a correctly-quoted nested command via argv is
    fragile. We construct one composed shell string and let ssh's stdio
    handling do the heavy lifting.
    """

    docker_host: str
    ssh_host: str = ""
    default_timeout: float = 300.0

    def _wrap(self, cmd: str) -> list[str]:
        prefix = f"export DOCKER_HOST={shlex.quote(self.docker_host)}; "
        full = prefix + cmd
        if self.ssh_host:
            return [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=10",
                self.ssh_host,
                f"bash -lc {shlex.quote(full)}",
            ]
        return ["bash", "-lc", full]

    def run(
        self,
        cmd: str,
        *,
        check: bool = True,
        timeout: float | None = None,
        stdin: bytes | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        argv = self._wrap(cmd)
        proc = subprocess.run(
            argv,
            input=stdin,
            timeout=timeout if timeout is not None else self.default_timeout,
            capture_output=capture_output,
        )
        if check and proc.returncode != 0:
            raise DockerExecError(
                argv,
                proc.returncode,
                (proc.stdout or b"").decode("utf-8", errors="replace"),
                (proc.stderr or b"").decode("utf-8", errors="replace"),
            )
        return proc

    def exec(
        self,
        container: str,
        cmd: str,
        *,
        check: bool = True,
        timeout: float | None = None,
        stdin: bytes | None = None,
        as_user: str | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        """`docker exec <container> bash -lc <cmd>` (or with -u as_user)."""
        u = f"-u {shlex.quote(as_user)} " if as_user else ""
        i = "-i" if stdin is not None else ""
        full = f"docker exec {i} {u}{shlex.quote(container)} bash -lc {shlex.quote(cmd)}"
        return self.run(full, check=check, timeout=timeout, stdin=stdin, capture_output=capture_output)

    def exec_raw(
        self,
        container: str,
        argv: Sequence[str],
        *,
        check: bool = True,
        timeout: float | None = None,
        stdin: bytes | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        """`docker exec <container> <argv...>` — no shell wrapping inside the container."""
        i = "-i" if stdin is not None else ""
        body = " ".join(shlex.quote(a) for a in argv)
        full = f"docker exec {i} {shlex.quote(container)} {body}"
        return self.run(full, check=check, timeout=timeout, stdin=stdin, capture_output=capture_output)

    def ls_containers(self) -> list[dict[str, str]]:
        """Return list of {name, image, status} dicts for all containers."""
        proc = self.run("docker ps -a --format '{{.Names}}|{{.Image}}|{{.Status}}'")
        out = proc.stdout.decode("utf-8", errors="replace")
        rows: list[dict[str, str]] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            rows.append({"name": parts[0], "image": parts[1], "status": parts[2]})
        return rows

    def container_exists(self, name: str) -> bool:
        return any(c["name"] == name for c in self.ls_containers())

    def image_exists(self, tag: str) -> bool:
        proc = self.run(f"docker image inspect {shlex.quote(tag)}", check=False)
        return proc.returncode == 0

    def commit(self, container: str, image_tag: str, *, pause: bool = True) -> None:
        """`docker commit` a running container into a new image tag.

        ``--pause`` (default) freezes the container during commit for filesystem
        consistency. The caller is responsible for separately quiescing app-level
        state (e.g., stopping sidekiq) if cross-DB consistency matters.
        """
        flag = "--pause" if pause else "--no-pause"
        self.run(f"docker commit {flag} {shlex.quote(container)} {shlex.quote(image_tag)}", timeout=900)

    def stop_rm(self, container: str, *, timeout_seconds: int = 30) -> None:
        self.run(f"docker stop -t {timeout_seconds} {shlex.quote(container)}", check=False, timeout=timeout_seconds + 30)
        self.run(f"docker rm -f {shlex.quote(container)}", check=False)

    def cp_from(self, container: str, src_path: str, dst_path: str) -> None:
        """`docker cp <container>:<src> <dst>` on the target machine."""
        self.run(
            f"docker cp {shlex.quote(container)}:{shlex.quote(src_path)} {shlex.quote(dst_path)}",
            timeout=600,
        )

    def cp_to(self, src_path: str, container: str, dst_path: str) -> None:
        self.run(
            f"docker cp {shlex.quote(src_path)} {shlex.quote(container)}:{shlex.quote(dst_path)}",
            timeout=600,
        )
