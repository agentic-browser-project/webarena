#!/usr/bin/env bash
# Step 1 — start the rootless Docker daemon for this deployment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/lib.sh"

log "=== 01: rootless docker ==="
ensure_rootless_docker
fix_rootless_dns
share_socket
docker info --format 'Server {{.ServerVersion}} | driver {{.Driver}} | root {{.DockerRootDir}}'
