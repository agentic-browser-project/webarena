#!/usr/bin/env bash
# Step 4 — create NUM_WORKERS replicas of each mutable site and configure each
# replica's own base_url. This is a thin wrapper around `python -m mp.bring_up`,
# which: tags golden images, docker-runs the replicas on stride ports, rewrites
# each replica's base_url (Magento setup:store-config:set / GitLab external_url),
# snapshots each datadir as <datadir>.golden for fast reset, and renders the
# per-worker task config files.
#
# Extra args are passed through to mp.bring_up, e.g.:
#   ./04_bring_up_replicas.sh --skip_goldens   # reuse already-present golden artifacts
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/lib.sh"

log "=== 04: bring up $NUM_WORKERS replicas/site (host=$HOST_IP) ==="
[ -d "$REPO/mp" ] || die "REPO=$REPO has no mp/ — point REPO at the webarena checkout"
[ -x "$PYTHON" ] || die "PYTHON=$PYTHON not found — create the venv (see README)"

# Write the authoritative config.json from config.env FIRST, so bring_up (and
# every later reset) uses this machine's own host/paths — not whatever config
# was shipped with the repo checkout. (mp.bring_up --host does not update
# readonly_url_overrides or the mp-state paths.)
write_config_json

cd "$REPO"
PYTHONPATH="$REPO" "$PYTHON" -m mp.bring_up \
  --num_workers "$NUM_WORKERS" \
  --host "$HOST_IP" \
  --docker_host "$DOCKER_HOST" \
  --golden_root "$GOLDEN_ROOT" \
  "$@"
rc=$?
log "mp.bring_up exit=$rc"

share_socket
share_paths
log "running containers:"; docker ps --format '  {{.Names}}\t{{.Status}}' | sort
exit "$rc"
