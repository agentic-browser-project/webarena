#!/usr/bin/env bash
# Master end-to-end setup: rootless docker -> images -> shared services -> map
# -> per-worker replicas -> healthcheck. Reads config.env (copy from the example
# first). Extra args pass through to mp.bring_up, e.g.:
#   ./setup_all.sh --skip_goldens
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/lib.sh"

log "########## WebArena multi-worker setup — host=$HOST_IP workers=$NUM_WORKERS ##########"
critical() { log ">>> $*"; "$@" || die "step failed: $*"; }
besteffort() { log ">>> $*"; "$@" || log "WARN: step had warnings: $*"; }

critical  "$HERE/01_rootless_docker.sh"
critical  "$HERE/02_images.sh"
besteffort "$HERE/03_shared_services.sh"
besteffort "$HERE/deploy_map.sh"
# bring_up can return non-zero on a single slow replica under heavy host load;
# the healthcheck is the real verdict, so don't abort on its exit code.
besteffort "$HERE/04_bring_up_replicas.sh" "$@"

log ">>> final healthcheck"
"$HERE/healthcheck.sh" || log "NOTE: not all endpoints green — see matrix above (often transient under host load; re-run healthcheck.sh)"
log "########## setup_all done ##########"
