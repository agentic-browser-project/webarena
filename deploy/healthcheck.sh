#!/usr/bin/env bash
# Probe every replica + shared service over HTTP and print a status matrix.
# 200/301/302 = serving. gitlab redirects to /users/sign_in (302) when healthy.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/lib.sh"

log "=== healthcheck: $NUM_WORKERS workers/site + shared (host=$HOST_IP) ==="
ok=0; tot=0
probe() {
  local label="$1" url="$2" code
  tot=$((tot+1))
  code=$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' --max-time 12 "$url" 2>/dev/null || echo 000)
  case "$code" in 200|301|302) ok=$((ok+1));; esac
  printf '  %-20s %s  %s\n' "$label" "$code" "$url"
}

for w in $(seq 0 $((NUM_WORKERS-1))); do
  for site in shopping shopping_admin reddit gitlab; do
    probe "$site w$w" "http://$HOST_IP:$(port_for "$site" "$w")/"
  done
done
probe "homepage"  "http://$HOST_IP:4399/"
probe "wikipedia" "http://$HOST_IP:8888/"
probe "map"       "http://$HOST_IP:13000/"

log "healthy: $ok/$tot"
[ "$ok" = "$tot" ]
