#!/usr/bin/env bash
# Step 3 — shared read-only services: wikipedia (kiwix-serve) + homepage (Flask),
# each fronted by a tcp_proxy bridge. Map is handled by deploy_map.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/lib.sh"

log "=== 03: shared services (wikipedia + homepage) ==="

# --- wikipedia ---------------------------------------------------------------
[ -s "$DOWNLOADS/$WIKI_ZIM" ] || die "missing $DOWNLOADS/$WIKI_ZIM (run 02_images.sh first)"
docker pull ghcr.io/kiwix/kiwix-serve:3.3.0 || true
run_container wikipedia --volume="$DOWNLOADS:/data" -p 127.0.0.1:18888:80 -d \
  ghcr.io/kiwix/kiwix-serve:3.3.0 "$WIKI_ZIM"
sleep 5
run_proxy wikipedia 8888 18888

# --- homepage ----------------------------------------------------------------
tpl="$HOMEPAGE_DIR/templates/index.html"
if [ -f "$tpl" ]; then
  perl -pi -e "s|<your-server-hostname>|http://$HOST_IP|g" "$tpl"
  perl -pi -e "s|http://[0-9.]+:3000|http://$HOST_IP:13000|g" "$tpl"
  [ -f "$PIDS/homepage.pid" ] && kill "$(cat "$PIDS/homepage.pid")" >/dev/null 2>&1 || true
  ( cd "$HOMEPAGE_DIR" && nohup "$VENV/bin/flask" run --host=127.0.0.1 --port=14399 \
      > "$LOGS/homepage.log" 2>&1 & echo $! > "$PIDS/homepage.pid" )
  sleep 3
  run_proxy homepage 4399 14399
else
  log "WARN: $tpl missing — clone web-arena-x/webarena-homepage into \$HOMEPAGE_DIR to enable the homepage"
fi

# --- health ------------------------------------------------------------------
wait_http "http://$HOST_IP:8888/"  60 && log "wikipedia OK" || log "wikipedia not ready"
wait_http "http://$HOST_IP:4399/"  30 && log "homepage OK"  || log "homepage not ready / not configured"
