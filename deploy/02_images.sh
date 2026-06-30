#!/usr/bin/env bash
# Step 2 — obtain the 4 populated site images + the wikipedia .zim, then tag the
# golden:* aliases that mp.bring_up / mp.reset expect. Source is selectable
# (metis download | local tars | peer docker-save-over-ssh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/lib.sh"

log "=== 02: site images (IMG_SOURCE=$IMG_SOURCE) ==="

download_metis() {
  local file="$1" out="$DOWNLOADS/$1"
  [ -s "$out" ] && { log "have $file"; return; }
  log "downloading $file from $METIS_HOST"
  curl --fail --location --continue-at - \
       --resolve "$METIS_HOST:80:$METIS_IP" \
       --output "$out" "http://$METIS_HOST/webarena-images/$file"
}
load_tar() {
  local tar="$1" img="${1%.tar}"
  docker image inspect "$img" >/dev/null 2>&1 && { log "image present: $img"; return; }
  [ -s "$DOWNLOADS/$tar" ] || die "missing tar: $DOWNLOADS/$tar"
  log "docker load $tar"; docker load --input "$DOWNLOADS/$tar"
}
peer_pull() {
  local img="$1"
  docker image inspect "$img" >/dev/null 2>&1 && { log "image present: $img"; return; }
  command -v ssh >/dev/null || die "ssh required for IMG_SOURCE=peer"
  log "peer save|load $img from $PEER"
  ssh -o Compression=no -o Ciphers=chacha20-poly1305@openssh.com "$PEER" \
      "DOCKER_HOST=$PEER_DOCKER_HOST docker save $img" | docker load
}

case "$IMG_SOURCE" in
  metis)
    for t in "${SITE_TARS[@]}"; do download_metis "$t"; load_tar "$t"; done
    download_metis "$WIKI_ZIM"
    ;;
  localtars)
    for t in "${SITE_TARS[@]}"; do load_tar "$t"; done
    [ -s "$DOWNLOADS/$WIKI_ZIM" ] || log "WARN: $DOWNLOADS/$WIKI_ZIM absent (needed by 03_shared_services.sh)"
    ;;
  peer)
    for pair in "${GOLDEN_ALIASES[@]}"; do peer_pull "${pair% *}"; done
    if [ ! -s "$DOWNLOADS/$WIKI_ZIM" ]; then
      log "rsync wiki zim from peer"
      rsync -a --info=progress2 "$PEER:$DOWNLOADS/$WIKI_ZIM" "$DOWNLOADS/" || log "WARN: wiki zim not fetched"
    fi
    ;;
  *) die "unknown IMG_SOURCE=$IMG_SOURCE (metis|localtars|peer)";;
esac

tag_golden_aliases
log "images now:"; docker images --format '  {{.Repository}}:{{.Tag}} {{.Size}}' \
  | grep -Ei 'golden|shopping|postmill|gitlab|verified-map' || true
