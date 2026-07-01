#!/usr/bin/env bash
# Deploy the official WebArena-Verified map (am1n3e/webarena-verified-map) on
# 0.0.0.0:13000. Map data (9 docker volumes) comes from S3 (default) or from a
# peer host (LAN tar|ssh|tar — much faster). Idempotent / resumable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/lib.sh"

MAP_DL="$BASE/wa-verified-downloads"
mkdir -p "$MAP_DL"
S3_BASE=https://webarena-map-server-data.s3.amazonaws.com
TARBALLS=(osm_tile_server.tar nominatim_volumes.tar osrm_routing.tar)

# vol|mount|src_tar|extract_path|strip   (empty tar => created empty)
VOLUMES=(
  "webarena_verified_map_tile_db|/data/database|osm_tile_server.tar|projects/ogma3/docker/volumes/osm-data/_data|6"
  "webarena_verified_map_routing_car|/data/routing/car|osrm_routing.tar|car|1"
  "webarena_verified_map_routing_bike|/data/routing/bike|osrm_routing.tar|bike|1"
  "webarena_verified_map_routing_foot|/data/routing/foot|osrm_routing.tar|foot|1"
  "webarena_verified_map_nominatim_db|/data/nominatim/postgres|nominatim_volumes.tar|projects/metis2/docker/docker/volumes/nominatim-data/_data|7"
  "webarena_verified_map_nominatim_flatnode|/data/nominatim/flatnode|nominatim_volumes.tar|projects/metis2/docker/docker/volumes/nominatim-flatnode/_data|7"
  "webarena_verified_map_website_db|/var/lib/postgresql/14/main|||"
  "webarena_verified_map_tiles|/data/tiles|||"
  "webarena_verified_map_style|/data/style|||"
)
ALL_VOLS=(webarena_verified_map_tile_db webarena_verified_map_routing_car
  webarena_verified_map_routing_bike webarena_verified_map_routing_foot
  webarena_verified_map_nominatim_db webarena_verified_map_nominatim_flatnode
  webarena_verified_map_website_db webarena_verified_map_tiles webarena_verified_map_style)

log "=== map deploy (MAP_SOURCE=$MAP_SOURCE) ==="
docker pull --platform linux/amd64 am1n3e/webarena-verified-map:latest \
  || ssh -o Compression=no "$PEER" "DOCKER_HOST=$PEER_DOCKER_HOST docker save am1n3e/webarena-verified-map:latest" | docker load

for v in "${ALL_VOLS[@]}"; do docker volume inspect "$v" >/dev/null 2>&1 || docker volume create "$v" >/dev/null; done

vol_populated() { [ -n "$(docker run --rm -v "$1:/d:ro" alpine sh -c 'ls -A /d 2>/dev/null | head -1' 2>/dev/null)" ]; }

if [ "$MAP_SOURCE" = "peer" ]; then
  command -v ssh >/dev/null || die "ssh required for MAP_SOURCE=peer"
  for v in "${ALL_VOLS[@]}"; do
    vol_populated "$v" && { log "SKIP $v (populated)"; continue; }
    log "transfer volume $v from $PEER"
    ssh -o Compression=no "$PEER" "DOCKER_HOST=$PEER_DOCKER_HOST docker run --rm -v $v:/src:ro alpine tar -C /src -cf - ." \
      | docker run --rm -i -v "$v:/dst" alpine tar -C /dst -xf -
  done
else
  # ---- S3 download + extract ----
  for name in "${TARBALLS[@]}"; do
    out="$MAP_DL/$name"; [ -s "$out" ] && { log "have $name"; continue; }
    log "download $name from S3"
    docker run --rm -v "$MAP_DL:/data" alpine sh -c \
      "apk add --no-cache curl >/dev/null 2>&1; curl -L --fail --retry 5 --retry-delay 10 -C - -o /data/$name '$S3_BASE/$name'"
  done
  for spec in "${VOLUMES[@]}"; do
    IFS='|' read -r vol mount tar extract strip <<<"$spec"; [ -z "$tar" ] && continue
    vol_populated "$vol" && { log "SKIP extract $vol"; continue; }
    log "extract $tar -> $vol (strip=$strip)"
    docker run --rm -v "$vol:/dst" -v "$MAP_DL:/src:ro" alpine \
      sh -c "cd /dst && tar --strip-components=$strip -xf /src/$tar $extract"
  done
fi

# stop any legacy fake-map proxy, then run the official container
[ -f "$PIDS/proxy-map.pid" ] && { kill "$(cat "$PIDS/proxy-map.pid")" 2>/dev/null || true; rm -f "$PIDS/proxy-map.pid"; }
run_container map --platform linux/amd64 -p 127.0.0.1:13030:8080 -p 127.0.0.1:13031:8877 \
  -v webarena_verified_map_tile_db:/data/database \
  -v webarena_verified_map_routing_car:/data/routing/car \
  -v webarena_verified_map_routing_bike:/data/routing/bike \
  -v webarena_verified_map_routing_foot:/data/routing/foot \
  -v webarena_verified_map_nominatim_db:/data/nominatim/postgres \
  -v webarena_verified_map_nominatim_flatnode:/data/nominatim/flatnode \
  -v webarena_verified_map_website_db:/var/lib/postgresql/14/main \
  -v webarena_verified_map_tiles:/data/tiles \
  -v webarena_verified_map_style:/data/style \
  -d am1n3e/webarena-verified-map:latest
run_proxy map 13000 13030

wait_http "http://$HOST_IP:13000/" 300 && log "map OK" || log "map not ready (tiles render on first request under load)"
