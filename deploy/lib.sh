#!/usr/bin/env bash
# Shared helpers + config loader for the WebArena deploy scripts.
# Sourced by every other script; not executed directly.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${WA_CONFIG:-$HERE/config.env}"
if [ ! -f "$CONF" ]; then
  echo "FATAL: $CONF not found. Copy config.env.example -> config.env and edit it." >&2
  exit 1
fi
# shellcheck disable=SC1090
. "$CONF"

# ---- Derived paths -----------------------------------------------------------
ROOTLESS="$BASE/rootless-docker"
DOWNLOADS="$BASE/downloads"
LOGS="$BASE/logs"
PIDS="$BASE/pids"
DOCKER_SOCK="$ROOTLESS/run/docker.sock"
export DOCKER_HOST="unix://$DOCKER_SOCK"
PYTHON="$VENV/bin/python"
TCP_PROXY="$HERE/tcp_proxy.py"

mkdir -p "$DOWNLOADS" "$LOGS" "$PIDS" "$ROOTLESS/run" "$ROOTLESS/data" "$ROOTLESS/exec" 2>/dev/null || true

# ---- The populated WebArena source images and their golden aliases -----------
# (src image tag) -> (golden alias used by mp.bring_up / mp.reset)
GOLDEN_ALIASES=(
  "shopping_final_0712:latest webarena-shopping-golden:latest"
  "shopping_admin_final_0719:latest webarena-shopping_admin-golden:latest"
  "postmill-populated-exposed-withimg:latest webarena-reddit-golden:latest"
  "gitlab-populated-final-port8023:latest webarena-gitlab-golden:latest"
)
# metis tar filenames (IMG_SOURCE=metis / localtars)
SITE_TARS=(
  shopping_final_0712.tar
  shopping_admin_final_0719.tar
  postmill-populated-exposed-withimg.tar
  gitlab-populated-final-port8023.tar
)
METIS_HOST=metis.lti.cs.cmu.edu
METIS_IP=128.2.205.52
WIKI_ZIM=wikipedia_en_all_maxi_2022-05.zim

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
die() { log "FATAL: $*"; exit 1; }

# Write an AUTHORITATIVE mp/config.json from config.env. This is the single
# source of truth for the harness (mp.bring_up / mp.reset / mp.reset_cli).
# Do NOT rely on bring_up's save_config: it runs last (skipped if a health
# check aborts) and it does NOT derive readonly_url_overrides or the mp-state
# paths from --host, so a copied config.json keeps the wrong host/paths.
write_config_json() {
  local mp_root; mp_root="$(dirname "$GOLDEN_ROOT")"   # e.g. .../webarena-mp
  mkdir -p "$mp_root"/{auth,config_files,results} 2>/dev/null || true
  cat > "$REPO/mp/config.json" <<JSON
{
  "auth_root": "$mp_root/auth",
  "config_files_root": "$mp_root/config_files",
  "docker_host": "unix://$DOCKER_SOCK",
  "golden_root": "$GOLDEN_ROOT",
  "host": "$HOST_IP",
  "map_replicas": 1,
  "num_workers": $NUM_WORKERS,
  "port_stride": 100,
  "readonly_url_overrides": {
    "homepage": "http://$HOST_IP:4399",
    "map": "http://$HOST_IP:13000",
    "wikipedia": "http://$HOST_IP:8888/${WIKI_ZIM%.zim}/A/User:The_other_Kiwix_guy/Landing"
  },
  "result_dir": "$mp_root/results",
  "ssh_host": ""
}
JSON
  log "wrote authoritative $REPO/mp/config.json (host=$HOST_IP workers=$NUM_WORKERS)"
}

# Start the rootless Docker daemon if it isn't already serving on our socket.
ensure_rootless_docker() {
  if [ -S "$DOCKER_SOCK" ] && docker version >/dev/null 2>&1; then
    log "rootless docker already up"
    return
  fi
  log "starting rootless docker (data-root=$ROOTLESS/data)"
  nohup env \
    XDG_RUNTIME_DIR="$ROOTLESS/run" \
    DOCKERD_ROOTLESS_ROOTLESSKIT_PORT_DRIVER=slirp4netns \
    dockerd-rootless.sh \
      --data-root "$ROOTLESS/data" \
      --exec-root "$ROOTLESS/exec" \
      --host "$DOCKER_HOST" \
      > "$LOGS/dockerd-rootless.log" 2>&1 &
  for _ in $(seq 1 60); do
    [ -S "$DOCKER_SOCK" ] && docker version >/dev/null 2>&1 && break
    sleep 2
  done
  docker version >/dev/null || die "rootless docker failed to start (see $LOGS/dockerd-rootless.log)"
}

# Rootless dockerd's embedded resolver sometimes ships a broken resolv.conf.
fix_rootless_dns() {
  local resolv="$ROOTLESS/run/dockerd-rootless/resolv.conf"
  [ -f "$resolv" ] && printf 'nameserver 8.8.8.8\nnameserver %s\n' "${HOST_IP%.*}.1" > "$resolv" || true
}

# Open the socket to all local users (so any account can run mp.reset). Optional.
share_socket() {
  [ "${SHARE_SOCKET:-0}" = "1" ] || return 0
  chmod 0666 "$DOCKER_SOCK" && log "socket opened 0666 (shared reset access)"
}

# Add the directory search bit (o+x) to every ancestor of $1, walking up to '/'.
# o+x (not o+r) means other users can traverse THROUGH to a known path without
# being able to list the directory, so private siblings (e.g. ~/.ssh at 0700)
# stay protected. Non-owned/system dirs already have o+x, so failures are benign.
_grant_traverse() {
  local p; p="$(cd "$1" 2>/dev/null && pwd)" || return 0
  while [ -n "$p" ] && [ "$p" != "/" ]; do
    chmod o+x "$p" 2>/dev/null || true
    p="$(dirname "$p")"
  done
}

# A 0666 socket is NOT enough for "any SSH user can reset": every ancestor dir
# of BOTH the socket and the repo must also be traversable, or other users can't
# reach them (e.g. a 0700/0750 $HOME blocks everything beneath it). When the
# socket is shared, grant o+x up both chains and make the harness world-readable
# so any local account can import mp.* and read config.json. Idempotent.
share_paths() {
  [ "${SHARE_SOCKET:-0}" = "1" ] || return 0
  _grant_traverse "$(dirname "$DOCKER_SOCK")"
  _grant_traverse "$REPO"
  chmod -R o+rX "$REPO/mp" 2>/dev/null || true
  log "shared paths: socket + repo ancestors made o+x, mp/ world-readable"
}

# Start a 0.0.0.0:<listen> -> 127.0.0.1:<target> bridge (idempotent per name).
run_proxy() {
  local name="$1" listen="$2" target="$3" pidf="$PIDS/proxy-$1.pid"
  [ -f "$pidf" ] && kill "$(cat "$pidf")" >/dev/null 2>&1 || true
  nohup python3 "$TCP_PROXY" --listen-port "$listen" --target-port "$target" \
    > "$LOGS/proxy-$name.log" 2>&1 &
  echo $! > "$pidf"
  log "proxy $name :$listen -> 127.0.0.1:$target (pid $(cat "$pidf"))"
}

# docker run, replacing any existing container of the same name.
run_container() {
  local name="$1"; shift
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run --name "$name" --restart unless-stopped "$@"
}

# Tag the populated source images as the golden:* aliases mp.* expects.
tag_golden_aliases() {
  local pair src alias
  for pair in "${GOLDEN_ALIASES[@]}"; do
    src=${pair% *}; alias=${pair#* }
    if docker image inspect "$src" >/dev/null 2>&1; then
      docker tag "$src" "$alias" && log "tagged $src -> $alias"
    else
      log "WARN: source image missing, cannot tag: $src"
    fi
  done
}

# Poll an HTTP endpoint until it returns 200/302 or times out (seconds).
wait_http() {
  local url="$1" timeout="${2:-180}" t=0 code
  while [ "$t" -lt "$timeout" ]; do
    code=$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null || echo 000)
    case "$code" in 200|301|302) return 0;; esac
    sleep 5; t=$((t+5))
  done
  return 1
}

# Host port for a mutable site replica (mirrors mp/config.py: base + 100*worker,
# with shopping_admin worker 3 shifted off the map-reserved :8080).
port_for() {
  local site="$1" w="$2" base
  case "$site" in
    shopping) base=7770;; shopping_admin) base=7780;; reddit) base=9999;; gitlab) base=8023;;
    *) echo 0; return;;
  esac
  local p=$((base + 100*w))
  { [ "$site" = "shopping_admin" ] && [ "$w" = 3 ]; } && p=8090   # :8080 reserved by map
  echo "$p"
}
