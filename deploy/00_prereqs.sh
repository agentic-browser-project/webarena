#!/usr/bin/env bash
# Step 0 — one-time prerequisites for rootless Docker on this machine.
# Installs the rootless packages, fixes the Ubuntu 23.10+/24.04 AppArmor
# unprivileged-userns restriction (which otherwise breaks rootlesskit),
# installs the rootless docker binaries, and registers a reboot-surviving
# systemd --user daemon whose data-root + socket live on the big disk (BASE),
# with the socket world-accessible (0666) so any local user can run reset.
#
# Needs sudo for the apt install + AppArmor profile + enable-linger.
# Idempotent: safe to re-run.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; . "$HERE/lib.sh"
U=$(id -u); ME=$(id -un)
export PATH="$HOME/bin:$PATH"
export XDG_RUNTIME_DIR="/run/user/$U"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$U/bus"

log "=== 00: prerequisites (rootless docker + AppArmor userns + systemd daemon) ==="
mkdir -p "$BASE"/{downloads,logs,pids} "$ROOTLESS"/{run,data,exec}

# 1) OS packages (rootless docker prereqs + python venv + flask deps). Needs root.
log "apt: uidmap fuse-overlayfs slirp4netns dbus-user-session python3-venv"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  uidmap fuse-overlayfs slirp4netns dbus-user-session python3-venv curl rsync
# Subordinate UID/GID ranges (needed by rootless). Add if absent.
grep -q "^$ME:" /etc/subuid || sudo usermod --add-subuids 100000-165535 "$ME"
grep -q "^$ME:" /etc/subgid || sudo usermod --add-subgids 100000-165535 "$ME"
sudo loginctl enable-linger "$ME"

# 2) AppArmor: allow rootlesskit to create a user namespace (Ubuntu 23.10+/24.04).
if [ -d /etc/apparmor.d ]; then
  RK="$HOME/bin/rootlesskit"
  PROF="$(printf '%s' "${RK#/}" | tr / .)"   # e.g. home.user.bin.rootlesskit
  log "apparmor: profile $PROF for $RK"
  sudo tee "/etc/apparmor.d/$PROF" >/dev/null <<AA
abi <abi/4.0>,
include <tunables/global>
$RK flags=(unconfined) {
  userns,
  include if exists <local/$PROF>
}
AA
  sudo systemctl restart apparmor.service 2>/dev/null || true
fi

# 3) Rootless docker binaries (installs dockerd-rootless.sh / rootlesskit into ~/bin).
if [ ! -x "$HOME/bin/dockerd-rootless.sh" ] && ! command -v dockerd-rootless.sh >/dev/null 2>&1; then
  log "installing rootless docker into ~/bin"
  curl -fsSL https://get.docker.com/rootless | FORCE_ROOTLESS_INSTALL=1 sh || \
    log "(setuptool step may warn — we run our own unit below)"
fi
DRS="$HOME/bin/dockerd-rootless.sh"; [ -x "$DRS" ] || DRS="$(command -v dockerd-rootless.sh || true)"
[ -x "$DRS" ] || die "dockerd-rootless.sh not installed (check apt prereqs + AppArmor)"

# 4) Drop the default rootless docker.service (it uses ~/.local/share/docker on /);
#    we run our own unit with data-root + socket on the big disk.
systemctl --user disable --now docker 2>/dev/null || true

# 5) Register the webarena-dockerd systemd --user unit (reboot-surviving via linger).
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/webarena-dockerd.service" <<UNIT
[Unit]
Description=WebArena rootless dockerd (data-root + socket on big disk)
Wants=dbus.socket
After=dbus.socket

[Service]
Environment=PATH=$HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=XDG_RUNTIME_DIR=/run/user/$U
Environment=DOCKERD_ROOTLESS_ROOTLESSKIT_PORT_DRIVER=slirp4netns
ExecStart=$DRS --data-root $ROOTLESS/data --exec-root $ROOTLESS/exec --host unix://$DOCKER_SOCK
ExecStartPost=/bin/sh -c 'for i in \$(seq 1 30); do [ -S $DOCKER_SOCK ] && break; sleep 1; done; chmod 0666 $DOCKER_SOCK 2>/dev/null || true'
Restart=always
RestartSec=5
Type=simple

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable webarena-dockerd
systemctl --user restart webarena-dockerd

# 6) Wait for the socket + verify.
for i in $(seq 1 40); do [ -S "$DOCKER_SOCK" ] && break; sleep 2; done
chmod 0666 "$DOCKER_SOCK" 2>/dev/null || true
DOCKER_HOST="unix://$DOCKER_SOCK" docker info \
  --format 'Server {{.ServerVersion}} | driver {{.Driver}} | root {{.DockerRootDir}}' 2>&1 | head -1
ls -l "$DOCKER_SOCK"
echo -n "unit enabled: "; systemctl --user is-enabled webarena-dockerd
echo -n "unit active:  "; systemctl --user is-active webarena-dockerd
log "=== 00 done — rootless docker up on $DOCKER_SOCK ==="
