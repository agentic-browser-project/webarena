#!/usr/bin/env bash
# Shared helpers for mp/launch_{tsa,dense,judge}.sh and mp/teardown_inference.sh.
#
# Sourced — not executed directly. Defines:
#
#   gpu_detect       — echoes the sm version reported by the GPU host
#   ssh_gpu CMD ...  — wraps a remote command in `ssh "$GPU_HOST"` with sane opts
#   wait_healthy URL TIMEOUT_S — polls /health until 200 or timeout
#   open_tunnel LOCAL REMOTE   — autossh -L LOCAL:127.0.0.1:REMOTE "$GPU_HOST"
#   kill_tunnel LOCAL          — kills any tunnel matching :LOCAL
#   ensure_tmux NAME CMD       — start tmux session NAME running CMD if absent
#   kill_tmux NAME             — kill the named tmux session on the GPU host
#
# Required env (caller MUST export — there is no operator-specific default):
#   GPU_HOST          — user@host (or user@ip) of the GPU machine
#                       e.g. `export GPU_HOST=alice@my-gpu.example.com`
#                       e.g. `export GPU_HOST=alice@10.0.0.42`  (when the orchestrator
#                            host lacks DNS resolution for the GPU host)
# Optional env:
#   TSA_REMOTE_ROOT   — path to the TreeSparseAttention_CW checkout on the GPU host.
#                       Default: $HOME/TreeSparseAttention_CW

set -euo pipefail

if [[ -z "${GPU_HOST:-}" ]]; then
    echo "[_inference_common.sh] ERROR: GPU_HOST is not set." >&2
    echo "  Export it before sourcing/launching, e.g.:" >&2
    echo "    export GPU_HOST=user@gpu-host.example.com" >&2
    return 1 2>/dev/null || exit 1
fi
TSA_REMOTE_ROOT="${TSA_REMOTE_ROOT:-\$HOME/TreeSparseAttention_CW}"

# ── SSH helpers ───────────────────────────────────────────────────────────────

ssh_gpu() {
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        "$GPU_HOST" "$@"
}

gpu_detect() {
    # Returns just digits, e.g. "120" for sm_12.0. Falls back to "unknown".
    ssh_gpu 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -n1 | awk -F. "{printf \"%s%s\", \$1, \$2}"' \
        2>/dev/null || echo "unknown"
}

# ── Health probe ──────────────────────────────────────────────────────────────

wait_healthy() {
    local url="$1"
    local timeout_s="${2:-600}"
    local elapsed=0
    while (( elapsed < timeout_s )); do
        if curl --silent --fail --max-time 5 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
        elapsed=$(( elapsed + 5 ))
    done
    echo "[wait_healthy] $url did not become ready within ${timeout_s}s" >&2
    return 1
}

# ── Tunnels (autossh preferred, plain ssh fallback) ───────────────────────────

_tunnel_pidfile() {
    local local_port="$1"
    echo "/tmp/wa_tunnel_${local_port}.pid"
}

open_tunnel() {
    local local_port="$1"
    local remote_port="$2"
    local pidfile
    pidfile="$(_tunnel_pidfile "$local_port")"
    # Reuse healthy tunnel if one already exists.
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        echo "[tunnel] :$local_port already established (pid $(cat "$pidfile"))"
        return 0
    fi
    rm -f "$pidfile"
    if command -v autossh >/dev/null 2>&1; then
        AUTOSSH_PIDFILE="$pidfile" autossh -M 0 -f -N \
            -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" \
            "$GPU_HOST"
    else
        nohup ssh -N \
            -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" \
            "$GPU_HOST" >/dev/null 2>&1 &
        echo $! > "$pidfile"
    fi
    sleep 1
    echo "[tunnel] opened 127.0.0.1:${local_port} → $GPU_HOST:${remote_port}"
}

kill_tunnel() {
    local local_port="$1"
    local pidfile
    pidfile="$(_tunnel_pidfile "$local_port")"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid="$(cat "$pidfile")"
        kill "$pid" 2>/dev/null || true
        rm -f "$pidfile"
    fi
    # Best-effort sweep for orphan tunnels. We use pgrep + per-PID `kill`
    # rather than `pkill -f` because pkill's pattern-match can hit the very
    # shell command currently running (whose argv contains the same regex
    # text), terminating the caller. pgrep skips self by default.
    local pids
    pids=$(pgrep -f "ssh.*-L 127.0.0.1:${local_port}:127.0.0.1:" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        for p in $pids; do
            # Final safety: confirm cmdline still matches before killing,
            # and skip our own PID/parent.
            if [[ "$p" != "$$" && "$p" != "$PPID" ]]; then
                kill "$p" 2>/dev/null || true
            fi
        done
    fi
}

# ── tmux session helpers (on GPU host) ────────────────────────────────────────

ensure_tmux() {
    local name="$1"
    local cmd="$2"
    ssh_gpu "tmux has-session -t ${name} 2>/dev/null || \
             tmux new-session -d -s ${name} '${cmd}'"
}

kill_tmux() {
    local name="$1"
    ssh_gpu "tmux kill-session -t ${name} 2>/dev/null" || true
}
