#!/usr/bin/env bash
# Tear down TSA / dense / judge servers + tunnels.
#
# Flags:
#   --keep-judge   leave the judge tmux session + tunnel alive (back-to-back
#                  TSA → dense runs save warm-up time this way).
#   --tsa-only     tear down only the TSA server + tunnel.
#   --dense-only   tear down only the dense server + tunnel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/_inference_common.sh"

KEEP_JUDGE=0
TARGET="all"
for arg in "$@"; do
    case "$arg" in
        --keep-judge) KEEP_JUDGE=1 ;;
        --tsa-only)   TARGET="tsa" ;;
        --dense-only) TARGET="dense" ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

TSA_PORT="${TSA_PORT:-10000}"
DENSE_PORT="${DENSE_PORT:-10001}"
JUDGE_PORT="${JUDGE_PORT:-10002}"

if [[ "$TARGET" == "all" || "$TARGET" == "tsa" ]]; then
    echo "[teardown] killing wa-tsa tmux + tunnel :$TSA_PORT"
    kill_tmux "wa-tsa"
    kill_tunnel "$TSA_PORT"
fi
if [[ "$TARGET" == "all" || "$TARGET" == "dense" ]]; then
    echo "[teardown] killing wa-dense tmux + tunnel :$DENSE_PORT"
    kill_tmux "wa-dense"
    kill_tunnel "$DENSE_PORT"
fi
if [[ "$TARGET" == "all" && $KEEP_JUDGE -eq 0 ]]; then
    echo "[teardown] killing wa-judge tmux + tunnel :$JUDGE_PORT"
    kill_tmux "wa-judge"
    kill_tunnel "$JUDGE_PORT"
fi

# Clean up the env file so subsequent shells don't pick up stale endpoints.
rm -f "$SCRIPT_DIR/.inference_env"
echo "[teardown] done"
