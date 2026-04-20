#!/usr/bin/env bash
# =============================================================
# nodeb_connect_watch.sh  --  run on NODE B inside WSL terminal
#
# Connects Node B to Node A's Ray head via Tailscale and
# auto-reconnects whenever the local raylet process dies.
#
# NEVER calls ray.init() or ray.shutdown() -- those were the
# root cause of Node B disconnecting on every verify run.
#
# Usage:
#   bash nodeb_connect_watch.sh --address 100.x.x.x:6379
#   (use Node A's Tailscale IP, shown in ray_cluster.ps1 output)
#
# Or just run the PowerShell script which does this automatically:
#   .\ray_cluster.ps1 -Role B -TailscaleIP 100.x.x.x
# =============================================================

set -uo pipefail

ADDRESS=""
VENV_PATH="$HOME/maor-equity/venv"

usage() {
    echo "Usage: bash nodeb_connect_watch.sh --address HOST:PORT [--venv PATH]"
    echo "Example: bash nodeb_connect_watch.sh --address 100.64.0.1:6379"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --address) ADDRESS="${2:-}"; shift 2 ;;
        --venv)    VENV_PATH="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$ADDRESS" ]]; then
    echo "ERROR: --address is required."
    usage
    exit 1
fi

RAY_BIN="$VENV_PATH/bin/ray"
if [[ ! -x "$RAY_BIN" ]]; then
    echo "ERROR: Ray not found at $RAY_BIN"
    echo "Set --venv to the correct venv path."
    exit 1
fi

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export CUDA_VISIBLE_DEVICES=0

# Add project root to PYTHONPATH so Ray worker processes can import 'agents'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
echo "  PYTHONPATH → $PROJECT_ROOT"

HEAD_HOST=$(echo "$ADDRESS" | cut -d: -f1)
HEAD_PORT=$(echo "$ADDRESS" | cut -d: -f2)

now() { date '+%H:%M:%S'; }

connect_worker() {
    echo "[$(now)] Stopping existing Ray..."
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    sleep 2
    echo "[$(now)] Connecting to head at $ADDRESS..."
    "$RAY_BIN" start \
        --address="$ADDRESS" \
        --num-gpus=1 \
        --num-cpus=4 \
        2>&1 | tail -6
    echo "[$(now)] Worker started. Watching for disconnects..."
}

trap 'echo "[$(now)] Watcher stopped."; "$RAY_BIN" stop 2>/dev/null; exit 0' INT TERM

echo ""
echo "  Node B Ray Worker Watchdog"
echo "  Head  : $ADDRESS"
echo "  Ray   : $RAY_BIN"
echo "  Checks: every 10s (pgrep raylet + TCP probe)"
echo "  Note  : never calls ray.init/shutdown -- safe for connection"
echo ""

connect_worker

while true; do
    sleep 10

    # Check 1: local raylet process alive?
    if ! pgrep -f raylet >/dev/null 2>&1; then
        echo "[$(now)] RAYLET DIED -- reconnecting..."
        connect_worker
        continue
    fi

    # Check 2: TCP path to Node A still open? (pure TCP, no Ray protocol)
    if ! nc -z -w4 "$HEAD_HOST" "$HEAD_PORT" 2>/dev/null; then
        echo "[$(now)] LOST CONNECTION to $HEAD_HOST:$HEAD_PORT -- reconnecting..."
        connect_worker
        continue
    fi

    echo "[$(now)] OK -- raylet alive, connected to $ADDRESS"
done
