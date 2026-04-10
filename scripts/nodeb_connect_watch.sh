#!/usr/bin/env bash
# =============================================================
# nodeb_connect_watch.sh  —  run on NODE B inside WSL terminal
# Auto-reconnects whenever the local raylet process dies.
# NEVER calls ray.init() or ray.shutdown() — those disrupt the
# worker connection registered with GCS.
# =============================================================

set -uo pipefail

ADDRESS=""
VENV_PATH="$HOME/maor-equity/venv"   # adjust if cloned elsewhere

usage() {
    echo "Usage: bash nodeb_connect_watch.sh --address HOST:PORT [--venv PATH]"
    echo "Example: bash nodeb_connect_watch.sh --address 0.tcp.ap.ngrok.io:12345"
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

NGROK_HOST=$(echo "$ADDRESS" | cut -d: -f1)
NGROK_PORT=$(echo "$ADDRESS" | cut -d: -f2)

now() { date '+%H:%M:%S'; }

connect_worker() {
    echo "[$(now)] Starting ray worker → $ADDRESS"
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
    sleep 2
    "$RAY_BIN" start \
        --address="$ADDRESS" \
        --num-gpus=1 \
        --num-cpus=4 \
        2>&1 | tail -5
    echo "[$(now)] ray start returned (worker running as daemon)"
}

trap 'echo "[$(now)] Watcher stopped."; exit 0' INT TERM

echo "[$(now)] Node B watcher started — address=$ADDRESS"
echo "[$(now)] Checking every 10s. Ctrl+C to stop."
echo ""

# Initial connect
connect_worker

while true; do
    sleep 10

    # ── Check 1: is the local raylet process alive? ─────────────────
    if ! pgrep -f "raylet" >/dev/null 2>&1; then
        echo "[$(now)] RAYLET DIED — reconnecting..."
        connect_worker
        continue
    fi

    # ── Check 2: can we still reach Node A's GCS via Ngrok? ─────────
    # Use nc with a 3s timeout — pure TCP, no Ray connection, no disruption.
    if ! nc -z -w3 "$NGROK_HOST" "$NGROK_PORT" 2>/dev/null; then
        echo "[$(now)] NGROK TUNNEL UNREACHABLE — reconnecting..."
        connect_worker
        continue
    fi

    echo "[$(now)] OK — raylet alive, tunnel reachable"
done
