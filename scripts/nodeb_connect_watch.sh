#!/usr/bin/env bash
# Keep Node B connected to Node A's Ray head through transient failures.

set -euo pipefail

PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
VENV_DEFAULT="$PROJECT_ROOT/venv"

ADDRESS=""
INTERVAL=8
VENV_PATH="$VENV_DEFAULT"

usage() {
    cat <<EOF
Usage: bash scripts/nodeb_connect_watch.sh --address <host:port> [--interval <sec>] [--venv <path>]

Examples:
  bash scripts/nodeb_connect_watch.sh --address 0.tcp.ap.ngrok.io:11311
  bash scripts/nodeb_connect_watch.sh --address 0.tcp.ap.ngrok.io:11311 --interval 10

Notes:
  - Expects Ray + Python packages installed in the given venv.
  - Keeps running until Ctrl+C and reconnects Node B automatically on drops.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --address)
            ADDRESS="${2:-}"
            shift 2
            ;;
        --interval)
            INTERVAL="${2:-}"
            shift 2
            ;;
        --venv)
            VENV_PATH="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$ADDRESS" ]]; then
    echo "Missing required --address <host:port>."
    usage
    exit 1
fi

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [[ "$INTERVAL" -lt 1 ]]; then
    echo "--interval must be a positive integer."
    exit 1
fi

PY_BIN="$VENV_PATH/bin/python3"
RAY_BIN="$VENV_PATH/bin/ray"

if [[ ! -x "$PY_BIN" ]]; then
    echo "Python not found at: $PY_BIN"
    exit 1
fi
if [[ ! -x "$RAY_BIN" ]]; then
    echo "Ray CLI not found at: $RAY_BIN"
    exit 1
fi

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1

LOCAL_IP="$(hostname -I | awk '{print $1}')"

now() {
    date '+%H:%M:%S'
}

probe_membership() {
    "$PY_BIN" - <<PY
import ray
import sys

address = "$ADDRESS"
local_ip = "$LOCAL_IP"

try:
    ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    ips = [n.get("NodeManagerAddress") for n in nodes]
    local_alive = any(ip == local_ip for ip in ips)
    print(f"alive_nodes={len(nodes)} local_alive={int(local_alive)} ips={ips}")
    sys.exit(0 if local_alive else 1)
except Exception as e:
    print(f"probe_error={type(e).__name__}: {e}")
    sys.exit(2)
finally:
    try:
        ray.shutdown()
    except Exception:
        pass
PY
}

stop_local_ray() {
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
}

connect_worker() {
    echo "[$(now)] reconnecting: ray start --address=$ADDRESS"
    set +e
    OUTPUT="$($RAY_BIN start --address="$ADDRESS" 2>&1)"
    RC=$?
    set -e
    if [[ $RC -eq 0 ]]; then
        echo "[$(now)] reconnect success"
    else
        echo "[$(now)] reconnect failed (exit $RC)"
        echo "$OUTPUT" | tail -n 12
    fi
}

trap 'echo "[$(now)] watcher stopped by user"; exit 0' INT TERM

echo "[$(now)] Node B watcher started"
echo "[$(now)] address=$ADDRESS local_ip=$LOCAL_IP interval=${INTERVAL}s"

while true; do
    if ! pgrep -f "raylet" >/dev/null 2>&1; then
        echo "[$(now)] local raylet not running"
        stop_local_ray
        connect_worker
        sleep "$INTERVAL"
        continue
    fi

    set +e
    PROBE_OUTPUT="$(probe_membership 2>&1)"
    PROBE_RC=$?
    set -e

    if [[ $PROBE_RC -eq 0 ]]; then
        echo "[$(now)] healthy $PROBE_OUTPUT"
    else
        echo "[$(now)] dropped  $PROBE_OUTPUT"
        stop_local_ray
        connect_worker
    fi

    sleep "$INTERVAL"
done
