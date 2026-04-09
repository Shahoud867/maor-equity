#!/usr/bin/env bash
# =============================================================
# nodea_start.sh  —  Run THIS inside YOUR OWN open WSL terminal
# Do NOT run via PowerShell or any automated tool.
# Keep the terminal open while Node B is connecting.
# =============================================================

# NOTE: intentionally NOT using set -e so Ray startup errors are
# caught and reported rather than silently killing the script.
set -uo pipefail

VENV="/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity/venv"
RAY="$VENV/bin/ray"
PROJECT="$(dirname "$(dirname "$(realpath "$0")")")"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║        NODE A  —  Ray + Ngrok Startup           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Kill everything cleanly ──────────────────────────
echo "[1/4] Stopping any existing Ray / Ngrok processes..."
export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export RAY_raylet_start_wait_time_s=60
export RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS=60

"$RAY" stop --force 2>/dev/null || true
sleep 1

pkill -9 -f gcs_server   2>/dev/null || true
pkill -9 -f raylet        2>/dev/null || true
pkill -9 -f plasma_store  2>/dev/null || true
pkill -9 -f ngrok         2>/dev/null || true
sleep 2

# ── Clean up ALL stale Ray socket / lock files ────────────────
echo "    Cleaning stale Ray tmp files..."
rm -rf /tmp/ray /tmp/ray_*                        2>/dev/null || true
rm -f  /tmp/plasma_store_socket*                  2>/dev/null || true
rm -f  /tmp/.ray_lock*                            2>/dev/null || true
rm -rf /tmp/session_*                             2>/dev/null || true

# Give OS time to release sockets
sleep 2

if ss -tlnp 2>/dev/null | grep -q ':6379'; then
    echo "ERROR: Port 6379 still occupied after cleanup."
    echo "Run: sudo fuser -k 6379/tcp && sudo fuser -k 6380/tcp"
    exit 1
fi
echo "    Port 6379 free. Clean state confirmed."

# ── Step 2: Start Ray head ────────────────────────────────────
echo ""
echo "[2/4] Starting Ray head node on port 6379..."
NODE_IP=$(hostname -I | awk '{print $1}')
echo "    WSL IP: $NODE_IP"

# These env-vars MUST stay exported for the ray subprocess
export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export RAY_raylet_start_wait_time_s=60
export RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS=60

# ── Heartbeat tolerance for high-latency Ngrok tunnel ────────
# Default: 30 heartbeats × 1000ms = 30s before marking a node dead.
# Ngrok adds 100–400ms RTT, causing spurious heartbeat drops.
# New: 100 heartbeats × 5000ms = 500s tolerance.
export RAY_num_heartbeats_timeout=100
export RAY_raylet_heartbeat_period_milliseconds=5000
export RAY_grpc_keepalive_time_ms=30000
export RAY_grpc_keepalive_timeout_ms=60000

"$RAY" start \
    --head \
    --port=6379 \
    --node-ip-address="$NODE_IP" \
    --disable-usage-stats \
    --include-dashboard=false \
    --num-cpus=2 \
    --object-store-memory=268435456 \
    --plasma-directory=/tmp \
    2>&1

RAY_EXIT=$?
if [ $RAY_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: 'ray start' exited with code $RAY_EXIT"
    echo ""
    echo "Common WSL2 fixes:"
    echo "  1) Run again — first attempt sometimes fails after a clean boot"
    echo "  2) Free memory:  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
    echo "  3) If error mentions 'plasma': rm -rf /tmp/ray /tmp/plasma*"
    echo "  4) Reopen a fresh WSL terminal and retry"
    exit 1
fi

sleep 5

# Verify Ray GCS is actually up
if ! ss -tlnp 2>/dev/null | grep -q ':6379'; then
    echo "ERROR: Ray GCS did not bind to port 6379."
    exit 1
fi

# Verify raylet is not defunct
RAYLET_LIVE=$(ps aux 2>/dev/null | grep '[r]aylet' | grep -v defunct | wc -l)
if ps aux 2>/dev/null | grep -q '[r]aylet'; then
    if [ "$RAYLET_LIVE" -eq 0 ]; then
        echo "ERROR: Raylet started but immediately became defunct."
        echo "Try: export RAY_DISABLE_JEMALLOC=1 before running this script"
        exit 1
    fi
    echo "    Raylet: alive ($RAYLET_LIVE process)."
fi
echo "    Ray GCS: listening on :6379."

# ── Step 3: Start Ngrok ──────────────────────────────────────
echo ""
echo "[3/4] Starting Ngrok TCP tunnel → port 6379..."
nohup ngrok tcp 6379 --log=stdout > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!

# Wait up to 30 seconds for the tunnel to establish
TUNNEL=""
for i in $(seq 1 30); do
    sleep 1
    TUNNEL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
        | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('tunnels', [])
    print(t[0]['public_url'] if t else '')
except:
    print('')
" 2>/dev/null || echo "")
    if [ -n "$TUNNEL" ]; then
        break
    fi
    printf "    Waiting for tunnel... %d/30\r" "$i"
done

if [ -z "$TUNNEL" ]; then
    echo ""
    echo "ERROR: Ngrok tunnel did not start within 30 seconds."
    echo "Ngrok log:"
    cat /tmp/ngrok.log
    echo ""
    echo "Possible fix: ngrok config add-authtoken <YOUR_TOKEN>"
    exit 1
fi

NGROK_HOST=$(echo "$TUNNEL" | sed 's|tcp://||' | cut -d: -f1)
NGROK_PORT=$(echo "$TUNNEL" | sed 's|tcp://||' | cut -d: -f2)

# ── Step 4: Verify Ngrok → Ray path ──────────────────────────
echo ""
echo "[4/4] Verifying Ngrok → Ray connectivity..."
if nc -z -w5 "$NGROK_HOST" "$NGROK_PORT" 2>/dev/null; then
    echo "    Tunnel reachable from external address. "
else
    echo "    WARNING: nc test inconclusive. Tunnel may still be warming up."
    echo "    Proceed anyway — Node B should try connecting."
fi

# ── Print the connection command for Node B ───────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  NODE A IS READY                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Send this EXACT block to your partner (Node B):            ║"
echo "║                                                              ║"
echo "║    export LD_PRELOAD=''                                      ║"
echo "║    export RAY_DISABLE_JEMALLOC=1                             ║"
echo "║    export CUDA_VISIBLE_DEVICES=0                             ║"
echo "║    export LD_PRELOAD=''                                       ║"
echo "║    export RAY_DISABLE_JEMALLOC=1                              ║"
echo "║    export CUDA_VISIBLE_DEVICES=0                              ║"
printf "║    ray start --address=%s:%s \\\\\n" "$NGROK_HOST" "$NGROK_PORT"
echo "║          --num-gpus=1 --num-cpus=4                            ║"
echo "║                                                              ║"
printf "║  Full address: %s\n" "$TUNNEL"
echo "║                                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  KEEP THIS TERMINAL OPEN while Node B is connected.         ║"
echo "║  Closing it will shut Ray down.                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "After partner connects, verify with:"
echo "  source $VENV/bin/activate && python $PROJECT/verify_cluster.py"
echo ""
echo "Watching Ray health every 30s (Ctrl+C stops watching — Ray stays running):"
echo "──────────────────────────────────────────────────────────────"

# Keep terminal alive and show live Ray + Ngrok health every 30s
while true; do
    sleep 30
    NOW=$(date '+%H:%M:%S')
    if ss -tlnp 2>/dev/null | grep -q ':6379'; then
        RAY_OK="Ray GCS: UP"
    else
        RAY_OK="Ray GCS: DOWN  <-- RESTART NEEDED"
    fi
    CURRENT_TUNNEL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
        | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('tunnels', [])
    print(t[0]['public_url'] if t else 'NO_TUNNEL')
except:
    print('NO_TUNNEL')
" 2>/dev/null || echo "NO_TUNNEL")
    # Count alive nodes. os._exit(0) skips ray.shutdown() atexit so Node B
    # stays connected. sys.stdout.flush() is required before os._exit()
    # because os._exit bypasses normal Python buffer flushing.
    NODES=$("$VENV/bin/python3" -c "
import ray, os, sys
try:
    ray.init(address='$NODE_IP:6379', ignore_reinit_error=True, logging_level='ERROR')
    n = len([x for x in ray.nodes() if x.get('Alive')])
    sys.stdout.write(str(n) + '\n')
    sys.stdout.flush()
except Exception:
    sys.stdout.write('?\n')
    sys.stdout.flush()
finally:
    os._exit(0)
" 2>/dev/null)
    echo "[$NOW] $RAY_OK | Ngrok: $CURRENT_TUNNEL | Alive nodes: $NODES"
done
