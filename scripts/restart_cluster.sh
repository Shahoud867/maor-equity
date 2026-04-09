#!/usr/bin/env bash
# restart_cluster.sh
# Fully stops, cleans, and restarts Ray head + Ngrok on Node A (WSL).
# Run this whenever the connection to Node B breaks.
# Usage: bash scripts/restart_cluster.sh

set -euo pipefail

VENV="/mnt/c/Users/shaho/OneDrive - FAST National University/Attachments/@Fast/Semester 6/PDC + NLP/maor-equity/venv"
RAY="$VENV/bin/ray"
PYTHON="$VENV/bin/python3"

echo ""
echo "========================================================"
echo "  STEP 1 — Kill everything (Ray + Ngrok)"
echo "========================================================"
export LD_PRELOAD=""
"$RAY" stop --force 2>/dev/null || true
sleep 1
pkill -9 -f gcs_server  2>/dev/null || true
pkill -9 -f raylet      2>/dev/null || true
pkill -9 -f plasma_store 2>/dev/null || true
pkill -9 -f ray/core    2>/dev/null || true
pkill -9 -f ngrok       2>/dev/null || true
sleep 3

# Confirm port 6379 is free
if ss -tlnp | grep -q 6379; then
    echo "ERROR: Port 6379 still in use after kill. Aborting."
    ss -tlnp | grep 6379
    exit 1
fi
echo "Port 6379 is FREE — clean state confirmed."

echo ""
echo "========================================================"
echo "  STEP 2 — Start Ray head node"
echo "========================================================"
export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1

# Get WSL IP (the actual interface Ray binds to)
NODE_A_IP=$(hostname -I | awk '{print $1}')
echo "Node A WSL IP: $NODE_A_IP"

"$RAY" start \
    --head \
    --port=6379 \
    --node-ip-address="$NODE_A_IP" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --disable-usage-stats \
    --num-cpus=2 \
    --object-store-memory=536870912 \
    --plasma-directory=/tmp \
    2>&1

sleep 4

# Verify Ray GCS is actually listening
if ! ss -tlnp | grep -q 6379; then
    echo "ERROR: Ray GCS did not bind to port 6379. Check logs above."
    exit 1
fi
echo "Ray GCS is LISTENING on port 6379."

# Verify Ray worker port (let raylet register)
sleep 3
if "$RAY" status --address="$NODE_A_IP:6379" 2>&1 | grep -q "Node status"; then
    echo "Ray cluster status: OK"
else
    echo "Ray status check (informational — may need a moment to stabilize):"
    "$RAY" status --address="$NODE_A_IP:6379" 2>&1 | head -10 || true
fi

echo ""
echo "========================================================"
echo "  STEP 3 — Start Ngrok TCP tunnel on port 6379"
echo "========================================================"
nohup ngrok tcp 6379 --log=stdout > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!
echo "Ngrok PID: $NGROK_PID"

# Wait up to 15 seconds for Ngrok to establish tunnel
echo "Waiting for Ngrok tunnel..."
for i in $(seq 1 15); do
    sleep 1
    TUNNEL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | \
             python3 -c "import sys,json; d=json.load(sys.stdin); \
             t=d.get('tunnels',[]); \
             print(t[0]['public_url'] if t else '')" 2>/dev/null || echo "")
    if [ -n "$TUNNEL" ]; then
        break
    fi
    echo "  Attempt $i/15 — waiting..."
done

if [ -z "$TUNNEL" ]; then
    echo "ERROR: Ngrok tunnel did not start within 15 seconds."
    echo "Check /tmp/ngrok.log:"
    cat /tmp/ngrok.log
    exit 1
fi

# Parse host and port from tcp://host:port
NGROK_HOST=$(echo "$TUNNEL" | sed 's|tcp://||' | cut -d: -f1)
NGROK_PORT=$(echo "$TUNNEL" | sed 's|tcp://||' | cut -d: -f2)

echo ""
echo "========================================================"
echo "  STEP 4 — Verify Ngrok can reach Ray"
echo "========================================================"
# Test that the tunnel actually forwards to Ray
if nc -z -w5 "$NGROK_HOST" "$NGROK_PORT" 2>/dev/null; then
    echo "Tunnel connectivity: OK (Ngrok → Ray port 6379 reachable)"
else
    echo "WARNING: nc test failed — tunnel may still be warming up."
    echo "Proceed anyway and tell Node B to try connecting."
fi

echo ""
echo "========================================================"
echo ""
echo "  NODE A IS READY"
echo ""
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │  Send this EXACT command to your partner:       │"
echo "  │                                                 │"
echo "  │  export LD_PRELOAD=''                           │"
echo "  │  export RAY_DISABLE_JEMALLOC=1                  │"
printf "  │  ray start --address=%s:%s  │\n" "$NGROK_HOST" "$NGROK_PORT"
echo "  │                                                 │"
echo "  │  Full Ngrok address: $TUNNEL"
echo "  └─────────────────────────────────────────────────┘"
echo ""
echo "  Ray Dashboard:    http://localhost:8265"
echo "  Ngrok Inspector:  http://localhost:4040"
echo "  Ray address:      $NODE_A_IP:6379"
echo ""
echo "  After partner connects, verify with:"
echo "    python verify_cluster.py"
echo "========================================================"
