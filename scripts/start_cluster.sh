#!/usr/bin/env bash
# start_cluster.sh - run inside WSL on Node A to bring up Ray + Ngrok
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_DIR/venv/bin/activate"

echo "=== Starting Ray head node ==="
# Prevent Ray from inheriting a broken LD_PRELOAD split by spaces in this path.
export LD_PRELOAD=""
RAY_DISABLE_JEMALLOC=1 ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats

echo ""
echo "=== Starting Ngrok TCP tunnel ==="
echo "Make sure you have run:  ngrok config add-authtoken YOUR_TOKEN"
echo ""

# If a static domain is set in environment, use it
if [ -n "$NGROK_STATIC_DOMAIN" ]; then
    ngrok tcp --domain="$NGROK_STATIC_DOMAIN" 6379 &
else
    ngrok tcp 6379 &
fi

echo ""
echo "Ngrok started in background. Check http://localhost:4040 for the tunnel URL."
echo "Ray Dashboard: http://localhost:8265"
echo ""
echo "Share the Ngrok TCP address with your partner (Node B)."
echo "Then verify the cluster:  python verify_cluster.py"