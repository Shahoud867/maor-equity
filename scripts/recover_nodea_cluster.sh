#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ ! -d "venv" ]]; then
  echo "ERROR: venv not found at $PROJECT_DIR/venv"
  exit 1
fi

source "venv/bin/activate"

echo "[1/5] Cleaning stale Ray/ngrok processes..."
if ! timeout 20s ray stop --force >/dev/null 2>&1; then
  pkill -f gcs_server >/dev/null 2>&1 || true
  pkill -f raylet >/dev/null 2>&1 || true
fi
pkill -f gcs_server >/dev/null 2>&1 || true
pkill -f raylet >/dev/null 2>&1 || true
pkill -f dashboard_agent >/dev/null 2>&1 || true
pkill -f runtime_env_agent >/dev/null 2>&1 || true
pkill -f dashboard.py >/dev/null 2>&1 || true
pkill -f monitor.py >/dev/null 2>&1 || true
pkill -f "ngrok tcp 6379" >/dev/null 2>&1 || true

echo "[2/5] Starting Ray head (dashboard disabled for stability)..."
export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export RAY_raylet_start_wait_time_s=180
if ! timeout 300s env LD_PRELOAD= RAY_DISABLE_JEMALLOC=1 RAY_raylet_start_wait_time_s=180 \
  ray start --head --port=6379 --include-dashboard=false --disable-usage-stats \
  >/tmp/nodea_ray_head.log 2>&1; then
  echo "RAY_START_CMD_FAILED"
  echo "--- /tmp/nodea_ray_head.log ---"
  sed -n '1,200p' /tmp/nodea_ray_head.log || true
  exit 1
fi

sleep 8

if ss -ltn | grep -q ":6379"; then
  echo "RAY_PORT_LISTENING=YES"
else
  echo "RAY_PORT_LISTENING=NO"
  echo "--- /tmp/nodea_ray_head.log ---"
  sed -n '1,200p' /tmp/nodea_ray_head.log || true
  exit 1
fi

if pgrep -af raylet >/dev/null 2>&1; then
  echo "RAYLET_PROCESS=YES"
else
  echo "RAYLET_PROCESS=NO"
  echo "--- /tmp/nodea_ray_head.log ---"
  sed -n '1,200p' /tmp/nodea_ray_head.log || true
  exit 1
fi

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ERROR: ngrok not found in PATH inside WSL"
  exit 1
fi

echo "[3/5] Starting ngrok tunnel for tcp://localhost:6379 ..."
nohup ngrok tcp 6379 >/tmp/nodea_ngrok.log 2>&1 < /dev/null &
sleep 4

echo "[4/5] Fetching public ngrok endpoint..."
ENDPOINT="$(python3 - <<'PY'
import json
import urllib.request

try:
    data = json.load(urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=5))
    urls = [t.get("public_url", "") for t in data.get("tunnels", []) if t.get("proto") == "tcp"]
    print(urls[0] if urls else "")
except Exception:
    print("")
PY
)"

if [[ -z "$ENDPOINT" ]]; then
  echo "NGROK_ENDPOINT=NOT_FOUND"
  echo "--- /tmp/nodea_ngrok.log ---"
  sed -n '1,200p' /tmp/nodea_ngrok.log || true
  exit 1
fi

HOST_PORT="${ENDPOINT#tcp://}"

echo "[5/5] Ready"
echo "NGROK_ENDPOINT=$HOST_PORT"
echo "NODE_B_RUN: export LD_PRELOAD=''; export RAY_DISABLE_JEMALLOC=1; ray stop --force || true; ray start --address=$HOST_PORT"
