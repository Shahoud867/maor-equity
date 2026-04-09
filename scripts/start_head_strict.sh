#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
source venv/bin/activate

# Clean stale processes
if ! timeout 20s ray stop --force >/dev/null 2>&1; then
  true
fi
pkill -f gcs_server >/dev/null 2>&1 || true
pkill -f raylet >/dev/null 2>&1 || true
pkill -f dashboard_agent >/dev/null 2>&1 || true
pkill -f runtime_env_agent >/dev/null 2>&1 || true
pkill -f dashboard.py >/dev/null 2>&1 || true
pkill -f monitor.py >/dev/null 2>&1 || true

export LD_PRELOAD=""
export RAY_DISABLE_JEMALLOC=1
export RAY_raylet_start_wait_time_s=180

# Start head and wait for completion output
set +e
timeout 300s ray start --head --include-dashboard=false --port=6379 --disable-usage-stats \
  --object-store-memory=536870912 --plasma-directory=/tmp
rc=$?
set -e

echo "RAY_START_EXIT=$rc"

echo "PORT_STATUS"
ss -ltn | grep ':6379' || true

echo "PROCS_STATUS"
pgrep -af gcs_server || true
pgrep -af raylet || true
pgrep -af dashboard.py || true
pgrep -af dashboard_agent || true
pgrep -af runtime_env_agent || true

# Sanity: raylet must exist for usable head node
if ! pgrep -af raylet >/dev/null 2>&1; then
  echo "RAYLET_MISSING=YES"
  latest=$(ls -dt /tmp/ray/session_* | head -n 1)
  echo "LATEST_SESSION=$latest"
  ls -1 "$latest"/logs | head -n 40 || true
  echo "---raylet.err---"
  sed -n '1,140p' "$latest"/logs/raylet.err || true
  echo "---dashboard_agent.log---"
  sed -n '1,140p' "$latest"/logs/dashboard_agent.log || true
  exit 1
fi

echo "RAYLET_MISSING=NO"
