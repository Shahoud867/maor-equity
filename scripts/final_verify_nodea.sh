#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
source venv/bin/activate

echo "=== CLEAN RAY RESTART ==="
if ! timeout 20s ray stop --force >/dev/null 2>&1; then
  pkill -f 'gcs_server|raylet|dashboard_agent|runtime_env_agent|monitor.py|dashboard.py' >/dev/null 2>&1 || true
fi
rm -rf /tmp/ray/session_* /tmp/ray/session_latest /tmp/ray/ray_current_cluster || true

export LD_PRELOAD=""
if timeout 90s env LD_PRELOAD= RAY_DISABLE_JEMALLOC=1 ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats; then
  echo "RAY_HEAD_START:OK"
else
  echo "RAY_HEAD_START:FAIL"
  exit 1
fi

echo "=== PROCESS CHECK ==="
ps -ef | grep -E '[g]cs_server|[r]aylet' || true

if [ -f /tmp/ray/ray_current_cluster ]; then
  echo "RAY_CURRENT_CLUSTER:$(cat /tmp/ray/ray_current_cluster)"
else
  echo "RAY_CURRENT_CLUSTER:MISSING"
fi

echo "=== VERIFY_CLUSTER ==="
if python3 verify_cluster.py; then
  echo "VERIFY_CLUSTER:PASS"
else
  code=$?
  echo "VERIFY_CLUSTER:FAIL:$code"
fi
