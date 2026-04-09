#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
source venv/bin/activate

if ! timeout 20s ray stop --force >/dev/null 2>&1; then
  pkill -f 'gcs_server|raylet|dashboard_agent|runtime_env_agent' >/dev/null 2>&1 || true
fi
export LD_PRELOAD=""
LD_PRELOAD= RAY_DISABLE_JEMALLOC=1 ray start --head --disable-usage-stats --port=6379 --dashboard-host=0.0.0.0 >/tmp/ray_start_nodea.log 2>&1 || {
  echo "RAY_START_FAILED"
  sed -n '1,200p' /tmp/ray_start_nodea.log
  exit 1
}

echo "RAY_START_OK"
ps -ef | grep -E '[g]cs_server|[r]aylet' || true

echo "RAY_INIT_CHECK_START"
if env RAY_DISABLE_JEMALLOC=1 timeout 30s python3 - <<'PY'
import ray
ray.init(address='auto')
print(ray.cluster_resources())
PY
then
  echo "RAY_INIT_CHECK_OK"
else
  echo "RAY_INIT_CHECK_TIMEOUT_OR_ERROR"
fi
