#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
source venv/bin/activate

echo "PYTHON_VERSION=$(python3 --version)"
echo "RAY_VERSION=$(python3 -c 'import ray; print(ray.__version__)')"
echo "NGROK_VERSION=$(ngrok --version)"
echo "RAY_STATUS_START"
if timeout 20s env RAY_DISABLE_JEMALLOC=1 ray status | sed -n '1,120p'; then
	:
else
	echo "RAY_STATUS_TIMEOUT_OR_ERROR"
	ray stop --force >/dev/null 2>&1 || true
fi
echo "RAY_STATUS_END"
