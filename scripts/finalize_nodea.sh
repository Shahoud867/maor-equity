#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
source venv/bin/activate

echo "=== Dependency verification ==="
python3 - <<'PY'
import ray
import datasets
import sec_edgar_downloader
print(f"ray={ray.__version__}")
print(f"datasets={datasets.__version__}")
print("sec-edgar-downloader=OK")
PY

echo "=== Editable install: edgar-crawler ==="
pip install -e edgar-crawler

echo "=== ECTSum dataset check/download ==="
python3 - <<'PY'
from pathlib import Path
from datasets import load_dataset, DownloadMode

out = Path('data/ectsum/ectsum_test.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists() and out.stat().st_size > 0:
    print(f"ECTSum already present: {out} ({out.stat().st_size} bytes)")
else:
    ds = load_dataset(
        'mrSoul7766/ECTSum',
        trust_remote_code=True,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    ds['test'].to_json(str(out))
    print(f"ECTSum downloaded: {len(ds['test'])} samples -> {out}")
PY

echo "=== Ray head check/start ==="
ray stop --force >/dev/null 2>&1 || true
# Prevent Ray from inheriting a broken LD_PRELOAD split by spaces in this path.
export LD_PRELOAD=""
RAY_DISABLE_JEMALLOC=1 ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats

echo "=== Final verification ==="
python3 --version
python3 -c 'import ray; print("Ray:", ray.__version__)'
ngrok --version
ls -1
