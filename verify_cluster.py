"""
verify_cluster.py  —  run on Node A after both nodes are connected.

Reads cluster state from /tmp/ray_cluster_state.json written by the
nodea_start.sh monitor every 30 seconds. This script NEVER connects to
Ray — eliminating the driver-GC cascade that was dropping Node B every
time this script ran.

Usage:
    python verify_cluster.py
"""
import json
import os
import sys
import time

STATE_FILE = "/tmp/ray_cluster_state.json"
MAX_AGE_S  = 120  # accept state files up to 120s old (monitor writes every 30s)


def _load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        print("FAIL: State file not found.")
        print(f"  Expected: {STATE_FILE}")
        print("  Is nodea_start.sh running? Wait for one monitor cycle (30s) then retry.")
        sys.exit(1)

    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception as e:
        print(f"FAIL: Could not read state file: {e}")
        sys.exit(1)

    age = time.time() - state.get("ts", 0)
    if age > MAX_AGE_S:
        print(f"FAIL: State file is {age:.0f}s old (max {MAX_AGE_S}s).")
        print("  nodea_start.sh monitor may have stopped.")
        sys.exit(1)

    return state


# ── Load state ──────────────────────────────────────────────────────────
state = _load_state()
age   = time.time() - state["ts"]

alive    = state.get("alive", [])       # list of {ip, cpu, gpu}
dead_ips = state.get("dead_ips", [])
cr       = state.get("cr", {})
cr_keys  = state.get("cr_keys", [])

# ── Print cluster state ─────────────────────────────────────────────────
print("=" * 60)
print(f"RAY CLUSTER STATE  (snapshot age: {age:.0f}s)")
print("=" * 60)
print(f"Total nodes (incl. stale): {len(alive) + len(dead_ips)}")
print(f"  Alive : {len(alive)}")
print(f"  Dead  : {len(dead_ips)}  (stale entries — ignore)")

for n in alive:
    print(f"  [ALIVE] {n['ip']}  CPU={n['cpu']}  GPU={n['gpu']}")
for ip in dead_ips:
    print(f"  [DEAD ] {ip}")

print()
print("Cluster resources:", cr)
print()

# ── Check 2 alive nodes ─────────────────────────────────────────────────
if len(alive) < 2:
    print(f"FAIL: Only {len(alive)} alive node(s) in latest snapshot.")
    print()
    print("If Node B just connected, wait 30s for the next monitor cycle then re-run.")
    print("Node B connection command:")
    print("  export LD_PRELOAD=''")
    print("  export RAY_DISABLE_JEMALLOC=1")
    print("  export CUDA_VISIBLE_DEVICES=0")
    print("  ray start --address=<TAILSCALE_IP>:6379 --num-gpus=1 --num-cpus=4")
    sys.exit(1)

print("Both nodes alive!")
print()

# ── Check GPU ───────────────────────────────────────────────────────────
gpu_total = cr.get("GPU", 0)
if gpu_total == 0:
    print("FAIL: No GPU in cluster resources.")
    print("Node B must reconnect with --num-gpus=1")
    sys.exit(1)

print(f"GPU resource registered: {gpu_total} GPU(s)\n")

# ── GPU model from accelerator_type key ────────────────────────────────
gpu_name = "Unknown"
for key in cr_keys:
    if key.startswith("accelerator_type:"):
        gpu_name = key.split("accelerator_type:", 1)[1]
        break

# ── VRAM budget ─────────────────────────────────────────────────────────
KNOWN_VRAM = {
    "T1000": 4096, "T400": 4096, "T600": 4096,
    "RTX 3060": 12288, "RTX 3070": 8192,
    "RTX 3080": 10240, "RTX 3090": 24576, "A100": 40960,
}
vram     = next((v for k, v in KNOWN_VRAM.items() if k in gpu_name), 4096)
budget   = 3 * 340 + 2100 + 400 + 120
headroom = vram - budget

node_b = next((n for n in alive if n["gpu"] > 0), None)
node_b_ip = node_b["ip"] if node_b else "?"

# ── Print success ───────────────────────────────────────────────────────
print("=" * 60)
print("SUCCESS — CLUSTER FULLY VERIFIED")
print("=" * 60)
print(f"  Node B IP : {node_b_ip}")
print(f"  GPU model : {gpu_name}")
print(f"  VRAM      : {vram} MB")
print(f"  Budget    : {budget} MB  (3×FinBERT + Phi-3-mini shared + overhead)")
print(f"  Headroom  : {headroom} MB  ({'OK' if headroom >= 0 else 'TIGHT'})")
print("=" * 60)
print()
print("Next steps:")
print("  1) Node B: python evaluation/vram_verify.py")
print("  2) Node A: python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json")
print("  3) Node A: python -m evaluation.latency_benchmark --tickers AAPL MSFT GOOGL")
