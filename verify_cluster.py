"""
verify_cluster.py -- run on Node A after both nodes are connected.

Connects directly to the running Ray cluster and verifies:
  - At least 2 alive nodes
  - GPU registered (from Node B)
  - Prints resource summary and VRAM budget

Usage:
    python verify_cluster.py
    python verify_cluster.py --address 100.x.x.x:6379
"""
import argparse
import os
import sys

os.environ.setdefault("RAY_DISABLE_JEMALLOC", "1")
os.environ.setdefault("LD_PRELOAD", "")

import ray

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--address", default="auto", help="Ray head address (default: auto)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------
print("=" * 60)
print("RAY CLUSTER VERIFICATION")
print("=" * 60)
print(f"  Connecting to Ray ({args.address})...")

try:
    ray.init(address=args.address, ignore_reinit_error=True, logging_level="ERROR")
except Exception as e:
    print(f"\nFAIL: Could not connect to Ray: {e}")
    print("\nSuggested fixes:")
    print("  1. Ensure nodeA.sh WSL window is open and shows Ray:UP")
    print("  2. Run with explicit address: python verify_cluster.py --address 100.x.x.x:6379")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Node status
# ---------------------------------------------------------------------------
try:
    all_nodes = ray.nodes()
except Exception as e:
    print(f"\nFAIL: Could not query nodes: {e}")
    sys.exit(1)

alive    = [n for n in all_nodes if n.get("Alive")]
dead     = [n for n in all_nodes if not n.get("Alive")]
cr       = dict(ray.cluster_resources())

print(f"  Connected OK.\n")
print(f"  Total nodes  : {len(all_nodes)}")
print(f"  Alive        : {len(alive)}")
print(f"  Dead (stale) : {len(dead)}")
print()

for n in alive:
    ip  = n.get("NodeManagerAddress", "?")
    res = n.get("Resources", {})
    cpu = res.get("CPU", 0)
    gpu = res.get("GPU", 0)
    print(f"  [ALIVE] {ip}  CPU={cpu}  GPU={gpu}")

for n in dead:
    ip = n.get("NodeManagerAddress", "?")
    print(f"  [DEAD]  {ip}  (stale entry -- ignore)")

print()

# ---------------------------------------------------------------------------
# Check: at least 2 alive nodes
# ---------------------------------------------------------------------------
if len(alive) < 2:
    print(f"FAIL: Only {len(alive)} alive node(s). Need at least 2.")
    print()
    print("If Node B just connected, wait 30s then re-run.")
    print("Node B manual join command (run in Node B WSL):")
    print("  export LD_PRELOAD=''")
    print("  export RAY_DISABLE_JEMALLOC=1")
    print("  export CUDA_VISIBLE_DEVICES=0")
    print("  ray start --address=<TAILSCALE_IP>:6379 --num-gpus=1 --num-cpus=4")
    ray.shutdown()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check: GPU registered
# ---------------------------------------------------------------------------
gpu_total = cr.get("GPU", 0)
if gpu_total == 0:
    print("FAIL: No GPU in cluster resources.")
    print("Node B must reconnect with --num-gpus=1")
    ray.shutdown()
    sys.exit(1)

print(f"  GPU resource registered: {gpu_total} GPU(s)")

# ---------------------------------------------------------------------------
# GPU model from accelerator_type resource key
# ---------------------------------------------------------------------------
gpu_name = "Unknown"
for key in cr.keys():
    if key.startswith("accelerator_type:"):
        gpu_name = key.split("accelerator_type:", 1)[1]
        break

# ---------------------------------------------------------------------------
# VRAM budget
# ---------------------------------------------------------------------------
KNOWN_VRAM = {
    "T1000": 4096, "T400": 4096, "T600": 4096,
    "RTX 3060": 12288, "RTX 3070": 8192,
    "RTX 3080": 10240, "RTX 3090": 24576, "A100": 40960,
}
vram     = next((v for k, v in KNOWN_VRAM.items() if k in gpu_name), 4096)
budget   = 3 * 340 + 2100 + 400 + 120
headroom = vram - budget

node_b    = next((n for n in alive if n.get("Resources", {}).get("GPU", 0) > 0), None)
node_b_ip = node_b.get("NodeManagerAddress", "?") if node_b else "?"

# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("SUCCESS -- CLUSTER FULLY VERIFIED")
print("=" * 60)
print(f"  Node B IP  : {node_b_ip}")
print(f"  GPU model  : {gpu_name}")
print(f"  VRAM       : {vram} MB")
print(f"  Budget     : {budget} MB  (3xFinBERT + Phi-3-mini shared + overhead)")
print(f"  Headroom   : {headroom} MB  ({'OK' if headroom >= 0 else 'TIGHT'})")
print("=" * 60)
print()
print("Next steps:")
print("  1) Node B: python evaluation/vram_verify.py")
print("  2) Node A: python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json")
print("  3) Node A: python -m evaluation.latency_benchmark --tickers AAPL MSFT GOOGL")

ray.shutdown()
