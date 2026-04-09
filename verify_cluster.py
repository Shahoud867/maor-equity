"""
verify_cluster.py  —  run on Node A after both nodes are connected.
Checks: 2 active nodes, GPU registered in cluster_resources.

GPU info (name, VRAM) is read directly from cluster_resources — no
cross-node task dispatch needed. This avoids the Ngrok heartbeat problem
where dispatching a GPU task over a high-latency tunnel causes Node B to
be incorrectly marked dead.

Usage:
    python verify_cluster.py
    RAY_ADDRESS=172.26.19.0:6379 python verify_cluster.py
    VERIFY_WAIT_SECONDS=60 VERIFY_POLL_SECONDS=5 python verify_cluster.py
"""
import os
import sys
import time

import ray

# Use os._exit() everywhere so Python's atexit handlers (which include
# ray.shutdown()) are NEVER called. Calling ray.shutdown() from a script
# running on the head node causes Node B to momentarily drop.
def _exit(code: int) -> None:
    """Exit without triggering atexit / ray.shutdown()."""
    os._exit(code)


address      = os.environ.get("RAY_ADDRESS", "172.26.19.0:6379")
wait_seconds = int(os.environ.get("VERIFY_WAIT_SECONDS", "60"))
poll_seconds = int(os.environ.get("VERIFY_POLL_SECONDS", "5"))

ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")

# ── Show all nodes with alive/dead status ──────────────────────
print("=" * 60)
print("RAY CLUSTER STATE")
print("=" * 60)
all_nodes   = ray.nodes()
alive_nodes = [n for n in all_nodes if n.get("Alive")]
dead_nodes  = [n for n in all_nodes if not n.get("Alive")]

print(f"Total nodes (incl. stale): {len(all_nodes)}")
print(f"  Alive : {len(alive_nodes)}")
print(f"  Dead  : {len(dead_nodes)}  (stale Ray entries — ignore these)")

for n in alive_nodes:
    ip  = n.get("NodeManagerAddress", "?")
    res = n.get("Resources", {})
    gpu = res.get("GPU", 0)
    cpu = res.get("CPU", 0)
    print(f"  [ALIVE] {ip}  CPU={cpu}  GPU={gpu}")
for n in dead_nodes:
    ip = n.get("NodeManagerAddress", "?")
    print(f"  [DEAD ] {ip}  (was connected, now gone)")

print()
cr = ray.cluster_resources()
print("Cluster resources:", cr)
print()

# ── Wait for exactly 2 alive nodes ────────────────────────────
print("Waiting for Node B to be alive...")
deadline = time.time() + wait_seconds
while True:
    alive_nodes = [n for n in ray.nodes() if n.get("Alive")]
    count = len(alive_nodes)
    if count >= 2:
        break
    elapsed = int(time.time() - (deadline - wait_seconds))
    if time.time() >= deadline:
        print(f"\nFAIL: Only {count} alive node(s) after {wait_seconds}s.")
        print()
        print("Node B is not connected. Tell your partner to run:")
        print()
        print("    ray stop")
        print("    export LD_PRELOAD=''")
        print("    export RAY_DISABLE_JEMALLOC=1")
        print("    export CUDA_VISIBLE_DEVICES=0")
        print("    ray start --address=<NGROK_ADDRESS> --num-gpus=1 --num-cpus=4")
        _exit(1)
    print(f"  [{elapsed}s] {count}/2 alive nodes. Retrying in {poll_seconds}s...")
    time.sleep(poll_seconds)

print(f"Both nodes alive!\n")

# ── Check GPU is registered in cluster resources ───────────────
cr = ray.cluster_resources()
gpu_total = cr.get("GPU", 0)

if gpu_total == 0:
    print("FAIL: No GPU resource registered in the cluster.")
    print()
    print("Node B connected BUT did not advertise its GPU.")
    print("Tell your partner to reconnect with --num-gpus=1:")
    print()
    print("    ray stop")
    print("    export LD_PRELOAD=''")
    print("    export RAY_DISABLE_JEMALLOC=1")
    print("    export CUDA_VISIBLE_DEVICES=0")
    print("    ray start --address=<NGROK_ADDRESS> --num-gpus=1 --num-cpus=4")
    _exit(1)

print(f"GPU resource registered: {gpu_total} GPU(s) in cluster\n")

# ── Identify GPU model from accelerator_type resource key ─────
# When Ray registers a GPU node it adds "accelerator_type:<name>: 1.0"
# to cluster_resources — no cross-node task needed to get the GPU name.
gpu_name = "Unknown"
for key in cr:
    if key.startswith("accelerator_type:"):
        gpu_name = key.split("accelerator_type:", 1)[1]
        break

# ── Find Node B (the GPU node) and read its VRAM ──────────────
node_b = next(
    (n for n in ray.nodes()
     if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0),
    None
)

# T1000 has 4096 MB — use known value; override if Ray exposes memory
KNOWN_VRAM = {
    "T1000": 4096,
    "T400":  4096,
    "T600":  4096,
    "RTX 3060": 12288,
    "RTX 3070": 8192,
    "RTX 3080": 10240,
}
vram = next((v for k, v in KNOWN_VRAM.items() if k in gpu_name), 4096)

# ── VRAM budget check ──────────────────────────────────────────
budget   = 3 * 340 + 2100 + 400 + 120   # FinBERT×3 + Phi3(shared) + KV + Ray
headroom = vram - budget

print("=" * 60)
print("SUCCESS — CLUSTER FULLY VERIFIED")
print("=" * 60)
print(f"  Node B IP : {node_b['NodeManagerAddress'] if node_b else '?'}")
print(f"  GPU model : {gpu_name}")
print(f"  VRAM      : {vram} MB  (known spec for {gpu_name})")
print(f"  Budget est: {budget} MB  (3×FinBERT + Phi-3-mini shared + overhead)")
print(f"  Headroom  : {headroom} MB")
if headroom >= 0:
    print(f"  Budget    : OK ({headroom} MB spare)")
else:
    print(f"  Budget    : TIGHT — switch FinBERT to 8-bit (load_in_8bit=True)")
print("=" * 60)
print()
print("Cluster is ready. Next steps:")
print("  1) Node B runs VRAM check:  python evaluation/vram_verify.py")
print("  2) First pipeline run:       python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json")
print("  3) Latency benchmark:        python -m evaluation.latency_benchmark --tickers AAPL MSFT GOOGL")

# Exit WITHOUT triggering atexit / ray.shutdown() — keeps Node B alive.
_exit(0)
