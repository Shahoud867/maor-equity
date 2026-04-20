"""
diagnose_cluster.py - Run on Node A to see full cluster state.
Usage: python scripts/diagnose_cluster.py
"""
import ray
import sys

ray.init(address="172.26.19.0:6379", ignore_reinit_error=True, logging_level="ERROR")

nodes = ray.nodes()
alive = [n for n in nodes if n.get("Alive")]
dead  = [n for n in nodes if not n.get("Alive")]

print("=" * 60)
print("FULL CLUSTER DIAGNOSIS")
print("=" * 60)

print(f"\nTotal nodes seen by Ray: {len(nodes)}")
print(f"  Alive: {len(alive)}")
print(f"  Dead / disconnected: {len(dead)}")

print("\n--- ALIVE NODES ---")
for n in alive:
    ip  = n.get("NodeManagerAddress", "unknown")
    res = n.get("Resources", {})
    gpu = res.get("GPU", 0)
    cpu = res.get("CPU", 0)
    mem = round(res.get("memory", 0) / 1e9, 1)
    print(f"  IP: {ip}")
    print(f"    CPU: {cpu}  GPU: {gpu}  Memory: {mem} GB")
    print(f"    Full resources: {res}")
    print()

if dead:
    print("--- DEAD / STALE NODES ---")
    for n in dead:
        ip = n.get("NodeManagerAddress", "unknown")
        print(f"  IP: {ip}  (disconnected - stale entry)")

print("\n--- CLUSTER RESOURCES (total) ---")
cr = ray.cluster_resources()
print(f"  {cr}")

print("\n--- DIAGNOSIS ---")
has_gpu = cr.get("GPU", 0) > 0

if len(alive) < 2:
    print("PROBLEM: Only 1 alive node. Node B is NOT connected.")
    print("  Fix: On Node B, re-run the ray start --address=... command.")
elif not has_gpu:
    print("PROBLEM: 2 nodes alive, but NO GPU resource registered.")
    print("  Cause: Node B connected without --num-gpus=1, OR")
    print("         CUDA is not visible to Ray on Node B.")
    print()
    print("  FIX for Node B - run these commands on Node B:")
    print("    ray stop")
    print("    export LD_PRELOAD=''")
    print("    export RAY_DISABLE_JEMALLOC=1")
    print("    export CUDA_VISIBLE_DEVICES=0")
    print("    ray start --address=<TAILSCALE_IP>:6379 --num-gpus=1 --num-cpus=4")
else:
    print("OK: 2 nodes alive, GPU registered.")
    print(f"  GPU count: {cr.get('GPU', 0)}")

print("=" * 60)
ray.shutdown()
