"""
verify_cluster.py  —  run on Node A after both nodes are connected.
Checks: 2 active nodes, GPU visible on Node B, cross-node GPU task executes.
"""
import ray, sys

ray.init(address="auto")
print("=== Ray Cluster Resources ===")
res   = ray.cluster_resources()
print(res, "\n")

nodes  = [n for n in ray.nodes() if n["Alive"]]
print(f"Active nodes: {len(nodes)}")
if len(nodes) < 2:
    print("WARNING: Only 1 node. Node B may not be connected yet.")
    sys.exit(1)
print("Both nodes alive!\n")

@ray.remote(num_gpus=0.1)
def check_gpu():
    import torch
    ok = torch.cuda.is_available()
    return {"gpu_available": ok,
            "gpu_name": torch.cuda.get_device_name(0) if ok else "N/A",
            "vram_mb":  torch.cuda.get_device_properties(0).total_memory // 1024**2 if ok else 0}

print("Dispatching cross-node GPU task ...")
r = ray.get(check_gpu.remote())
print(f"Result: {r}")

if r["gpu_available"]:
    print(f"\nSUCCESS: {r['gpu_name']}  {r['vram_mb']} MB VRAM")
    budget = 3*340 + 1800 + 200
    print(f"VRAM budget:  estimated {budget} MB  /  available {r['vram_mb']} MB")
    if budget <= r["vram_mb"]:
        print("Budget: OK")
    else:
        print("Budget: TIGHT — consider 8-bit fallback for FinBERT")
else:
    print("GPU NOT available in cross-node task — check CUDA on Node B")
    sys.exit(1)