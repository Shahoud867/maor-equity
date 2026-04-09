#!/usr/bin/env python3
"""Live consistency watcher for Ray node membership and GPU reachability.

This script is read-only against the running cluster.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime

import ray


def get_ngrok_tcp_url(timeout: float = 2.0) -> str:
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=timeout) as resp:
            payload = json.load(resp)
        tunnels = payload.get("tunnels", [])
        for tunnel in tunnels:
            if tunnel.get("proto") == "tcp" and tunnel.get("public_url"):
                return str(tunnel["public_url"])
    except Exception:
        pass
    return "N/A"


@ray.remote(num_gpus=0.1)
def gpu_probe() -> dict:
    import torch

    ok = torch.cuda.is_available()
    return {
        "gpu_available": ok,
        "gpu_name": torch.cuda.get_device_name(0) if ok else "N/A",
        "vram_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2 if ok else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch Ray node/GPU consistency")
    parser.add_argument("--address", default="172.26.19.0:6379", help="Ray GCS address")
    parser.add_argument("--interval", type=int, default=10, help="Seconds between samples")
    parser.add_argument("--samples", type=int, default=12, help="Number of samples")
    args = parser.parse_args()

    print("Watching synchronized cluster state...")
    print("Columns: timestamp | ngrok | alive_nodes | node_ips | gpu_probe")

    for _ in range(args.samples):
        ts = datetime.now().strftime("%H:%M:%S")
        ngrok_url = get_ngrok_tcp_url()

        try:
            ray.init(address=args.address, ignore_reinit_error=True, logging_level="ERROR")
            nodes = [n for n in ray.nodes() if n.get("Alive")]
            ips = [n.get("NodeManagerAddress", "?") for n in nodes]

            gpu_status = "skipped"
            if len(nodes) >= 2:
                try:
                    result = ray.get(gpu_probe.remote(), timeout=20)
                    gpu_status = (
                        f"ok={result['gpu_available']} name={result['gpu_name']} "
                        f"vram_mb={result['vram_mb']}"
                    )
                except Exception as exc:
                    gpu_status = f"error={type(exc).__name__}:{exc}"

            print(
                f"[{ts}] ngrok={ngrok_url} alive_nodes={len(nodes)} "
                f"node_ips={ips} gpu_probe={gpu_status}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[{ts}] ngrok={ngrok_url} ray_error={type(exc).__name__}:{exc}",
                flush=True,
            )
        finally:
            try:
                ray.shutdown()
            except Exception:
                pass

        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
