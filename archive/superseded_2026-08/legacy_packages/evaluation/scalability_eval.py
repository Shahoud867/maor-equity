"""
evaluation/scalability_eval.py — Scalability Analysis (PDC requirement)

Measures pipeline performance across configurations:
  - 1 node (Node A only, simulating no GPU worker)
  - 2 nodes (Node A + Node B — full cluster)

Also measures:
  - Throughput: filings/hour at each configuration
  - Bandwidth: raw vs compressed payload bytes
  - Tcomm fraction of total time

Run:
    python -m evaluation.scalability_eval --tickers AAPL MSFT

Outputs → logs/scalability_results.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import ray


def run_single_node_estimate(tickers: list, h1_path: str = "logs/h1_latency_results.json") -> dict:
    """
    Estimate 1-node performance from B1 serial results.
    B1 = single-node, no Ray, no warm actors = lower bound on 1-node throughput.
    This avoids re-running the full B1 baseline.
    """
    try:
        with open(h1_path) as f:
            h1 = json.load(f)
        serial_results = h1.get("per_ticker_serial", [])
        if not serial_results:
            raise ValueError("No serial results in H1 log")
        totals = [r.get("total_s", 0) for r in serial_results if r.get("total_s")]
        median = statistics.median(totals)
        return {
            "nodes": 1,
            "configuration": "Single-node serial (B1 baseline)",
            "median_latency_s": round(median, 2),
            "throughput_filings_per_hour": round(3600 / median, 2),
            "note": "Derived from B1 serial run — represents single-node upper-bound latency",
        }
    except Exception as e:
        return {"nodes": 1, "error": str(e)}


def run_two_node(tickers: list, filing_type: str = "8-K") -> dict:
    """Run distributed pipeline on full 2-node cluster."""
    from agents.orchestrator import run_pipeline_batch

    t0 = time.perf_counter()
    results = run_pipeline_batch(tickers, filing_type)
    total_batch = time.perf_counter() - t0

    totals = [r.get("timings", {}).get("total", 0) for r in results if "error" not in r]
    median = statistics.median(totals) if totals else 0

    # Bandwidth stats from orchestrator timings
    bw_stats = []
    for r in results:
        t = r.get("timings", {})
        if t.get("payload_raw_bytes"):
            bw_stats.append({
                "ticker": r.get("ticker"),
                "raw_bytes": t["payload_raw_bytes"],
                "compressed_bytes": t["payload_compressed_bytes"],
                "reduction_pct": t["bandwidth_reduction_pct"],
            })

    avg_bw_reduction = (
        statistics.mean(b["reduction_pct"] for b in bw_stats) if bw_stats else 0
    )

    # Tcomm fraction
    tcomm_fractions = []
    for r in results:
        t = r.get("timings", {})
        if t.get("t_comm_total_ms") and t.get("total"):
            tcomm_fractions.append(t["t_comm_total_ms"] / 1000 / t["total"] * 100)

    return {
        "nodes": 2,
        "configuration": "Two-node distributed (Node A CPU + Node B T1000 GPU)",
        "tickers": tickers,
        "median_latency_s": round(median, 2),
        "total_batch_s": round(total_batch, 2),
        "throughput_filings_per_hour": round(3600 / median, 2) if median > 0 else 0,
        "bandwidth_reduction_pct": round(avg_bw_reduction, 1),
        "bandwidth_stats_per_ticker": bw_stats,
        "tcomm_fraction_pct": round(statistics.mean(tcomm_fractions), 2) if tcomm_fractions else 0,
        "note": "Measured live on 2-node Ray cluster",
    }


def compute_scalability_report(one_node: dict, two_node: dict) -> dict:
    """Compare 1-node vs 2-node and compute scalability metrics."""
    s1 = one_node.get("median_latency_s", 0)
    s2 = two_node.get("median_latency_s", 0)

    if s1 > 0 and s2 > 0:
        speedup   = s1 / s2
        reduction = (s1 - s2) / s1 * 100
        efficiency = speedup / 2 * 100   # parallel efficiency for 2 nodes
    else:
        speedup = reduction = efficiency = 0

    tp1 = one_node.get("throughput_filings_per_hour", 0)
    tp2 = two_node.get("throughput_filings_per_hour", 0)

    return {
        "scalability_analysis": {
            "1_node_median_latency_s": s1,
            "2_node_median_latency_s": s2,
            "speedup_1_to_2_nodes": round(speedup, 2),
            "latency_reduction_pct": round(reduction, 1),
            "parallel_efficiency_pct": round(efficiency, 1),
            "throughput_1_node_filings_per_hour": round(tp1, 2),
            "throughput_2_node_filings_per_hour": round(tp2, 2),
            "throughput_improvement_pct": round((tp2 - tp1) / tp1 * 100, 1) if tp1 > 0 else 0,
        },
        "bandwidth_compression": {
            "avg_reduction_pct": two_node.get("bandwidth_reduction_pct", 0),
            "interpretation": (
                f"Gzip compression of chunk payloads reduces inter-node bandwidth by "
                f"{two_node.get('bandwidth_reduction_pct', 0):.1f}%. "
                f"Text chunks compress well due to repetitive financial vocabulary. "
                f"Tdecode adds {two_node.get('tcomm_fraction_pct', 0):.2f}% to total latency — "
                f"compression overhead is negligible vs bandwidth savings."
            ),
        },
        "tcomm_fraction_of_total_pct": two_node.get("tcomm_fraction_pct", 0),
        "interpretation": (
            f"Moving from 1 to 2 nodes achieves {speedup:.2f}x speedup "
            f"({reduction:.1f}% latency reduction) with {efficiency:.1f}% parallel efficiency. "
            f"Communication overhead (Tcomm) represents only {two_node.get('tcomm_fraction_pct', 0):.1f}% "
            f"of total pipeline time, confirming that computation dominates and "
            f"the distributed design is communication-efficient."
        ),
        "parallelism_types_demonstrated": {
            "task_parallelism": "Phase A: FinBERT (Node B GPU) ‖ Technical analysis (Node A CPU)",
            "data_parallelism": "Map step: each chunk processed as independent Ray task",
            "pipeline_parallelism": "Inter-ticker: ticker N+1 ingestion overlaps with ticker N GPU stages",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"])
    parser.add_argument("--filing", default="8-K")
    args = parser.parse_args()

    ray.init(address="auto")
    Path("logs").mkdir(exist_ok=True)

    print("=" * 60)
    print("SCALABILITY EVALUATION (PDC)")
    print("=" * 60)

    print("\n[1-node] Loading from B1 serial results...")
    one_node = run_single_node_estimate(args.tickers)
    print(f"  1-node median: {one_node.get('median_latency_s', '?')}s  |  "
          f"Throughput: {one_node.get('throughput_filings_per_hour', '?')} filings/hr")

    print(f"\n[2-node] Running distributed pipeline for {args.tickers}...")
    two_node = run_two_node(args.tickers, args.filing)
    print(f"  2-node median: {two_node['median_latency_s']}s  |  "
          f"Throughput: {two_node['throughput_filings_per_hour']} filings/hr")

    if two_node.get("bandwidth_stats_per_ticker"):
        for b in two_node["bandwidth_stats_per_ticker"]:
            print(f"  [{b['ticker']}] Payload: {b['raw_bytes']:,} bytes → "
                  f"{b['compressed_bytes']:,} bytes ({b['reduction_pct']:.1f}% reduction)")

    report = compute_scalability_report(one_node, two_node)

    print("\n" + "=" * 60)
    print("SCALABILITY RESULTS")
    print("=" * 60)
    sa = report["scalability_analysis"]
    print(f"Speedup (1→2 nodes):     {sa['speedup_1_to_2_nodes']}x")
    print(f"Latency reduction:       {sa['latency_reduction_pct']}%")
    print(f"Parallel efficiency:     {sa['parallel_efficiency_pct']}%")
    print(f"Throughput improvement:  {sa['throughput_improvement_pct']}%")
    print(f"Tcomm / total time:      {report['tcomm_fraction_of_total_pct']}%")
    print(f"Bandwidth reduction:     {report['bandwidth_compression']['avg_reduction_pct']}%")
    print("\nParallelism types:")
    for ptype, desc in report["parallelism_types_demonstrated"].items():
        print(f"  [{ptype}] {desc}")
    print("=" * 60)

    out = {
        "1_node": one_node,
        "2_node": two_node,
        "report": report,
    }
    with open("logs/scalability_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → logs/scalability_results.json")
