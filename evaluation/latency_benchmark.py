"""
evaluation/latency_benchmark.py — H1 Evaluation
Compares distributed pipeline latency vs B1 serial baseline.

H1: Distributed pipeline achieves 30–50% lower median latency than B1.

Run on Node A after cluster is connected:
    python -m evaluation.latency_benchmark --tickers AAPL MSFT GOOGL
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import ray


def run_distributed(tickers: list) -> list:
    """Run the full distributed pipeline for each ticker. Returns list of timing dicts."""
    from agents.orchestrator import run_pipeline
    results = []
    for tk in tickers:
        print(f"[Distributed] Running {tk} ...")
        t0 = time.perf_counter()
        out = run_pipeline(tk, "8-K")
        elapsed = time.perf_counter() - t0
        timings = out.get("timings", {})
        results.append({
            "ticker": tk,
            "method": "distributed",
            "total_s": timings.get("total", elapsed),
            "parallel_stage_s": timings.get("parallel_stage"),
            "guardrail_s": timings.get("guardrail"),
            "chunk_filter_ms": timings.get("chunk_filter_ms"),
            "n_chunks_before": out.get("n_chunks_raw"),
            "n_chunks_after": out.get("n_chunks_filtered"),
            "chunk_reduction_pct": out.get("chunk_reduction_pct"),
            "t_comm_total_ms": timings.get("t_comm_total_ms"),
            "t_encode_ms": timings.get("t_encode_ms"),
            "t_serialize_ms": timings.get("t_serialize_ms"),
            "t_transfer_ms": timings.get("t_transfer_ms"),
            "t_deserialize_ms": timings.get("t_deserialize_ms"),
        })
        print(f"  → {results[-1]['total_s']:.2f}s  "
              f"(chunks: {results[-1]['n_chunks_before']} → {results[-1]['n_chunks_after']}, "
              f"{results[-1]['chunk_reduction_pct']}% reduced)")
    return results


def run_serial(tickers: list) -> list:
    """Run B1 serial baseline for each ticker. Returns list of timing dicts."""
    from baselines.b1_serial_pipeline import run_serial as _serial
    results = []
    for tk in tickers:
        print(f"[Serial B1] Running {tk} ...")
        out = _serial(tk, "8-K")
        results.append({
            "ticker": tk,
            "method": "B1_serial",
            "total_s": out["timings"]["total"],
            "ingestion_s": out["timings"].get("ingestion"),
            "sentiment_s": out["timings"].get("sentiment"),
            "summarisation_s": out["timings"].get("summarisation"),
            "technical_s": out["timings"].get("technical"),
            "guardrail_s": out["timings"].get("guardrail"),
        })
        print(f"  → {results[-1]['total_s']:.2f}s")
    return results


def compute_stats(distributed: list, serial: list) -> dict:
    d_totals = [r["total_s"] for r in distributed]
    s_totals = [r["total_s"] for r in serial]

    d_median = statistics.median(d_totals)
    s_median = statistics.median(s_totals)
    reduction_pct = (s_median - d_median) / s_median * 100

    d_p95 = sorted(d_totals)[int(len(d_totals) * 0.95)] if len(d_totals) > 1 else d_totals[0]
    s_p95 = sorted(s_totals)[int(len(s_totals) * 0.95)] if len(s_totals) > 1 else s_totals[0]

    # Tcomm decomposition (from distributed results)
    comm_results = [r for r in distributed if r.get("t_comm_total_ms")]
    avg_comm = {}
    if comm_results:
        for key in ["t_encode_ms", "t_serialize_ms", "t_transfer_ms",
                    "t_deserialize_ms", "t_comm_total_ms"]:
            avg_comm[key] = statistics.mean(r[key] for r in comm_results if r.get(key))

    # Chunk filtering stats
    filter_results = [r for r in distributed if r.get("n_chunks_before")]
    avg_filter = {}
    if filter_results:
        avg_filter = {
            "avg_chunks_before": round(statistics.mean(r["n_chunks_before"] for r in filter_results), 1),
            "avg_chunks_after": round(statistics.mean(r["n_chunks_after"] for r in filter_results), 1),
            "avg_reduction_pct": round(statistics.mean(r["chunk_reduction_pct"] for r in filter_results), 1),
            "avg_filter_ms": round(statistics.mean(r["chunk_filter_ms"] for r in filter_results
                                                    if r.get("chunk_filter_ms")), 1),
        }

    # Speedup attribution — what fraction of gain comes from each source
    filter_gain_note = ""
    if avg_filter:
        est_phi3_per_chunk_s = 15      # ~15s per chunk on T1000 4-bit
        chunks_saved = avg_filter.get("avg_chunks_before", 0) - avg_filter.get("avg_chunks_after", 0)
        est_time_saved_s = chunks_saved * est_phi3_per_chunk_s
        filter_gain_note = (
            f"Chunk filtering saves ~{chunks_saved:.0f} Phi-3-mini calls ≈ "
            f"{est_time_saved_s:.0f}s estimated GPU time"
        )

    return {
        "H1_hypothesis": "Distributed achieves 30-50% lower median latency than B1",
        "distributed_median_s": round(d_median, 2),
        "serial_median_s": round(s_median, 2),
        "latency_reduction_pct": round(reduction_pct, 1),
        "H1_passed": 30 <= reduction_pct <= 50,
        "H1_note": (
            f"PASS ({reduction_pct:.1f}% reduction)" if 30 <= reduction_pct <= 50
            else f"PARTIAL ({reduction_pct:.1f}% — target 30–50%)" if reduction_pct > 0
            else f"FAIL (distributed is {abs(reduction_pct):.1f}% SLOWER)"
        ),
        "distributed_p95_s": round(d_p95, 2),
        "serial_p95_s": round(s_p95, 2),
        "speedup_sources": {
            "parallelism": "FinBERT + technical run alongside Phi-3-mini (not sequential)",
            "chunk_filtering": filter_gain_note,
            "model_preloading": "Distributed actors stay loaded; B1 loads models cold each run",
        },
        "chunk_filter_stats": avg_filter,
        "tcomm_decomposition_avg_ms": avg_comm,
        "per_ticker_distributed": distributed,
        "per_ticker_serial": serial,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--skip-serial", action="store_true",
                        help="Skip serial baseline (use saved b1_results.json)")
    args = parser.parse_args()

    ray.init(address="auto")
    Path("logs").mkdir(exist_ok=True)

    # Run distributed
    dist_results = run_distributed(args.tickers)

    # Run or load serial
    if args.skip_serial:
        with open("logs/b1_results.json") as f:
            serial_raw = json.load(f)
        serial_results = [{"ticker": r["ticker"], "method": "B1_serial",
                           "total_s": r["timings"]["total"],
                           **{k: r["timings"].get(k) for k in
                              ["ingestion", "sentiment", "summarisation", "technical", "guardrail"]}}
                          for r in serial_raw]
    else:
        serial_results = run_serial(args.tickers)
        with open("logs/b1_results.json", "w") as f:
            json.dump(serial_results, f, indent=2)

    stats = compute_stats(dist_results, serial_results)

    print("\n" + "="*60)
    print("H1 LATENCY BENCHMARK RESULTS")
    print("="*60)
    print(f"Distributed median:  {stats['distributed_median_s']}s")
    print(f"Serial B1 median:    {stats['serial_median_s']}s")
    print(f"Reduction:           {stats['latency_reduction_pct']}%")
    print(f"Result:              {stats['H1_note']}")

    if stats.get("chunk_filter_stats"):
        cf = stats["chunk_filter_stats"]
        print(f"\nChunk filter contribution:")
        print(f"  Avg chunks before: {cf['avg_chunks_before']}")
        print(f"  Avg chunks after:  {cf['avg_chunks_after']}")
        print(f"  Avg reduction:     {cf['avg_reduction_pct']}%")
        print(f"  Filter cost:       {cf['avg_filter_ms']} ms (CPU, Node A)")
        print(f"  {stats['speedup_sources']['chunk_filtering']}")

    if stats["tcomm_decomposition_avg_ms"]:
        print("\nTcomm decomposition (avg ms):")
        for k, v in stats["tcomm_decomposition_avg_ms"].items():
            print(f"  {k}: {v:.2f} ms")

    print("\nSpeedup sources:")
    for src, note in stats["speedup_sources"].items():
        print(f"  [{src}] {note}")
    print("="*60)

    with open("logs/h1_latency_results.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved → logs/h1_latency_results.json")
