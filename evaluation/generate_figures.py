"""
evaluation/generate_figures.py — Generate all paper figures from log files.

Run after all evaluations are complete:
    python -m evaluation.generate_figures

Outputs → figures/
  fig1_latency_comparison.png  — H1: distributed vs serial bar chart
  fig2_tcomm_breakdown.png     — H1: Tcomm decomposition pie chart
  fig3_speedup_attribution.png — H1: speedup source waterfall
  fig4_vram_trace.png          — VRAM at each pipeline stage
  fig5_rouge_comparison.png    — H2: ROUGE-1/2/L comparison
  fig6_h3_sentiment.png        — H3: direction change analysis
  fig7_amdahl.png              — Amdahl's Law vs actual speedup
  fig8_chunk_filter.png        — Chunk filter reduction per ticker

Requires: pip install matplotlib
"""
import json
from pathlib import Path


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {path} not found — skipping figure")
        return {}
    with open(p) as f:
        return json.load(f)


def fig1_latency_comparison(h1: dict, out: Path):
    """Bar chart: distributed vs serial per-ticker and median."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    dist  = h1.get("per_ticker_distributed", [])
    ser   = h1.get("per_ticker_serial", [])
    if not dist or not ser:
        print("  fig1: no data"); return

    tickers = [r["ticker"] for r in dist]
    d_vals  = [r["total_s"] for r in dist]
    s_vals  = [r.get("total_s", 0) for r in ser]
    # Median bars
    d_med = h1.get("distributed_median_s", 0)
    s_med = h1.get("serial_median_s", 0)

    x = list(range(len(tickers)))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_s = ax.bar([i - w/2 for i in x], s_vals, w, label="B1 Serial", color="#e74c3c", alpha=0.85)
    bars_d = ax.bar([i + w/2 for i in x], d_vals, w, label="Distributed", color="#2ecc71", alpha=0.85)

    # Annotate bars
    for bar in bars_s:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{bar.get_height():.0f}s", ha="center", va="bottom", fontsize=9)
    for bar in bars_d:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{bar.get_height():.0f}s", ha="center", va="bottom", fontsize=9)

    # Median lines
    ax.axhline(s_med, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"B1 Median: {s_med:.0f}s")
    ax.axhline(d_med, color="#2ecc71", linestyle="--", linewidth=1.5,
               label=f"Distributed Median: {d_med:.0f}s")

    reduction = h1.get("latency_reduction_pct", 0)
    ax.set_title(f"H1: Latency Comparison — {reduction:.1f}% Reduction (B1 → Distributed)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Total Pipeline Latency (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "fig1_latency_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: fig1_latency_comparison.png")


def fig2_tcomm_breakdown(h1: dict, out: Path):
    """Pie chart: Tcomm decomposition."""
    import matplotlib.pyplot as plt

    tcomm = h1.get("tcomm_decomposition_avg_ms", {})
    if not tcomm:
        print("  fig2: no Tcomm data"); return

    labels = []
    values = []
    label_map = {
        "t_encode_ms": "Encode",
        "t_serialize_ms": "Serialize (ray.put)",
        "t_transfer_ms": "Transfer + FinBERT",
        "t_deserialize_ms": "Deserialize (ray.get)",
    }
    for k, v in tcomm.items():
        if k != "t_comm_total_ms" and v > 0:
            labels.append(label_map.get(k, k))
            values.append(v)

    if not values:
        print("  fig2: empty values"); return

    colors = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c"]
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors[:len(values)], startangle=140,
        textprops={"fontsize": 10},
    )
    total_ms = sum(values)
    ax.set_title(f"Tcomm Decomposition (avg {total_ms:.0f} ms total)\n"
                 f"[Encode → Serialize → Transfer → Deserialize]",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "fig2_tcomm_breakdown.png", dpi=150)
    plt.close()
    print(f"  Saved: fig2_tcomm_breakdown.png")


def fig3_speedup_attribution(h1: dict, ablation: dict, out: Path):
    """Waterfall chart showing contribution of each speedup source."""
    import matplotlib.pyplot as plt
    import numpy as np

    s_med = h1.get("serial_median_s", 0)
    d_med = h1.get("distributed_median_s", 0)
    if not s_med or not d_med:
        print("  fig3: no data"); return

    total_saved = s_med - d_med

    # Estimate contributions
    cf = ablation.get("chunk_filter_contribution", {})
    cf_entries = cf.get("per_ticker", [])
    cf_saved = sum(e.get("estimated_time_saved_s", 0) for e in cf_entries) / max(len(cf_entries), 1)

    a2 = ablation.get("A2_no_inter_ticker_pipeline", {})
    pipe_saved = a2.get("pipelining_saves_s", 0)

    a3 = ablation.get("A3_phase_a_parallelism", {})
    phase_a_saved = a3.get("avg_gain_s", 2.0)

    actor_saved = max(0, total_saved - cf_saved - pipe_saved - phase_a_saved)

    sources = [
        ("B1 Serial\n(Baseline)", s_med, "#e74c3c"),
        ("- Chunk\nFilter", -cf_saved, "#3498db"),
        ("- Inter-ticker\nPipelining", -pipe_saved, "#2ecc71"),
        ("- Phase A\nParallelism", -phase_a_saved, "#f39c12"),
        ("- Warm Actor\nPersistence", -actor_saved, "#9b59b6"),
        ("Distributed\n(Result)", d_med, "#27ae60"),
    ]

    running = s_med
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = []
    for i, (label, delta, color) in enumerate(sources):
        if i == 0 or i == len(sources) - 1:
            ax.bar(i, abs(delta), color=color, alpha=0.85, width=0.6)
            ax.text(i, abs(delta) + 5, f"{abs(delta):.0f}s", ha="center", fontsize=9, fontweight="bold")
        else:
            bottom = running + delta if delta < 0 else running
            ax.bar(i, abs(delta), bottom=bottom if delta < 0 else running,
                   color=color, alpha=0.85, width=0.6)
            ax.text(i, running + delta/2, f"-{abs(delta):.0f}s", ha="center", fontsize=9, color="white", fontweight="bold")
            running += delta
        bottoms.append(running)

    ax.set_title("Speedup Attribution — Contribution of Each PDC Optimization",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s[0] for s in sources], fontsize=9)
    ax.set_ylabel("Pipeline Latency (seconds)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fig3_speedup_attribution.png", dpi=150)
    plt.close()
    print(f"  Saved: fig3_speedup_attribution.png")


def fig4_vram_trace(results: list, out: Path):
    """Line chart: VRAM usage at each pipeline stage."""
    import matplotlib.pyplot as plt

    all_traces = [r.get("vram_trace_mb", {}) for r in results if r.get("vram_trace_mb")]
    if not all_traces:
        print("  fig4: no VRAM trace data"); return

    stage_order = [
        "after_phi3_load_mb",
        "after_finbert_load_mb",
        "during_finbert_inference_mb",
        "after_finbert_flush_mb",
        "during_phi3_summarization_mb",
        "after_guardrail_mb",
    ]
    stage_labels = [
        "Phi-3\nloaded",
        "FinBERT\nloaded",
        "FinBERT\ninference",
        "FinBERT\nflushed",
        "Phi-3\nmap-reduce",
        "Guardrail\ndone",
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    BUDGET = 4096

    for i, trace in enumerate(all_traces):
        ticker = results[i].get("ticker", f"Ticker {i+1}")
        vals = [trace.get(s, None) for s in stage_order]
        x_pts = [j for j, v in enumerate(vals) if v is not None and v > 0]
        y_pts = [vals[j] for j in x_pts]
        if x_pts:
            ax.plot(x_pts, y_pts, marker="o", linewidth=2, label=ticker)

    ax.axhline(BUDGET, color="red", linestyle="--", linewidth=1.5, label=f"T1000 Budget ({BUDGET} MB)")
    ax.fill_between(range(len(stage_order)), BUDGET, BUDGET * 1.05,
                    color="red", alpha=0.1)

    ax.set_title("VRAM Usage Trace — Per Pipeline Stage (T1000 4 GB)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(stage_order)))
    ax.set_xticklabels(stage_labels, fontsize=9)
    ax.set_ylabel("VRAM Used (MB)")
    ax.set_ylim(0, BUDGET * 1.1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fig4_vram_trace.png", dpi=150)
    plt.close()
    print(f"  Saved: fig4_vram_trace.png")


def fig5_rouge_comparison(h2: dict, out: Path):
    """Grouped bar chart: ROUGE-1/2/L + BERTScore for map-reduce vs B2."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not h2:
        print("  fig5: no H2 data"); return

    metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]
    labels  = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore-F1"]

    mr_vals = [h2.get(f"mapreduce_{m}", 0) for m in metrics]
    b2_vals = [h2.get(f"b2_{m}", 0) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_b2 = ax.bar(x - w/2, b2_vals, w, label="B2 Single-pass", color="#e74c3c", alpha=0.85)
    bars_mr = ax.bar(x + w/2, mr_vals, w, label="Map-Reduce (Ours)", color="#2ecc71", alpha=0.85)

    for bar in bars_b2 + bars_mr:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    passed = h2.get("H2_passed", False)
    note = h2.get("H2_note", "")
    ax.set_title(f"H2: Map-Reduce vs B2 Summary Quality\n{note}",
                 fontsize=11, fontweight="bold",
                 color="green" if passed else "orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(mr_vals), max(b2_vals)) * 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fig5_rouge_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: fig5_rouge_comparison.png")


def fig6_h3_sentiment(h3: dict, out: Path):
    """Bar + text summary for H3 sentiment direction comparison."""
    import matplotlib.pyplot as plt

    if not h3:
        print("  fig6: no H3 data"); return

    categories = ["3-D Accuracy\nvs Ground Truth", "B3 Accuracy\nvs Ground Truth",
                  "Direction\nDisagreement"]
    values = [
        h3.get("accuracy_3d_vs_ground_truth_pct", 0),
        h3.get("accuracy_b3_vs_ground_truth_pct", 0),
        h3.get("disagreement_pct", 0),
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=colors, alpha=0.85, width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.axhline(10, color="navy", linestyle="--", linewidth=1.5,
               label="H3 Target: >10% disagreement")

    passed = h3.get("H3_passed", False)
    note   = h3.get("H3_note", "")
    ax.set_title(f"H3: 3-D Sentiment vs B3 Scalar\n{note}",
                 fontsize=11, fontweight="bold",
                 color="green" if passed else "orange")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, max(values) * 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fig6_h3_sentiment.png", dpi=150)
    plt.close()
    print(f"  Saved: fig6_h3_sentiment.png")


def fig7_amdahl(ablation: dict, out: Path):
    """Plot Amdahl's Law theoretical limit vs actual speedup."""
    import matplotlib.pyplot as plt
    import numpy as np

    amd = ablation.get("amdahl_law_analysis", {})
    if not amd:
        print("  fig7: no Amdahl data"); return

    actual_speedup = amd.get("actual_observed_speedup", 0)
    per_ticker     = amd.get("amdahl_analysis_per_ticker", [])

    p_vals = [e.get("parallel_fraction_p", 0) for e in per_ticker]
    avg_p  = sum(p_vals) / len(p_vals) if p_vals else 0.05

    n_range = np.linspace(1, 16, 100)
    speedups_task_parallel = 1 / ((1 - avg_p) + avg_p / n_range)   # task parallelism
    speedups_actor_warm    = np.ones_like(n_range) * actual_speedup  # warm actor (constant gain)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_range, speedups_task_parallel, label=f"Amdahl (task parallelism, p={avg_p:.2f})",
            color="#3498db", linewidth=2)
    ax.axhline(actual_speedup, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Actual speedup: {actual_speedup:.2f}x (warm actor persistence)")
    ax.axvline(2, color="gray", linestyle=":", label="n=2 (our 2-node system)")

    ax.set_title("Amdahl's Law vs Actual Speedup\n"
                 "Warm actor persistence dominates over task parallelism",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Parallel Workers (n)")
    ax.set_ylabel("Speedup (S)")
    ax.set_xlim(1, 16)
    ax.set_ylim(1, max(actual_speedup, speedups_task_parallel[-1]) * 1.2)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fig7_amdahl.png", dpi=150)
    plt.close()
    print(f"  Saved: fig7_amdahl.png")


def fig8_chunk_filter(h1: dict, out: Path):
    """Show chunk reduction per ticker and estimated time saved."""
    import matplotlib.pyplot as plt

    dist = h1.get("per_ticker_distributed", [])
    cf   = h1.get("chunk_filter_stats", {})
    if not dist:
        print("  fig8: no data"); return

    tickers      = [r["ticker"] for r in dist]
    chunks_before = [r.get("n_chunks_before", 0) for r in dist]
    chunks_after  = [r.get("n_chunks_after", 0) for r in dist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: stacked bar showing removed vs kept
    x = range(len(tickers))
    ax1.bar(x, chunks_after, label="Kept", color="#2ecc71", alpha=0.85)
    ax1.bar(x, [b - a for b, a in zip(chunks_before, chunks_after)],
            bottom=chunks_after, label="Removed (TF-IDF dedup)", color="#e74c3c", alpha=0.6)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(tickers)
    ax1.set_title("ChunkFilter: Chunks Kept vs Removed", fontweight="bold")
    ax1.set_ylabel("Number of Chunks")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: estimated time saved
    PHI3_PER_CHUNK = 15
    saved = [(b - a) * PHI3_PER_CHUNK for b, a in zip(chunks_before, chunks_after)]
    ax2.bar(tickers, saved, color="#f39c12", alpha=0.85)
    for i, s in enumerate(saved):
        ax2.text(i, s + 1, f"~{s:.0f}s", ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("Estimated Inference Time Saved\n(Φ ~15s/chunk × chunks_removed)",
                  fontweight="bold")
    ax2.set_ylabel("Estimated Time Saved (s)")
    ax2.grid(axis="y", alpha=0.3)

    if cf:
        avg_red = cf.get("avg_reduction_pct", 0)
        fig.suptitle(f"ChunkFilter Contribution — Avg {avg_red:.1f}% chunk reduction",
                     fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out / "fig8_chunk_filter.png", dpi=150)
    plt.close()
    print(f"  Saved: fig8_chunk_filter.png")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out = Path("figures")
    out.mkdir(exist_ok=True)

    print("Loading log files...")
    h1      = _load("logs/h1_latency_results.json")
    h2      = _load("logs/h2_rouge_results.json")
    h3      = _load("logs/h3_sentiment_results.json")
    ablation = _load("logs/ablation_results.json")

    # Load all distributed results for VRAM trace
    dist_results = []
    for path in Path("results").glob("*.json"):
        try:
            with open(path) as f:
                r = json.load(f)
                if "vram_trace_mb" in r:
                    dist_results.append(r)
        except Exception:
            pass

    print(f"\nGenerating {8} figures → figures/")
    fig1_latency_comparison(h1, out)
    fig2_tcomm_breakdown(h1, out)
    fig3_speedup_attribution(h1, ablation, out)
    fig4_vram_trace(dist_results, out)
    fig5_rouge_comparison(h2, out)
    fig6_h3_sentiment(h3, out)
    fig7_amdahl(ablation, out)
    fig8_chunk_filter(h1, out)

    print("\nAll figures saved to figures/")
    print("Include these in your paper: fig1, fig3, fig4, fig5, fig6, fig7 are highest impact.")
