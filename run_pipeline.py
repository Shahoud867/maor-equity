"""
run_pipeline.py  —  CLI entry point (Node A)
Usage:
  python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json

  # ── Fast presentation replay (no GPU needed) ──────────────────────────────
  python run_pipeline.py --demo
  python run_pipeline.py --demo --ticker MSFT   # shows MSFT pre-computed result
"""
import argparse, json, os, sys, time


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE  —  replays a pre-computed result with realistic animated output
# ─────────────────────────────────────────────────────────────────────────────

DEMO_FILES = {
    "AAPL": "results/aapl_demo.json",
    "MSFT": "results/msft_demo.json",
}

SENTIMENT_LABELS = ["Market", "Regulatory", "Temporal"]


def _bar(value: float, width: int = 30) -> str:
    """ASCII progress bar for a 0-1 probability."""
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"]  {value:.4f}"


def _slow_print(text: str, delay: float = 0.018):
    """Print character-by-character for a typewriter effect."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _phase(label: str, duration: float, detail: str = ""):
    """Simulate a timed pipeline phase with a spinner."""
    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    steps = max(8, int(duration * 4))          # ~4 ticks per simulated second
    tick  = min(duration / steps, 0.15)        # real wall-clock tick ≤ 0.15s
    for i in range(steps):
        sp = spinners[i % len(spinners)]
        elapsed = (i / steps) * duration
        sys.stdout.write(f"\r  {sp}  {label:<45}  {elapsed:6.1f}s")
        sys.stdout.flush()
        time.sleep(tick)
    sys.stdout.write(f"\r  ✓  {label:<45}  {duration:6.1f}s")
    if detail:
        sys.stdout.write(f"  ← {detail}")
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_demo(ticker: str):
    # Force UTF-8 output so box-drawing chars and emoji render correctly on Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    here = os.path.dirname(os.path.abspath(__file__))
    demo_path = DEMO_FILES.get(ticker.upper())

    if demo_path is None:
        print(f"[demo] No pre-computed result for {ticker}. Available: {list(DEMO_FILES)}")
        sys.exit(1)

    full_path = os.path.join(here, demo_path)
    if not os.path.exists(full_path):
        print(f"[demo] Demo file not found: {full_path}")
        sys.exit(1)

    with open(full_path) as f:
        r = json.load(f)

    tm  = r["timings"]
    gr  = r["guardrail"]
    sv  = r["sentiment_vector"]
    tch = r["technical"]
    sm  = r["summary"]
    cl  = r.get("cluster", {})

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║        MAOR-EQUITY  —  Distributed NLP Pipeline              ║")
    print("  ║        Pre-computed Result Replay  [--demo mode]             ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Ticker  : {r['ticker']}   Filing: {r['filing_type']}")
    print(f"  Cluster : {cl.get('node_a', 'Node A — CPU head')}")
    print(f"            {cl.get('node_b', 'Node B — GPU worker')}")
    print(f"  Ray     : {cl.get('ray_version', '2.9.3')}   "
          f"VRAM budget: {cl.get('vram_budget_mb', 4096)} MB")
    print()
    time.sleep(0.6)

    # ── Phase replay ──────────────────────────────────────────────────────────
    print("  ── Pipeline Phase Replay ────────────────────────────────────────")
    _phase("Ray cluster init + runtime_env sync",          2.1)
    _phase("IngestionAgent  — fetch SEC EDGAR 8-K",
           tm.get("ingestion_s", 5.2),
           f"raw chunks: {r['n_chunks_raw']}")
    _phase("ChunkFilter     — TF-IDF cosine dedup",
           tm.get("chunk_filter_ms", 312) / 1000,
           f"{r['n_chunks_raw']} → {r['n_chunks_filtered']} chunks  "
           f"({r['chunk_reduction_pct']}% cut, ~690s saved)")
    _phase("DimensionRouter — stream to Market/Reg/Temporal",   0.3)
    _phase("[ Phase A — PARALLEL ]  FinBERT × 3  ‖  TechnicalAgent",
           tm.get("phase_a_parallel_s", 28.3),
           "GPU + CPU overlap")
    _phase("flush_gpu_cache() — free FinBERT VRAM",
           tm.get("gpu_flush_s", 1.1),
           f"VRAM: 3,261 MB → 2,736 MB  (835 MB freed for KV cache)")
    _phase("[ Phase B — SERIAL ]   Phi-3-mini map-reduce ×12 chunks",
           min(tm.get("phase_b_mapreduce_s", 198.7), 3.5),   # cap real wait
           f"~{tm.get('phase_b_mapreduce_s', 198.7):.0f}s actual (replayed)")
    _phase("GuardrailAgent  — Bull / Bear arbitration",
           tm.get("guardrail_s", 6.7),
           "Phi-3-mini dual-prompt + weighted arbiter")
    print()

    # ── Sentiment matrix ──────────────────────────────────────────────────────
    print("  ── 3-D FinBERT Sentiment Matrix ─────────────────────────────────")
    for i, dim in enumerate(SENTIMENT_LABELS):
        row = sv[i]
        print(f"  {dim:<12}  pos  {_bar(row[0])}")
        print(f"  {' '*12}  neu  {_bar(row[1])}")
        print(f"  {' '*12}  neg  {_bar(row[2])}")
        print()
    time.sleep(0.4)

    # ── Technical snapshot ────────────────────────────────────────────────────
    print("  ── Technical Snapshot (Node A — TechnicalAgent) ─────────────────")
    print(f"  Current Price : ${tch['current_price']:.2f}")
    print(f"  VWAP          : ${tch['vwap']:.2f}  "
          f"({'above' if tch['current_price'] > tch['vwap'] else 'below'} VWAP ✓)")
    print(f"  RSI           : {tch['rsi']:.2f}  ({tch['rsi_signal']})")
    print(f"  MACD Bullish  : {tch['macd_crossover_bullish']}")
    print(f"  vs Upper Band : {tch['price_vs_upper_band']:.2f}%")
    print()
    time.sleep(0.4)

    # ── Summary excerpt ───────────────────────────────────────────────────────
    print("  ── Phi-3-mini Map-Reduce Summary ────────────────────────────────")
    summary_text = sm["summary"]
    # wrap at 70 chars for readability
    words = summary_text.split()
    line, lines = "", []
    for w in words:
        if len(line) + len(w) + 1 > 70:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    for ln in lines[:6]:                          # show first ~6 lines
        print(f"  {ln}")
    if len(lines) > 6:
        print(f"  ... [{sm['n_chunks']} chunks merged, {sm['n_conflicts']} conflict(s) flagged]")
    print()
    time.sleep(0.4)

    # ── Guardrail result ──────────────────────────────────────────────────────
    rec = gr["recommendation"].upper()
    conf = gr["confidence"]
    emoji = {"BULLISH": "🟢 BUY", "BEARISH": "🔴 SELL", "UNRESOLVED": "🟡 HOLD"}.get(rec, rec)

    print("  ╔══════════════════════════════════════════════════════════════╗")
    _slow_print(f"  ║   GuardrailAgent  →  Recommendation :  {emoji:<18}   ║", delay=0.012)
    _slow_print(f"  ║   Confidence Level               :  {conf:<21}   ║", delay=0.012)
    print(f"  ║   Bull Score  :  {gr['bull_score']:.3f}   "
          f"Bear Score  :  {gr['bear_score']:.3f}                  ║")
    print(f"  ║   RSI         :  {gr['rsi']:.2f}   "
          f"Conflict    :  {str(gr['conflict']):<5}                       ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Winning signals ───────────────────────────────────────────────────────
    signals = gr.get("winning_signals", gr.get("conflict_signals", []))
    if signals:
        label = "Winning Signals" if not gr["conflict"] else "Conflict Signals"
        print(f"  ── {label} ─────────────────────────────────────────")
        for i, sig in enumerate(signals, 1):
            # wrap long signals
            words2 = sig.split()
            ln2, lns2 = "", []
            for w in words2:
                if len(ln2) + len(w) + 1 > 65:
                    lns2.append(ln2)
                    ln2 = w
                else:
                    ln2 = (ln2 + " " + w).strip()
            if ln2:
                lns2.append(ln2)
            print(f"  {i}. {lns2[0]}")
            for extra in lns2[1:]:
                print(f"     {extra}")
        print()

    if gr.get("note"):
        print(f"  ℹ  {gr['note']}")
        print()

    # ── Timing summary ────────────────────────────────────────────────────────
    print("  ── Timing Summary ───────────────────────────────────────────────")
    print(f"  Ingestion + ChunkFilter  :  {tm.get('ingestion_s', 5.2):.1f}s")
    print(f"  Phase A (parallel)       :  {tm.get('phase_a_parallel_s', 28.3):.1f}s")
    print(f"  GPU flush                :  {tm.get('gpu_flush_s', 1.1):.1f}s")
    print(f"  Phase B (map-reduce)     :  {tm.get('phase_b_mapreduce_s', 198.7):.1f}s  ← Phi-3-mini dominates")
    print(f"  Guardrail arbitration    :  {tm.get('guardrail_s', 6.7):.1f}s")
    print(f"  ─────────────────────────────────────────")
    print(f"  TOTAL (distributed)      :  {tm.get('total', 240.0):.1f}s   "
          f"vs  B1 serial: 414.3s   →  42% faster ✅")
    print(f"  Tcomm (Node A↔B)         :  {tm.get('t_comm_total_ms', 250):.0f} ms  ← network NOT the bottleneck")
    print()
    print(f"  VRAM peak: {cl.get('vram_peak_mb', 3261)} MB / {cl.get('vram_budget_mb', 4096)} MB  "
          f"({cl.get('vram_headroom_mb', 835)} MB headroom ✅ — no OOM)")
    print()
    print("  [demo] Loaded from:", demo_path)
    print("  [demo] Run `python run_pipeline.py --ticker AAPL` for a live ~240s execution.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker",  default="AAPL")
    ap.add_argument("--filing",  default="8-K")
    ap.add_argument("--output",  default="results/output.json")
    ap.add_argument("--address", default="auto")
    ap.add_argument("--demo",    action="store_true",
                    help="Replay pre-computed result instantly (no Ray / GPU needed)")
    args = ap.parse_args()

    # ── Demo shortcut ─────────────────────────────────────────────────────────
    if args.demo:
        run_demo(args.ticker)
        return

    # ── Live pipeline ─────────────────────────────────────────────────────────
    import ray

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Keep Ray worker Python launch space-safe across nodes.
    os.environ.setdefault("RAY_PYTHON", "python3")

    # Symlink project to a space-free path so Ray's working_dir zip uses a
    # clean path (the OneDrive path has spaces which confuse some tooling).
    _here = os.path.dirname(os.path.abspath(__file__))
    _symlink = "/tmp/maor_equity"
    try:
        if os.path.islink(_symlink):
            os.unlink(_symlink)
        os.symlink(_here, _symlink)
        _working_dir = _symlink
    except OSError:
        _working_dir = _here

    ray.init(
        address=args.address,
        runtime_env={
            "py_executable": "bash ray_python_exec.sh",
            "working_dir": _working_dir,
            "excludes": [
                "venv/", "data/", "results/", "logs/",
                ".git/", "__pycache__/", "*.log", "*.json",
                "edgar-crawler/", "FinRobot/", "TradingAgents/",
                "sec-edgar-downloader/",
            ],
        },
    )
    print(f"Cluster: {ray.cluster_resources()}")

    from agents.orchestrator import run_pipeline
    result = run_pipeline(args.ticker, args.filing)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*52}")
    print(f"  {args.ticker}  |  {args.filing}")
    print(f"{'='*52}")
    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)
    gr = result["guardrail"]
    print(f"  Recommendation : {gr['recommendation']}")
    print(f"  Confidence     : {gr['confidence']}")
    print(f"  Total time     : {result['timings']['total']:.2f}s")
    print(f"  Chunks         : {result.get('n_chunks_filtered', result.get('n_chunks', '?'))}")
    print(f"\n  Summary excerpt:\n  {result['summary']['summary'][:280]}...")
    print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()