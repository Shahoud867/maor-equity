"""
orchestrator.py  —  Node A (head node entry point)
LangGraph-style DAG: Ingestion → [Sentiment ‖ Technical ‖ Summarization] → Guardrail
Run:  python agents/orchestrator.py
"""
import time
import ray
import numpy as np

from agents.ingestion_agent      import IngestionAgent
from agents.sentiment_agent      import FinBERTBundle, DimensionRouter, aggregate_sentiment_vector
from agents.summarization_agent  import SummarizationAgent
from agents.technical_agent      import TechnicalAnalysisAgent
from agents.guardrail_agent      import GuardrailAgent, Phi3ModelActor
from optimization.chunk_filter   import ChunkFilter


def run_pipeline(ticker: str, filing_type: str = "8-K") -> dict:
    """Full distributed pipeline. Returns result dict."""
    t0_total = time.perf_counter()
    timings  = {}

    # Phi-3-mini loaded ONCE, shared by summarizer and guardrail (VRAM fix)
    phi3        = Phi3ModelActor.remote()

    # GPU actors on Node B
    finbert     = FinBERTBundle.remote()
    summarizer  = SummarizationAgent.remote(phi3)
    guardrail   = GuardrailAgent.remote(phi3)

    # I/O-bound agents run locally on Node A: sec_edgar_downloader and yfinance
    # are only installed on Node A. These complete in seconds; FinBERT/Phi-3-mini
    # dominate latency, so local execution does not hurt end-to-end time.
    ingestion    = IngestionAgent()
    tech_agent   = TechnicalAnalysisAgent()
    router       = DimensionRouter()
    chunk_filter = ChunkFilter()

    # ── Stage 1: Ingestion (Node A, local) ───────────────────────────
    t0 = time.perf_counter()
    filing = ingestion.fetch_filing(ticker, filing_type)
    if "error" in filing:
        return {"error": filing["error"]}
    chunks_raw = ingestion.chunk_document(filing["text"])
    timings["ingestion"] = time.perf_counter() - t0

    # ── Stage 1b: Chunk filtering (Node A, CPU) ──────────────────────
    # Drops near-duplicate overlapping chunks before sending to Node B GPU.
    # This is the primary lever for H1: fewer chunks = fewer Phi-3-mini calls.
    t0 = time.perf_counter()
    filter_result = chunk_filter.filter(chunks_raw, filing_type)
    chunks = filter_result["chunks"]
    timings["chunk_filter_ms"] = (time.perf_counter() - t0) * 1000
    timings["n_chunks_before_filter"] = filter_result["n_before"]
    timings["n_chunks_after_filter"]  = filter_result["n_after"]
    timings["chunk_reduction_pct"]    = filter_result["reduction_pct"]

    # ── Stage 2: Dimension routing (Node A, instant) ─────────────────
    routed = router.route(chunks)

    # ── Stage 3: Parallel fan-out  ←─ CORE PDC CONTRIBUTION ──────────
    t0 = time.perf_counter()

    # Tcomm measurement: encode → serialize (put) → submit → deserialize (get) → decode
    t_encode = time.perf_counter()
    payload_mkt = np.array([len(t) for t in routed["market"]])   # proxy encode
    payload_reg = np.array([len(t) for t in routed["regulatory"]])
    payload_tmp = np.array([len(t) for t in routed["temporal"]])
    timings["t_encode_ms"] = (time.perf_counter() - t_encode) * 1000

    t_serialize = time.perf_counter()
    ref_mkt_payload = ray.put(routed["market"])
    ref_reg_payload = ray.put(routed["regulatory"])
    ref_tmp_payload = ray.put(routed["temporal"])
    timings["t_serialize_ms"] = (time.perf_counter() - t_serialize) * 1000

    t_transfer = time.perf_counter()

    # GPU SERIALIZATION: FinBERT and Phi-3-mini share one physical T1000.
    # Running them concurrently causes OOM. We serialize them:
    #   Phase A — FinBERT (Node B GPU) + Technical (Node A CPU) run in parallel.
    #   Phase B — Phi-3-mini summarization (Node B GPU) starts only after
    #             FinBERT finishes and its GPU cache is flushed.
    # This preserves cross-node parallelism (A CPU ‖ B GPU) while preventing
    # same-GPU concurrency between FinBERT and Phi-3-mini.

    # Phase A: FinBERT on Node B GPU  ‖  Technical analysis on Node A CPU
    # tech_agent.compute_indicators() is an I/O-bound yfinance call (~1-2 s).
    # Submitting the Ray remote first lets it start on Node B while we compute
    # technical indicators locally; both finish before we call ray.get().
    r_bundle = finbert.classify_all.remote(ref_mkt_payload, ref_reg_payload,
                                           ref_tmp_payload)
    tech_r   = tech_agent.compute_indicators(ticker)   # local, completes fast

    mkt_r, reg_r, tmp_r = ray.get(r_bundle)          # wait for FinBERT
    ray.get(finbert.flush_gpu_cache.remote())          # release cached allocs
    timings["t_transfer_ms"] = (time.perf_counter() - t_transfer) * 1000

    # Phase B: Phi-3-mini summarization now has full GPU headroom
    t_deserialize = time.perf_counter()
    r_sum  = summarizer.process_document.remote(chunks, f"{ticker}_{filing_type}")
    sum_r  = ray.get(r_sum)
    timings["t_deserialize_ms"] = (time.perf_counter() - t_deserialize) * 1000

    timings["t_comm_total_ms"] = (
        timings["t_encode_ms"] + timings["t_serialize_ms"] +
        timings["t_transfer_ms"] + timings["t_deserialize_ms"]
    )
    timings["parallel_stage"] = time.perf_counter() - t0

    # ── Stage 4: Aggregate sentiment matrix ──────────────────────────
    sv = aggregate_sentiment_vector(mkt_r, reg_r, tmp_r)

    # ── Stage 5: Guardrail (Node B) ──────────────────────────────────
    t0 = time.perf_counter()
    gr = ray.get(guardrail.assess.remote(sum_r["summary"], sv, tech_r))
    timings["guardrail"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t0_total
    return {
        "ticker": ticker, "filing_type": filing_type,
        "n_chunks_raw": filter_result["n_before"],
        "n_chunks_filtered": filter_result["n_after"],
        "chunk_reduction_pct": filter_result["reduction_pct"],
        "sentiment_vector": sv.tolist(),
        "summary": sum_r, "technical": tech_r, "guardrail": gr,
        "timings": timings,
    }


if __name__ == "__main__":
    import json
    ray.init(address="auto")
    print("Running pipeline for AAPL 8-K ...")
    out = run_pipeline("AAPL", "8-K")
    print(json.dumps(out, indent=2, default=str))