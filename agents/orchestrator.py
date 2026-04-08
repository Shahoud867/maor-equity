"""
orchestrator.py  —  Node A (head node entry point)
LangGraph-style DAG: Ingestion → [Sentiment ‖ Technical ‖ Summarization] → Guardrail
Run:  python agents/orchestrator.py
"""
import time
import ray
import numpy as np

from agents.ingestion_agent      import IngestionAgent
from agents.sentiment_agent      import FinBERTActor, DimensionRouter, aggregate_sentiment_vector
from agents.summarization_agent  import SummarizationAgent
from agents.technical_agent      import TechnicalAnalysisAgent
from agents.guardrail_agent      import GuardrailAgent


def run_pipeline(ticker: str, filing_type: str = "8-K") -> dict:
    """Full distributed pipeline. Returns result dict."""
    t0_total = time.perf_counter()
    timings  = {}

    # Initialise remote actors (Node placement handled by Ray scheduler)
    ingestion   = IngestionAgent.remote()
    tech        = TechnicalAnalysisAgent.remote()
    sent_mkt    = FinBERTActor.remote("ProsusAI/finbert",          "market")
    sent_reg    = FinBERTActor.remote("yiyanghkust/finbert-tone",  "regulatory")
    sent_tmp    = FinBERTActor.remote("ProsusAI/finbert",          "temporal")
    summarizer  = SummarizationAgent.remote()
    guardrail   = GuardrailAgent.remote()
    router      = DimensionRouter()

    # ── Stage 1: Ingestion (Node A) ──────────────────────────────────
    t0 = time.perf_counter()
    filing = ray.get(ingestion.fetch_filing.remote(ticker, filing_type))
    if "error" in filing:
        return {"error": filing["error"]}
    chunks = ray.get(ingestion.chunk_document.remote(filing["text"]))
    timings["ingestion"] = time.perf_counter() - t0

    # ── Stage 2: Dimension routing (Node A, instant) ─────────────────
    routed = router.route(chunks)

    # ── Stage 3: Parallel fan-out  ←─ CORE PDC CONTRIBUTION ──────────
    t0 = time.perf_counter()
    r_mkt  = sent_mkt.classify_batch.remote(routed["market"])
    r_reg  = sent_reg.classify_batch.remote(routed["regulatory"])
    r_tmp  = sent_tmp.classify_batch.remote(routed["temporal"])
    r_sum  = summarizer.process_document.remote(chunks, f"{ticker}_{filing_type}")
    r_tech = tech.compute_indicators.remote(ticker)

    mkt_r, reg_r, tmp_r = ray.get([r_mkt, r_reg, r_tmp])
    sum_r  = ray.get(r_sum)
    tech_r = ray.get(r_tech)
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
        "n_chunks": len(chunks),
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