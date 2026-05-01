# Diagram 4: Sequence Diagram (Key Process)

This sequence captures one end-to-end execution for a single ticker request.
It highlights control flow, cross-node calls, and synchronization points that protect GPU memory.
The critical architectural pattern is parallel Phase A followed by serialized Phase B.
The result combines all evidence streams into a final guardrail decision object.

```mermaid
sequenceDiagram
    autonumber
    actor U as User CLI
    participant RP as run_pipeline.py
    participant ORC as Orchestrator (Node A)
    participant ING as IngestionAgent (Node A)
    participant F as FinBERTBundle (Node B)
    participant T as TechnicalAnalysisAgent (Node A)
    participant S as SummarizationAgent (Node B)
    participant G as GuardrailAgent (Node B)
    participant P as Phi3ModelActor (Node B)

    U->>RP: python run_pipeline.py --ticker AAPL --filing 8-K
    RP->>ORC: run_pipeline(ticker, filing_type)

    ORC->>P: initialize shared Phi-3 actor
    ORC->>F: initialize FinBERT bundle

    ORC->>ING: fetch_filing.remote()
    ING-->>ORC: filing text
    ORC->>ORC: chunk_document + ChunkFilter + DimensionRouter

    par Phase A parallel
        ORC->>F: classify_all.remote(market, regulatory, temporal)
        F-->>ORC: sentiment outputs
    and
        ORC->>T: compute_indicators.remote(ticker)
        T-->>ORC: RSI/MACD/BB/VWAP
    end

    ORC->>F: flush_gpu_cache.remote()

    ORC->>S: process_document.remote(filtered_chunks, doc_id)
    S->>P: generate(map/reduce prompts)
    P-->>S: summary text
    S-->>ORC: structured summary

    ORC->>G: assess.remote(summary, sentiment_vector, technical)
    G->>P: generate(bull and bear prompts)
    P-->>G: model responses
    G-->>ORC: recommendation + confidence + rationale

    ORC-->>RP: result dict
    RP-->>U: JSON output + console summary
```

Key design decisions:
- Warm up `Phi3ModelActor` before downstream steps to avoid repeated model initialization.
- Execute FinBERT and technical analysis concurrently for latency reduction.
- Flush GPU cache before summarization to maintain stable execution on limited VRAM.
