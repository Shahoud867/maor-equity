# Diagram 1: High-Level System Architecture

This diagram presents the full two-node architecture of the MAOR-Equity pipeline.
Node A hosts orchestration, SEC ingestion, chunk filtering, and technical indicators.
Node B hosts GPU inference: FinBERT sentiment, Phi-3 summarization, and guardrail arbitration.
The design intentionally serializes GPU phases after `flush_gpu_cache()` to stay within the 4 GB T1000 limit.
Ray object references are used for inter-node data exchange and execution control.

```mermaid
flowchart LR
    U[User CLI<br/>run_pipeline.py] --> O

    subgraph A[Node A: Ray Head + CPU Services]
        direction TB
        O[Orchestrator<br/>agents/orchestrator.py]
        I[IngestionAgent<br/>fetch_filing + chunk_document]
        C[ChunkFilter<br/>optimization/chunk_filter.py]
        R[DimensionRouter<br/>market/regulatory/temporal]
        T[TechnicalAnalysisAgent<br/>RSI MACD Bollinger VWAP]
    end

    subgraph B[Node B: Ray Worker + NVIDIA T1000 4 GB]
        direction TB
        F[FinBERTBundle<br/>ProsusAI + finbert-tone]
        X[flush_gpu_cache barrier]
        P[Phi3ModelActor<br/>shared Phi-3-mini model]
        S[SummarizationAgent<br/>map-reduce over chunks]
        G[GuardrailAgent<br/>bull vs bear arbitration]
    end

    SEC[(SEC EDGAR)] --> I
    MKT[(yfinance)] --> T

    I --> C --> R --> O
    O -->|Phase A| F
    O -->|Phase A parallel| T
    F --> X --> S
    S --> G
    T --> G
    O --> P
    S --> P
    G --> P
    G --> OUT[Final JSON Output<br/>recommendation + confidence + timings]

    style A fill:#e8f1ff,stroke:#1f5aa6,stroke-width:2px
    style B fill:#fff0f0,stroke:#b93a32,stroke-width:2px
    style O fill:#dbe8ff,stroke:#1f5aa6
    style F fill:#ffe0de,stroke:#b93a32
    style P fill:#ffe0de,stroke:#b93a32
    style OUT fill:#e9f8ef,stroke:#1e7a3e
```

Key design decisions:
- Split CPU-heavy orchestration and data prep (Node A) from GPU-heavy inference (Node B).
- Keep `SummarizationAgent` and `GuardrailAgent` on a shared `Phi3ModelActor` to avoid duplicate model memory.
- Enforce Phase A -> flush -> Phase B to prevent GPU OOM during mixed-model execution.
