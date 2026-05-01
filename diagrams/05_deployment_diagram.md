# Diagram 5: Deployment Diagram

This deployment view maps logical services to physical and runtime infrastructure.
It reflects the current two-node Ray setup with Node A as head and Node B as GPU worker.
External APIs remain decoupled through HTTPS, while inter-node scheduling and object exchange use Ray.
The layout highlights where GPU memory pressure is managed through actor placement.

```mermaid
flowchart LR
    subgraph EXTERNAL[External Services]
        SEC[SEC EDGAR API]
        YF[yfinance API]
        HF[HuggingFace Hub]
    end

    subgraph NODEA[Node A: Windows + WSL2 Ubuntu]
        RH[Ray Head:6380\nDashboard:8265]
        CLI[run_pipeline.py]
        ORC[Orchestrator]
        IA[IngestionAgent]
        TA[TechnicalAnalysisAgent]
        CF[ChunkFilter + DimensionRouter]
    end

    subgraph NODEB[Node B: Ubuntu + NVIDIA T1000 4 GB]
        RW[Ray Worker]
        FB[FinBERTBundle]
        PHI[Phi3ModelActor shared]
        SA[SummarizationAgent]
        GA[GuardrailAgent]
    end

    SEC --> IA
    YF --> TA
    HF --> FB
    HF --> PHI

    CLI --> ORC
    ORC --> IA
    ORC --> CF
    ORC --> TA
    ORC --> FB
    ORC --> SA
    ORC --> GA

    SA --> PHI
    GA --> PHI
    RH <-->|Ray cluster traffic| RW

    ORC --> RES[(results and logs)]

    style NODEA fill:#e8f1ff,stroke:#1f5aa6,stroke-width:2px
    style NODEB fill:#fff0f0,stroke:#b93a32,stroke-width:2px
    style EXTERNAL fill:#fff8e5,stroke:#a67400,stroke-width:2px
    style RES fill:#e9f8ef,stroke:#1e7a3e
```

Key design decisions:
- Deploy Ray head on Node A to keep orchestration and ingestion close to the CLI entry point.
- Isolate GPU actors on Node B and centralize LLM use via a shared model actor.
- Keep external dependencies outside cluster internals to simplify scaling and fault isolation.
