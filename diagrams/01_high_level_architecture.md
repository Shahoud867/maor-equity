# Diagram 1: High-Level System Architecture

**Description:**
Two-node heterogeneous cluster processing SEC EDGAR filings through a distributed NLP pipeline.
Node A (CPU) handles ingestion, chunk filtering, and orchestration via the Ray head node.
Node B (GPU T1000 4GB) runs all three model-inference phases: FinBERT, Phi-3-mini, and Guardrail.
Phase serialization (`flush_gpu_cache()`) between Phase A and Phase B prevents VRAM OOM (peak: 3,261 MB < 4,096 MB budget).
The ChunkFilter is the dominant performance contributor — 79% chunk reduction saves ~690s per run.

---

```mermaid
graph TB
    subgraph INPUT["📥 INPUT LAYER"]
        SEC[("SEC EDGAR API\n8-K · 10-K Filings")]
        YF[("yfinance API\nOHLCV Market Data")]
    end

    subgraph NODE_A["🖥️ NODE A — Intel CPU · 32 GB RAM · WSL2/Windows 11"]
        direction TB
        ORC["🎯 Orchestrator\n(Ray Head Node — Port 6380)\nDAG Coordinator · Timer · VRAM Tracker"]
        ING["📄 IngestionAgent\nSEC EDGAR Download\n512-token chunks · 64-stride overlap"]
        CF["🔍 ChunkFilter\nTF-IDF Cosine Deduplication\n58 → 12 chunks  ·  79% reduction  ·  ~690s saved"]
        DR["🗂️ DimensionRouter\nRegex-based Section Tagger\nMarket · Regulatory · Temporal"]
        TA["📊 TechnicalAnalysisAgent\nRSI-14 · MACD · Bollinger Bands\nyfinance 90-day window"]
    end

    subgraph NODE_B["🖥️ NODE B — NVIDIA T1000 · 4,096 MB VRAM · Ubuntu 22.04"]
        direction TB
        subgraph PHASE_A["⚡ PHASE A — Parallel with TechnicalAgent"]
            SA["🧠 FinBERTBundle\n4-bit NF4 Quantization · 525 MB VRAM\nMarket (ProsusAI) · Regulatory (yiyanghkust) · Temporal (ProsusAI)\n3-D Sentiment Matrix  M ∈ R^(3×3)  ·  ~15.7s"]
        end
        FLUSH["💨 flush_gpu_cache()\nVRAM: 3,261 MB → 2,736 MB\nHeadroom for Phi-3 KV Cache"]
        subgraph PHASE_B["📝 PHASE B — Sequential after GPU flush"]
            SUM["📋 SummarizationAgent\nPhi-3-mini 3.8B · 4-bit NF4 · 2,736 MB VRAM\nMap-Reduce over 12 filtered chunks  ·  ~353.5s"]
        end
        GA["🛡️ GuardrailAgent\nBull/Bear Dual-Prompt Arbitration\nRSI + Sentiment Conflict Detection\nShared Phi3ModelActor"]
    end

    subgraph OUTPUT["📤 OUTPUT LAYER"]
        REC["💡 Investment Recommendation\nDirection: BUY · HOLD · SELL\nConfidence: HIGH · MEDIUM · UNRESOLVED\n+ Sentiment Vector + Summary + Timings"]
    end

    SEC -->|"Raw HTML/XML"| ING
    YF -->|"OHLCV prices"| TA
    ING -->|"Clean text  50K+ tokens"| CF
    CF -->|"12 filtered chunks"| DR
    DR --> ORC
    ORC -->|"Ray.remote() tasks"| SA
    ORC -->|"Ray.remote() tasks"| TA
    SA -->|"Phase A complete"| FLUSH
    FLUSH -->|"VRAM freed"| SUM
    TA -->|"RSI · MACD · BB indicators"| GA
    SUM -->|"Structured summary + [CONFLICT] tags"| GA
    GA -->|"Final decision JSON"| REC

    style NODE_A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style NODE_B fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style PHASE_A fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px
    style PHASE_B fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style INPUT fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    style OUTPUT fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style FLUSH fill:#fff3e0,stroke:#e65100,stroke-width:1px
```

---

**Key Performance Numbers:**

| Metric | Value |
|--------|-------|
| End-to-end speedup (H1) | **1.72×** vs B1 serial baseline |
| Task parallelism alone (Amdahl) | 1.019× (p = 0.038) |
| ChunkFilter savings | ~690s · 79% chunk reduction |
| Peak VRAM usage | 3,261 MB / 4,096 MB budget |
| VRAM headroom | 835 MB |
