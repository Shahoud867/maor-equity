# Diagram 3: Data Flow Diagram (DFD)

This DFD explains how information moves and transforms from external data sources to the final recommendation.
It separates processes from data stores, making traceability and verification straightforward.
The central fusion stage combines three artifact types: sentiment matrix, summary, and technical indicators.
The output is a structured JSON object persisted to logs and returned to the CLI.

## Level 0: Context

```mermaid
flowchart LR
    U[Analyst/User] -->|ticker + filing type| SYS[MAOR-Equity Pipeline]
    SYS -->|recommendation + confidence + evidence| U
    SEC[(SEC EDGAR)] -->|filing text| SYS
    YF[(yfinance)] -->|price history| SYS
    HF[(HuggingFace)] -->|model weights| SYS
```

## Level 1: Process Flow

```mermaid
flowchart LR
    SEC[(SEC EDGAR)] --> P1
    YF[(yfinance)] --> P4
    HF[(HuggingFace cache)] --> P3
    HF --> P5

    P1[P1 IngestionAgent\nfetch + clean filing text] -->
    P2[P2 ChunkFilter\nchunk + deduplicate]

    P2 -->|routed chunks| P3[P3 FinBERTBundle\nmarket/regulatory/temporal sentiment]
    P2 -->|filtered chunks| P5[P5 SummarizationAgent\nmap-reduce summary]
    P4[P4 TechnicalAnalysisAgent\nRSI/MACD/BB/VWAP] --> P6

    P3 -->|sentiment matrix 3x3| P6[P6 GuardrailAgent\nLLM + rule-based arbitration]
    P5 -->|structured summary| P6

    P6 --> OUT[(results/*.json)]
    P6 --> CLI[run_pipeline.py output]

    style P1 fill:#e8f1ff,stroke:#1f5aa6
    style P2 fill:#e8f1ff,stroke:#1f5aa6
    style P3 fill:#fff0f0,stroke:#b93a32
    style P5 fill:#fff0f0,stroke:#b93a32
    style P6 fill:#fff0f0,stroke:#b93a32
    style P4 fill:#e8f1ff,stroke:#1f5aa6
    style OUT fill:#e9f8ef,stroke:#1e7a3e
```

Key design decisions:
- Keep chunk filtering before GPU stages to reduce inference cost and latency.
- Treat sentiment, summary, and technical analytics as independent evidence streams.
- Persist decision artifacts to `results` and `logs` for reproducibility and evaluation.
