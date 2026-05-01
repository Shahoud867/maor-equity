# Diagram 3: Data Flow Diagram (Level 1 DFD)

**Description:**
Shows how data transforms as it moves through the pipeline — from raw SEC filing to final recommendation.
Each process node represents a stateless transformation: raw HTML → clean tokens → filtered chunks → sentiment matrix → summary → decision.
Cross-node Ray transfers carry NumPy arrays and JSON dicts (not raw text) — ~80% bandwidth reduction vs naive string transfer.
The 3×3 sentiment matrix `M[dimension][polarity]` is the central NLP artifact consumed by the GuardrailAgent.
All intermediate results are stored to `logs/` as JSON for hypothesis validation and auditability.

---

```mermaid
flowchart LR
    DS1[("🗄️ SEC EDGAR\nDatabase\nsec.gov REST API")]
    DS2[("📈 yfinance\nMarket Data API\n90-day OHLCV")]
    DS3[("🤗 HuggingFace Hub\nProsusAI/finbert\nyiyanghkust/finbert-tone\nPhi-3-mini-4k-instruct")]
    DS4[("💾 logs/\nJSON Result Store\nh1/h2/h3 · vram_verify")]

    P1["P1\n📄 Download & Parse\nHTTP fetch · BeautifulSoup\nHTML → clean text"]
    P2["P2\n🔍 Chunk & Filter\n512-token sliding window\nTF-IDF dedup · 79% reduction"]
    P3["P3\n🧠 3-D Sentiment\nClassification\nFinBERT × 3 dimensions\n4-bit NF4 · ~15.7s"]
    P4["P4\n📊 Technical\nIndicator Calc\nRSI · MACD · Bollinger\n~4.98s"]
    P5["P5\n📋 Map-Reduce\nSummarization\nPhi-3-mini 4-bit\n~353.5s"]
    P6["P6\n🛡️ Guardrail\nArbitration\nDual-prompt Bull/Bear\nRuleBasedArbiter"]

    DS1 -->|"Raw HTML/XML\n8-K · 10-K\n~50K+ tokens"| P1
    DS2 -->|"OHLCV prices\n90-day window"| P4
    DS3 -->|"Model weights\nFinBERT 3× [4-bit NF4]"| P3
    DS3 -->|"Model weights\nPhi-3-mini [4-bit NF4]"| P5

    P1 -->|"Clean text string\n50K–200K chars"| P2

    P2 -->|"12 filtered chunks\n512-token each\n[Node A → Node B via Ray Object Store]"| P3
    P2 -->|"12 filtered chunks\n[Node A → Node B via Ray Object Store]"| P5

    P3 -->|"Sentiment Matrix\nM ∈ R^(3×3)\n{positive · neutral · negative}\n× {market · regulatory · temporal}\n[Ray object ref]"| P6

    P4 -->|"Indicators dict\nRSI-14 · MACD signal · BB%\n[Node A → Node B · ~250ms Tcomm]"| P6

    P5 -->|"Structured summary string\n+ [CONFLICT] markup tags\n[Ray object ref]"| P6

    P6 -->|"Recommendation JSON\n{direction · confidence · flags\n sentiment_vector · summary\n timings · vram_trace}"| OUT["📤 Output Layer\nConsole + JSON file\nrun_pipeline.py return"]

    P3 -->|"H2: ROUGE-L · BERTScore"| DS4
    P5 -->|"H1: latency · speedup"| DS4
    P6 -->|"H3: divergence count"| DS4
    P4 -->|"Baseline indicators"| DS4

    style DS1 fill:#fff9c4,stroke:#f9a825
    style DS2 fill:#fff9c4,stroke:#f9a825
    style DS3 fill:#fff9c4,stroke:#f9a825
    style DS4 fill:#fff9c4,stroke:#f9a825
    style OUT fill:#c8e6c9,stroke:#2e7d32
    style P3 fill:#fce4ec,stroke:#c62828
    style P5 fill:#fce4ec,stroke:#c62828
    style P6 fill:#fce4ec,stroke:#c62828
    style P1 fill:#e3f2fd,stroke:#1565c0
    style P2 fill:#e3f2fd,stroke:#1565c0
    style P4 fill:#e3f2fd,stroke:#1565c0
```

---

## Level 0 — Context Diagram

```mermaid
flowchart LR
    USER(["👤 Analyst / User"])
    SYS["🏭 maor-equity\nDistributed Financial\nNLP Pipeline"]
    SEC[("SEC EDGAR")]
    MKT[("Market Data")]

    USER -->|"ticker + filing_type"| SYS
    SYS -->|"BUY/HOLD/SELL + confidence + rationale"| USER
    SEC -->|"SEC filings"| SYS
    MKT -->|"price data"| SYS

    style SYS fill:#e8eaf6,stroke:#3949ab,stroke-width:3px
```

---

**Data Transformation Summary:**

| Stage | Input | Output | Size Change |
|-------|-------|--------|-------------|
| P1 Download & Parse | Raw HTML (50K+ tokens) | Clean text string | -30% (HTML stripped) |
| P2 Chunk & Filter | Clean text | 12 × 512-token chunks | **-79%** (TF-IDF dedup) |
| P3 FinBERT | 12 chunks | 3×3 sentiment matrix | → 9 float values |
| P4 Technical | 90-day OHLCV | 3 indicator values | → scalar dict |
| P5 Phi-3-mini | 12 chunks | 1 summary string | Map-reduce compression |
| P6 Guardrail | Matrix + Summary + Indicators | Recommendation JSON | Final decision |
