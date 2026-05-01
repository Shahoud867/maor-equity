# Diagram 2: Component / Module Diagram

**Description:**
Illustrates the internal structure of each agent module and their typed interfaces.
Every agent is a persistent Ray Actor — initialized once per run, reused across batch tickers.
The Orchestrator holds remote references to all actors and drives the Phase A → flush → Phase B DAG.
SummarizationAgent and GuardrailAgent share a single `Phi3ModelActor` to eliminate duplicate model loading (~2,736 MB saved).
`BitsAndBytesConfig` (4-bit NF4 + double quantization) is the shared quantization strategy across all GPU actors.

---

```mermaid
classDiagram
    class Orchestrator {
        +run_pipeline(ticker: str, filing_type: str) dict
        +run_pipeline_batch(tickers: list) list
        -_phase_a_parallel(chunks, ticker) tuple
        -_phase_b_serial(chunks) str
        -_collect_timings() dict
        -vram_trace: list[dict]
        -timings: dict[str, float]
        -actors: dict[str, ray.ActorHandle]
    }

    class IngestionAgent {
        +download_filing(ticker: str, form: str) str
        +chunk_document(text: str) list[str]
        +chunk_filter(chunks: list) list[str]
        +tag_dimensions(chunks: list) dict
        -chunk_size: int = 512
        -stride: int = 64
        -cap_8k: int = 12
        -cap_10k: int = 20
        -tfidf: TfidfVectorizer
    }

    class ChunkFilter {
        +filter(chunks: list) list[str]
        +tfidf_dedup(chunks: list, threshold: float) list
        -vectorizer: TfidfVectorizer
        -similarity_threshold: float = 0.85
        -AAPL_raw: 58
        -AAPL_filtered: 12
        -MSFT_raw: 117
        -MSFT_filtered: 12
    }

    class FinBERTBundle {
        +classify_all(chunks: list, dim_map: dict) dict
        +classify_market(text: str) SentimentResult
        +classify_regulatory(text: str) SentimentResult
        +classify_temporal(text: str) SentimentResult
        +build_matrix() ndarray
        -model_market: AutoModelForSeqClass [4-bit NF4]
        -model_regulatory: AutoModelForSeqClass [4-bit NF4]
        -model_temporal: AutoModelForSeqClass [4-bit NF4]
        -quantization_cfg: BitsAndBytesConfig
        -device: cuda = 0
        -VRAM_usage: 525 MB
    }

    class SummarizationAgent {
        +process(chunks: list[str]) str
        +map_step(chunk: str) str
        +reduce_step(summaries: list[str]) str
        -phi3_actor: Phi3ModelActor
        -map_prompt_template: str
        -reduce_prompt_template: str
        -avg_time_per_run: 353.5s
    }

    class TechnicalAnalysisAgent {
        +compute_indicators(ticker: str) dict
        +rsi(prices: Series, period: int) float
        +macd(prices: Series) tuple[float, float]
        +bollinger_bands(prices: Series) tuple[float, float, float]
        -data_source: yfinance
        -lookback_days: int = 90
        -rsi_period: int = 14
        -avg_time: 4.98s
    }

    class GuardrailAgent {
        +arbitrate(sentiment: dict, summary: str, indicators: dict) dict
        +bull_assessment(context: dict) dict
        +bear_assessment(context: dict) dict
        +resolve_conflict(bull: dict, bear: dict) dict
        -phi3_actor: Phi3ModelActor
        -arbiter: RuleBasedArbiter
        -confidence_levels: HIGH · MED · UNRESOLVED
    }

    class Phi3ModelActor {
        +generate(prompt: str, max_new_tokens: int) str
        +warm_up() None
        -model: AutoModelForCausalLM [4-bit NF4]
        -tokenizer: AutoTokenizer
        -device: cuda = 0
        -VRAM_usage: 2736 MB
        -model_id: microsoft/Phi-3-mini-4k-instruct
    }

    class RuleBasedArbiter {
        +resolve(bull: dict, bear: dict, rsi: float, macd: float) dict
        +HIGH_CONF: str = consensus
        +MED_CONF: str = technical_aligned
        +UNRESOLVED: str = conflict_flag
        -rsi_overbought: float = 70.0
        -rsi_oversold: float = 30.0
    }

    class BitsAndBytesConfig {
        +load_in_4bit: bool = True
        +bnb_4bit_quant_type: str = nf4
        +bnb_4bit_use_double_quant: bool = True
        +bnb_4bit_compute_dtype: dtype = float16
    }

    Orchestrator --> IngestionAgent : "Ray.remote() ·  Node A CPU"
    Orchestrator --> FinBERTBundle : "Ray.remote() · Node B GPU"
    Orchestrator --> SummarizationAgent : "Ray.remote() · Node B GPU"
    Orchestrator --> TechnicalAnalysisAgent : "Ray.remote() · Node A CPU"
    Orchestrator --> GuardrailAgent : "Ray.remote() · Node B GPU"
    IngestionAgent --> ChunkFilter : "delegates"
    SummarizationAgent --> Phi3ModelActor : "shared actor handle"
    GuardrailAgent --> Phi3ModelActor : "shared actor handle"
    GuardrailAgent --> RuleBasedArbiter : "uses"
    FinBERTBundle ..> BitsAndBytesConfig : "4-bit NF4 quantization"
    Phi3ModelActor ..> BitsAndBytesConfig : "4-bit NF4 quantization"
```

---

**Actor Placement Summary:**

| Actor | Node | Resource | VRAM |
|-------|------|----------|------|
| IngestionAgent | Node A | CPU | — |
| TechnicalAnalysisAgent | Node A | CPU | — |
| FinBERTBundle | Node B | GPU (cuda:0) | 525 MB |
| SummarizationAgent | Node B | GPU (cuda:0) | shared via Phi3ModelActor |
| GuardrailAgent | Node B | GPU (cuda:0) | shared via Phi3ModelActor |
| **Phi3ModelActor** (shared) | Node B | GPU (cuda:0) | **2,736 MB** |
