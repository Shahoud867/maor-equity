# Diagram 4: Sequence Diagram — End-to-End Pipeline Execution

**Description:**
Shows the full temporal ordering of all agent interactions for a single ticker (e.g., AAPL 8-K filing).
Phase A parallelism is the key distributed computing contribution: FinBERT (Node B GPU) and TechnicalAgent (Node A CPU) execute concurrently via `ray.get([ref_a, ref_b])`.
`flush_gpu_cache()` is the critical synchronization barrier — clears FinBERT KV cache so Phi-3-mini can load without OOM.
The Guardrail's dual-prompt pattern (Bull first, Bear second, then rule-based resolution) is the final decision gate.
Total estimated wall-clock time: ~489s median (AAPL: ~240s, MSFT: ~738s).

---

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User / run_pipeline.py
    participant ORC as 🎯 Orchestrator<br/>(Node A — Ray Head)
    participant ING as 📄 IngestionAgent<br/>(Node A — CPU Actor)
    participant CF as 🔍 ChunkFilter<br/>(Node A — in-process)
    participant FB as 🧠 FinBERTBundle<br/>(Node B — GPU Actor)
    participant TA as 📊 TechnicalAgent<br/>(Node A — CPU Actor)
    participant SUM as 📋 SummarizationAgent<br/>(Node B — GPU Actor)
    participant GA as 🛡️ GuardrailAgent<br/>(Node B — GPU Actor)
    participant LOG as 💾 logs/ JSON

    User->>ORC: run_pipeline("AAPL", "8-K")
    Note over ORC: Start timer · init VRAM trace

    rect rgb(227, 242, 253)
        Note over ORC,CF: INGESTION PHASE — Node A CPU
        ORC->>ING: download_filing.remote("AAPL", "8-K")
        ING-->>ORC: raw_text (~56K tokens, HTML-cleaned)

        ORC->>CF: chunk_filter(raw_text)
        Note over CF: 1. Sliding window: 512 tokens, stride 64<br/>2. TF-IDF cosine similarity matrix<br/>3. Greedy dedup threshold θ = 0.85<br/>58 chunks → 12 chunks  (~690s workload eliminated)
        CF-->>ORC: filtered_chunks[12] + dimension_map
    end

    rect rgb(243, 229, 245)
        Note over ORC,TA: PHASE A — Parallel Execution (Ray async)
        ORC->>FB: ref_fb = classify_all.remote(filtered_chunks, dim_map)
        Note over FB: Market FinBERT → 12 chunks<br/>Regulatory FinBERT → 12 chunks<br/>Temporal FinBERT → 12 chunks<br/>Aggregate → M[3×3] matrix<br/>~15.7s · 525 MB VRAM
        ORC->>TA: ref_ta = compute_indicators.remote("AAPL")
        Note over TA: RSI-14: prices[-14:]<br/>MACD: EMA12 - EMA26 + signal<br/>Bollinger: SMA20 ± 2σ<br/>yfinance 90-day · ~4.98s

        ORC->>ORC: ray.get([ref_fb, ref_ta])  ← PARALLEL BARRIER
        FB-->>ORC: sentiment_matrix M[3×3]<br/>{market: [+0.72, n0.18, -0.10],<br/> regulatory: [...], temporal: [...]}
        TA-->>ORC: {rsi: 58.3, macd: +0.23, bb_pct: 0.61}
    end

    rect rgb(255, 243, 224)
        Note over ORC,FB: GPU CACHE FLUSH — VRAM Serialization Point
        ORC->>FB: flush_gpu_cache.remote()
        Note over FB: torch.cuda.empty_cache()<br/>del model_market, model_regulatory, model_temporal<br/>gc.collect()<br/>VRAM: 3,261 MB → 2,736 MB  (headroom: 835 MB freed)
        FB-->>ORC: flush_complete
    end

    rect rgb(232, 245, 233)
        Note over ORC,SUM: PHASE B — Sequential Summarization (Node B GPU)
        ORC->>SUM: process.remote(filtered_chunks)
        loop Map Step — 12 chunks
            SUM->>SUM: phi3.generate(map_prompt + chunk_i)
            Note over SUM: Phi-3-mini 4-bit NF4<br/>max_new_tokens: 150<br/>per-chunk mini-summary
        end
        SUM->>SUM: phi3.generate(reduce_prompt + all_summaries)
        Note over SUM: Final structured summary<br/>+ [CONFLICT] tags where detected<br/>~353.5s total · 2,736 MB VRAM
        SUM-->>ORC: structured_summary (string)
    end

    rect rgb(252, 228, 236)
        Note over ORC,GA: GUARDRAIL PHASE — Dual-Prompt Arbitration (Node B GPU)
        ORC->>GA: arbitrate.remote(sentiment_matrix, summary, indicators)
        GA->>GA: phi3.generate(BULL_PROMPT + context)
        Note over GA: Bull assessment JSON:<br/>{direction: BUY, confidence: 0.78, reasoning: ...}
        GA->>GA: phi3.generate(BEAR_PROMPT + context)
        Note over GA: Bear assessment JSON:<br/>{direction: HOLD, confidence: 0.61, reasoning: ...}
        GA->>GA: RuleBasedArbiter.resolve(bull, bear, rsi=58.3, macd=+0.23)
        Note over GA: Consensus check:<br/>RSI not overbought (< 70) → confirms bull<br/>MACD positive → aligned<br/>→ confidence: HIGH
        GA-->>ORC: {direction: BUY, confidence: HIGH,<br/>sentiment_vector: [...], flags: [], risk_notes: [...]}
    end

    ORC->>LOG: write h1_latency_results.json
    ORC->>LOG: write vram_verify.json
    ORC-->>User: Final recommendation dict<br/>{direction, confidence, sentiment_matrix,<br/>summary, indicators, timings, vram_trace}

    Note over User,LOG: Total wall-clock (AAPL estimated): ~240s<br/>Speedup vs B1 serial baseline: 1.72×
```

---

**Timing Breakdown (AAPL Estimated):**

| Phase | Actor | Node | Duration |
|-------|-------|------|----------|
| Ingestion + ChunkFilter | IngestionAgent | A (CPU) | ~45s |
| Phase A: FinBERT (parallel) | FinBERTBundle | B (GPU) | ~15.7s |
| Phase A: Technical (parallel) | TechnicalAgent | A (CPU) | ~4.98s |
| GPU cache flush | FinBERTBundle | B (GPU) | ~2s |
| Phase B: Summarization | SummarizationAgent | B (GPU) | ~353.5s |
| Guardrail arbitration | GuardrailAgent | B (GPU) | ~15s |
| Ray communication overhead | Object Store | A↔B | ~250ms total |
| **Total (AAPL estimated)** | | | **~240s** |
| **B1 Serial baseline (AAPL)** | | | **414.3s** |
| **Speedup** | | | **1.72×** |
