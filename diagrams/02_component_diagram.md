# Diagram 2: Component / Module Diagram

This component view maps the main Python modules and actor relationships in the current codebase.
It emphasizes interface-level dependencies rather than infrastructure, making it useful for design reviews.
The orchestrator is the control plane; all agents remain specialized and loosely coupled.
The most important decision is a shared `Phi3ModelActor` used by both summarization and guardrail paths.

```mermaid
classDiagram
        class run_pipeline_py {
            +main()
            +ray.init(runtime_env)
        }

        class orchestrator_py {
            +run_pipeline(ticker, filing_type) dict
            +run_pipeline_batch(tickers, filing_type) list
            -_vram_mb() float
        }

        class IngestionAgent {
            +fetch_filing(ticker, filing_type, limit) dict
            +chunk_document(text, chunk_size, stride) list
        }

        class ChunkFilter {
            +filter(chunks, filing_type) dict
        }

        class DimensionRouter {
            +route(chunks) dict
        }

        class FinBERTBundle {
            +classify_all(market_texts, reg_texts, tmp_texts) tuple
            +flush_gpu_cache() bool
        }

        class SummarizationAgent {
            +process_document(chunks, doc_id) dict
            +map_chunk(chunk, doc_id) dict
            +reduce(summaries, depth) dict
        }

        class TechnicalAnalysisAgent {
            +compute_indicators(ticker, period) dict
        }

        class GuardrailAgent {
            +assess(summary, sentiment_vector, tech) dict
            -_arbitrate(bull, bear, tech) dict
        }

        class Phi3ModelActor {
            +generate(prompt, max_new_tokens) str
        }

        run_pipeline_py --> orchestrator_py : invokes
        orchestrator_py --> IngestionAgent : Ray actor
        orchestrator_py --> ChunkFilter : local module
        orchestrator_py --> DimensionRouter : local module
        orchestrator_py --> FinBERTBundle : Ray actor
        orchestrator_py --> SummarizationAgent : Ray actor
        orchestrator_py --> TechnicalAnalysisAgent : Ray actor
        orchestrator_py --> GuardrailAgent : Ray actor
        SummarizationAgent --> Phi3ModelActor : shared handle
        GuardrailAgent --> Phi3ModelActor : shared handle
```

Key design decisions:
- Keep orchestration logic centralized in `agents/orchestrator.py` for deterministic phase control.
- Use small, purpose-built actors with explicit responsibilities to reduce coupling.
- Reuse the same Phi-3 actor across downstream modules to optimize VRAM and startup overhead.
