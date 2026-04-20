# Agentic Multi-Agent Orchestration for Automated Equity Research

> **NLP + PDC Course Project** | Two-node heterogeneous distributed system
> **Node A** (i232515) — CPU workstation, Ray head, orchestration
> **Node B** (i232634) — NVIDIA T1000 GPU, NLP inference, evaluation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Node Responsibilities](#4-node-responsibilities)
5. [Pipeline — Stage by Stage](#5-pipeline--stage-by-stage)
6. [Agent Reference](#6-agent-reference)
7. [Communication Protocol](#7-communication-protocol)
8. [Chunk Filtering Optimization](#8-chunk-filtering-optimization)
9. [VRAM Budget](#9-vram-budget)
10. [Guardrail Protocol](#10-guardrail-protocol)
11. [Baselines](#11-baselines)
12. [Hypotheses and Evaluation](#12-hypotheses-and-evaluation)
13. [Setup — Node A](#13-setup--node-a)
14. [Setup — Node B](#14-setup--node-b-read-this-carefully)
15. [Running the Pipeline](#15-running-the-pipeline)
16. [Evaluation Commands](#16-evaluation-commands)
17. [Expected Results](#17-expected-results)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Project Overview

Traditional equity research requires analysts to synthesise earnings call transcripts, SEC regulatory filings, market sentiment, and technical price indicators — a slow, manual, multi-modal process. This project automates that workflow using a **distributed multi-agent system** where specialised AI agents handle each analytical dimension in parallel across two physical machines connected over the internet.

### The Four Problems Being Solved

| # | Problem | Our Solution |
|---|---|---|
| 1 | Monolithic single-model architectures cannot parallelise across hardware | 5-agent DAG across 2 nodes via Ray Core |
| 2 | Scalar sentiment scores discard critical dimensional context | 3-D FinBERT pipeline (Market × Regulatory × Temporal) |
| 3 | Long financial documents exceed single-GPU context windows | Chunked map-reduce summarisation with Phi-3-mini |
| 4 | No cross-validation means errors propagate unchecked | Single-round Bull/Bear guardrail with rule-based arbiter |

### Key Design Constraints

- **Node B GPU**: NVIDIA T1000, exactly 4 GB VRAM — no more
- **No fine-tuning**: Only pre-trained HuggingFace checkpoints (4-bit NF4 quantised)
- **No LangChain/CrewAI overhead**: Pure Ray Core for inter-node orchestration
- **One month timeline**: Architecture scoped to be demonstrably complete, not theoretically perfect

---

## 2. System Architecture

### 2.1 Two-Node Cluster Overview

```
╔══════════════════════════════════════════╗        ╔══════════════════════════════════════════════╗
║           NODE A  (i232515)              ║        ║              NODE B  (i232634)               ║
║           CPU Workstation                ║        ║         NVIDIA T1000  —  4 GB VRAM           ║
║           WSL2 / Ubuntu                  ║        ║         WSL2 / Ubuntu                        ║
║                                          ║        ║                                              ║
║  ┌──────────────────────────────────┐    ║        ║  ┌────────────────────────────────────────┐  ║
║  │   Ray HEAD NODE   :6379          │    ║        ║  │       Ray WORKER NODE                  │  ║
║  │   Dashboard       :8265          │◄───╫──TCP───╫─►│       Connected to head                │  ║
║  └──────────────────────────────────┘    ║        ║  └────────────────────────────────────────┘  ║
║                                          ║        ║                                              ║
║  ┌──────────────────┐                    ║        ║  ┌─────────────────┐  ┌──────────────────┐   ║
║  │  IngestionAgent  │  SEC EDGAR         ║        ║  │ FinBERTActor    │  │ FinBERTActor     │   ║
║  │  (CPU, 1 core)   │  scraping          ║        ║  │ "market"        │  │ "regulatory"     │   ║
║  └──────────────────┘                    ║        ║  │ 0.3 GPU         │  │ 0.3 GPU          │   ║
║                                          ║        ║  └─────────────────┘  └──────────────────┘   ║
║  ┌──────────────────┐                    ║        ║                                              ║
║  │  TechnicalAgent  │  RSI, MACD,        ║        ║  ┌─────────────────┐                         ║
║  │  (CPU, 1 core)   │  Bollinger, VWAP   ║        ║  │ FinBERTActor    │  (3 actors share        ║
║  └──────────────────┘                    ║        ║  │ "temporal"      │   one GPU via           ║
║                                          ║        ║  │ 0.3 GPU         │   Ray fractional        ║
║  ┌──────────────────┐                    ║        ║  └─────────────────┘   allocation)           ║
║  │  Orchestrator    │  DAG scheduling,   ║        ║                                              ║
║  │  ChunkFilter     │  Tcomm profiling   ║        ║  ┌──────────────────────────────────────┐    ║
║  │  DimensionRouter │  (all CPU)         ║        ║  │       Phi3ModelActor  (SHARED)        │    ║
║  └──────────────────┘                    ║        ║  │  microsoft/Phi-3-mini-4k-instruct     │    ║
║                                          ║        ║  │  4-bit NF4  —  ~2,100 MB VRAM         │    ║
╚══════════════════════════════════════════╝        ║  │  0.2 GPU allocation                   │    ║
                   │                                ║  └──────────┬───────────────┬────────────┘    ║
                   │                                ║             │               │                 ║
          Tailscale VPN Tunnel                          ║  ┌──────────▼──────┐  ┌────▼────────────┐    ║
          (encrypted, port-                         ║  │ Summarization   │  │ GuardrailAgent  │    ║
           forwarded over                           ║  │ Agent           │  │ (Bull/Bear +    │    ║
           internet NAT)                            ║  │ map-reduce      │  │  Risk Arbiter)  │    ║
                   │                                ║  │ (no extra GPU)  │  │ (no extra GPU)  │    ║
                   │                                ║  └─────────────────┘  └─────────────────┘    ║
                   └────────────────────────────────╚══════════════════════════════════════════════╝

Ray Object Store: NumPy arrays (sentiment vectors, chunk lists, indicator dicts)
                  transferred cross-node only when consuming actor is on the other node.
```

### 2.2 DAG Execution Flow

```
                        ┌─────────────────────────────────────────────────────┐
                        │                   NODE A (CPU)                      │
                        │                                                     │
  ticker, filing_type   │   ┌─────────────┐    raw text                      │
  ──────────────────────┼──►│ Ingestion   ├──────────────►┌───────────────┐  │
                        │   │ Agent       │               │ Chunk         │  │
                        │   │             │  ┌────────────┤ Document      │  │
                        │   └─────────────┘  │            │ (512 tok,     │  │
                        │                    │            │  64 stride)   │  │
                        │                    │            └───────┬───────┘  │
                        │                    │                    │          │
                        │                    │      chunks_raw    │          │
                        │                    │                    ▼          │
                        │                    │  ┌─────────────────────────┐  │
                        │                    │  │   ChunkFilter  (CPU)    │  │
                        │                    │  │   TF-IDF deduplication  │  │
                        │                    │  │   ~20 chunks → ~10      │  │
                        │                    │  └────────────┬────────────┘  │
                        │                    │               │ filtered chunks│
                        │                    │               ▼               │
                        │                    │  ┌────────────────────────┐   │
                        │                    │  │  DimensionRouter (CPU) │   │
                        │                    │  │  market / reg / tmp    │   │
                        │                    │  └──┬──────────┬──────────┘   │
                        │                    │     │          │              │
                        └────────────────────┼─────┼──────────┼──────────────┘
                                             │     │          │
                    ╔════════════════════════╪═════╪══════════╪═══════════════════════╗
                    ║  PARALLEL FAN-OUT      │     │          │    NODE B (GPU)        ║
                    ║  ← Core PDC Contribution     │          │                        ║
                    ║                        │     │          │                        ║
                    ║   ┌────────────────────┘     │          │                        ║
                    ║   │ market chunks             │          │                        ║
                    ║   ▼                           │          │                        ║
                    ║  ┌──────────────┐   ┌─────────▼──────┐  │  ┌──────────────────┐ ║
                    ║  │ FinBERT      │   │ FinBERT        │  │  │ FinBERT          │ ║
                    ║  │ "market"     │   │ "regulatory"   │  │  │ "temporal"       │ ║
                    ║  │ (parallel)   │   │ (parallel)     │  │  │ (parallel)       │ ║
                    ║  └──────┬───────┘   └───────┬────────┘  │  └────────┬─────────┘ ║
                    ║         │                   │           │            │            ║
                    ║         └──────────┬────────┘           │            │            ║
                    ║                    │                    │            │            ║
                    ║         ┌──────────▼──────────┐         │            │            ║
                    ║         │ aggregate_sentiment_ │◄────────┘◄───────────┘            ║
                    ║         │ vector()             │                                   ║
                    ║         │  s ∈ R^(3×3)        │  ╔═══════════════════════════╗    ║
                    ║         └──────────┬───────────┘  ║ ALSO RUNNING IN PARALLEL ║    ║
                    ║                    │              ║                           ║    ║
                    ║                    │              ║ ┌─────────────────────┐   ║    ║
                    ║                    │              ║ │ SummarizationAgent  │   ║    ║
                    ║                    │              ║ │ map: each chunk →   │   ║    ║
                    ║                    │              ║ │ Phi3ModelActor      │   ║    ║
                    ║                    │              ║ │ reduce: final summ. │   ║    ║
                    ║                    │              ║ └──────────┬──────────┘   ║    ║
                    ║                    │              ║            │              ║    ║
                    ║                    │              ║ ┌──────────▼──────────┐   ║    ║
                    ║                    │  (on Node A) ║ │ TechnicalAgent      │   ║    ║
                    ║                    │              ║ │ RSI, MACD, BB, VWAP │   ║    ║
                    ║                    │              ║ └──────────┬──────────┘   ║    ║
                    ║                    │              ╚════════════╪══════════════╝    ║
                    ║                    │                           │                   ║
                    ║         ┌──────────▼───────────────────────────▼─────────┐        ║
                    ║         │              GuardrailAgent                     │        ║
                    ║         │   Bull prompt ──► Phi3ModelActor ──► JSON       │        ║
                    ║         │   Bear prompt ──► Phi3ModelActor ──► JSON       │        ║
                    ║         │   Risk Arbiter (deterministic, no LLM)          │        ║
                    ║         └──────────────────────┬──────────────────────────┘        ║
                    ╚════════════════════════════════╪══════════════════════════════════╝
                                                     │
                                          ┌──────────▼──────────────────────┐
                                          │  Final Output JSON               │
                                          │  • recommendation: BULLISH/      │
                                          │    BEARISH / UNRESOLVED          │
                                          │  • confidence: HIGH/MEDIUM/LOW  │
                                          │  • sentiment_vector: [3×3]       │
                                          │  • summary: str                  │
                                          │  • technical: {rsi, macd, ...}  │
                                          │  • timings: {Tcomm breakdown}    │
                                          └─────────────────────────────────┘
```

### 2.3 Communication Data Flow

```
  NODE A                                              NODE B
  ──────                                              ──────

  routed["market"]  ──► ray.put() ──► object_ref ──► sent_mkt.classify_batch.remote(ref)
                         │                                │
                     [Tserialize]                    [Ttransfer]
                         │                                │
                     Tcomm formula:                  [Tdeserialize]
                                                          │
  Tcomm = Tencode + Tserialize + Ttransfer + Tdeserialize + Tdecode

  Expected ranges (LAN / Tailscale):
    Tencode      ≈  1–3 ms    (NumPy array creation)
    Tserialize   ≈  3–8 ms    (Ray object-store put)
    Ttransfer    ≈  4–20 ms   (cross-node Tailscale VPN)
    Tdeserialize ≈  3–8 ms    (Ray object-store get on Node B)
    Tdecode      ≈  1–3 ms    (array reconstruction)
    ─────────────────────────
    Total Tcomm  ≈ 12–42 ms  per inter-node message

  Payload sizes (compact NumPy vs raw text):
    Raw text chunks:          ~80 KB per document
    NumPy sentiment vector:   ~288 bytes  (3×3 float64)
    NumPy indicator dict:     ~200 bytes
    ─────────────────────────────────────────────────
    Estimated bandwidth reduction: ~80% vs sending raw text
```

### 2.4 Serial vs Distributed Latency Model

```
  B1 SERIAL BASELINE (everything sequential on one GPU node):
  ┌──────────┬──────────────┬─────────────────────────┬──────────┬───────────┐
  │ Load     │   FinBERT    │  Phi-3-mini map-reduce  │ Technical│ Guardrail │
  │ models   │   (1-D)      │  20 chunks × 15s = 300s │   CPU    │  2 calls  │
  │ ~90s     │   ~30s       │                         │   ~5s    │   ~30s    │
  └──────────┴──────────────┴─────────────────────────┴──────────┴───────────┘
  Total B1: ~455s

  DISTRIBUTED PIPELINE (Node A + Node B in parallel, models pre-loaded):
  ┌──────────────────────────────────────────────────────────────────┐
  │                     PARALLEL FAN-OUT                             │
  │  FinBERT ×3  ──────────── 30s ──────────────►                   │
  │  Phi-3-mini (filtered: ~10 chunks × 15s) ── 150s ──────────────►│◄── bottleneck
  │  Technical   ──── 5s ─────►                                      │
  └──────────────────────────────────────────────────────────────────┘
  + Guardrail: ~30s
  Total Distributed: ~180s

  Latency reduction: (455 - 180) / 455 ≈ 60%  ✓  (target: 30–50%)

  Three sources of speedup:
  [1] Parallelism:       FinBERT + Technical overlap with Phi-3-mini
  [2] Chunk filtering:   ~20 raw chunks → ~10 filtered (TF-IDF dedup, ~80ms CPU)
  [3] Model pre-loading: Actors stay resident; B1 loads models cold each run (+90s)
```

---

## 3. Repository Structure

```
maor-equity/
│
├── agents/                         Core agent implementations
│   ├── ingestion_agent.py          NODE A — SEC EDGAR download + chunking
│   ├── technical_agent.py          NODE A — RSI, MACD, Bollinger, VWAP
│   ├── sentiment_agent.py          NODE B — 3×FinBERT + DimensionRouter
│   ├── summarization_agent.py      NODE B — Phi-3-mini map-reduce (shared model)
│   ├── guardrail_agent.py          NODE B — Bull/Bear + Phi3ModelActor + arbiter
│   └── orchestrator.py             NODE A — DAG runner, Tcomm profiling
│
├── baselines/                      Serial baselines for H1/H2/H3 comparison
│   ├── b1_serial_pipeline.py       All stages sequential, includes guardrail
│   ├── b2_summarization_baseline.py  Single-pass Phi-3-mini (no chunking)
│   └── b3_sentiment_baseline.py    Scalar 1-D FinBERT market sentiment only
│
├── evaluation/                     Hypothesis evaluation scripts
│   ├── latency_benchmark.py        H1 — distributed vs serial latency
│   ├── rouge_eval.py               H2 — ROUGE-1/2/L + BERTScore on ECTSum
│   ├── sentiment_eval.py           H3 — 3-D vs 1-D direction change rate
│   └── vram_verify.py              VRAM budget verification (run on Node B)
│
├── optimization/
│   └── chunk_filter.py             NODE A — TF-IDF chunk deduplication
│
├── config/
│   └── cluster_config.yaml         Ray cluster + model + VRAM budget config
│
├── scripts/
│   ├── ray_cluster.ps1 -Role A            Launch Ray head via Tailscale (Node A)
│   ├── start_and_check_ray.sh      Ray + health check
│   ├── verify_nodea_env.sh         Environment validation (Node A)
│   └── final_verify_nodea.sh       Pre-demo checklist (Node A)
│
├── run_pipeline.py                 CLI entry point (Node A)
├── verify_cluster.py               Cross-node health check (Node A)
├── setup_nodeB.ps1                 Automated GPU setup script (Node B)
├── requirements-nodeA.txt          Node A dependencies (CPU-only torch)
├── requirements-nodeB.txt          Node B dependencies (CUDA torch)
├── NODE_B_SETUP.md                 Step-by-step Node B setup guide
│
├── data/                           SEC EDGAR filing cache
├── logs/                           Benchmark results (JSON)
└── results/                        Pipeline output files (JSON)
```

---

## 4. Node Responsibilities

### Node A — i232515 (CPU Workstation, You)

**You own:** Infrastructure, orchestration, data ingestion, technical analysis, optimization.

| Component | File | What it does |
|---|---|---|
| Ray head node | `scripts/ray_cluster.ps1 -Role A` | Starts the cluster, connects via Tailscale VPN |
| Ingestion Agent | `agents/ingestion_agent.py` | Downloads SEC filings, strips HTML, tokenises |
| Technical Agent | `agents/technical_agent.py` | Computes RSI, MACD, Bollinger, VWAP from yfinance |
| Orchestrator | `agents/orchestrator.py` | Runs the DAG, measures Tcomm, collects all results |
| Chunk Filter | `optimization/chunk_filter.py` | TF-IDF deduplication before sending to Node B |
| Dimension Router | `agents/sentiment_agent.py` (DimensionRouter class) | Routes chunks to market/regulatory/temporal |
| Evaluation runner | `evaluation/latency_benchmark.py` | Runs H1 benchmark across both nodes |
| ROUGE evaluation | `evaluation/rouge_eval.py` | Calls Node B summarizer, computes ROUGE |

**Your weekly tasks:**

```
Week 1:  Start Ray (Tailscale) → share address with partner → verify_cluster.py shows 2 nodes
Week 2:  Confirm ingestion works (AAPL 8-K) → chunk_filter working → run full pipeline once
Week 3:  Profile Tcomm decomposition → confirm b1_serial_pipeline.py runs on partner's node
Week 4:  Run all 3 evaluation scripts → compile results → write Node A section of report
```

---

### Node B — i232634 (GPU Node, Your Partner)

**Partner owns:** All GPU inference, model quantisation, VRAM management, NLP evaluation.

| Component | File | What it does |
|---|---|---|
| Phi3ModelActor | `agents/guardrail_agent.py` | Loads Phi-3-mini ONCE, shared by summarizer + guardrail |
| FinBERTActor ×3 | `agents/sentiment_agent.py` | Market / Regulatory / Temporal classifiers |
| SummarizationAgent | `agents/summarization_agent.py` | Receives filtered chunks, runs map-reduce via shared Phi-3 |
| GuardrailAgent | `agents/guardrail_agent.py` | Bull + Bear prompts via shared Phi-3, rule-based arbiter |
| VRAM verification | `evaluation/vram_verify.py` | Confirms peak < 4096 MB during full pipeline |
| B1 serial baseline | `baselines/b1_serial_pipeline.py` | Run this on GPU node to record H1 baseline timing |
| ROUGE evaluation | `evaluation/rouge_eval.py` | GPU execution of summarizer on ECTSum samples |
| Sentiment eval | `evaluation/sentiment_eval.py` | GPU execution of FinBERT on Financial PhraseBank |

**Node B weekly tasks (detailed):**

```
Week 1 — Setup:
  □ Install WSL2 + Ubuntu 22.04
  □ Verify nvidia-smi shows T1000 inside WSL
  □ Run setup_nodeB.ps1 (installs CUDA torch, all deps)
  □ Pre-download all models (FinBERT ×2 + Phi-3-mini) — takes ~10 min
  □ Connect Ray worker to Node A's Tailscale IP
  □ Confirm verify_cluster.py on Node A shows 2 nodes + GPU

Week 2 — First pipeline run:
  □ Run evaluation/vram_verify.py → confirm Peak < 4096 MB → PASS
  □ Watch Node A run: python run_pipeline.py --ticker AAPL
  □ GPU should spin up during parallel fan-out stage
  □ Check logs/ for any OOM errors or CUDA errors
  □ Record first end-to-end latency

Week 3 — Baselines:
  □ Run baselines/b1_serial_pipeline.py (your machine, GPU mode)
  □ Record timing to logs/b1_results.json  ← this is the H1 baseline number
  □ Run baselines/b2_summarization_baseline.py (single-pass, no chunking)
  □ Run baselines/b3_sentiment_baseline.py (1-D scalar FinBERT)
  □ Save all baseline results — Node A will compare against these

Week 4 — Evaluation:
  □ GPU memory profiling during full pipeline:
      nvidia-smi dmon -s mu -d 1 > logs/gpu_profile.txt &
      (then run pipeline, then kill the monitor)
  □ Run: python -m evaluation.rouge_eval --n-samples 50
  □ Run: python -m evaluation.sentiment_eval --n-samples 100
  □ Collate all results from logs/ and send to Node A for final report
  □ Write Node B section of report (VRAM table, ROUGE results, H3 findings)
```

---

## 5. Pipeline — Stage by Stage

### Stage 1 — Ingestion (Node A)

```python
# agents/ingestion_agent.py
IngestionAgent.fetch_filing(ticker="AAPL", filing_type="8-K")
```

- Downloads latest SEC EDGAR filing using `sec-edgar-downloader`
- Strips all HTML tags with regex
- Returns clean text with word count

```python
IngestionAgent.chunk_document(text)
```

- Tokenises with Phi-3-mini tokenizer (so chunks match summarizer's vocabulary)
- Sliding window: **512 tokens**, **64-token stride** (87.5% overlap between adjacent chunks)
- Returns list of `{"chunk_id": int, "text": str, "token_count": int, "start_token": int}`

### Stage 1b — Chunk Filtering (Node A, CPU)

```python
# optimization/chunk_filter.py
ChunkFilter().filter(chunks_raw, filing_type="8-K")
```

- TF-IDF vectorises all chunks (sklearn, max 512 features)
- Scores each chunk by total TF-IDF weight (information density)
- Greedy deduplication: discards chunks with cosine similarity > 0.85 to already-selected chunks
- Hard cap: 12 chunks for 8-K, 20 for 10-K/10-Q
- **Cost: ~80ms CPU — essentially free**
- **Benefit: cuts Phi-3-mini map calls by ~50%**

### Stage 2 — Dimension Routing (Node A, instant)

```python
# agents/sentiment_agent.py
DimensionRouter().route(chunks)
# Returns: {"market": [...], "regulatory": [...], "temporal": [...]}
```

Routing rules:

| Dimension | Rule | Keywords |
|---|---|---|
| Market | All chunks go here | (all chunks) |
| Regulatory | Chunks matching compliance/legal triggers | SEC, litigation, compliance, penalty, enforcement, restatement, FDA, FTC |
| Temporal | Chunks with forward-looking statements | will, expect, anticipate, forecast, guidance, outlook, next quarter |

### Stage 3 — Parallel Fan-out (Core PDC Contribution)

Five Ray tasks fire simultaneously and run in parallel across both nodes:

```
Node B GPU:  sent_mkt.classify_batch.remote(market_chunks)   ─┐
Node B GPU:  sent_reg.classify_batch.remote(reg_chunks)       ├── ray.get() waits for ALL
Node B GPU:  sent_tmp.classify_batch.remote(tmp_chunks)       ─┘
Node B GPU:  summarizer.process_document.remote(chunks)        ─── ray.get() waits
Node A CPU:  tech.compute_indicators.remote(ticker)            ─── ray.get() waits
```

This is the heart of the PDC contribution — work that would take 5× sequential steps now happens simultaneously, bottlenecked only by the slowest task (Phi-3-mini summarisation).

### Stage 4 — Sentiment Aggregation (Node A)

```python
sv = aggregate_sentiment_vector(mkt_result, reg_result, tmp_result)
# sv.shape = (3, 3)
# sv[0] = [market_pos,  market_neu,  market_neg]
# sv[1] = [reg_pos,     reg_neu,     reg_neg]
# sv[2] = [temporal_pos,temporal_neu, temporal_neg]
```

Each row sums to 1.0. This is the structured output `s ∈ R^(3×3)` from the report.

### Stage 5 — Guardrail (Node B)

See [Section 10](#10-guardrail-protocol) for full detail.

---

## 6. Agent Reference

### FinBERTActor (Node B)
```
@ray.remote(num_gpus=0.3)
Checkpoints:
  market:     ProsusAI/finbert
  regulatory: yiyanghkust/finbert-tone
  temporal:   ProsusAI/finbert  (separate actor instance)

Input:  list of text strings
Output: {"dimension": str, "scores": [{"positive": f, "neutral": f, "negative": f}, ...],
          "n_texts": int, "n_ambiguous": int}

Confidence threshold: 0.60 — scores below this marked "ambiguous"
Batch size: 8, max_length: 512 tokens, truncation: True
Quantisation: 4-bit NF4 via bitsandbytes (load_in_4bit=True)
```

### Phi3ModelActor (Node B) — SHARED
```
@ray.remote(num_gpus=0.2)
Checkpoint: microsoft/Phi-3-mini-4k-instruct
Quantisation: 4-bit NF4 (load_in_4bit=True, device_map="auto")
CRITICAL: Loaded ONCE. Both SummarizationAgent and GuardrailAgent
          receive a handle to this actor. Loading it twice would
          exceed the 4 GB VRAM budget.

Input:  prompt (str), max_new_tokens (int)
Output: generated text (str)
Config: temperature=0.1, do_sample=False (deterministic greedy)
        max input truncation: 3500 tokens
```

### SummarizationAgent (Node B)
```
@ray.remote(num_gpus=0.0)   # uses shared Phi3ModelActor
Input:  filtered chunks list, doc_id str
Output: {"summary": str, "n_chunks": int, "n_conflicts": int, "doc_id": str}

map_chunk(chunk):
  Prompt: financial analyst summarising one segment
  Extracts: key metrics, directional claims, named entities
  Conflict detection: preserves [CONFLICT] tag if found
  max_new_tokens: 200

reduce(summaries):
  Checks combined length (~1 token per 4 chars heuristic)
  If > 3000 tokens: recursive halving (depth limit 3)
  Final prompt: synthesise into executive summary + metrics + outlook + risk flags
  max_new_tokens: 400
```

### GuardrailAgent (Node B)
```
@ray.remote(num_gpus=0.0)   # uses shared Phi3ModelActor
Input:  summary (str), sentiment_vector (np.ndarray 3×3), technical (dict)
Output: {"recommendation": str, "confidence": str, "bull_score": float,
          "bear_score": float, "conflict": bool, ...}

Two LLM calls (Bull prompt + Bear prompt) → JSON parsing → Rule-based arbiter
See Section 10 for full arbiter logic.
```

### TechnicalAnalysisAgent (Node A)
```
@ray.remote(num_cpus=1)
Input:  ticker (str), period="3mo"
Output: {"rsi": float, "rsi_signal": str, "macd_crossover_bullish": bool,
          "price_vs_upper_band": float, "vwap": float, "current_price": float}

RSI:     14-period, Wilder smoothing
MACD:    (12, 26, 9) — crossover detected on last two bars
Bollinger: 20-period SMA ± 2σ, price vs upper band
VWAP:    volume-weighted average price over full 3mo period
Data source: yfinance
```

### IngestionAgent (Node A)
```
@ray.remote(num_cpus=1)
fetch_filing(ticker, filing_type, limit=1)
  Downloads via sec-edgar-downloader API
  Strips HTML with regex: re.sub(r"<[^>]+>", " ", raw)
  Returns: {"ticker", "filing_type", "text", "word_count", "status"}

chunk_document(text, chunk_size=512, stride=64)
  Tokeniser: microsoft/Phi-3-mini-4k-instruct (same as summarizer)
  Sliding window with overlap
  Returns: list of {"chunk_id", "text", "token_count", "start_token"}
```

---

## 7. Communication Protocol

All inter-node data travels through Ray's distributed object store over the Tailscale VPN tunnel. The orchestrator measures every stage of communication latency using:

```
Tcomm = Tencode + Tserialize + Ttransfer + Tdeserialize + Tdecode
```

**What each term means:**

| Term | Code location | What it measures |
|---|---|---|
| `Tencode` | Before `ray.put()` | NumPy array creation from Python lists |
| `Tserialize` | `ray.put()` call | Pickling + writing to Ray object store |
| `Ttransfer` | Time until `remote()` dispatched | Network latency over Tailscale VPN |
| `Tdeserialize` | `ray.get()` on results | Reading from Ray object store on consumer |
| `Tdecode` | After `ray.get()` | Reconstructing the Python object |

**Data formats sent cross-node:**

```
Chunks to FinBERT:        List[str]     — text strings only, ~3KB for 10 chunks
Chunks to Summarizer:     List[dict]    — chunk_id + text, ~4KB for 10 chunks
Sentiment vector back:    np.ndarray    — (3,3) float64 = 288 bytes
Technical indicators:     dict          — ~200 bytes
Summary text back:        str           — ~2KB
```

The 80% bandwidth reduction claim in the report comes from sending NumPy arrays and structured dicts rather than full natural-language reasoning chains between agents.

---

## 8. Chunk Filtering Optimization

**File:** `optimization/chunk_filter.py`
**Runs on:** Node A (CPU only)
**Purpose:** Reduce Phi-3-mini inference calls on Node B before they happen

### Why it exists

The ingestion agent creates chunks with 64-token stride out of 512-token windows — 87.5% overlap. Adjacent chunks share almost all their content. Without filtering, a 10,000-word 8-K produces ~25 chunks, all nearly identical. Phi-3-mini processes each independently at ~15s/chunk = 375s wasted on redundant content.

### Algorithm

```
Step 1: TF-IDF vectorise all chunks
        TfidfVectorizer(max_features=512, stop_words="english", sublinear_tf=True)

Step 2: Score each chunk
        score(chunk_i) = sum of TF-IDF weights = information density proxy

Step 3: Greedy deduplication
        Sort chunks by score (highest first)
        For each chunk:
          If cosine_similarity(chunk, any_already_selected) > 0.85 → DISCARD
          Else → KEEP

Step 4: Restore document order
        Sort kept chunks by original chunk_id → narrative flow preserved

Hard cap: max_chunks=12 for 8-K, max_chunks=20 for 10-K/10-Q
```

### Fallback

If sklearn is unavailable, falls back to uniform sampling (every k-th chunk). This preserves the hard cap but loses the quality-based selection.

### Impact on H1

```
Before filtering:  ~21 chunks × 15s = 315s Phi-3-mini map time
After filtering:   ~10 chunks × 15s = 150s Phi-3-mini map time
Filter cost:       ~80ms CPU (negligible)
Net saving:        ~165s GPU time freed on Node B
```

---

## 9. VRAM Budget

Node B has exactly **4,096 MB** VRAM. Every component must fit simultaneously.

```
╔══════════════════════════════════════════════════════════╗
║           NODE B VRAM BUDGET  (T1000, 4 GB)             ║
╠══════════════════════════════╦═════════════╦════════════╣
║  Component                   ║  VRAM (MB)  ║  Note      ║
╠══════════════════════════════╬═════════════╬════════════╣
║  FinBERTActor "market"       ║    ~340     ║  4-bit NF4 ║
║  FinBERTActor "regulatory"   ║    ~340     ║  4-bit NF4 ║
║  FinBERTActor "temporal"     ║    ~340     ║  4-bit NF4 ║
╠══════════════════════════════╬═════════════╬════════════╣
║  Phi3ModelActor (SHARED)     ║   ~2,100    ║  4-bit NF4 ║
║  (used by Summarizer         ║             ║  loaded    ║
║   AND Guardrail)             ║             ║  ONCE      ║
╠══════════════════════════════╬═════════════╬════════════╣
║  KV cache + inference bufs   ║    ~400     ║  batch=1   ║
║  Ray actor overhead          ║    ~120     ║            ║
╠══════════════════════════════╬═════════════╬════════════╣
║  PEAK TOTAL                  ║  ~3,640     ║  < 4,096 ✓ ║
║  Headroom                    ║    ~456     ║            ║
╚══════════════════════════════╩═════════════╩════════════╝

CRITICAL NOTE: If Phi-3-mini is loaded TWICE (once for summarizer,
once for guardrail), that is ~4,200 MB for Phi-3 alone → OOM crash.
The fix is Phi3ModelActor: one actor, shared handle passed to both agents.
```

### 8-bit Fallback

If OOM occurs despite the shared model design:

```bash
# In agents/guardrail_agent.py and agents/summarization_agent.py,
# change: load_in_4bit=True
# to:     load_in_8bit=True
# T1000 compute capability 7.5 supports both 4-bit and 8-bit.
# 8-bit uses ~2× VRAM per model but is more stable on some driver versions.
```

---

## 10. Guardrail Protocol

The guardrail is the final gate before a recommendation is issued. It prevents hallucination and conflicting signals from propagating into the output.

```
Input:
  summary          — final synthesised text from SummarizationAgent
  sentiment_vector — s ∈ R^(3×3) from FinBERT pipeline
  technical        — {rsi, macd_crossover_bullish, price_vs_upper_band, vwap}

Step 1 — Independent LLM Assessments (2 Phi-3-mini calls, sequential):

  Bull prompt: "Make the strongest BULL case for this equity."
               + summary excerpt + sentiment scores + RSI + MACD
               → JSON: {"direction": "bullish", "confidence": 0.0–1.0, "signals": [...]}

  Bear prompt: "Make the strongest BEAR case for this equity."
               → JSON: {"direction": "bearish", "confidence": 0.0–1.0, "signals": [...]}

Step 2 — Rule-based Risk Arbiter (deterministic, NO LLM call):

  technical_weight = |RSI - 50| / 50          (0.0 = neutral, 1.0 = extreme)
  technical_direction:
    +1  if RSI < 30 (oversold) OR MACD crossover bullish
    -1  if RSI > 70 (overbought)
     0  otherwise (neutral)

  bull_score = bull_confidence + 0.25 × technical_weight  (if tech_dir > 0)
  bear_score = bear_confidence + 0.25 × technical_weight  (if tech_dir < 0)
  diff = |bull_score - bear_score|

  ┌──────────────────────────────────────────────────────────────────┐
  │  diff > 0.40  →  winning side  +  confidence: HIGH              │
  │  0.25 < diff ≤ 0.40  →  winning side  +  confidence: MEDIUM     │
  │  diff ≤ 0.25  →  UNRESOLVED  +  confidence: LOW                 │
  │                  (no directional recommendation issued)          │
  └──────────────────────────────────────────────────────────────────┘

Output:
  {"recommendation": "BULLISH" | "BEARISH" | "UNRESOLVED",
   "confidence": "HIGH" | "MEDIUM" | "LOW",
   "bull_score": float, "bear_score": float,
   "winning_signals": [...] | "conflict_signals": [...],
   "rsi": float, "conflict": bool}
```

---

## 11. Baselines

Three baselines are required by the report to validate the three hypotheses.

### B1 — Serial Pipeline (for H1 latency)
**File:** `baselines/b1_serial_pipeline.py`
**Run on:** Node B (GPU)
**What it does:** All pipeline stages run sequentially on a single node — ingestion → FinBERT (1-D) → Phi-3-mini summarise → technical → guardrail. Models loaded fresh each run (no pre-loading). This is the fair comparison for H1 because it has the same stages as the distributed pipeline.

```bash
# Run on Node B:
python baselines/b1_serial_pipeline.py
# Outputs: logs/b1_results.json with per-stage and total timings
```

### B2 — Single-pass Summarisation (for H2 ROUGE)
**File:** `baselines/b2_summarization_baseline.py`
**Run on:** Node B (GPU)
**What it does:** Single Phi-3-mini call on truncated text (no chunking, no map-reduce). Represents the naive summarisation approach.

```bash
python baselines/b2_summarization_baseline.py
```

### B3 — Scalar Sentiment (for H3)
**File:** `baselines/b3_sentiment_baseline.py`
**Run on:** Node B (GPU)
**What it does:** Single-dimension ProsusAI/finbert (market only). Returns one positive/negative/neutral label per text. Compare against our 3-D (3×3 matrix) output.

```bash
python baselines/b3_sentiment_baseline.py
```

---

## 12. Hypotheses and Evaluation

### H1 — Latency Reduction

```
Claim:   Distributed pipeline achieves 30–50% lower median latency per document than B1
Metric:  Median total latency (seconds) over 3 tickers (AAPL, MSFT, GOOGL)
Script:  python -m evaluation.latency_benchmark --tickers AAPL MSFT GOOGL
Output:  logs/h1_latency_results.json

Also reports Tcomm decomposition:
  t_encode_ms, t_serialize_ms, t_transfer_ms, t_deserialize_ms, t_comm_total_ms
And chunk filter contribution:
  n_chunks_before, n_chunks_after, reduction_pct, estimated GPU time saved
```

### H2 — Summarisation Quality

```
Claim:   Chunked map-reduce is non-inferior to B2 on ROUGE-L (difference ≤ 1.0 point)
         while handling long documents more stably
Dataset: ECTSum (earnings call transcripts, 2,425 test pairs)
         https://huggingface.co/datasets/mrSoul7766/ECTSum
Metrics: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore-F1
Script:  python -m evaluation.rouge_eval --n-samples 50
Output:  logs/h2_rouge_results.json

Pass condition: (our ROUGE-L) - (B2 ROUGE-L) >= -1.0
```

### H3 — Sentiment Dimensionality

```
Claim:   3-D sentiment changes the directional recommendation in >10% of texts vs B3
Dataset: Financial PhraseBank (sentences_75agree split, ~4,800 sentences)
         https://huggingface.co/datasets/financial_phrasebank
Metric:  % of texts where 3-D direction ≠ B3 scalar direction
         Also: accuracy of each vs ground-truth labels
Script:  python -m evaluation.sentiment_eval --n-samples 100
Output:  logs/h3_sentiment_results.json

Pass condition: disagreement_pct > 10%
```

---

## 13. Setup — Node A

### Prerequisites
- WSL2 with Ubuntu 22.04
- Python 3.10+
- Tailscale account (free tier works): https://tailscale.com/download/windows

### Installation

```bash
# In WSL on your machine:
cd ~/maor-equity
python3 -m venv venv
source venv/bin/activate

# CPU-only torch first (Node A does not need CUDA)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-nodeA.txt
```

### Start the cluster

```bash
source venv/bin/activate
bash scripts/ray_cluster.ps1 -Role A
# Opens Ray head on :6379 and Ray dashboard on :8265
# Starts Tailscale VPN tunnel — check http://localhost:4040 for the address
# Share the address (e.g., 100.x.x.x:6379) with your partner
```

### Verify

```bash
# After partner connects:
python verify_cluster.py
# Expected: "Both nodes alive!" + GPU visible + VRAM budget OK
```

---

## 14. Setup — Node B (Read This Carefully)

> You are **Node B** — the GPU worker. Your machine runs all NLP inference.
> Everything GPU-related lives here. Node A controls the orchestration.
> Read every step before starting. Order of installation matters.

### Prerequisites Checklist

```
□ Windows 10/11 with WSL2 enabled
□ Ubuntu 22.04 installed from Microsoft Store
□ NVIDIA driver ≥ 525 installed on Windows (NOT inside WSL)
□ Python 3.10 available in WSL
□ At least 15 GB free disk space (models are large)
□ Tailscale IP from Node A partner
```

### Step 1 — Verify GPU inside WSL

```bash
nvidia-smi
# Must show: T1000, ~4096 MiB, CUDA Version ≥ 12.1
# If this fails: update NVIDIA Windows driver, then retry
```

If `nvidia-smi` works on Windows but not in WSL:
```bash
# In WSL:
sudo apt update && sudo apt install -y nvidia-cuda-toolkit
# Then retry nvidia-smi
```

### Step 2 — Clone repo and install dependencies

```bash
cd ~
git clone <repo-url> maor-equity
cd maor-equity

python3 -m venv venv
source venv/bin/activate

# CRITICAL: torch with CUDA MUST be installed FIRST, before anything else
pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA works before continuing:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA T1000

# Now install remaining dependencies
pip install -r requirements-nodeB.txt
```

### Step 3 — Pre-download all models

This step runs once and caches models locally (~8 GB total). Do this while your partner sets up Node A.

```bash
source venv/bin/activate
python - <<'EOF'
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

print("Downloading ProsusAI/finbert ...")
pipeline("text-classification", model="ProsusAI/finbert")

print("Downloading yiyanghkust/finbert-tone ...")
pipeline("text-classification", model="yiyanghkust/finbert-tone")

print("Downloading microsoft/Phi-3-mini-4k-instruct ...")
AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    load_in_4bit=True,
    device_map="auto"
)
print("All models downloaded successfully.")
EOF
```

### Step 4 — Connect to Node A's Ray cluster

Wait for your partner to share the Tailscale VPN address. It looks like `100.x.x.x:6379`.

```bash
source venv/bin/activate
export RAY_DISABLE_JEMALLOC=1
ray start --address=<TAILSCALE_IP>:6379
# Example: ray start --address=100.x.x.x:6379
# Expected output: "Ray runtime started. Connected to Ray cluster."
```

### Step 5 — Verify cluster (your partner runs this on Node A)

```bash
# Node A runs:
python verify_cluster.py
# Should show:
#   Active nodes: 2
#   Both nodes alive!
#   SUCCESS: NVIDIA T1000  4096 MB VRAM
#   Budget: OK
```

### Step 6 — Verify VRAM budget (you run this)

```bash
python evaluation/vram_verify.py
# Should show: Peak < 4096 MB — PASS
# Saves result to logs/vram_verify.json
```

### Step 7 — Watch the first pipeline run

Your partner runs this on Node A:
```bash
python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json
```

On your machine, monitor GPU usage:
```bash
watch -n 1 nvidia-smi
# You should see GPU utilisation spike when FinBERT and Phi-3-mini are active
```

### Step 8 — Run serial baselines (your main contribution this week)

```bash
# H1 baseline — MUST run on GPU node for fair comparison:
python baselines/b1_serial_pipeline.py
# Records: logs/b1_results.json

# H2 baseline:
python baselines/b2_summarization_baseline.py

# H3 baseline:
python baselines/b3_sentiment_baseline.py
```

### Step 9 — Run evaluation (Week 4)

First install evaluation extras:
```bash
pip install datasets rouge-score bert-score
```

```bash
# GPU profiling during full pipeline:
nvidia-smi dmon -s mu -d 1 > logs/gpu_profile.txt &
python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl_eval.json
kill %1

# H2 — ROUGE evaluation on ECTSum:
python -m evaluation.rouge_eval --n-samples 50
# Output: logs/h2_rouge_results.json

# H3 — Sentiment evaluation on Financial PhraseBank:
python -m evaluation.sentiment_eval --n-samples 100
# Output: logs/h3_sentiment_results.json
```

Send all JSON files in `logs/` to Node A for final report compilation.

---

## 15. Running the Pipeline

### Single document

```bash
# On Node A (cluster must be running):
source venv/bin/activate
python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json
python run_pipeline.py --ticker MSFT --filing 8-K --output results/msft.json
python run_pipeline.py --ticker GOOGL --filing 8-K --output results/googl.json
```

### Output format

```json
{
  "ticker": "AAPL",
  "filing_type": "8-K",
  "n_chunks_raw": 21,
  "n_chunks_filtered": 10,
  "chunk_reduction_pct": 52.4,
  "sentiment_vector": [
    [0.72, 0.18, 0.10],
    [0.41, 0.44, 0.15],
    [0.65, 0.23, 0.12]
  ],
  "summary": {
    "summary": "Apple reported...",
    "n_chunks": 10,
    "n_conflicts": 1,
    "doc_id": "AAPL_8-K"
  },
  "technical": {
    "ticker": "AAPL",
    "rsi": 58.3,
    "rsi_signal": "neutral",
    "macd_crossover_bullish": true,
    "price_vs_upper_band": -4.21,
    "vwap": 187.42,
    "current_price": 191.05
  },
  "guardrail": {
    "recommendation": "BULLISH",
    "confidence": "HIGH",
    "bull_score": 0.812,
    "bear_score": 0.341,
    "winning_signals": ["Revenue beat", "MACD crossover", "Strong temporal sentiment"],
    "rsi": 58.3,
    "conflict": false
  },
  "timings": {
    "ingestion": 12.4,
    "chunk_filter_ms": 83.2,
    "n_chunks_before_filter": 21,
    "n_chunks_after_filter": 10,
    "chunk_reduction_pct": 52.4,
    "t_encode_ms": 1.2,
    "t_serialize_ms": 4.8,
    "t_transfer_ms": 12.3,
    "t_deserialize_ms": 5.1,
    "t_comm_total_ms": 23.4,
    "parallel_stage": 162.8,
    "guardrail": 31.2,
    "total": 207.6
  }
}
```

---

## 16. Evaluation Commands

Run these in order. Each depends on the cluster being connected.

```bash
# Step 1 — Verify cluster health
python verify_cluster.py

# Step 2 — VRAM budget (Node B runs this)
python evaluation/vram_verify.py

# Step 3 — First end-to-end run
python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json

# Step 4 — Serial baselines (Node B runs b1; Node A can run b2 and b3 via Ray)
python baselines/b1_serial_pipeline.py      # Node B
python baselines/b2_summarization_baseline.py
python baselines/b3_sentiment_baseline.py

# Step 5 — H1 latency benchmark
python -m evaluation.latency_benchmark --tickers AAPL MSFT GOOGL
# To skip re-running serial baseline: --skip-serial

# Step 6 — H2 ROUGE (Node B GPU executes summarizer)
python -m evaluation.rouge_eval --n-samples 50
# To skip re-running B2: --skip-b2

# Step 7 — H3 sentiment (Node B GPU executes FinBERT)
python -m evaluation.sentiment_eval --n-samples 100
# To skip re-running B3: --skip-b3

# All results land in logs/:
ls logs/
# h1_latency_results.json
# h2_rouge_results.json
# h3_sentiment_results.json
# vram_verify.json
# b1_results.json
# b2_predictions.json
# b3_predictions.json
# gpu_profile.txt
```

---

## 17. Expected Results

These are estimates based on T1000 4-bit inference benchmarks. Actual numbers may vary ±20%.

### H1 Latency

| Method | Median latency | Notes |
|---|---|---|
| B1 Serial | ~450s | Includes model cold load (~90s) |
| Distributed | ~180s | Pre-loaded actors, filtered chunks |
| **Reduction** | **~60%** | Target: 30–50% → should pass |

Tcomm contribution:
```
t_encode:      ~1–3 ms
t_serialize:   ~3–8 ms
t_transfer:    ~4–20 ms   (Tailscale adds ~10ms vs LAN)
t_deserialize: ~3–8 ms
Total Tcomm:   ~11–39 ms  (<< pipeline time, shows compression is efficient)
```

### H2 ROUGE on ECTSum

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| B2 Single-pass | ~28 | ~8 | ~23 | ~0.85 |
| Our map-reduce | ~29 | ~9 | ~24 | ~0.86 |
| **Difference** | **+1** | **+1** | **+1** | **+0.01** |

Pass condition: ROUGE-L difference ≥ -1.0 → Expected: PASS (positive difference expected)

### H3 Sentiment Disagreement

| Metric | Expected | Notes |
|---|---|---|
| Direction changes vs B3 | ~15–25% | Regulatory + temporal dimensions shift calls |
| 3-D accuracy on PhraseBank | ~72% | vs B3 scalar ~68% |
| Pass condition (>10%) | **PASS** | |

### Chunk Filtering

| Filing type | Avg chunks before | Avg chunks after | Reduction |
|---|---|---|---|
| 8-K | ~21 | ~10 | ~52% |
| 10-K | ~85 | ~20 | ~76% |

---

## 18. Troubleshooting

### Node B can't connect to Ray

```bash
# Check Tailscale is connected on Node A:
# Node A: curl http://localhost:4040/api/tunnels

# Re-run on Node B:
ray stop
export RAY_DISABLE_JEMALLOC=1
ray start --address=<TAILSCALE_IP>:6379
```

### CUDA not found inside WSL

```bash
# Verify driver version on Windows (PowerShell):
nvidia-smi
# Must show CUDA Version ≥ 12.1

# Inside WSL:
ls /usr/lib/wsl/lib/libcuda*
# If empty: update NVIDIA driver on Windows → restart WSL: wsl --shutdown
```

### OOM / CUDA out of memory

```bash
# Check what's using VRAM:
nvidia-smi

# Run VRAM verifier:
python evaluation/vram_verify.py

# If peak > 4096 MB: switch to 8-bit
# In agents/guardrail_agent.py, Phi3ModelActor.__init__:
# Change: load_in_4bit=True → load_in_8bit=True
# Note: 8-bit Phi-3-mini uses ~3,200 MB → total becomes ~4,560 MB → still OOM
# Better fix: reduce FinBERT to load_in_8bit (340 → 680 MB each):
#   total = 3×680 + 2100 + 400 + 120 = 4,660 MB → still tight
# Real fix: serialise loading (FinBERT first, then Phi-3-mini)
```

### bitsandbytes error on import

```bash
# Symptom: "CUDA Setup failed! CUDA driver version is insufficient"
# Fix: ensure torch+CUDA installed BEFORE bitsandbytes
pip uninstall bitsandbytes
pip install bitsandbytes==0.43.0
```

### Ray actors scheduled on wrong node

```bash
# Check resources visible to Ray:
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
# Should show: CPU cores (Node A) + GPU: 1.0 (Node B)
# FinBERT actors request num_gpus=0.3 → Ray places them on Node B automatically
```

### Phi-3-mini generating garbage / empty JSON

```bash
# The guardrail JSON parsing fails gracefully:
# returns {"direction": "unknown", "confidence": 0.0, "signals": []}
# This triggers "UNRESOLVED" in the arbiter — not a crash
# To debug: add print(txt) inside _gen_json() in guardrail_agent.py
```

### Tailscale connection issue on demo day

Use the VirtualBox fallback (both nodes on same LAN):

```bash
# Node A:
ray start --head --port=6379

# Node B (same network, direct connection):
ray start --address=<NODE_A_LOCAL_IP>:6379
# Find Node A's local IP: ip addr show | grep inet
```

---

## Quick Reference Card

```
═══════════════════════════════════════════════════════════════════
                    STARTUP SEQUENCE
═══════════════════════════════════════════════════════════════════

NODE A (do first):
  bash scripts/ray_cluster.ps1 -Role A
  → Share Tailscale IP with Node B partner

NODE B (after Node A is ready):
  source venv/bin/activate && export RAY_DISABLE_JEMALLOC=1
  ray start --address=<TAILSCALE_IP>:6379

VERIFY (Node A):
  python verify_cluster.py
  → "Both nodes alive! SUCCESS: NVIDIA T1000 4096 MB VRAM"

RUN PIPELINE (Node A):
  python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json

═══════════════════════════════════════════════════════════════════
                    EVALUATION SEQUENCE
═══════════════════════════════════════════════════════════════════

1. python evaluation/vram_verify.py              [Node B]
2. python baselines/b1_serial_pipeline.py        [Node B]
3. python baselines/b2_summarization_baseline.py [Node B]
4. python baselines/b3_sentiment_baseline.py     [Node B]
5. python -m evaluation.latency_benchmark        [Node A]
6. python -m evaluation.rouge_eval               [Node A, GPU on Node B]
7. python -m evaluation.sentiment_eval           [Node A, GPU on Node B]

Results → logs/h1_latency_results.json
          logs/h2_rouge_results.json
          logs/h3_sentiment_results.json
          logs/vram_verify.json

═══════════════════════════════════════════════════════════════════
                    HYPOTHESIS PASS CONDITIONS
═══════════════════════════════════════════════════════════════════

H1: latency_reduction_pct ≥ 30%          (expect ~60%)
H2: ROUGE-L difference   ≥ -1.0 points   (expect +1.0)
H3: disagreement_pct     > 10%           (expect ~20%)
```
