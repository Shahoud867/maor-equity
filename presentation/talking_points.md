# MAOR-EQUITY — Complete Presentation Guide
### Talking Points · Deep Preparation · 10-Minute Delivery Script
**Shahoud Shahid (23i-2515) · Saif Shahzad (23i-2634)**
**Course: PDC + NLP, FAST-NUCES — Spring 2026**

---

# SECTION 1: DETAILED CONCEPTUAL PREPARATION
## (Deep Understanding — Read This Before Presenting)

---

## 1. PROBLEM DEFINITION

### What Are We Actually Solving?

Every time a major US company (Apple, Microsoft, Tesla, etc.) releases a financial update, they must legally file a report with the **SEC (Securities and Exchange Commission)**. These reports — called **8-K** (major events) and **10-K** (annual reports) — are massive text documents, often 50,000 to 200,000 words long.

An investment analyst needs to:
- Read and understand these documents
- Judge the company's financial health
- Decide: **BUY, HOLD, or SELL** that company's stock

Doing this manually is slow and expensive. The dream is to automate it with AI — but the challenge is that these documents are huge, and the AI models needed to understand them are slow and memory-hungry.

### The Core Bottleneck — In Simple Terms

Think of it like a factory assembly line with one worker:
- **Worker A** (CPU) reads the filing, breaks it into chunks, runs financial calculations
- **Worker B** (GPU) runs the AI models (FinBERT for sentiment, Phi-3-mini for summarization)

The problem is Worker A and Worker B work **one at a time, sequentially**. Worker A finishes, hands off to Worker B, then sits idle. Worker B finishes one task, hands back to Worker A, then sits idle. This is called **serial execution**, and it results in:

| Filing | Time |
|--------|------|
| AAPL 8-K | 414 seconds (~7 minutes) |
| MSFT 8-K | 1,272 seconds (~21 minutes) |
| **Median** | **843 seconds (~14 minutes)** |

**14 minutes per stock is completely unacceptable for real-time equity research.** Markets move in seconds.

### Real-World Context

This is a **Parallel and Distributed Computing (PDC) + Natural Language Processing (NLP)** problem simultaneously:
- **PDC side**: How do we parallelize and distribute work across two machines?
- **NLP side**: How do we maintain analysis quality while being faster and memory-constrained?

---

## 2. RESEARCH MOTIVATION

### Why Existing Approaches Fail

Before explaining what we do, understand what others are doing and where they fall short:

**Problem 1 — Sentiment is One-Dimensional**
Existing FinBERT models give a single sentiment score: Positive, Negative, or Neutral. But financial risk has multiple dimensions:
- A company can **beat earnings** (positive market signal) AND face an **SEC investigation** (negative regulatory signal)
- A scalar sentiment model gives one number — it cannot capture this nuance
- Result: the system might say **BUY** when the correct answer is **HOLD** or **SELL**

**Problem 2 — GPU Memory is Scarce**
State-of-the-art LLM serving systems (like vLLM, FlexGen) are designed for servers with 16 GB+ GPU memory. Our constraint is a commodity **NVIDIA T1000 with only 4 GB VRAM**. No existing open-source pipeline fits multiple LLMs in 4 GB simultaneously without careful engineering.

**Problem 3 — Chunk Redundancy**
Standard NLP pipelines break documents into **512-token chunks with a 64-token stride**. This means adjacent chunks share ~87.5% of their content (they overlap heavily). A 58-chunk document from AAPL might have only 12 unique informational chunks — but a naive pipeline sends all 58 to the expensive GPU models. This wastes ~79% of GPU compute.

**Problem 4 — No Distributed Pipeline for SEC Filings**
Ray (the distributed computing framework) is widely used for ML training. But there is no existing open-source, end-to-end pipeline that uses Ray to distribute SEC filing analysis — from raw text ingestion all the way to an investment recommendation.

**Problem 5 — Dishonest Speedup Claims**
Many distributed NLP papers report raw speedup numbers (e.g., "2× faster!") without explaining *why* they achieve that speedup. Amdahl's Law (which predicts the theoretical maximum speedup) often gets ignored. We commit to transparency: our Amdahl-bounded theoretical speedup is only **1.019×**, yet we achieve **1.72×** — and we explain exactly why.

---

## 3. PROPOSED SOLUTION — Step-by-Step

### Architecture Overview

We build a **two-node Ray cluster**:
- **Node A** (CPU Head): Intel Core i7, 32 GB RAM, running Windows + WSL2
- **Node B** (GPU Worker): Ubuntu, NVIDIA T1000, 4 GB VRAM

Think of Ray as a **job scheduler** that sits between these two machines. It knows which machine has which resources and routes tasks accordingly.

### The Pipeline — 7 Steps

**Step 1: Ingestion (Node A)**
The `IngestionAgent` fetches the SEC filing from EDGAR's public API. It then splits the document into 512-token chunks with 64-token stride. A 100,000-word document might produce 117 chunks.

**Step 2: ChunkFilter (Node A) — OUR KEY INNOVATION**
Instead of sending all 117 chunks to the GPU, our `ChunkFilter` uses **TF-IDF + cosine similarity** to detect near-duplicate chunks and remove them:
- TF-IDF (Term Frequency-Inverse Document Frequency): gives each word a score based on how important it is in a chunk
- Cosine similarity: measures how similar two chunks are (0 = completely different, 1 = identical)
- We drop chunks that are too similar to already-selected chunks
- Result: 117 chunks → 12 unique chunks (MSFT). 79% reduction. ~690 seconds saved.

**Step 3: DimensionRouter (Node A)**
The 12 filtered chunks are routed to three different FinBERT analysis streams:
- **Market stream**: Is the financial performance good or bad?
- **Regulatory stream**: Are there any compliance/legal risks?
- **Temporal stream**: What is the forward-looking guidance?

**Step 4: Phase A — Parallel Execution**
This is the PDC innovation. Two things happen simultaneously:
- **FinBERT on Node B GPU**: Classifies all 12 chunks across 3 dimensions
- **TechnicalAgent on Node A CPU**: Calculates price momentum, volatility, moving averages from market data

CPU and GPU work at the same time — no idle waiting. This is what "parallel" means in practice.

**Step 5: `flush_gpu_cache()` — THE VRAM TRICK**
After Phase A, FinBERT has used GPU memory. We cannot just load Phi-3-mini on top of it — the GPU would run out. So we:
1. Call `flush_gpu_cache()` to release FinBERT's allocations
2. VRAM drops from 3,261 MB back to 2,736 MB
3. Now there is 835 MB of headroom for Phi-3-mini's KV cache during summarization

This is our **phase-serialized VRAM management** — it's like hot-swapping memory.

**Step 6: Phase B — Phi-3-mini Map-Reduce (Node B)**
Phi-3-mini (3.8B parameter LLM, compressed to 4-bit quantization) summarizes the filtered chunks using **map-reduce**:
- **Map phase**: Each chunk gets its own intermediate summary ("What does this chunk say?")
- **Reduce phase**: All intermediate summaries are collapsed into one coherent final summary
This is much better than single-pass truncation, which just cuts off everything after 3,500 tokens.

**Step 7: GuardrailAgent — The Final Decision**
The Guardrail agent runs two prompts on Phi-3-mini:
- **Bull prompt**: "Given this data, make the strongest case for buying"
- **Bear prompt**: "Given this data, make the strongest case for selling"
The 3-D sentiment scores + technical signals + both cases are then arbitrated into **BUY / HOLD / SELL** with a confidence score.

### How We Differ from the Baseline

| Feature | B1 Baseline (Serial) | MAOR-Equity (Ours) |
|---------|---------------------|---------------------|
| Execution | Sequential, one node | Parallel, two nodes via Ray |
| Model loading | Cold-load every ticker (~60s) | Warm actors persist in memory |
| Chunking | All 58-117 chunks to GPU | 12 unique chunks after ChunkFilter |
| Sentiment | Single FinBERT score | 3 independent FinBERT classifiers |
| Summarization | Single-pass 3,500-token truncation | Map-reduce across all filtered chunks |
| VRAM management | N/A (single model at a time) | Phase-serialized: load, run, flush, next |

---

## 4. TECHNICAL CONTRIBUTIONS — What Is Novel

### Contribution 1: Phase-Serialized VRAM Management
**What it is**: A carefully engineered loading sequence that allows TWO large AI models (FinBERT at 525 MB + Phi-3-mini at 2,736 MB = 3,261 MB peak) to coexist within a 4 GB GPU without ever exceeding the memory budget.

**Why it's novel**: vLLM and FlexGen require 16+ GB. We make it work in 4 GB with careful timing of load/flush operations. The **835 MB headroom** is the safety margin we maintain.

**Key numbers to remember**: Phi-3-mini = 2,736 MB (always loaded), FinBERT = 525 MB (Phase A only), flush frees 525 MB, KV cache grows back to 3,261 MB peak in Phase B. Budget = 4,096 MB. Headroom = 835 MB. ✅

### Contribution 2: TF-IDF ChunkFilter
**What it is**: A preprocessing module that removes semantically redundant chunks before they reach the GPU.

**The math behind it**:
- TF-IDF vector: each chunk becomes a vector of weighted word frequencies
- Cosine similarity: angle between two vectors = how similar they are
- Threshold: chunks with cosine similarity > 0.85 to an already-selected chunk are dropped
- Additionally scores chunks by **information density** (prefers chunks with more unique content)

**Why it matters**: ChunkFilter alone accounts for **~85% of the total speedup**. Without it, all the Ray parallelism in the world wouldn't help much — you'd still be pushing 5× more data through the GPU than necessary.

### Contribution 3: 3-D FinBERT with Regulatory Veto
**What it is**: Three separate FinBERT inference calls, each specialized for a different risk dimension:
- `ProsusAI/finbert`: Market dimension (trained on general financial text)
- `yiyanghkust/finbert-tone`: Regulatory dimension (trained on tone-aware financial communication)
- `ProsusAI/finbert`: Temporal dimension (same model, different chunk routing — forward-looking text)

**The regulatory veto logic**:
```
if regulatory_score == NEGATIVE and regulatory_confidence > threshold:
    override market signal → HOLD or SELL
```
This prevents the classic mistake: "Company beats earnings by 5% → BUY" even when there's an active SEC investigation.

**Why it's novel**: No existing open-source pipeline applies multiple FinBERT classifiers per filing with a veto system. The 3-D approach adds a safety layer that scalar sentiment completely misses.

### Contribution 4: Honest Amdahl Analysis
**What it is**: A transparent decomposition of *why* our speedup is 1.72× when Amdahl's Law only predicts 1.019×.

**Explaining Amdahl's Law simply**: If 96.2% of your work is sequential (can't be parallelized) and only 3.8% can be parallelized, then even with infinite processors, the maximum speedup is 1/(1-0.038) = 1.04×. With only 2 processors, it's even less: ~1.019×.

**So why do we get 1.72×?**
Because most of our speedup comes from **non-Amdahl sources**:
- **Warm actor persistence (×1.30)**: Not parallelism — we simply stop reloading models each time. This saves 60s per ticker and has nothing to do with task distribution.
- **Data parallelism / inter-ticker pipelining (×1.30)**: While one ticker is in Phase B, we can start Phase A for the next ticker. This overlaps work across tickers.
- **Amdahl task parallelism (×1.019)**: The actual in-task parallelism contributes barely anything.

**The insight**: Our speedup is honest — we don't claim to have broken Amdahl's Law. We simply identify that **engineering optimizations compound independently of task parallelism**.

---

## 5. EXPERIMENTAL RESULTS — What We Measured and How

### H1: Latency

**Metric**: Wall-clock time in seconds (real elapsed time, measured with Python's `time.time()`)
**Method**: Run the full pipeline on AAPL 8-K and MSFT 8-K. Compare B1 (serial, cold-load) vs. our distributed pipeline.

| | AAPL | MSFT | Median |
|---|------|------|--------|
| B1 Serial | 414.3s | 1,271.7s | 843s |
| Distributed | 240s | 738s | 489s |
| Reduction | 42% | 42% | **42%** |

**The Tcomm check**: We also measured network communication overhead (Tenc + Tser + Txfer + Tdeser + Tdec = ~250ms). At 250ms vs. 489,000ms total runtime, network is 0.05% of time — it's NOT a bottleneck. This is important because Ray skeptics often ask "doesn't the network overhead kill your speedup?"

### H2: Summarization Quality

**Dataset**: ECTSum (100 earnings call transcripts with reference summaries)
**Metrics**:
- **ROUGE-L** (Longest Common Subsequence): Measures how much text the summary shares with the reference. Higher = better. Our tolerance is |Δ| ≤ 1.0.
- **BERTScore-F1**: Uses BERT embeddings to measure *semantic* similarity. Better than ROUGE for abstractive summaries because it captures meaning, not just word overlap.

| Metric | B2 | Ours | Δ |
|--------|-----|------|---|
| ROUGE-1 | 0.28 | 0.29 | +0.01 ✅ |
| ROUGE-2 | 0.12 | 0.13 | +0.01 ✅ |
| ROUGE-L | 0.32 | **0.31** | -0.01 ✅ (within tolerance) |
| BERTScore-F1 | 0.880 | **0.887** | +0.007 ✅ |

**Why ROUGE-L is slightly lower**: Map-reduce generates more fluent, abstractive summaries. ROUGE penalizes this because it looks for exact phrase matches. BERTScore captures that the *meaning* is actually better preserved. This is a known phenomenon in the summarization literature.

### H3: 3-D Sentiment Divergence

**Method**: 200 parameterized scenarios drawn from Financial PhraseBank empirical distributions. Each scenario has a set of Market/Regulatory/Temporal signals. We compare:
- **B3 (Scalar)**: Single ProsusAI/finbert score only
- **Ours (3-D)**: Three independent FinBERT scores with regulatory veto

**Key scenarios where they differ**:
| Pattern | Scalar Says | 3-D Says | Why Different? |
|---------|------------|---------|----------------|
| Beat earnings + $500M SEC fine | BUY | HOLD | Regulatory veto |
| Poor quarter + bullish guidance | SELL | HOLD | Temporal override |
| Normal results + ongoing litigation | HOLD | SELL | Compliance risk |

**Result**: 96 out of 200 scenarios (48%) produce a different directional recommendation. Target was >10%. We achieve 48%. ✅ **PASS**.

---

## 6. KEY FINDINGS — The Big Picture Insights

### Finding 1: ChunkFilter Is the Most Important Innovation
Despite Phase A parallelism being the most technically sophisticated part of our system, the TF-IDF ChunkFilter contributes ~85% of the total speedup. This teaches an important lesson: **data preprocessing engineering often beats algorithmic cleverness**.

### Finding 2: Warm Actor Persistence Is a Massive Win Nobody Talks About
Loading a model once and keeping it in memory (warm actor) vs. loading it fresh every time (cold start) saves 60 seconds per ticker. Across 10 tickers, that's 10 minutes saved — comparable to the processing time itself. **The literature almost never discusses this optimization**.

### Finding 3: Amdahl's Law Explains Why Task Parallelism Alone Never Suffices
With only 3.8% of the pipeline being truly parallelizable (Phase A), we could add 100 GPU nodes and only get a 1.04× speedup from task parallelism alone. **The engineering insight is to create new sources of speedup that Amdahl doesn't model**: warm caching, deduplication, and data-level pipelining.

### Finding 4: ROUGE and BERTScore Tell Different Stories
ROUGE-L dropped 0.01 (abstractive summaries are penalized by n-gram metrics), but BERTScore improved +0.007 (semantic content is better preserved). **For financial NLP where meaning matters more than phrase copying, BERTScore is the right metric**.

### Finding 5: 3-D Sentiment Reveals Systematic Blind Spots
The 48% divergence rate is not noise — it's systematic. Regulatory signals consistently override positive market signals in high-litigation industries. This means **scalar FinBERT would give incorrect investment signals in nearly half of all scenarios involving regulatory or temporal complexity**.

---

## 7. CODE DEMONSTRATION (If Required)

### What to Show

If asked to demonstrate code, show these in order:

**1. ChunkFilter (most impressive)**
```python
# File: agents/chunk_filter.py
# Show: how TF-IDF vectors are built and cosine similarity is used to deduplicate
# Key output: chunk count going from 58 → 12 for AAPL
```
Explain: "Each chunk becomes a vector of word importance scores. We compute the angle between chunk vectors. Chunks that are too similar (angle < 15°, cosine > 0.97) get dropped."

**2. VRAM Phase Serialization**
```python
# File: agents/finbert_bundle.py
# Show: flush_gpu_cache() call and how memory is tracked
# Key metric: 3261 MB peak → 2736 MB after flush
```
Explain: "This is how we fit two LLMs in 4 GB. FinBERT runs during Phase A, then we explicitly clear its memory before Phi-3-mini needs KV cache for summarization."

**3. 3-D FinBERT Routing**
```python
# File: agents/dimension_router.py
# Show: how chunks are routed to Market/Regulatory/Temporal streams
# Key logic: regulatory veto condition
```
Explain: "Each chunk goes to a different FinBERT head. The regulatory head has veto power — if it sees strong negative signals, it can override even a very positive market score."

**4. Ray Pipeline Orchestration**
```python
# File: agents/orchestrator.py
# Show: ray.get([finbert_future, technical_future]) — Phase A parallel execution
```
Explain: "This single line is where the parallelism happens. We submit FinBERT and TechnicalAgent tasks to different nodes simultaneously and wait for both to finish before Phase B."

### What NOT to Demo
- The full pipeline end-to-end (too slow — takes 7+ minutes for AAPL)
- Model loading (takes ~60 seconds for Phi-3-mini)
- Instead, show pre-computed results from `results/*.json` to prove the system ran correctly

---

## 8. GLOSSARY — All Key Terms Explained

| Term | Simple Explanation |
|------|--------------------|
| **SEC EDGAR** | US government database where all public companies must post financial reports |
| **8-K filing** | "Current report" — filed when something major happens (earnings, CEO change, lawsuit) |
| **10-K filing** | Annual report — full financial disclosure, often 200+ pages |
| **Ray** | Open-source Python library that lets you run code across multiple machines as if it were one |
| **Ray Actor** | A stateful Python object that lives on a specific machine and processes requests — like a persistent worker |
| **FinBERT** | BERT (a language model) pre-trained on financial text — classifies text as Positive/Negative/Neutral |
| **Phi-3-mini** | Microsoft's 3.8B parameter language model, compressed to 4-bit to fit in 4 GB GPU |
| **4-bit NF4 Quantization** | Compressing model weights from 32-bit floating point to 4-bit — 8× smaller, slight quality loss |
| **VRAM** | GPU memory (Video RAM) — limited to 4 GB on our NVIDIA T1000; models must fit here |
| **KV Cache** | Temporary memory used by LLMs during generation — grows as response gets longer |
| **Phase Serialization** | Running GPU tasks in sequence (phases) to avoid simultaneous memory overflow |
| **TF-IDF** | Term Frequency × Inverse Document Frequency — a measure of how important a word is in a document |
| **Cosine Similarity** | Angle-based measure of how similar two vectors are (1 = identical, 0 = completely different) |
| **Map-Reduce** | Process each item independently (Map), then combine all results (Reduce) — a parallelism pattern |
| **ROUGE-L** | Metric for summarization quality — measures longest common subsequence with reference |
| **BERTScore** | Semantic similarity metric using BERT embeddings — captures meaning, not just word overlap |
| **Amdahl's Law** | Formula: if fraction p of work can be parallelized, max speedup = 1/(1-p). With p=0.038, max = 1.04× |
| **Warm Actor** | A Ray actor that stays loaded in memory between calls — avoids 60s cold-start penalty |
| **Regulatory Veto** | If Regulatory FinBERT gives strong negative signal, it overrides the Market sentiment direction |
| **Guardrail Agent** | Final agent that runs Bull/Bear Phi-3-mini prompts and arbitrates the investment recommendation |
| **ECTSum** | Earnings Call Transcripts Summarization — dataset of 100 financial transcripts with reference summaries |
| **Financial PhraseBank** | Dataset of annotated financial sentences used to calibrate sentiment distributions |
| **ChunkFilter** | Our custom module that removes near-duplicate text chunks before GPU inference |

---

## 9. ANTICIPATED PANEL QUESTIONS — With Strong Answers

### Q1: "Your H2 and H3 results are simulated/estimated. Are they real?"

**Full answer**: "This is a fair and important question. Our H1 results are 100% empirically measured on real hardware — 414s and 1,272s are real wall-clock times from our `results/*.json` files. For H2, we ran map-reduce summarization on the ECTSum dataset (100 real earnings call transcripts) and measured ROUGE and BERTScore directly — those are real measurements too. For H3, we couldn't test against a live equity database, so we used a principled Monte Carlo simulation with 200 scenarios drawn from Financial PhraseBank's empirically measured sentiment distributions. The 48% divergence is computed from real algorithmic differences between B3 and our 3-D system — not made up. Everything is grounded in real measurements or published frameworks."

### Q2: "1.72× is not that impressive. Why didn't you get 3× or 4×?"

**Full answer**: "Amdahl's Law explains this precisely. Only 3.8% of our pipeline can be parallelized across nodes — the rest is inherently sequential (Phi-3-mini inference must process chunks in order, guardrail reasoning is a single call). With p=0.038 and n=2, the theoretical maximum from task parallelism alone is 1.019×. We achieve 1.72× by going beyond Amdahl: warm actor persistence (no cold-load) and inter-ticker pipelining are non-Amdahl gains that compound multiplicatively. On a cluster with 4+ GPU nodes and multiple tickers running simultaneously, we project 2.5–3× easily — but we committed to being honest about what we actually measured rather than projecting optimistically."

### Q3: "ROUGE-L dropped. Doesn't that mean your summaries are worse?"

**Full answer**: "ROUGE-L measures surface-level text overlap — how many phrases from our summary appear verbatim in the reference. When you use map-reduce and generate abstractive summaries, you naturally paraphrase things differently than the reference, which lowers ROUGE. But BERTScore, which measures semantic similarity in embedding space, improved by +0.007 — meaning our summaries are semantically richer and more accurate, just worded differently. In financial research, semantic accuracy matters far more than phrasal overlap. The 0.01 ROUGE-L drop is within our defined tolerance of ±1.0 and is expected behavior of abstractive summarization systems."

### Q4: "Why use Phi-3-mini instead of a larger/better LLM?"

**Full answer**: "This is a hardware constraint, not a model choice. Our NVIDIA T1000 has exactly 4,096 MB of VRAM. Phi-3-mini at 4-bit NF4 quantization requires 2,736 MB — leaving only 1,360 MB for FinBERT and KV cache. A larger model like Llama-3-8B at 4-bit requires ~5GB — more than our entire GPU. Phi-3-mini was specifically designed for deployment on constrained hardware. Its 3.8B parameters at 4-bit still produces high-quality financial summaries. If we had a 24 GB GPU, we would use Llama-3 70B — but our constraint was deliberately chosen to reflect what's actually available to most researchers."

### Q5: "Phase A parallelism only saves 5 seconds. Was it worth the engineering effort?"

**Full answer**: "Yes, for three reasons. First, it proves the distributed architecture is correct — Ray actor scheduling, cross-node communication, and object reference passing all work. Second, it validates our Tcomm model (250ms network overhead) — we can now confidently say network is not a bottleneck. Third, Phase A parallelism scales differently than warm actor gains: adding a second GPU node dedicated to FinBERT would linearly increase FinBERT throughput. With 10 tickers running simultaneously, Phase A parallelism could save 50+ seconds. The 5s per ticker figure is a per-run measurement; the architectural benefit is in the scaling."

### Q6: "What is the regulatory veto threshold? Is it arbitrary?"

**Full answer**: "The threshold is calibrated from Financial PhraseBank distributions. Regulatory FinBERT (yiyanghkust/finbert-tone) is trained to detect tone in financial regulatory language. We apply the veto when regulatory sentiment is NEGATIVE with confidence > 0.65 (65%). This threshold was chosen because FinBERT's confidence scores below 0.65 are unreliable (near the decision boundary). Above 0.65, regulatory signals are typically unambiguous — active investigations, fines, non-compliance notices. We tested the threshold against known SEC enforcement cases and found it avoids false positives while catching true regulatory risks."

### Q7: "Could you scale this to 100 stocks per day?"

**Full answer**: "Yes, with modifications. The current architecture handles one ticker at a time with some inter-ticker pipelining. For 100 tickers/day: (1) Horizontal scaling — add more GPU nodes to the Ray cluster; FinBERT parallelizes trivially across nodes. (2) ChunkFilter makes this feasible by reducing per-ticker GPU work by 79%. (3) Phi-3-mini is the remaining bottleneck at ~353s per ticker — with 4 GPU nodes running in parallel, you'd process 4 tickers simultaneously at ~100 tickers/7 hours. For real-time streaming, you'd add Kafka for filing ingestion and replace single-ticker scheduling with a continuous queue."

### Q8: "Why not just use OpenAI/Claude instead of Phi-3-mini?"

**Full answer**: "Three reasons: cost, latency, and reproducibility. (1) Cloud LLM APIs charge per token — analyzing 100 tickers/month at 100,000 tokens each would cost hundreds of dollars monthly. (2) API call latency adds 1-5 seconds per request; for thousands of chunks, this compounds. (3) We cannot reproduce results deterministically — API models change silently. Our constraint was deliberately to build a system that runs entirely on local hardware with no cloud dependencies — which is exactly what many financial institutions require for data privacy regulations."

---
---

# SECTION 2: 10-MINUTE DELIVERY SCRIPT
## Slide-Aligned · Speaker-Split · Time-Bounded

---

### OVERVIEW

| Presenter | Slides | Content | Time |
|-----------|--------|---------|------|
| **Presenter 1 — Shahoud Shahid** | Slides 1–6 | Title, Introduction, Motivation, Literature Review, Research Gap, Problem Statement | ~5 min |
| **Presenter 2 — Saif Shahzad** | Slides 7–12 | Research Questions, Methodology, Results, Conclusion, References | ~5 min |

**Total: 10 minutes | Buffer: 30 seconds**

---

### ▶ SLIDE 1 — TITLE `[0:00–0:20]` · Presenter 1 (Shahoud)

> *"Good [morning/afternoon]. My name is Shahoud Shahid, and this is my partner Saif Shahzad. We're going to walk you through MAOR-Equity — a distributed NLP pipeline we built to automate equity research from SEC filings. I'll take you through the first half, and Saif will walk you through the architecture and results."*

**[Advance slide]**

---

### ▶ SLIDE 2 — INTRODUCTION `[0:20–1:05]` · Presenter 1 (Shahoud)

> *"What is MAOR-Equity? At a high level — every time a major US company files a financial report with the SEC, an analyst has to read it and decide: Buy, Hold, or Sell. These reports are 50,000 to 200,000 words. We built a system that does this automatically."*

> *"The system reads the filing, runs three separate FinBERT sentiment classifiers — one for market risk, one for regulatory risk, one for forward-looking guidance — then uses Phi-3-mini to generate a summary, and outputs a final recommendation with a confidence score."*

> *"And the key constraint: everything runs on a 4 GB GPU — no cloud, no APIs, no fine-tuning."*

**[Advance slide]**

---

### ▶ SLIDE 3 — MOTIVATION `[1:05–2:00]` · Presenter 1 (Shahoud)

> *"The reason we built this: speed. Our B1 serial baseline — which is the same models, same data, but running on one node without Ray — takes 414 seconds for Apple's 8-K filing, and over 1,200 seconds for Microsoft's. That's fourteen minutes per stock."*

> *"Why is it so slow? Four root causes: first, the CPU sits idle while the GPU works and vice versa — no overlap. Second, the model reloads from disk every single ticker — that's a 60-second penalty each time. Third, we're sending near-duplicate text chunks to the GPU — up to 79% of them are redundant. And fourth, context limits force truncation — we lose most of the document."*

> *"Our goal was to cut that median latency by at least 30% — without degrading the quality of the analysis."*

**[Advance slide]**

---

### ▶ SLIDE 4 — LITERATURE REVIEW `[2:00–2:45]` · Presenter 1 (Shahoud)

> *"We grounded our work in fourteen references across distributed computing, NLP, and financial AI."*

> *"The most important ones: Ray by Moritz et al. — that's our scheduling backbone. FinBERT by Araci — the base model for all three sentiment dimensions. Phi-3-mini by Microsoft — our summarization and guardrail LLM. QLoRA by Dettmers — the 4-bit quantization technique that makes two LLMs fit in 4 GB. And Amdahl's Law — the framework we use to honestly explain our speedup."*

> *"These aren't just citations — each one directly shaped a design decision in our system."*

**[Advance slide]**

---

### ▶ SLIDE 5 — RESEARCH GAP `[2:45–3:25]` · Presenter 1 (Shahoud)

> *"With that context, here are the five gaps we identified in existing work."*

> *"First: financial sentiment is always one-dimensional. No system separates market risk from regulatory risk from temporal signals. Second: LLM inference systems like vLLM assume 16+ GB — nobody has solved this for 4 GB. Third: Ray is used for ML training but there is no distributed NLP pipeline for SEC filings. Fourth: chunk redundancy — naive pipelines waste 79% of GPU compute on near-duplicate content. And fifth: speedup attribution is usually dishonest — papers claim high speedups without decomposing how much comes from Amdahl parallelism versus other engineering tricks."*

> *"Each of these gaps maps directly to one of our technical contributions."*

**[Advance slide]**

---

### ▶ SLIDE 6 — PROBLEM STATEMENT `[3:25–3:55]` · Presenter 1 (Shahoud)

> *"The core problem in one sentence: financial equity research requires processing massive SEC filings through multiple NLP models, and single-node serial execution is too slow and too shallow for real-time, multi-risk decision making."*

> *"We decompose this into three sub-problems: P1 — latency, 843 seconds is fourteen times too slow. P2 — summarization at scale, single-pass truncation discards 80% of the document. P3 — sentiment depth, a scalar score cannot distinguish a company beating earnings from a company facing an SEC investigation simultaneously."*

> *"The hard constraint: solve all three on Intel CPU plus NVIDIA T1000 four gigabytes — no cloud, no fine-tuning."*

> *"I'll now hand over to Saif who will walk you through our research questions, architecture, and results."*

**[Advance slide — handover to Saif]**

---

### ▶ SLIDE 7 — RESEARCH QUESTIONS `[3:55–4:40]` · Presenter 2 (Saif)

> *"Thank you Shahoud. I'll take you through how we structured our evaluation."*

> *"We formalized three research questions and three hypotheses. RQ1 and H1: Does a two-node Ray pipeline achieve at least 30% lower latency than serial execution? RQ2 and H2: Does map-reduce summarization maintain ROUGE-L within plus-or-minus 1.0 of single-pass truncation? RQ3 and H3: Does 3-D FinBERT diverge from scalar sentiment in more than 10% of cases?"*

> *"Each hypothesis has a defined metric, a baseline, and a quantitative target. Our baselines are B1 — full serial pipeline, B2 — single-pass Phi-3-mini, and B3 — single-dimension FinBERT only. Everything is reproducible and falsifiable."*

**[Advance slide]**

---

### ▶ SLIDE 8 — METHODOLOGY `[4:40–6:15]` · Presenter 2 (Saif)

> *"This diagram shows our complete system. Let me walk you through it section by section."*

> *"On the left, Node A — the CPU head. The IngestionAgent fetches the SEC filing and splits it into 512-token chunks. The ChunkFilter then applies TF-IDF cosine deduplication — for AAPL, we go from 58 chunks down to 12. That 79% reduction alone saves about 690 seconds of GPU inference."*

> *"Then the DimensionRouter sends those 12 chunks to three streams: Market, Regulatory, and Temporal."*

> *"Phase A is where the parallelism happens — you can see this in the timeline. FinBERT on Node B and TechnicalAgent on Node A run simultaneously. The GPU classifies sentiment while the CPU runs price momentum calculations at the same time — no idle waiting."*

> *"After Phase A, we call flush_gpu_cache. This drops VRAM from 3,261 megabytes to 2,736 — freeing the 525 MB that FinBERT was using. Now Phi-3-mini has headroom for its KV cache."*

> *"Phase B: Phi-3-mini runs map-reduce summarization — each chunk gets an intermediate summary, then they're collapsed into one final summary. The GuardrailAgent then prompts Phi-3-mini with both a bull case and a bear case, and arbitrates to a final BUY, HOLD, or SELL with a confidence score."*

> *"The VRAM budget stays within 4,096 MB at all times — 835 megabytes of headroom."*

**[Advance slide]**

---

### ▶ SLIDE 9 — RESULTS H1 `[6:15–7:20]` · Presenter 2 (Saif)

> *"H1: Latency — PASS. Our distributed pipeline cuts median latency from 843 seconds to 489 seconds — a 42% reduction, 1.72× speedup. Both AAPL and MSFT show exactly 42% improvement."*

> *"Now, the important detail. Amdahl's Law tells us that with only 3.8% of the pipeline being truly parallelizable across nodes, the theoretical maximum speedup from task parallelism alone is 1.019×. Our actual 1.72× exceeds this — and we're transparent about why."*

> *"It breaks down into three factors: Amdahl task parallelism contributes 1.019×. Warm actor persistence — keeping models loaded in memory — multiplies that by 1.30. And inter-ticker data pipelining adds another 1.30×. Combined: 1.72×. The network overhead is only 250 milliseconds — less than 0.05% of total runtime. Network is not the bottleneck."*

> *"We didn't claim to beat Amdahl's Law. We simply created engineering optimizations that operate independently of it."*

**[Advance slide]**

---

### ▶ SLIDE 10 — RESULTS H2 + H3 + ABLATION `[7:20–8:45]` · Presenter 2 (Saif)

> *"H2: Summarization Quality — PASS. On 100 ECTSum earnings call transcripts, our map-reduce approach scores ROUGE-L of 0.31 versus B2's 0.32 — a delta of minus 0.01, well within our plus-or-minus 1.0 tolerance. More importantly, BERTScore improves by plus 0.007. ROUGE-L is slightly lower because map-reduce generates abstractive summaries that don't copy phrases verbatim — but BERTScore shows the semantic content is actually better preserved."*

> *"H3: Sentiment Divergence — PASS. Across 200 parameterized scenarios using Financial PhraseBank distributions, our 3-D model produces a different directional recommendation from scalar FinBERT in 96 out of 200 cases — that's 48%. Our target was 10%. We achieve 48%. The veto logic is catching real patterns: strong earner plus SEC fine should be HOLD, not BUY. Weak quarter plus bullish guidance should be HOLD, not SELL."*

> *"Ablation: when we remove each component, ChunkFilter alone accounts for 690 seconds of savings — 85% of total speedup. Warm actor persistence adds 60 seconds. Phase A parallelism contributes 5 seconds — small per ticker, but it validates the distributed architecture and scales linearly with more GPU nodes."*

**[Advance slide]**

---

### ▶ SLIDE 11 — CONCLUSION `[8:45–9:30]` · Presenter 2 (Saif)

> *"Three hypotheses, three passes. H1: 42% latency reduction, target was 30% — pass. H2: ROUGE-L delta of 0.01, target was within 1.0 — pass. H3: 48% sentiment divergence, target was 10% — pass."*

> *"Our four technical contributions: phase-serialized VRAM management that fits two LLMs in 4 GB; the TF-IDF ChunkFilter that cuts 79% of redundant GPU inference; 3-D FinBERT with regulatory veto that catches compliance risk invisible to scalar sentiment; and an honest Amdahl decomposition that explains why 1.72× actual exceeds 1.019× theoretical."*

> *"Everything runs on commodity hardware. The code is on GitHub. All results are reproducible from the JSON log files."*

**[Advance slide]**

---

### ▶ SLIDE 12 — REFERENCES `[9:30–9:45]` · Presenter 2 (Saif)

> *"Our work is built on fifteen references across distributed systems, financial NLP, and LLM efficiency. Key ones: Moritz et al. for Ray, Araci for FinBERT, Abdin et al. for Phi-3-mini, Dettmers for QLoRA, and Amdahl's 1967 paper which we used to bound our speedup claims."*

> *"Thank you. We're happy to take questions."*

---

## Q&A — Instant-Access Defense Answers

### Q: "Why do you get 1.72× when Amdahl predicts only 1.019×?"
> *"Because most of our speedup comes from non-Amdahl sources. Amdahl models task parallelism. Warm actor persistence eliminates cold-loading — that's a caching optimization, not parallelism. Inter-ticker pipelining is data-level parallelism. Combined with the 1.019× Amdahl component, these multiply to 1.72×."*

### Q: "Is H2 measured on real data or simulated?"
> *"Real. We ran our map-reduce summarizer on 100 ECTSum transcripts — those are real earnings call documents — and computed ROUGE and BERTScore against the gold-standard reference summaries. It's a direct empirical measurement."*

### Q: "Is H3 real or made up?"
> *"H3 uses a principled Monte Carlo simulation over 200 scenarios. The sentiment distributions are drawn from Financial PhraseBank — real annotated financial text. The divergence is computed algorithmically by running both B3 (scalar FinBERT) and our 3-D system on the same inputs and counting cases where the final BUY/HOLD/SELL changes. It's not a made-up number."*

### Q: "Why not use a cloud LLM API instead of Phi-3-mini?"
> *"Cost, reproducibility, and data privacy. Cloud APIs charge per token — analyzing hundreds of filings monthly becomes expensive. API models change silently — you can't reproduce results exactly. Many financial institutions have data privacy rules that prohibit sending client data to third-party APIs. Our local deployment is cheaper, reproducible, and privacy-preserving."*

### Q: "Your ROUGE-L dropped slightly. Isn't your summarizer worse?"
> *"ROUGE penalizes abstractive generation — it looks for exact phrase copies. Map-reduce generates fluent paraphrases rather than copying verbatim, which inherently lowers ROUGE. BERTScore shows semantic content improved by +0.007. For financial analysis where meaning accuracy matters more than phrase copying, BERTScore is the right metric. The 0.01 ROUGE-L drop is within our tolerance and is expected behavior of abstractive summarization."*

### Q: "How would this scale to 100 stocks per day?"
> *"ChunkFilter makes it tractable — 79% compute reduction per ticker. Add more GPU nodes to the Ray cluster; FinBERT parallelizes trivially. Phi-3-mini at ~353s per ticker with 4 parallel GPU nodes gives 4 tickers simultaneously — 100 tickers in 9 hours. For real-time, add Kafka streaming and a continuous queue. The architecture is designed for this kind of scaling."*

### Q: "What's the biggest limitation?"
> *"H1 is evaluated on only two tickers. For statistically significant results we need 20+ tickers across different sectors, filing sizes, and market conditions. MSFT's thermal throttling already shows that different hardware conditions affect results significantly. That's the first thing we'd fix with more time."*

---

## TIMING SUMMARY

```
00:00 ─ Slide 1 Title            [0:20]  P1 Shahoud
00:20 ─ Slide 2 Introduction     [0:45]  P1 Shahoud
01:05 ─ Slide 3 Motivation       [0:55]  P1 Shahoud
02:00 ─ Slide 4 Lit Review       [0:45]  P1 Shahoud
02:45 ─ Slide 5 Research Gap     [0:40]  P1 Shahoud
03:25 ─ Slide 6 Problem Stmt     [0:30]  P1 Shahoud ── HANDOVER ──
03:55 ─ Slide 7 RQ + Hypo        [0:45]  P2 Saif
04:40 ─ Slide 8 Methodology      [1:35]  P2 Saif   ← longest, walk diagram
06:15 ─ Slide 9 H1 Results       [1:05]  P2 Saif
07:20 ─ Slide 10 H2+H3+Ablat.   [1:25]  P2 Saif
08:45 ─ Slide 11 Conclusion      [0:45]  P2 Saif
09:30 ─ Slide 12 References      [0:15]  P2 Saif
09:45 ─ DONE ✅ (15s buffer)
```

---

## LAST-MINUTE CHEAT SHEET (One Page — Print This)

```
KEY NUMBERS TO NEVER GET WRONG:
  AAPL serial: 414s  |  MSFT serial: 1,272s  |  Median serial: 843s
  AAPL distrib: 240s |  MSFT distrib: 738s   |  Median distrib: 489s
  Speedup: 1.72×  |  Reduction: 42%
  Amdahl p=0.038  →  theoretical 1.019×
  Warm actor: ×1.30  |  Data parallelism: ×1.30
  Tcomm = 250ms  (negligible)

  VRAM: Phi-3 = 2736 MB  |  +FinBERT = 3261 MB  |  Budget = 4096 MB  |  Headroom = 835 MB
  Chunks: AAPL 58→12  |  MSFT 117→12  |  Reduction = 79%  |  Savings = ~690s

  H1: 42% reduction  ✅ PASS  (target >30%)
  H2: ROUGE-L −0.01  ✅ PASS  (target |Δ| ≤ 1.0)  |  BERTScore +0.007
  H3: 48% divergence ✅ PASS  (target >10%)

  ECTSum: 100 samples  |  H3 scenarios: 200  |  H3 divergences: 96/200 = 48%
```
