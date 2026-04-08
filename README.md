# Agentic Multi-Agent Orchestration for Automated Equity Research

> **NLP + PDC Project** | Two-node distributed system | Ray + FinBERT + Phi-3-mini

---

## Architecture

\\\
Node A (your machine — WSL2)          Node B (partner — WSL2 + NVIDIA T1000)
─────────────────────────────         ───────────────────────────────────────
Ray Head Node  :6379                  Ray Worker Node
Ingestion Agent       (CPU)           FinBERT Market    (GPU 0.3)
Technical Agent       (CPU)           FinBERT Regulatory(GPU 0.3)
Orchestrator / DAG    (CPU)           FinBERT Temporal  (GPU 0.3)
                                      Phi-3-mini Summ.  (GPU 0.5)
                                      Guardrail Agent   (GPU 0.2)

            ← Ngrok TCP tunnel (encrypted) →
\\\

---

## Quick Start

### Node A (head node — run ONCE)
\\\powershell
# In PowerShell (from repo root):
.\\setup_nodeA.ps1 -NgrokAuthToken "YOUR_NGROK_TOKEN"
\\\

Then in WSL:
\\\ash
bash scripts/start_cluster.sh
# Note the Ngrok TCP address and share it with your partner
\\\

### Node B (GPU worker — run ONCE after cloning)
\\\powershell
# Clone the repo first, then:
.\\setup_nodeB.ps1 -NgrokHost "0.tcp.ngrok.io" -NgrokPort 12345
\\\

### Verify cluster (Node A)
\\\ash
source venv/bin/activate
python verify_cluster.py
\\\

### Run the full pipeline
\\\ash
source venv/bin/activate
python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json
python run_pipeline.py --ticker MSFT --filing 8-K --output results/msft.json
python run_pipeline.py --ticker GOOGL --filing 8-K --output results/googl.json
\\\

---

## Hypotheses

| ID | Claim | Target metric |
|----|-------|---------------|
| H1 | Distributed pipeline is faster than serial | ≥ 30 % latency reduction |
| H2 | Chunked map-reduce improves long-doc quality | ROUGE-L ≥ B2 on ECTSum long docs |
| H3 | 3-D sentiment changes ≥ 10 % of investment calls vs scalar | > 10 % filings |

---

## VRAM Budget (Node B — T1000 4 GB)

| Component | MB (4-bit) |
|-----------|-----------|
| FinBERT × 3 actors | 1 020 |
| Phi-3-mini | 1 800 |
| Ray overhead | 200 |
| Safety buffer | 416 |
| **Total** | **~3 436** |

---

## VirtualBox Fallback
If Ngrok is unstable on demo day:
\\\ash
# Node A VM (192.168.56.10):
ray start --head --port=6379

# Node B VM (192.168.56.11):
ray start --address=192.168.56.10:6379
\\\

---

## Day-by-Day Commands
\\\ash
# Days 1-3  — Verify cluster
python verify_cluster.py

# Days 4-7  — Baselines
python baselines/b3_sentiment_baseline.py   # Node B
python baselines/b2_summarization_baseline.py # Node B
python baselines/b1_serial_pipeline.py       # Node A

# Days 11-16 — Distributed pipeline
python agents/orchestrator.py

# Days 22-26 — Evaluation
python evaluation/latency_benchmark.py
python evaluation/rouge_eval.py
python evaluation/sentiment_eval.py

# Days 27-30 — Final runs
python run_pipeline.py --ticker AAPL  --filing 8-K --output results/aapl.json
python run_pipeline.py --ticker MSFT  --filing 8-K --output results/msft.json
python run_pipeline.py --ticker GOOGL --filing 8-K --output results/googl.json
\\\