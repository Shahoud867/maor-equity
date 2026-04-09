# Node B Setup — GPU Partner Instructions

You are **Node B**: NVIDIA T1000, 4 GB VRAM.
Node A (your partner) runs the Ray head node and orchestrator.

## Prerequisites Checklist
- [ ] WSL2 installed on Windows (`wsl --install`)
- [ ] Ubuntu 22.04 from Microsoft Store
- [ ] NVIDIA driver ≥ 525 installed on Windows (not inside WSL)
- [ ] Python 3.10 available in WSL (`python3 --version`)

## Step 1 — Verify GPU inside WSL
```bash
nvidia-smi
# Must show: T1000, ~4096 MiB total, CUDA Version ≥ 12.1
```
If this fails, your NVIDIA driver or WSL2 GPU passthrough is not configured.
Fix: update NVIDIA driver on Windows, then in WSL run: `sudo apt install nvidia-cuda-toolkit`

## Step 2 — Clone and set up the repo
```bash
cd ~
git clone <repo-url> maor-equity
cd maor-equity
python3 -m venv venv
source venv/bin/activate
# Install torch with CUDA FIRST — order matters
pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-nodeB.txt
```

## Step 3 — Pre-download models (do this once, takes ~10 min)
```bash
source venv/bin/activate
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# FinBERT models
pipeline("text-classification", model="ProsusAI/finbert")
pipeline("text-classification", model="yiyanghkust/finbert-tone")
# Phi-3-mini
AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                      load_in_4bit=True, device_map="auto")
print("All models downloaded successfully.")
EOF
```

## Step 4 — Connect to Node A's Ray cluster
Wait for your partner to send you the Ngrok address (looks like `0.tcp.ngrok.io:12345`).

```bash
source venv/bin/activate
export LD_PRELOAD=''
export RAY_DISABLE_JEMALLOC=1
bash scripts/nodeb_connect_watch.sh --address=<NGROK_ADDRESS_FROM_PARTNER>
# Preferred: keeps Node B attached and auto-reconnects if Ray drops.
```

Fallback (one-shot connect):
```bash
ray start --address=<NGROK_ADDRESS_FROM_PARTNER>
```

## Step 5 — Verify cluster (ask partner to run this on Node A)
Your partner runs:
```bash
python verify_cluster.py
# Should show: 2 nodes, GPU visible, VRAM OK
```

## Step 6 — Run VRAM verification (you run this)
```bash
python evaluation/vram_verify.py
# Should show: Peak < 4096 MB — PASS
```

## Step 7 — Run B1 serial baseline (you run this, GPU required)
```bash
python baselines/b1_serial_pipeline.py
# Records timing to logs/b1_results.json
# Note the total time — this is the H1 comparison point
```

## Step 8 — Your evaluation tasks (Week 4)
Once the cluster is running end-to-end, you run:
```bash
# GPU profiling during full pipeline:
nvidia-smi dmon -s mu -d 1 > logs/gpu_profile.txt &
python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json
# Kill the monitor after: kill %1

# H2 ROUGE evaluation (needs ECTSum dataset):
python -m evaluation.rouge_eval --n-samples 50

# H3 sentiment evaluation (needs Financial PhraseBank):
python -m evaluation.sentiment_eval --n-samples 100
```

## VRAM Budget Reference
| Component | VRAM (approx.) |
|---|---|
| 3× FinBERT (4-bit) | ~1,020 MB |
| Phi-3-mini shared (4-bit) | ~2,100 MB |
| KV cache + buffers | ~400 MB |
| Ray overhead | ~120 MB |
| **Peak total** | **~3,640 MB < 4,096 MB** |

**Important:** Phi-3-mini is loaded ONCE and shared between the Summarizer and Guardrail agent.
Do not run old code that loads them separately — it will OOM.

## Troubleshooting
- **OOM error**: Run `python evaluation/vram_verify.py` to see what's using memory
- **Ray connection refused**: Make sure partner's Ngrok is running, then restart `bash scripts/nodeb_connect_watch.sh --address=...`
- **Models not found**: Re-run Step 3 model download
- **bitsandbytes error**: Verify torch+CUDA installed before bitsandbytes — check `pip list | grep torch`
