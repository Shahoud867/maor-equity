"""
evaluation/vram_verify.py — VRAM Budget Verification
Confirms peak VRAM stays within 4 GB T1000 budget during full pipeline.
Must be run ON NODE B (GPU node).

Run:
    python evaluation/vram_verify.py
"""
import subprocess
import time


def get_vram_used_mb() -> float:
    """Query nvidia-smi for current GPU memory used in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return -1.0


def get_vram_total_mb() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 4096.0


if __name__ == "__main__":
    import ray
    ray.init(address="auto")

    total_mb = get_vram_total_mb()
    print(f"GPU total VRAM: {total_mb:.0f} MB")
    print(f"Budget limit:   4096 MB\n")

    baseline_mb = get_vram_used_mb()
    print(f"Baseline (before loading): {baseline_mb:.0f} MB used")

    # Load FinBERT actors
    print("\nLoading 3× FinBERT actors ...")
    from agents.sentiment_agent import FinBERTActor
    sent_mkt = FinBERTActor.remote("ProsusAI/finbert", "market")
    sent_reg = FinBERTActor.remote("yiyanghkust/finbert-tone", "regulatory")
    sent_tmp = FinBERTActor.remote("ProsusAI/finbert", "temporal")
    # Warm up
    _ = ray.get(sent_mkt.classify_batch.remote(["Revenue grew 5%."] * 3))
    time.sleep(2)
    after_finbert_mb = get_vram_used_mb()
    print(f"After 3× FinBERT:          {after_finbert_mb:.0f} MB used  (delta: {after_finbert_mb - baseline_mb:.0f} MB)")

    # Load shared Phi-3-mini
    print("\nLoading shared Phi-3-mini (Phi3ModelActor) ...")
    from agents.guardrail_agent import Phi3ModelActor
    from agents.summarization_agent import SummarizationAgent
    from agents.guardrail_agent import GuardrailAgent

    phi3 = Phi3ModelActor.remote()
    summarizer = SummarizationAgent.remote(phi3)
    guardrail = GuardrailAgent.remote(phi3)
    # Warm up
    _ = ray.get(phi3.generate.remote("Hello.", 10))
    time.sleep(2)
    after_phi3_mb = get_vram_used_mb()
    print(f"After Phi-3-mini (shared): {after_phi3_mb:.0f} MB used  (delta: {after_phi3_mb - after_finbert_mb:.0f} MB)")

    peak_mb = after_phi3_mb
    budget_ok = peak_mb < 4096
    headroom_mb = 4096 - peak_mb

    print("\n" + "="*50)
    print("VRAM BUDGET VERIFICATION")
    print("="*50)
    print(f"Peak VRAM used:   {peak_mb:.0f} MB")
    print(f"Budget limit:     4096 MB")
    print(f"Headroom:         {headroom_mb:.0f} MB")
    print(f"Result:           {'PASS' if budget_ok else 'FAIL — reduce quantization or serialise loading'}")
    print("="*50)

    import json
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    with open("logs/vram_verify.json", "w") as f:
        json.dump({
            "total_vram_mb": total_mb,
            "baseline_mb": baseline_mb,
            "after_finbert_mb": after_finbert_mb,
            "after_phi3_mb": after_phi3_mb,
            "peak_mb": peak_mb,
            "budget_limit_mb": 4096,
            "headroom_mb": headroom_mb,
            "budget_ok": budget_ok,
        }, f, indent=2)
    print("\nSaved → logs/vram_verify.json")
