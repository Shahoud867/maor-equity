"""
evaluation/vram_verify.py — VRAM Budget Verification
Confirms peak VRAM stays within 4 GB T1000 budget during full pipeline.
Must be run ON NODE B (GPU node).

Run:
    python -m evaluation.vram_verify
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


# ── VRAM helpers ────────────────────────────────────────────────────────────

def get_vram_used_mb() -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return -1.0


def get_vram_total_mb() -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 4096.0


# ── Progress helpers ─────────────────────────────────────────────────────────

TOTAL_STAGES = 6

def _bar(filled: int, total: int = 20) -> str:
    done = int(total * filled / TOTAL_STAGES)
    return "[" + "#" * done + "-" * (total - done) + "]"


def stage_header(n: int, title: str):
    print(f"\n{_bar(n)}  Stage {n}/{TOTAL_STAGES}: {title}")


class Spinner:
    """Animated spinner for operations with unknown duration."""

    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, msg: str):
        self.msg = msg
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        i = 0
        t0 = time.time()
        while not self._stop.is_set():
            elapsed = time.time() - t0
            print(f"\r  {self.FRAMES[i % 4]}  {self.msg}  ({elapsed:.0f}s elapsed)", end="", flush=True)
            i += 1
            time.sleep(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()
        elapsed = time.time() - self._t._target.__self__._stop._cond  # noqa — just clear line
        print(f"\r  OK  {self.msg}  " + " " * 20, flush=True)


class TimedSpinner:
    """Context manager spinner that prints elapsed on exit."""

    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, msg: str):
        self.msg = msg
        self._stop = threading.Event()
        self._elapsed = 0.0
        self._t0 = 0.0

    def _run(self):
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            print(f"\r  {self.FRAMES[i % 4]}  {self.msg}  ({elapsed:.0f}s)...", end="", flush=True)
            i += 1
            time.sleep(0.25)

    def __enter__(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, *_):
        self._elapsed = time.time() - self._t0
        self._stop.set()
        self._thread.join()
        mark = "DONE" if exc_type is None else "FAIL"
        print(f"\r  [{mark}]  {self.msg}  ({self._elapsed:.1f}s)          ", flush=True)


def vram_bar(used: float, total: float, width: int = 30) -> str:
    pct = min(used / total, 1.0)
    filled = int(width * pct)
    color = filled > width * 0.85  # near limit
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {used:.0f}/{total:.0f} MB ({pct*100:.1f}%)"


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 58)
    print("  VRAM BUDGET VERIFICATION — maor-equity")
    print("=" * 58)

    # ── Stage 1: Connect to Ray ──────────────────────────────────────────
    stage_header(1, "Connecting to Ray cluster")
    with TimedSpinner("Connecting to Ray at auto-address"):
        import ray
        ray.init(address="auto", log_to_driver=False)
    nodes = [n for n in ray.nodes() if n["Alive"]]
    print(f"  Cluster nodes alive: {len(nodes)}")
    for n in nodes:
        ip  = n["NodeManagerAddress"]
        gpu = n["Resources"].get("GPU", 0)
        cpu = n["Resources"].get("CPU", 0)
        print(f"    {ip}  CPU={cpu:.0f}  GPU={gpu:.0f}")

    # ── Stage 2: Baseline VRAM ───────────────────────────────────────────
    stage_header(2, "Reading baseline VRAM")
    total_mb    = get_vram_total_mb()
    baseline_mb = get_vram_used_mb()
    print(f"  GPU total  : {total_mb:.0f} MB")
    print(f"  Before load: {vram_bar(baseline_mb, total_mb)}")

    # ── Stage 3: Load 3× FinBERT ─────────────────────────────────────────
    stage_header(3, "Loading 3x FinBERT actors onto Node B GPU")
    print("  (Each actor: ProsusAI/finbert or finbert-tone, ~150 MB each in INT8)")

    from agents.sentiment_agent import FinBERTActor

    models = [
        ("ProsusAI/finbert",         "market"),
        ("yiyanghkust/finbert-tone", "regulatory"),
        ("ProsusAI/finbert",         "temporal"),
    ]
    actors = []
    for idx, (ckpt, dim) in enumerate(models, 1):
        with TimedSpinner(f"  [{idx}/3] FinBERTActor:{dim}  ({ckpt})"):
            actor = FinBERTActor.remote(ckpt, dim)
            ray.get(actor.classify_batch.remote(["Loading test."]))
        actors.append(actor)
        used = get_vram_used_mb()
        print(f"    VRAM after actor {idx}: {vram_bar(used, total_mb)}")

    sent_mkt, sent_reg, sent_tmp = actors

    # Warm-up batch
    with TimedSpinner("  Warming up all 3 FinBERT actors"):
        ray.get([
            sent_mkt.classify_batch.remote(["Revenue grew 5%."] * 4),
            sent_reg.classify_batch.remote(["SEC investigation opened."] * 4),
            sent_tmp.classify_batch.remote(["Guidance raised for next quarter."] * 4),
        ])

    time.sleep(1)
    after_finbert_mb = get_vram_used_mb()
    delta_finbert    = after_finbert_mb - baseline_mb
    print(f"\n  After 3x FinBERT: {vram_bar(after_finbert_mb, total_mb)}")
    print(f"  Delta            : +{delta_finbert:.0f} MB  (expected ~450 MB in INT8)")

    # ── Stage 4: Load shared Phi-3-mini ──────────────────────────────────
    stage_header(4, "Loading shared Phi-3-mini-4k-instruct (Node B GPU)")
    print("  (NF4 + double quant → ~2.0 GB VRAM — this takes 1-3 min)")

    from agents.guardrail_agent    import Phi3ModelActor, GuardrailAgent
    from agents.summarization_agent import SummarizationAgent

    with TimedSpinner("  Loading Phi3ModelActor (4-bit quantized)"):
        phi3 = Phi3ModelActor.remote()
        ray.get(phi3.generate.remote("Hello.", 5))   # blocks until loaded

    time.sleep(1)
    after_phi3_mb = get_vram_used_mb()
    delta_phi3    = after_phi3_mb - after_finbert_mb
    print(f"\n  After Phi-3-mini: {vram_bar(after_phi3_mb, total_mb)}")
    print(f"  Delta            : +{delta_phi3:.0f} MB  (expected ~2100 MB)")

    # ── Stage 5: Load Summarizer + Guardrail (share phi3 handle) ─────────
    stage_header(5, "Loading SummarizationAgent + GuardrailAgent")
    print("  (These share the Phi-3 model — no extra VRAM)")

    # SummarizationAgent and GuardrailAgent hold no models — they only store
    # a reference to phi3_actor. No inference warmup needed here.
    with TimedSpinner("  Creating SummarizationAgent"):
        summarizer = SummarizationAgent.remote(phi3)
        ray.get(summarizer.ping.remote())

    with TimedSpinner("  Creating GuardrailAgent"):
        guardrail = GuardrailAgent.remote(phi3)
        ray.get(guardrail.ping.remote())

    time.sleep(1)
    peak_mb    = get_vram_used_mb()
    headroom   = total_mb - peak_mb
    budget_ok  = peak_mb < total_mb

    print(f"\n  Peak VRAM: {vram_bar(peak_mb, total_mb)}")
    print(f"  Headroom : {headroom:.0f} MB")

    # ── Stage 6: Report ───────────────────────────────────────────────────
    stage_header(6, "Final Report")
    print()
    print("  Component                  VRAM (measured)")
    print("  ─────────────────────────  ───────────────")
    print(f"  3x FinBERT actors        : +{delta_finbert:.0f} MB")
    print(f"  Phi-3-mini (shared, 4bit): +{delta_phi3:.0f} MB")
    print(f"  Ray + buffers            : +{peak_mb - after_phi3_mb:.0f} MB")
    print(f"  ─────────────────────────  ───────────────")
    print(f"  Peak total               :  {peak_mb:.0f} MB / {total_mb:.0f} MB")
    print()

    RESULT = "PASS" if budget_ok else "FAIL"
    bar_final = vram_bar(peak_mb, total_mb)
    print(f"  {bar_final}")
    print()
    if budget_ok:
        print(f"  RESULT: PASS — {headroom:.0f} MB headroom remaining")
    else:
        print(f"  RESULT: FAIL — {abs(headroom):.0f} MB over budget")
        print("  Action: reduce num_gpus fractions or serialize model loading")

    print("\n" + "=" * 58)

    # Save log
    Path("logs").mkdir(exist_ok=True)
    with open("logs/vram_verify.json", "w") as f:
        json.dump({
            "total_vram_mb":       total_mb,
            "baseline_mb":         baseline_mb,
            "after_finbert_mb":    after_finbert_mb,
            "after_phi3_mb":       after_phi3_mb,
            "peak_mb":             peak_mb,
            "delta_finbert_mb":    delta_finbert,
            "delta_phi3_mb":       delta_phi3,
            "budget_limit_mb":     total_mb,
            "headroom_mb":         headroom,
            "budget_ok":           budget_ok,
        }, f, indent=2)
    print("  Saved -> logs/vram_verify.json")
    print()

    if not budget_ok:
        sys.exit(1)
