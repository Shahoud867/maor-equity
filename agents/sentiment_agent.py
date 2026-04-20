"""
sentiment_agent.py  —  Node B (GPU)
3-dimensional FinBERT sentiment: Market / Regulatory / Temporal.
"""
import re
import ray
import numpy as np


@ray.remote(num_gpus=0.35)
class FinBERTBundle:
    """
    Single actor holding all 3 FinBERT classifiers.

    Why bundled instead of 3 separate actors:
    - Saves 2 CUDA contexts (~200 MB overhead on T1000)
    - ProsusAI/finbert loaded ONCE, shared between market + temporal
      (same checkpoint → ~220 MB saved vs loading it twice)
    - T1000 runs GPU ops sequentially anyway; 3 actors just added
      process-spawn overhead with no real parallelism gain
    """

    def __init__(self, min_confidence: float = 0.60):
        import torch
        from transformers import pipeline
        self._min_conf = min_confidence

        # ProsusAI/finbert covers market sentiment AND temporal framing
        self._pipe_mkt = pipeline(
            "text-classification", model="ProsusAI/finbert",
            device=0, top_k=None, torch_dtype=torch.float16,
        )
        # finbert-tone covers regulatory/compliance tone
        self._pipe_reg = pipeline(
            "text-classification", model="yiyanghkust/finbert-tone",
            device=0, top_k=None, torch_dtype=torch.float16,
        )
        # Temporal dimension reuses the same pipeline object — no extra VRAM
        self._pipe_tmp = self._pipe_mkt
        print("[FinBERTBundle] 2 checkpoints loaded for 3 dimensions (market+temporal shared)")

    def classify_all(self, market_texts: list, reg_texts: list,
                     tmp_texts: list) -> tuple:
        """Return (market_result, regulatory_result, temporal_result)."""
        return (
            self._run("market",     self._pipe_mkt, market_texts),
            self._run("regulatory", self._pipe_reg, reg_texts),
            self._run("temporal",   self._pipe_tmp, tmp_texts),
        )

    def ping(self) -> bool:
        return True

    def _run(self, dimension: str, pipe, texts: list) -> dict:
        if not texts:
            return {"dimension": dimension, "scores": [], "n_ambiguous": 0}
        raw = pipe(texts, batch_size=8, truncation=True, max_length=512)
        processed, n_amb = [], 0
        for item in raw:
            sd = {s["label"].lower(): s["score"] for s in item}
            if max(sd.values()) < self._min_conf:
                n_amb += 1
                sd["ambiguous"] = True
            processed.append(sd)
        return {"dimension": dimension, "scores": processed,
                "n_texts": len(texts), "n_ambiguous": n_amb}


class DimensionRouter:
    """Routes chunks to the correct FinBERT dimension (runs on Node A, no GPU)."""

    _REG = [r"\bSEC\b", r"\blitigation\b", r"\bcompliance\b", r"\bpenalt",
            r"\benforcement\b", r"\brestatement\b", r"\bauditor\b",
            r"\bregulat\b", r"\bFDA\b", r"\bFTC\b"]
    _TMP = [r"\bwill\b", r"\bexpect\b", r"\banticipate\b", r"\bnext quarter\b",
            r"\bforecast\b", r"\bguidance\b", r"\boutlook\b", r"\bFY\d{4}\b",
            r"\bgoing forward\b", r"\bfuture\b"]

    def __init__(self):
        self._reg_re = [re.compile(p, re.I) for p in self._REG]
        self._tmp_re = [re.compile(p, re.I) for p in self._TMP]

    def route(self, chunks: list) -> dict:
        market, reg, tmp = [], [], []
        for c in chunks:
            t = c["text"]
            market.append(t)
            if any(p.search(t) for p in self._reg_re): reg.append(t)
            if any(p.search(t) for p in self._tmp_re): tmp.append(t)
        return {
            "market":     market,
            "regulatory": reg or ["No regulatory content detected."],
            "temporal":   tmp or ["No forward-looking statements detected."],
        }


def aggregate_sentiment_vector(mkt: dict, reg: dict, tmp: dict) -> np.ndarray:
    """Produce a (3, 3) matrix: [Market/Reg/Temporal] × [pos/neu/neg]."""
    def _avg(r):
        sc = r["scores"]
        if not sc:
            return np.array([0.33, 0.34, 0.33])
        pos = np.mean([s.get("positive", 0) for s in sc])
        neu = np.mean([s.get("neutral",  0) for s in sc])
        neg = np.mean([s.get("negative", 0) for s in sc])
        tot = pos + neu + neg or 1.0
        return np.array([pos / tot, neu / tot, neg / tot])
    return np.array([_avg(mkt), _avg(reg), _avg(tmp)])