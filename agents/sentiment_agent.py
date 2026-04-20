"""
sentiment_agent.py  —  Node B (GPU)
3-dimensional FinBERT sentiment: Market / Regulatory / Temporal.
"""
import re
import ray
import numpy as np


@ray.remote(num_gpus=0.25)          # 3 actors share GPU  →  0.75 GPU total
class FinBERTActor:
    """One FinBERT classifier for one sentiment dimension."""

    def __init__(self, checkpoint: str, dimension: str, min_confidence: float = 0.60):
        from transformers import pipeline
        self.dimension = dimension
        self.min_conf  = min_confidence
        self.pipe = pipeline(
            "text-classification", model=checkpoint,
            device=0, top_k=None,
        )
        print(f"[FinBERTActor:{dimension}] loaded {checkpoint}")

    def classify_batch(self, texts: list) -> dict:
        if not texts:
            return {"dimension": self.dimension, "scores": [], "n_ambiguous": 0}
        raw = self.pipe(texts, batch_size=8, truncation=True, max_length=512)
        processed, n_amb = [], 0
        for item in raw:
            sd = {s["label"].lower(): s["score"] for s in item}
            if max(sd.values()) < self.min_conf:
                n_amb += 1
                sd["ambiguous"] = True
            processed.append(sd)
        return {"dimension": self.dimension, "scores": processed,
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