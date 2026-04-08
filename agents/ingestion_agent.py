"""
ingestion_agent.py  —  Node A (CPU)
Downloads SEC EDGAR filings, strips HTML, and produces overlapping token chunks.
"""
import re
import ray
from pathlib import Path
from sec_edgar_downloader import Downloader


@ray.remote(num_cpus=1)
class IngestionAgent:
    """Node A: Downloads and pre-processes SEC filings."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.dl = Downloader("EquityResearchProject", "research@project.local", str(self.data_dir))

    # ------------------------------------------------------------------
    def fetch_filing(self, ticker: str, filing_type: str = "8-K", limit: int = 1) -> dict:
        try:
            self.dl.get(filing_type, ticker, limit=limit)
            filing_path = self.data_dir / "sec-edgar-filings" / ticker / filing_type
            files = list(filing_path.rglob("*.txt"))
            if not files:
                return {"error": f"No {filing_type} found for {ticker}"}
            raw = files[0].read_text(errors="replace")
            clean = re.sub(r"<[^>]+>", " ", raw)
            clean = re.sub(r"\s+", " ", clean).strip()
            return {"ticker": ticker, "filing_type": filing_type,
                    "text": clean, "word_count": len(clean.split()), "status": "ok"}
        except Exception as exc:
            return {"error": str(exc), "ticker": ticker}

    # ------------------------------------------------------------------
    def chunk_document(self, text: str, chunk_size: int = 512, stride: int = 64) -> list:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        tokens = tok.encode(text)
        chunks, i, cid = [], 0, 0
        while i < len(tokens):
            t = tokens[i: i + chunk_size]
            chunks.append({"chunk_id": cid, "text": tok.decode(t, skip_special_tokens=True),
                           "token_count": len(t), "start_token": i})
            i += chunk_size - stride
            cid += 1
        return chunks