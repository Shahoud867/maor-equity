"""B1: All pipeline stages sequential on a single node — latency baseline."""
import json, time, statistics


def run_serial(ticker: str, filing_type: str = "8-K") -> dict:
    from sec_edgar_downloader import Downloader
    from pathlib import Path
    import re, torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import yfinance as yf

    timings = {}
    T0 = time.time()

    # Stage 1 — Ingestion
    t = time.time()
    dl = Downloader("EquityResearchProject", "research@project.local", "data")
    dl.get(filing_type, ticker, limit=1)
    fp   = Path(f"data/sec-edgar-filings/{ticker}/{filing_type}")
    fs   = list(fp.rglob("*.txt"))
    text = re.sub(r"<[^>]+>", " ", fs[0].read_text(errors="replace")) if fs else ""
    timings["ingestion"] = time.time() - t

    # Stage 2 — Sentiment (sequential, GPU)
    t    = time.time()
    pipe = pipeline("text-classification", model="ProsusAI/finbert", device=0)
    _    = pipe([text[:2000]], truncation=True, max_length=512)
    timings["sentiment"] = time.time() - t

    # Stage 3 — Summarisation (sequential, GPU)
    t   = time.time()
    tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    mdl = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", load_in_4bit=True, device_map="auto"
    )
    inp = tok(text[:2000], return_tensors="pt").to("cuda")
    with torch.no_grad():
        mdl.generate(**inp, max_new_tokens=200)
    timings["summarisation"] = time.time() - t

    # Stage 4 — Technical (sequential, CPU)
    t   = time.time()
    h   = yf.Ticker(ticker).history(period="3mo")
    if not h.empty:
        c  = h["Close"]
        dv = c.diff(); g = dv.clip(lower=0).rolling(14).mean()
        l  = (-dv.clip(upper=0)).rolling(14).mean()
        _  = (100 - 100 / (1 + g / l.replace(0, float("inf")))).iloc[-1]
    timings["technical"] = time.time() - t

    timings["total"] = time.time() - T0
    return {"method": "B1_serial", "ticker": ticker, "timings": timings}


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]
    results = []
    for tk in tickers:
        print(f"Serial pipeline → {tk} ...")
        r = run_serial(tk)
        results.append(r)
        print(f"  total: {r['timings']['total']:.2f}s")

    totals = [r["timings"]["total"] for r in results]
    print(f"\n>>> B1 median latency: {statistics.median(totals):.2f}s  (H1 baseline)")
    with open("logs/b1_results.json", "w") as f:
        json.dump(results, f, indent=2)