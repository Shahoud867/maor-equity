"""
B1: All pipeline stages sequential on a SINGLE node — latency baseline for H1.

Fair comparison rules (matches distributed pipeline exactly):
  - Full document chunked (512-token windows, 64-token stride)
  - TF-IDF dedup filter, capped at MAX_CHUNKS (same as distributed pipeline)
  - All filtered chunks through FinBERT (3 dimensions, batched)
  - All filtered chunks through Phi-3-mini map-reduce
  - Technical + guardrail identical to distributed pipeline
  - Everything sequential — no Ray, no cross-node transfer

Run on Node B BEFORE starting Ray cluster (models compete for VRAM otherwise).
"""
import gc, json, re, statistics, time
from pathlib import Path

MAX_CHUNKS = 25   # cap matches what distributed pipeline sees after filtering


def _chunk_text(text: str, chunk_size: int = 512, stride: int = 64) -> list:
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import AutoTokenizer
    tok    = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    tokens = tok.encode(text)
    chunks, i, cid = [], 0, 0
    while i < len(tokens):
        t = tokens[i: i + chunk_size]
        chunks.append({"chunk_id": cid,
                       "text": tok.decode(t, skip_special_tokens=True),
                       "token_count": len(t)})
        i += chunk_size - stride
        cid += 1
    return chunks


def _filter_chunks(chunks: list, cap: int = MAX_CHUNKS) -> list:
    """TF-IDF dedup then hard cap — matches ChunkFilter + pipeline behaviour."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        texts = [c["text"] for c in chunks]
        if len(texts) < 2:
            return chunks[:cap]
        tfidf = TfidfVectorizer(max_features=500, stop_words="english")
        mat   = tfidf.fit_transform(texts).toarray()
        keep  = [0]
        for i in range(1, len(chunks)):
            if len(keep) >= cap:
                break
            sims = cosine_similarity(mat[i:i+1], mat[keep])[0]
            if sims.max() < 0.85:
                keep.append(i)
        return [chunks[i] for i in keep]
    except ImportError:
        step = max(1, len(chunks) // cap)
        return chunks[::step][:cap]


def run_serial(ticker: str, filing_type: str = "8-K") -> dict:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import yfinance as yf

    timings = {}
    T0 = time.time()

    # ── Stage 1: Ingestion ────────────────────────────────────────────
    print(f"  [{ticker}] stage 1: ingestion...")
    t  = time.time()
    from sec_edgar_downloader import Downloader
    dl = Downloader("EquityResearchProject", "research@project.local", "data")
    dl.get(filing_type, ticker, limit=1)
    fp   = Path(f"data/sec-edgar-filings/{ticker}/{filing_type}")
    fs   = list(fp.rglob("*.txt"))
    text = re.sub(r"<[^>]+>", " ", fs[0].read_text(errors="replace")) if fs else ""
    text = re.sub(r"\s+", " ", text).strip()
    timings["ingestion"] = time.time() - t
    print(f"  [{ticker}] ingestion done ({len(text.split())} words, {timings['ingestion']:.1f}s)")

    # ── Stage 1b: Chunking + filtering ───────────────────────────────
    print(f"  [{ticker}] chunking + filtering...")
    t          = time.time()
    chunks_raw = _chunk_text(text)
    chunks     = _filter_chunks(chunks_raw, cap=MAX_CHUNKS)
    timings["chunk_filter_ms"] = (time.time() - t) * 1000
    timings["n_chunks_before"] = len(chunks_raw)
    timings["n_chunks_after"]  = len(chunks)
    print(f"  [{ticker}] chunks: {len(chunks_raw)} → {len(chunks)} (capped at {MAX_CHUNKS})")

    # ── Stage 2: FinBERT — all chunks, sequential ─────────────────────
    print(f"  [{ticker}] stage 2: FinBERT sentiment ({len(chunks)} chunks)...")
    t       = time.time()
    finbert = pipeline("text-classification", model="ProsusAI/finbert",
                       device=0, top_k=None, torch_dtype=torch.float16)
    texts   = [c["text"] for c in chunks]
    finbert(texts,        batch_size=8, truncation=True, max_length=512)
    finbert(texts[:3],    batch_size=8, truncation=True, max_length=512)
    finbert(texts[:3],    batch_size=8, truncation=True, max_length=512)
    del finbert
    gc.collect()
    torch.cuda.empty_cache()
    timings["sentiment"] = time.time() - t
    print(f"  [{ticker}] sentiment done ({timings['sentiment']:.1f}s)")

    # ── Stage 3: Phi-3-mini map-reduce — all chunks, sequential ──────
    print(f"  [{ticker}] stage 3: loading Phi-3-mini...")
    t   = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                        trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", quantization_config=bnb,
        device_map={"": 0}, trust_remote_code=True,
    )
    print(f"  [{ticker}] Phi-3-mini loaded, starting map ({len(chunks)} chunks)...")

    def _gen(prompt, max_new=200):
        inp = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=3500).to("cuda")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return tok.decode(out[0][inp["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    chunk_summaries = []
    for idx, c in enumerate(chunks):
        print(f"  [{ticker}] map chunk {idx+1}/{len(chunks)}...", flush=True)
        prompt = (
            f"You are a financial analyst reviewing an SEC filing ({ticker}_{filing_type}).\n"
            "Summarize the following filing segment.\n"
            "Extract: (1) key financial metrics, (2) directional claims, (3) named entities.\n"
            "Output: 3-5 bullet points. Use only information in the text below.\n\n"
            f"Filing segment:\n{c['text'][:3200]}\n\nSummary:"
        )
        chunk_summaries.append({"chunk_id": c["chunk_id"],
                                 "summary": _gen(prompt, 200)})

    print(f"  [{ticker}] reduce step...")
    combined = "\n".join(f"Chunk {s['chunk_id']}: {s['summary']}"
                         for s in chunk_summaries)
    reduce_prompt = (
        "Synthesize the following SEC filing chunk summaries into a final equity research note.\n\n"
        f"Chunk summaries:\n{combined[:3200]}\n\n"
        "Produce:\n1. Executive summary (2 sentences)\n"
        "2. Key financial metrics\n3. Directional outlook\n\nFinal Summary:"
    )
    final_summary = _gen(reduce_prompt, 400)
    timings["summarisation"] = time.time() - t
    print(f"  [{ticker}] summarisation done ({timings['summarisation']:.1f}s)")

    # ── Stage 4: Technical ────────────────────────────────────────────
    print(f"  [{ticker}] stage 4: technical...")
    t  = time.time()
    h  = yf.Ticker(ticker).history(period="3mo")
    if not h.empty:
        c2  = h["Close"]; dv = c2.diff()
        g   = dv.clip(lower=0).rolling(14).mean()
        l   = (-dv.clip(upper=0)).rolling(14).mean()
        _   = (100 - 100 / (1 + g / l.replace(0, float("inf")))).iloc[-1]
    timings["technical"] = time.time() - t

    # ── Stage 5: Guardrail ────────────────────────────────────────────
    print(f"  [{ticker}] stage 5: guardrail...")
    t = time.time()
    for case in ["BULL", "BEAR"]:
        _gen(
            f"Make the strongest {case} case for this equity.\n"
            f"Summary: {final_summary[:800]}\n"
            'JSON only: {"direction":"bullish","confidence":0.0,"signals":["s1","s2"]}',
            150,
        )
    timings["guardrail"] = time.time() - t

    timings["total"] = time.time() - T0
    print(f"  [{ticker}] DONE — total {timings['total']:.1f}s")
    return {"method": "B1_serial_full_doc", "ticker": ticker,
            "n_chunks": len(chunks), "timings": timings}


if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    tickers = ["AAPL", "MSFT", "GOOGL"]
    results = []
    for tk in tickers:
        print(f"\n{'='*50}\nSerial pipeline → {tk}\n{'='*50}")
        r = run_serial(tk)
        results.append(r)

    totals = [r["timings"]["total"] for r in results]
    print(f"\n>>> B1 median latency: {statistics.median(totals):.2f}s")
    with open("logs/b1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved → logs/b1_results.json")
