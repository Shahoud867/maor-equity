"""B3: Single-dimension scalar FinBERT — the sentiment baseline to beat."""
import json, time
from transformers import pipeline


def run(texts: list) -> dict:
    t0   = time.time()
    pipe = pipeline("text-classification", model="ProsusAI/finbert", device=0)
    out  = pipe(texts, truncation=True, max_length=512)
    return {"method": "B3_scalar_sentiment", "results": out,
            "latency_s": time.time() - t0, "dimensions": 1}


if __name__ == "__main__":
    samples = [
        "Revenue grew 12% year-over-year, beating analyst expectations.",
        "The company faces significant regulatory headwinds from the SEC investigation.",
        "Management expects strong growth in the next two quarters.",
    ]
    r = run(samples)
    print(json.dumps(r, indent=2))
    print(f"\n>>> Record B3 latency: {r['latency_s']:.2f}s  (this is your H3 baseline)")