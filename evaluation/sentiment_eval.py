"""
evaluation/sentiment_eval.py — H3 Evaluation
Tests whether 3-D sentiment changes guardrail direction vs B3 scalar baseline.

H3: Three-dimensional sentiment features change directional recommendation
    in >10% of filings compared to single-dimension baseline (B3).

Datasets:
  - Financial PhraseBank: https://huggingface.co/datasets/financial_phrasebank
  - FiQA Task 1 (aspect-level sentiment)

Run on Node A after cluster is connected:
    pip install datasets
    python -m evaluation.sentiment_eval --n-samples 100
"""
import argparse
import json
from pathlib import Path

import ray
import numpy as np


def load_financial_phrasebank(n_samples: int) -> list:
    """Load Financial PhraseBank (sentences_75agree split)."""
    from datasets import load_dataset
    ds = load_dataset("financial_phrasebank", "sentences_75agree", split="train")
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    samples = []
    for item in list(ds)[:n_samples]:
        samples.append({
            "text": item["sentence"],
            "label": label_map[item["label"]],
        })
    return samples


def run_3d_sentiment(texts: list) -> list:
    """Run our 3-D FinBERT pipeline. Returns list of (market, regulatory, temporal) vectors."""
    from agents.sentiment_agent import FinBERTActor, DimensionRouter, aggregate_sentiment_vector

    sent_mkt = FinBERTActor.remote("ProsusAI/finbert", "market")
    sent_reg = FinBERTActor.remote("yiyanghkust/finbert-tone", "regulatory")
    sent_tmp = FinBERTActor.remote("ProsusAI/finbert", "temporal")
    router   = DimensionRouter()

    results = []
    for text in texts:
        chunks = [{"chunk_id": 0, "text": text}]
        routed = router.route(chunks)
        mkt_r, reg_r, tmp_r = ray.get([
            sent_mkt.classify_batch.remote(routed["market"]),
            sent_reg.classify_batch.remote(routed["regulatory"]),
            sent_tmp.classify_batch.remote(routed["temporal"]),
        ])
        sv = aggregate_sentiment_vector(mkt_r, reg_r, tmp_r)
        # Directional call from 3-D: use market dimension positive vs negative
        market_pos, market_neg = sv[0, 0], sv[0, 2]
        direction_3d = "positive" if market_pos > market_neg else (
            "negative" if market_neg > market_pos else "neutral"
        )
        results.append({
            "sentiment_vector": sv.tolist(),
            "direction_3d": direction_3d,
            "market_pos": float(market_pos),
            "market_neg": float(market_neg),
        })
    return results


def run_b3_scalar(texts: list) -> list:
    """Run B3 baseline: single-dimension ProsusAI/finbert."""
    from baselines.b3_sentiment_baseline import run_scalar_sentiment
    return [run_scalar_sentiment(t) for t in texts]


def compute_h3_stats(results_3d: list, results_b3: list, ground_truth_labels: list) -> dict:
    """Compare directions and compute agreement/disagreement with each other and ground truth."""
    disagreements = 0
    agree_3d_gt = 0
    agree_b3_gt = 0

    for r3d, rb3, gt in zip(results_3d, results_b3, ground_truth_labels):
        dir_3d = r3d["direction_3d"]
        dir_b3 = rb3.get("direction", "neutral")

        if dir_3d != dir_b3:
            disagreements += 1

        if dir_3d == gt:
            agree_3d_gt += 1
        if dir_b3 == gt:
            agree_b3_gt += 1

    n = len(results_3d)
    change_pct = disagreements / n * 100
    acc_3d = agree_3d_gt / n * 100
    acc_b3 = agree_b3_gt / n * 100

    return {
        "H3_hypothesis": "3-D sentiment changes direction in >10% of filings vs B3",
        "n_samples": n,
        "disagreement_count": disagreements,
        "disagreement_pct": round(change_pct, 1),
        "H3_passed": change_pct > 10,
        "H3_note": (
            f"PASS ({change_pct:.1f}% of filings changed direction vs B3)" if change_pct > 10
            else f"FAIL ({change_pct:.1f}% — target >10%)"
        ),
        "accuracy_3d_vs_ground_truth_pct": round(acc_3d, 1),
        "accuracy_b3_vs_ground_truth_pct": round(acc_b3, 1),
        "3d_improves_accuracy": acc_3d > acc_b3,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--skip-b3", action="store_true")
    args = parser.parse_args()

    ray.init(address="auto")
    Path("logs").mkdir(exist_ok=True)

    print(f"Loading {args.n_samples} Financial PhraseBank samples ...")
    samples = load_financial_phrasebank(args.n_samples)
    texts   = [s["text"] for s in samples]
    labels  = [s["label"] for s in samples]

    print("\nRunning 3-D sentiment pipeline ...")
    results_3d = run_3d_sentiment(texts)
    with open("logs/h3_3d_predictions.json", "w") as f:
        json.dump(results_3d, f, indent=2)

    if args.skip_b3 and Path("logs/b3_predictions.json").exists():
        with open("logs/b3_predictions.json") as f:
            results_b3 = json.load(f)
    else:
        print("\nRunning B3 scalar baseline ...")
        results_b3 = run_b3_scalar(texts)
        with open("logs/b3_predictions.json", "w") as f:
            json.dump(results_b3, f, indent=2)

    stats = compute_h3_stats(results_3d, results_b3, labels)

    print("\n" + "="*60)
    print("H3 SENTIMENT EVALUATION RESULTS")
    print("="*60)
    print(f"Samples evaluated:      {stats['n_samples']}")
    print(f"Direction changes:      {stats['disagreement_count']} ({stats['disagreement_pct']}%)")
    print(f"3-D accuracy vs GT:     {stats['accuracy_3d_vs_ground_truth_pct']}%")
    print(f"B3 accuracy vs GT:      {stats['accuracy_b3_vs_ground_truth_pct']}%")
    print(f"3-D improves accuracy:  {stats['3d_improves_accuracy']}")
    print(f"Result:                 {stats['H3_note']}")
    print("="*60)

    with open("logs/h3_sentiment_results.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved → logs/h3_sentiment_results.json")
