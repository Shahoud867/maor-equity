"""
evaluation/rouge_eval.py — H2 Evaluation
Measures summarisation quality (ROUGE-1/2/L + BERTScore) on ECTSum dataset.

H2: Chunked map-reduce summarisation is non-inferior to B2 single-pass
    (ROUGE-L difference ≤ 1.0 point) while handling long documents better.

Datasets:
  - ECTSum: earnings call transcript summarisation benchmark (2,425 pairs)
    https://huggingface.co/datasets/mrSoul7766/ECTSum

Run on Node A after cluster is connected:
    pip install datasets rouge-score bert-score
    python -m evaluation.rouge_eval --n-samples 50
"""
import argparse
import json
import statistics
from pathlib import Path


def load_ectsum(n_samples: int):
    """Load ECTSum test split. Returns list of {transcript, reference_summary}."""
    from datasets import load_dataset
    ds = load_dataset("mrSoul7766/ECTSum", split="test")
    samples = []
    for item in list(ds)[:n_samples]:
        # ECTSum fields: 'text' = transcript, 'summary' = bullet-point summary
        samples.append({
            "text": item.get("text", item.get("input", "")),
            "reference": item.get("summary", item.get("output", "")),
        })
    return samples


def compute_rouge(predictions: list, references: list) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "ROUGE-1": round(statistics.mean(r1) * 100, 2),
        "ROUGE-2": round(statistics.mean(r2) * 100, 2),
        "ROUGE-L": round(statistics.mean(rl) * 100, 2),
    }


def compute_bertscore(predictions: list, references: list) -> dict:
    from bert_score import score
    P, R, F1 = score(predictions, references, lang="en",
                     model_type="distilbert-base-uncased", verbose=False)
    return {"BERTScore-F1": round(F1.mean().item(), 4)}


def run_chunked_mapreduce(samples: list) -> list:
    """Run our map-reduce summarizer (distributed, Node B)."""
    import ray
    from agents.guardrail_agent import Phi3ModelActor
    from agents.summarization_agent import SummarizationAgent
    from agents.ingestion_agent import IngestionAgent

    phi3 = Phi3ModelActor.remote()
    summarizer = SummarizationAgent.remote(phi3)
    ingestion = IngestionAgent.remote()

    predictions = []
    for i, s in enumerate(samples):
        print(f"  [map-reduce] sample {i+1}/{len(samples)} ...")
        chunks = ray.get(ingestion.chunk_document.remote(s["text"]))
        out = ray.get(summarizer.process_document.remote(chunks, f"ectsum_{i}"))
        predictions.append(out["summary"])
    return predictions


def run_b2_single_pass(samples: list) -> list:
    """Run B2 baseline: single-pass Phi-3-mini zero-shot summarization."""
    from baselines.b2_summarization_baseline import run_single_pass
    predictions = []
    for i, s in enumerate(samples):
        print(f"  [B2 single-pass] sample {i+1}/{len(samples)} ...")
        out = run_single_pass(s["text"])
        predictions.append(out)
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of ECTSum samples to evaluate")
    parser.add_argument("--skip-b2", action="store_true",
                        help="Skip B2 baseline (load from logs/b2_predictions.json)")
    args = parser.parse_args()

    import ray
    ray.init(address="auto")
    Path("logs").mkdir(exist_ok=True)

    print(f"Loading {args.n_samples} ECTSum samples ...")
    samples = load_ectsum(args.n_samples)
    references = [s["reference"] for s in samples]

    # Our system
    print("\nRunning chunked map-reduce summarizer ...")
    mapreduce_preds = run_chunked_mapreduce(samples)
    with open("logs/mapreduce_predictions.json", "w") as f:
        json.dump(mapreduce_preds, f, indent=2)

    # B2 baseline
    if args.skip_b2 and Path("logs/b2_predictions.json").exists():
        with open("logs/b2_predictions.json") as f:
            b2_preds = json.load(f)
    else:
        print("\nRunning B2 single-pass baseline ...")
        b2_preds = run_b2_single_pass(samples)
        with open("logs/b2_predictions.json", "w") as f:
            json.dump(b2_preds, f, indent=2)

    # Score both
    print("\nComputing ROUGE scores ...")
    mapreduce_rouge = compute_rouge(mapreduce_preds, references)
    b2_rouge = compute_rouge(b2_preds, references)

    print("Computing BERTScore ...")
    mapreduce_bert = compute_bertscore(mapreduce_preds, references)
    b2_bert = compute_bertscore(b2_preds, references)

    rouge_l_diff = mapreduce_rouge["ROUGE-L"] - b2_rouge["ROUGE-L"]
    h2_passed = rouge_l_diff >= -1.0  # non-inferior: our system ≤ 1.0 point below B2

    results = {
        "H2_hypothesis": "Chunked map-reduce ROUGE-L non-inferior to B2 (diff ≤ 1.0 point)",
        "n_samples": len(samples),
        "map_reduce": {**mapreduce_rouge, **mapreduce_bert},
        "B2_single_pass": {**b2_rouge, **b2_bert},
        "ROUGE_L_difference (ours - B2)": round(rouge_l_diff, 2),
        "H2_passed": h2_passed,
        "H2_note": (
            f"PASS (our ROUGE-L is {rouge_l_diff:+.2f} vs B2)" if h2_passed
            else f"FAIL (our ROUGE-L is {rouge_l_diff:+.2f} vs B2, below -1.0 threshold)"
        ),
    }

    print("\n" + "="*60)
    print("H2 ROUGE EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<20} {'Map-Reduce':>12} {'B2 Single-Pass':>15}")
    print("-"*50)
    for m in ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore-F1"]:
        print(f"{m:<20} {results['map_reduce'][m]:>12} {results['B2_single_pass'][m]:>15}")
    print("-"*50)
    print(f"ROUGE-L difference:  {rouge_l_diff:+.2f}")
    print(f"Result:              {results['H2_note']}")
    print("="*60)

    with open("logs/h2_rouge_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → logs/h2_rouge_results.json")
