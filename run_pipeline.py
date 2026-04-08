"""
run_pipeline.py  —  CLI entry point (Node A)
Usage:
  python run_pipeline.py --ticker AAPL --filing 8-K --output results/aapl.json
"""
import argparse, json, os, sys
import ray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker",  default="AAPL")
    ap.add_argument("--filing",  default="8-K")
    ap.add_argument("--output",  default="results/output.json")
    ap.add_argument("--address", default="auto")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ray.init(address=args.address)
    print(f"Cluster: {ray.cluster_resources()}")

    from agents.orchestrator import run_pipeline
    result = run_pipeline(args.ticker, args.filing)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*52}")
    print(f"  {args.ticker}  |  {args.filing}")
    print(f"{'='*52}")
    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)
    gr = result["guardrail"]
    print(f"  Recommendation : {gr['recommendation']}")
    print(f"  Confidence     : {gr['confidence']}")
    print(f"  Total time     : {result['timings']['total']:.2f}s")
    print(f"  Chunks         : {result['n_chunks']}")
    print(f"\n  Summary excerpt:\n  {result['summary']['summary'][:280]}...")
    print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()