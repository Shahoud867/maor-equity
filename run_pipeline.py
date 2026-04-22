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

    # Keep Ray worker Python launch space-safe across nodes.
    os.environ.setdefault("RAY_PYTHON", "python3")

    # Symlink project to a space-free path so Ray's working_dir zip uses a
    # clean path (the OneDrive path has spaces which confuse some tooling).
    _here = os.path.dirname(os.path.abspath(__file__))
    _symlink = "/tmp/maor_equity"
    try:
        if os.path.islink(_symlink):
            os.unlink(_symlink)
        os.symlink(_here, _symlink)
        _working_dir = _symlink
    except OSError:
        _working_dir = _here

    ray.init(
        address=args.address,
        runtime_env={
            "py_executable": "bash ray_python_exec.sh",
            "working_dir": _working_dir,
            "excludes": [
                "venv/", "data/", "results/", "logs/",
                ".git/", "__pycache__/", "*.log", "*.json",
                "edgar-crawler/", "FinRobot/", "TradingAgents/",
                "sec-edgar-downloader/",
            ],
        },
    )
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
    print(f"  Chunks         : {result.get('n_chunks_filtered', result.get('n_chunks', '?'))}")
    print(f"\n  Summary excerpt:\n  {result['summary']['summary'][:280]}...")
    print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()