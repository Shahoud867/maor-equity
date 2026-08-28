"""Single entry point for every experiment.

    python -m maor.cli <command> [options]

Design rules:

* One command per experiment, each writing exactly one provenance-stamped result
  file. There is no path by which a number reaches ``results/`` without recording
  how it was produced.
* ``--dry-run`` on every expensive command, so the wiring can be checked without
  spending GPU time.
* Hardware is verified before work starts, not discovered when it fails.
* Nothing runs unbounded: expensive stages have configurable timeouts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from . import hardware
from .config import Config, ConfigError
from .provenance import (
    EvidenceClass,
    Provenance,
    Timer,
    config_hash,
    write_result,
)

log = logging.getLogger("maor")


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def setup_logging(verbosity: int = 0) -> None:
    level = logging.WARNING if verbosity == 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These libraries log every HTTP request at INFO, which drowns the run log.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {}
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        overrides[key.strip()] = _coerce(raw.strip())
    cfg = Config.load(getattr(args, "config", None), **overrides)
    cfg.paths.ensure()
    return cfg


def _coerce(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _resolve_device(cfg: Config, hw: hardware.HardwareInfo) -> str:
    requested = cfg.execution.device
    if requested == "cuda":
        if not hw.has_cuda:
            raise SystemExit(
                "execution.device=cuda but no CUDA device is available.\n"
                f"Detected: {hw.describe()}\n"
                "Run `python -m maor.cli doctor` for details, or use "
                "--set execution.device=cpu."
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if hw.has_cuda else "cpu"


def _make_provenance(
    experiment: str,
    cfg: Config,
    hw: hardware.HardwareInfo,
    *,
    evidence: EvidenceClass = EvidenceClass.MEASURED,
    caveats: list[str] | None = None,
    derived_from: list[str] | None = None,
) -> Provenance:
    return Provenance(
        evidence_class=evidence,
        experiment=experiment,
        config_sha=config_hash(cfg),
        seed=cfg.execution.seed,
        hardware=hw.to_dict(),
        caveats=caveats or [],
        derived_from=derived_from or [],
    )


def _emit(path: Path, payload: dict[str, Any]) -> None:
    print(f"\nwrote {path}")
    print(json.dumps(payload, indent=2, default=str)[:2000])


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    """Verify hardware, dependencies and data before anything expensive runs."""
    from .data import datasets as ds

    cfg = _load_config(args)
    hw = hardware.probe()

    print("=" * 68)
    print("maor-equity environment check")
    print("=" * 68)

    print(f"\n[hardware]\n  {hw.describe()}")
    print(f"  CUDA available : {hw.has_cuda}")
    print(f"  torch          : {hw.torch_version or 'not installed'}")
    print(f"  torch CUDA     : {hw.cuda_version or 'n/a'}")
    for g in hw.gpus:
        print(f"  GPU {g.index}: {g.name}  {g.total_mb:.0f} MB  cc={g.compute_capability}")

    print("\n[vram budget]")
    if hw.has_cuda or cfg.vram.total_mb:
        budget = hardware.VRAMBudget.from_hardware(
            hw,
            usable_fraction=cfg.vram.usable_fraction,
            override_total_mb=cfg.vram.total_mb,
            device=cfg.vram.device,
        )
        print(f"  total          : {budget.total_mb:.0f} MB")
        print(f"  usable ({cfg.vram.usable_fraction:.0%})  : {budget.usable_mb:.0f} MB")
        print(f"  sentiment needs: {cfg.models.sentiment_estimated_vram_mb:.0f} MB")
        print(f"  summariser     : {cfg.models.summarizer_estimated_vram_mb:.0f} MB")
        both = (
            cfg.models.sentiment_estimated_vram_mb
            + cfg.models.summarizer_estimated_vram_mb
        )
        fits = both <= budget.usable_mb
        print(f"  co-resident    : {both:.0f} MB -> {'fits' if fits else 'DOES NOT FIT'}")
        print(
            f"  phase-serialised: {cfg.vram.phase_serialised}"
            + ("" if fits or cfg.vram.phase_serialised else "  <-- REQUIRED, currently off")
        )
    else:
        print("  no GPU detected; GPU experiments will not run here")

    print("\n[dependencies]")
    required = {
        "torch": "core",
        "transformers": "core",
        "datasets": "H2/H3 data",
        "sklearn": "chunk filter",
        "scipy": "chunk filter",
        "rouge_score": "H2 metrics",
        "numpy": "core",
        "yaml": "config",
        "bert_score": "H2 BERTScore (optional)",
        "matplotlib": "figures (optional)",
        "ray": "distributed mode (optional)",
    }
    missing_core = []
    for mod, why in required.items():
        try:
            __import__(mod)
            status = "ok"
        except Exception:
            status = "MISSING"
            if why == "core":
                missing_core.append(mod)
        print(f"  {mod:<14} {status:<8} ({why})")

    print("\n[data]")
    for name, info in ds.validate_available(cfg).items():
        print(f"  {name:<22} {info.get('status')}")
        for k in ("path", "n_lines", "repo", "remedy", "error"):
            if k in info:
                print(f"      {k}: {info[k]}")

    print("\n[config]")
    print(f"  config sha     : {config_hash(cfg)}")
    print(f"  seed           : {cfg.execution.seed}")
    print(f"  execution mode : {cfg.execution.mode}")
    print(f"  device         : {cfg.execution.device} -> {_resolve_device(cfg, hw)}")

    print("\n" + "=" * 68)
    if missing_core:
        print(f"BLOCKED: missing core dependencies: {', '.join(missing_core)}")
        return 1
    if not hw.has_cuda:
        print("READY for CPU experiments (H3, chunk-filter, ingestion).")
        print("GPU experiments (H1 distributed, H2 summarisation, VRAM) need CUDA.")
    else:
        print("READY for all experiments.")
    return 0


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------


def cmd_h3(args: argparse.Namespace) -> int:
    from .agents.sentiment import DimensionRouter
    from .commands import h3_sequence
    from .data.datasets import load_financial_phrasebank

    cfg = _load_config(args)
    hw = hardware.probe()
    device = _resolve_device(cfg, hw)

    print(f"H3 sentiment dimensionality | device={device} | n={args.n_samples or 'all'}")

    if args.dry_run:
        data = load_financial_phrasebank(n_samples=args.n_samples, seed=cfg.execution.seed)
        print(f"  dataset: {len(data)} sentences, {data.label_distribution()}")
        print("  dry-run: dataset loads and routing is exercised; no model is loaded")
        coverage = DimensionRouter().route(data.texts)["coverage"]
        print(f"  routing coverage: {json.dumps(coverage, indent=2)}")
        return 0

    payload = h3_sequence(
        cfg,
        device,
        n_samples=args.n_samples,
        n_resamples=args.n_resamples,
    )
    out = Path(payload.pop("_result_path"))

    r = payload["results"]
    print("\n" + "-" * 62)
    print(f"  B3 scalar accuracy       : {r['accuracy_b3_pct']:.2f}%")
    print(f"  Multidimensional accuracy: {r['accuracy_multidimensional_pct']:.2f}%")
    print(f"  Delta                    : {r['accuracy_delta_pp']:+.2f} pp")
    print(f"  Direction divergence     : {r['direction_divergence_pct']:.2f}%")
    ci = payload["statistics"]["accuracy_delta_ci_pp"]
    print(f"  Delta 95% CI             : [{ci['lower']:+.2f}, {ci['upper']:+.2f}] pp")
    print(f"  Permutation p            : {payload['statistics']['accuracy_permutation_test']['p_value']}")
    for t in payload["hypothesis_tests"]:
        print(f"  {t['hypothesis_id']}: {t['result']}  ({t['metric_name']} = {t['observed']} {t['units']})")
        for w in t["sanity_warnings"]:
            print(f"      WARNING: {w}")
    print("-" * 62)
    for threat in payload.get("validity_threats", []):
        print(f"  [{threat['severity'].upper()}] {threat['threat']}")
    print(f"\nwrote {out}")
    return 0


# ---------------------------------------------------------------------------
# chunk filter
# ---------------------------------------------------------------------------


def cmd_chunkfilter(args: argparse.Namespace) -> int:
    """Measure the ChunkFilter quality/cost trade-off. CPU only, runs anywhere."""
    from .evaluation.chunk_filter_eval import run_chunk_filter_study

    cfg = _load_config(args)
    hw = hardware.probe()

    with Timer() as timer:
        payload = run_chunk_filter_study(cfg, n_documents=args.n_documents)

    payload["wall_clock_s"] = round(timer.elapsed_s, 2)
    prov = _make_provenance("chunk_filter_study", cfg, hw)
    prov.duration_s = round(timer.elapsed_s, 2)
    out = Path(cfg.paths.results_dir) / "chunk_filter_study.json"
    write_result(out, payload, provenance=prov)
    _emit(out, payload["summary"])
    return 0


# ---------------------------------------------------------------------------
# vram verify
# ---------------------------------------------------------------------------


def cmd_vram(args: argparse.Namespace) -> int:
    from .evaluation.vram_verify import run_vram_verification

    cfg = _load_config(args)
    hw = hardware.probe()
    if not hw.has_cuda:
        print("No CUDA device present. VRAM verification requires a GPU.")
        print(f"Detected: {hw.describe()}")
        print("\nThis command is ready to run; execute it on the GPU node.")
        return 2

    with Timer() as timer:
        payload = run_vram_verification(cfg, dry_run=args.dry_run)
    payload["wall_clock_s"] = round(timer.elapsed_s, 2)
    prov = _make_provenance("vram_verification", cfg, hw)
    prov.duration_s = round(timer.elapsed_s, 2)
    out = Path(cfg.paths.results_dir) / "vram_verification.json"
    write_result(out, payload, provenance=prov)
    _emit(out, payload)
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def cmd_fetch_models(args: argparse.Namespace) -> int:
    from .commands import fetch_models

    return fetch_models(_load_config(args))


def cmd_smoke(args: argparse.Namespace) -> int:
    from .commands import smoke

    cfg = _load_config(args)
    return smoke(cfg, _resolve_device(cfg, hardware.probe()))


def cmd_h2(args: argparse.Namespace) -> int:
    from .commands import h2_summarisation

    cfg = _load_config(args)
    hw = hardware.probe()
    device = _resolve_device(cfg, hw)
    if device == "cpu":
        print(
            "H2 requires generation from a 3.8B model. On CPU this is impractically "
            "slow (hours per document).\n"
            f"Detected: {hw.describe()}\n"
            "This command is ready; run it on the GPU node — see docs/GPU_RUNBOOK.md."
        )
        return 2
    return h2_summarisation(cfg, device, args.n_samples, args.no_bertscore)


def cmd_h1(args: argparse.Namespace) -> int:
    from .commands import h1_latency

    cfg = _load_config(args)
    hw = hardware.probe()
    device = _resolve_device(cfg, hw)
    if device == "cpu":
        print(
            "H1 measures the latency of the full pipeline, including generation "
            "from a 3.8B model. CPU timings would not transfer to the GPU node.\n"
            f"Detected: {hw.describe()}\n"
            "This command is ready; run it on the GPU node — see docs/GPU_RUNBOOK.md."
        )
        return 2
    return h1_latency(
        cfg, device, args.tickers, args.repeats or cfg.execution.n_repeats, args.ablation
    )


def cmd_verify_cluster(args: argparse.Namespace) -> int:
    from .commands import verify_cluster

    return verify_cluster(_load_config(args))


def cmd_report(args: argparse.Namespace) -> int:
    from .reporting import build_report

    return build_report(_load_config(args))


def cmd_gpu_audit(args: argparse.Namespace) -> int:
    from .sequence import gpu_audit

    return gpu_audit(_load_config(args))


def cmd_run_all(args: argparse.Namespace) -> int:
    from .sequence import run_all

    # This command runs unattended for hours. Silence would be indistinguishable
    # from a hang, so progress logging is on unless the caller asked otherwise.
    if getattr(args, "verbose", 0) == 0:
        setup_logging(1)

    cfg = _load_config(args)
    device = _resolve_device(cfg, hardware.probe())
    return run_all(cfg, device, resume=not args.no_resume, only=args.only)


def _global_options() -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand.

    Attached to the top-level parser *and* as a parent of every subparser, so
    ``cli --config X h3-sentiment`` and ``cli h3-sentiment --config X`` both work.
    Argparse otherwise accepts only the first form, and every example in the GPU
    runbook is written in the second — the natural one to type.
    """
    shared = argparse.ArgumentParser(add_help=False)
    # SUPPRESS, not None: a parent parser's defaults are applied again by the
    # subparser, so a plain default would overwrite a value given *before* the
    # subcommand with None. With SUPPRESS the attribute is only set when the flag
    # is actually present, and the earlier value survives. Callers read these
    # with getattr(..., default) since the attribute may be absent entirely.
    shared.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="path to a YAML config",
    )
    shared.add_argument(
        "--set",
        action="append",
        default=argparse.SUPPRESS,
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set execution.device=cpu",
    )
    shared.add_argument(
        "-v", "--verbose", action="count", default=argparse.SUPPRESS
    )
    return shared


def build_parser() -> argparse.ArgumentParser:
    shared = _global_options()
    p = argparse.ArgumentParser(
        prog="python -m maor.cli",
        description="maor-equity experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[shared],
        epilog=(
            "Every command writes a provenance-stamped result to results/.\n"
            "Start with `doctor` to verify the environment."
        ),
    )

    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", parents=[shared], help="verify hardware, dependencies and data")
    d.set_defaults(func=cmd_doctor)

    h3 = sub.add_parser("h3-sentiment", parents=[shared], help="H3: multi-dimensional vs scalar sentiment (CPU-capable)")
    h3.add_argument("--n-samples", type=int, default=None, help="default: whole dataset")
    h3.add_argument("--n-resamples", type=int, default=10_000, help="bootstrap resamples")
    h3.add_argument("--dry-run", action="store_true")
    h3.set_defaults(func=cmd_h3)

    cf = sub.add_parser("chunk-filter", parents=[shared], help="ChunkFilter cost/coverage study (CPU)")
    cf.add_argument("--n-documents", type=int, default=None)
    cf.set_defaults(func=cmd_chunkfilter)

    vr = sub.add_parser("vram-verify", parents=[shared], help="measure real VRAM against the budget (GPU)")
    vr.add_argument("--dry-run", action="store_true")
    vr.set_defaults(func=cmd_vram)

    fm = sub.add_parser("fetch-models", parents=[shared], help="pre-download all checkpoints")
    fm.set_defaults(func=cmd_fetch_models)

    sm = sub.add_parser("smoke", parents=[shared], help="exercise every code path, small and fast")
    sm.set_defaults(func=cmd_smoke)

    h2 = sub.add_parser("h2-summarisation", parents=[shared], help="H2: map-reduce vs single-pass on ECTSum (GPU)")
    h2.add_argument("--n-samples", type=int, default=100)
    h2.add_argument("--no-bertscore", action="store_true", help="skip BERTScore")
    h2.set_defaults(func=cmd_h2)

    h1 = sub.add_parser("h1-latency", parents=[shared], help="H1: latency and the parallelism ceiling (GPU)")
    h1.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL"])
    h1.add_argument("--repeats", type=int, default=None, help="default: execution.n_repeats")
    h1.add_argument(
        "--ablation",
        choices=["minimal", "local", "full"],
        default="local",
        help="'local' (default) crosses the implemented factors: chunk filter x "
        "warm start. 'full' adds the distribution factor and requires "
        "execution.mode=ray.",
    )
    h1.set_defaults(func=cmd_h1)

    vc = sub.add_parser("verify-cluster", parents=[shared], help="check a Ray cluster is usable")
    vc.set_defaults(func=cmd_verify_cluster)

    rp = sub.add_parser("report", parents=[shared], help="regenerate tables and RESULTS_STATUS.md")
    rp.set_defaults(func=cmd_report)

    ga = sub.add_parser("gpu-audit", parents=[shared], help="report device memory, residency and workload feasibility"
    )
    ga.set_defaults(func=cmd_gpu_audit)

    ra = sub.add_parser("run-all", parents=[shared],
        help="run the full experiment sequence with cleanup between runs",
    )
    ra.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="NAME",
        help="run only these experiments (chunk_filter h3_sentiment "
        "vram_verification h2_summarisation h1_latency)",
    )
    ra.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore the checkpoint and re-run everything",
    )
    ra.set_defaults(func=cmd_run_all)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "verbose", 0))
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
