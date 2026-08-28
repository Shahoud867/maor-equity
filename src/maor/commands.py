"""Command implementations that need heavier imports than the CLI shell.

Kept out of ``cli.py`` so that ``--help`` and ``doctor`` stay fast and work even
when optional dependencies are missing.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from . import hardware
from .execution.timeouts import TimeoutGuard, run_with_timeout
from .gpu.lifecycle import ModelRegistry
from .gpu.memory import snapshot as gpu_snapshot
from .hardware import release_vram
from .provenance import EvidenceClass, Provenance, Timer, config_hash, write_result

log = logging.getLogger(__name__)


def _provenance(
    experiment: str,
    cfg: Any,
    hw: hardware.HardwareInfo,
    *,
    evidence: EvidenceClass = EvidenceClass.MEASURED,
    caveats: list[str] | None = None,
) -> Provenance:
    return Provenance(
        evidence_class=evidence,
        experiment=experiment,
        config_sha=config_hash(cfg),
        seed=cfg.execution.seed,
        hardware=hw.to_dict(),
        caveats=caveats or [],
    )


def _budget(cfg: Any, hw: hardware.HardwareInfo) -> hardware.VRAMBudget:
    return hardware.VRAMBudget.from_hardware(
        hw,
        usable_fraction=cfg.vram.usable_fraction,
        override_total_mb=cfg.vram.total_mb,
        device=cfg.vram.device,
    )


# ---------------------------------------------------------------------------
# fetch-models
# ---------------------------------------------------------------------------


def fetch_models(cfg: Any) -> int:
    """Pre-download every checkpoint so a run never stalls mid-download."""
    from huggingface_hub import snapshot_download

    checkpoints = [
        ("sentiment_market", cfg.models.sentiment_market),
        ("sentiment_regulatory", cfg.models.sentiment_regulatory),
        ("summarizer", cfg.models.summarizer),
    ]
    print("Pre-downloading checkpoints (this is the slow step; do it once)\n")
    failures = 0
    for role, repo in checkpoints:
        print(f"  {role:<22} {repo}")
        try:
            path = snapshot_download(repo)
            size_mb = sum(
                f.stat().st_size for f in Path(path).rglob("*") if f.is_file()
            ) / (1024 * 1024)
            print(f"  {'':<22} cached, {size_mb:,.0f} MB\n")
        except Exception as exc:
            failures += 1
            print(f"  {'':<22} FAILED: {type(exc).__name__}: {exc}\n")

    if failures:
        print(f"{failures} checkpoint(s) failed. Check network access or set HF_TOKEN.")
        return 1
    print("All checkpoints cached.")
    return 0


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def smoke(cfg: Any, device: str) -> int:
    """Exercise every code path end to end, small and fast.

    Proves the wiring before GPU hours are committed. Output is a wiring check,
    not evidence, and says so in the result file.
    """
    from .agents.sentiment import DimensionRouter, SentimentBundle, build_matrix
    from .data.chunking import ChunkFilter, chunk_document
    from .data.datasets import load_ectsum

    hw = hardware.probe()
    budget = _budget(cfg, hw)
    checks: list[dict[str, Any]] = []

    def check(name: str, fn) -> Any:
        t0 = time.perf_counter()
        try:
            value = fn()
            checks.append(
                {"check": name, "status": "ok", "seconds": round(time.perf_counter() - t0, 2)}
            )
            print(f"  {name:<34} ok    ({time.perf_counter() - t0:.1f}s)")
            return value
        except Exception as exc:
            checks.append(
                {
                    "check": name,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
            print(f"  {name:<34} FAILED  {type(exc).__name__}: {exc}")
            raise

    print(f"Smoke test | device={device} | {hw.describe()}\n")

    try:
        data = check("load ECTSum", lambda: load_ectsum(cfg.data.ectsum_path, n_samples=2))
        document = data.documents[0]

        chunks = check(
            "chunk document",
            lambda: chunk_document(
                document,
                window_tokens=cfg.chunking.window_tokens,
                stride_tokens=cfg.chunking.stride_tokens,
            ),
        )
        filtered = check(
            "filter chunks",
            lambda: ChunkFilter(
                sim_threshold=cfg.chunking.sim_threshold,
                max_chunks_8k=cfg.chunking.max_chunks_8k,
            ).filter(chunks),
        )
        routed = check(
            "route dimensions",
            lambda: DimensionRouter().route(filtered.chunks),
        )

        bundle = check(
            "load sentiment models",
            lambda: SentimentBundle(
                market_checkpoint=cfg.models.sentiment_market,
                regulatory_checkpoint=cfg.models.sentiment_regulatory,
                device=device,
                quantisation=cfg.models.sentiment_quantisation,
                batch_size=cfg.models.sentiment_batch_size,
            ).load(),
        )
        results = check(
            "classify sentiment", lambda: bundle.classify_all(routed["routed"])
        )
        matrix = check("build sentiment matrix", lambda: build_matrix(results))
        check("release sentiment models", lambda: bundle.unload())

        vram_after_phase_a = hardware.current_vram_mb(cfg.vram.device)

        summariser_ok = False
        if device == "cuda" or cfg.models.summarizer_quantisation == "none":
            from .agents.summarisation import MapReduceSummariser, SummarisationModel

            model = check(
                "load summariser",
                lambda: SummarisationModel(
                    checkpoint=cfg.models.summarizer,
                    device=device,
                    quantisation=cfg.models.summarizer_quantisation,
                    max_input_tokens=cfg.models.max_input_tokens,
                ).load(),
            )
            summary = check(
                "map-reduce summarise",
                lambda: MapReduceSummariser(
                    model,
                    map_max_new_tokens=cfg.models.map_max_new_tokens,
                    reduce_max_new_tokens=cfg.models.reduce_max_new_tokens,
                ).summarise([c.text for c in filtered.chunks[:2]]),
            )

            from .agents.guardrail import GuardrailAgent

            verdict = check(
                "guardrail assess",
                lambda: GuardrailAgent(
                    model, max_new_tokens=cfg.models.guardrail_max_new_tokens
                ).assess(summary.summary, matrix, {"rsi": 55.0}),
            )
            check("release summariser", lambda: model.unload())
            summariser_ok = True
        else:
            print(
                "  skipping summariser on CPU: Phi-3-mini generation is impractically "
                "slow here.\n  Set models.summarizer_quantisation=none and a small "
                "model to exercise it locally."
            )

    except Exception:
        _write_smoke(cfg, hw, checks, passed=False)
        print("\nSMOKE TEST FAILED — see the failing check above.")
        return 1

    payload = _write_smoke(
        cfg,
        hw,
        checks,
        passed=True,
        extra={
            "n_chunks_raw": filtered.n_before,
            "n_chunks_kept": filtered.n_after,
            "sentiment_dimensions_present": matrix.n_present,
            "vram_after_phase_a_mb": round(vram_after_phase_a, 1),
            "summariser_exercised": summariser_ok,
            "budget": budget.snapshot(),
        },
    )
    print(f"\nSMOKE TEST PASSED — {payload}")
    return 0


def _write_smoke(
    cfg: Any,
    hw: hardware.HardwareInfo,
    checks: list[dict[str, Any]],
    *,
    passed: bool,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(cfg.paths.results_dir) / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = _provenance(
        "smoke_test",
        cfg,
        hw,
        caveats=[
            "Wiring check only. Sample sizes are far too small to support any "
            "claim; these numbers are not evidence."
        ],
    )
    payload = {
        "experiment": "smoke_test",
        "passed": passed,
        "checks": checks,
        **(extra or {}),
    }
    return write_result(out_dir / "smoke.json", payload, provenance=prov)


# ---------------------------------------------------------------------------
# h2
# ---------------------------------------------------------------------------


def h2_summarisation(cfg: Any, device: str, n_samples: int, no_bertscore: bool) -> int:
    from .agents.summarisation import SummarisationModel
    from .data.datasets import load_ectsum
    from .evaluation.h2_summarisation import run_h2

    hw = hardware.probe()
    budget = _budget(cfg, hw)

    data = load_ectsum(cfg.data.ectsum_path, n_samples=n_samples, seed=cfg.execution.seed)
    print(f"H2 summarisation | device={device} | n={len(data)} documents")
    print(f"  mean document length: {data.source['mean_document_words']} words")

    model = SummarisationModel(
        checkpoint=cfg.models.summarizer,
        device=device,
        quantisation=cfg.models.summarizer_quantisation,
        max_input_tokens=cfg.models.max_input_tokens,
        trust_remote_code=cfg.models.trust_remote_code,
        do_sample=cfg.models.do_sample,
        temperature=cfg.models.temperature,
    )

    with Timer() as timer:
        with budget.phase("summariser", cfg.models.summarizer_estimated_vram_mb):
            model.load()
            result = run_h2(
                model=model,
                documents=data.documents,
                references=data.references,
                dataset_source=data.source,
                config=cfg,
                compute_bertscore_metric=not no_bertscore,
                seed=cfg.execution.seed,
            )
            model.unload()

    payload = result.payload
    payload["wall_clock_s"] = round(timer.elapsed_s, 2)
    prov = _provenance(
        "H2_summarisation_quality",
        cfg,
        hw,
        caveats=[t["threat"] for t in payload.get("validity_threats", [])],
    )
    prov.duration_s = round(timer.elapsed_s, 2)

    results_dir = Path(cfg.paths.results_dir)
    out = write_result(results_dir / "h2_summarisation.json", payload, provenance=prov)

    # Predictions saved separately so scores can be recomputed without the model.
    (results_dir / "h2_predictions.json").write_text(
        json.dumps(
            {
                "map_reduce": result.map_reduce_predictions,
                "baseline_b2": result.baseline_predictions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mr = payload["systems"]["map_reduce"]["rouge"]
    b2 = payload["systems"]["baseline_b2_single_pass"]["rouge"]
    d = payload["rouge_l_delta"]
    print("\n" + "-" * 62)
    print(f"  {'metric':<16}{'map-reduce':>14}{'B2 single-pass':>18}")
    for key, label in (("rouge_1", "ROUGE-1"), ("rouge_2", "ROUGE-2"), ("rouge_l", "ROUGE-L")):
        print(f"  {label:<16}{mr[key]:>14.3f}{b2[key]:>18.3f}")
    print(f"\n  ROUGE-L delta : {d['point']:+.3f} [{d['ci_lower']:+.3f}, {d['ci_upper']:+.3f}]")
    for t in payload["hypothesis_tests"]:
        print(f"  {t['hypothesis_id']}: {t['result']}")
    print("-" * 62)
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# h1
# ---------------------------------------------------------------------------


def h1_latency(
    cfg: Any, device: str, tickers: list[str], repeats: int, ablation: str
) -> int:
    from .data.datasets import load_ectsum
    from .evaluation.h1_latency import (
        Condition,
        RunOutcome,
        default_conditions,
        local_conditions,
        run_h1,
    )
    from .pipeline.orchestrator import Pipeline

    hw = hardware.probe()
    budget = _budget(cfg, hw)

    # Documents stand in for filings so the benchmark does not depend on live
    # SEC availability, which would make repeats non-comparable.
    data = load_ectsum(cfg.data.ectsum_path, n_samples=len(tickers), seed=cfg.execution.seed)
    documents = dict(zip(tickers, data.documents))

    if ablation == "full":
        conditions = default_conditions(include_cold=True)
    elif ablation == "local":
        conditions = local_conditions(include_cold=True)
    else:
        conditions = [
            Condition(distributed=False, filter_enabled=True, warm_start=True),
            Condition(distributed=False, filter_enabled=False, warm_start=True),
        ]

    if any(c.distributed for c in conditions) and cfg.execution.mode != "ray":
        print(
            "\nThe 'full' ablation includes distributed conditions, but the Ray\n"
            "execution path is not implemented. Those cells would run identical\n"
            "single-node code and report a distribution effect of ~0 that nothing\n"
            "distributed actually produced — a fabricated null result.\n\n"
            "Use --ablation local to cross the factors that are implemented\n"
            "(chunk filter x warm start). The distribution factor stays PENDING\n"
            "in docs/RESULTS_STATUS.md until a Ray path exists to measure it."
        )
        return 2

    print(f"H1 latency | device={device} | {len(conditions)} conditions "
          f"x {len(tickers)} documents x {repeats} repeats")

    # Exactly one pipeline is resident at a time. Caching one per condition
    # would keep up to six warm pipelines alive simultaneously, each holding its
    # own ~2.8 GB summariser — roughly 17 GB on a 4 GB card. The pipeline is
    # rebuilt when the condition changes, and the previous one is closed first.
    current: dict[str, Any] = {"key": None, "pipeline": None}
    memory_marks: list[dict[str, Any]] = []

    def _close_current() -> None:
        pipe = current.get("pipeline")
        if pipe is None:
            return
        try:
            pipe.close()
        except Exception as exc:
            log.warning("error closing pipeline %s: %s", current.get("key"), exc)
        finally:
            current["pipeline"] = None
            current["key"] = None
            release_vram()

    def execute(condition: Condition, ticker: str, repeat: int) -> RunOutcome:
        key = condition.name
        if current["key"] != key:
            _close_current()
            memory_marks.append(
                {"event": f"before {key}", **gpu_snapshot(cfg.vram.device).to_dict()}
            )
            current["pipeline"] = Pipeline(
                cfg,
                budget=budget,
                device=device,
                warm_start=condition.warm_start,
                filter_enabled=condition.filter_enabled,
                distributed=condition.distributed,
            )
            current["key"] = key

        pipe = current["pipeline"]
        t0 = time.perf_counter()
        with TimeoutGuard(
            f"h1:{key}:{ticker}", cfg.execution.stage_timeout_s, interrupt=True
        ):
            output = pipe.run(ticker=ticker, document=documents[ticker])
        return RunOutcome(
            condition=condition,
            ticker=ticker,
            repeat=repeat,
            total_s=time.perf_counter() - t0,
            recorder=output.recorder,
            metadata={"warnings": output.warnings},
        )

    with Timer() as timer:
        try:
            payload = run_h1(
                execute=execute,
                tickers=tickers,
                conditions=conditions,
                n_repeats=repeats,
                seed=cfg.execution.seed,
            )
        finally:
            _close_current()
            memory_marks.append(
                {"event": "after all conditions", **gpu_snapshot(cfg.vram.device).to_dict()}
            )
            registry_audit = ModelRegistry.instance().audit(cfg.vram.device)
            if registry_audit["tracked_models"]:
                log.warning(
                    "models still resident after H1: %s", registry_audit["tracked_models"]
                )
                ModelRegistry.instance().release_all(cfg.vram.device)

    payload["gpu_memory"] = {
        "marks": memory_marks,
        "registry_audit_after": registry_audit,
        "note": (
            "One pipeline is resident at a time; the previous is closed before the "
            "next condition is constructed. Allocation at 'after all conditions' "
            "should be near the value before the first condition."
        ),
    }

    payload["wall_clock_s"] = round(timer.elapsed_s, 2)
    prov = _provenance("H1_latency_and_parallelism_ceiling", cfg, hw)
    prov.duration_s = round(timer.elapsed_s, 2)
    out = write_result(
        Path(cfg.paths.results_dir) / "h1_latency.json", payload, provenance=prov
    )

    print("\n" + "-" * 62)
    for name, summary in payload["conditions"].items():
        print(f"  {name:<32} median {summary['total_s']['median']:>9.2f} s")
    print()
    for key, contrast in payload["contrasts"].items():
        print(f"  {contrast['label']:<44} speedup {contrast['speedup']}")
    ceiling = payload["parallelism_ceiling"]
    print(f"\n  Amdahl bound (n=2) : {ceiling['amdahl_bound_n2']}")
    print(f"  Measured speedup   : {ceiling['measured_speedup']}")
    for t in payload["hypothesis_tests"]:
        print(f"  {t['hypothesis_id']}: {t['result']}")
    print("-" * 62)
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# verify-cluster
# ---------------------------------------------------------------------------


def verify_cluster(cfg: Any) -> int:
    """Check a Ray cluster is usable before running distributed experiments."""
    try:
        import ray
    except ImportError:
        print("ray is not installed. pip install 'ray[default]'")
        return 1

    address = cfg.execution.ray_address or "auto"
    print(f"Connecting to Ray at {address} ...")
    try:
        ray.init(address=address, ignore_reinit_error=True, logging_level=logging.ERROR)
    except Exception as exc:
        print(f"Could not connect: {type(exc).__name__}: {exc}")
        print("\nStart a head node with:  ray start --head --port=6380")
        return 1

    try:
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        resources = ray.cluster_resources()
        print(f"\n  alive nodes : {len(nodes)}")
        for n in nodes:
            print(f"    {n.get('NodeManagerAddress')}  "
                  f"cpus={n.get('Resources', {}).get('CPU')}  "
                  f"gpus={n.get('Resources', {}).get('GPU', 0)}")
        print(f"  cluster CPU : {resources.get('CPU')}")
        print(f"  cluster GPU : {resources.get('GPU', 0)}")

        @ray.remote
        def _ping() -> str:
            import socket

            return socket.gethostname()

        hosts = ray.get([_ping.remote() for _ in range(max(4, len(nodes) * 2))])
        print(f"  reachable hosts: {sorted(set(hosts))}")

        if len(nodes) < 2:
            print("\n  Only one node. Distributed experiments will not exercise the "
                  "network path; single-node results remain valid for everything "
                  "except the cross-node communication measurement.")
        else:
            print("\n  Cluster ready for distributed experiments.")
        return 0
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# h3 (shared by the CLI command and the sequence runner)
# ---------------------------------------------------------------------------


def h3_sequence(
    cfg: Any,
    device: str,
    *,
    n_samples: int | None = None,
    n_resamples: int = 10_000,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run H3 and write the result. One implementation, two callers.

    The model is released in ``finally`` so an exception part-way through
    classification cannot leave checkpoints resident for the next experiment.
    """
    from .agents.sentiment import DimensionRouter, SentimentBundle
    from .data.datasets import load_financial_phrasebank
    from .evaluation.h3_sentiment import run_h3
    from .gpu.limits import plan_workload

    hw = hardware.probe()
    data = load_financial_phrasebank(n_samples=n_samples, seed=cfg.execution.seed)

    if not quiet:
        print(f"  dataset: {len(data)} sentences, {data.label_distribution()}")
        print(f"  split identified as: {data.source['identified_agreement_split']}")

    # Decide before loading whether this fits, rather than discovering it during
    # the first forward pass.
    plan = plan_workload(
        "h3-sentiment",
        model_mb=cfg.models.sentiment_estimated_vram_mb,
        device=cfg.vram.device,
        usable_fraction=cfg.vram.usable_fraction,
        total_mb_override=cfg.vram.total_mb,
        requested_batch_size=cfg.models.sentiment_batch_size,
        already_resident_mb=ModelRegistry.instance().resident_mb(cfg.vram.device),
    )
    plan.raise_if_blocked()
    for note in plan.adjustments:
        log.info("h3 plan: %s", note)

    bundle = SentimentBundle(
        market_checkpoint=cfg.models.sentiment_market,
        regulatory_checkpoint=cfg.models.sentiment_regulatory,
        device=device,
        quantisation=cfg.models.sentiment_quantisation,
        batch_size=plan.batch_size,
        max_length=cfg.models.sentiment_max_length,
    )

    release_info: Any = None
    with Timer() as timer:
        try:
            run_with_timeout(
                bundle.load,
                cfg.execution.model_load_timeout_s,
                label="h3 sentiment model load",
            )
            if not quiet:
                print(f"  distinct checkpoints resident: {bundle.n_distinct_checkpoints}")
            with TimeoutGuard(
                "h3-classify", cfg.execution.stage_timeout_s * 4, interrupt=True
            ):
                result = run_h3(
                    bundle=bundle,
                    router=DimensionRouter(),
                    texts=data.texts,
                    gold_labels=data.labels,
                    dataset_source=data.source,
                    seed=cfg.execution.seed,
                    n_resamples=n_resamples,
                )
        finally:
            release_info = bundle.unload()

    payload = result.payload
    payload["wall_clock_s"] = round(timer.elapsed_s, 2)
    payload["execution"] = {
        "device": device,
        "batch_size": plan.batch_size,
        "plan": plan.to_dict(),
        "model_release": release_info.to_dict() if release_info is not None else None,
        "gpu_memory_after": gpu_snapshot(cfg.vram.device, label="h3:after").to_dict(),
    }

    prov = _provenance(
        "H3_sentiment_dimensionality",
        cfg,
        hw,
        caveats=[t["threat"] for t in payload.get("validity_threats", [])],
    )
    prov.duration_s = round(timer.elapsed_s, 2)
    out = write_result(
        Path(cfg.paths.results_dir) / "h3_sentiment.json", payload, provenance=prov
    )
    payload["_result_path"] = str(out)
    return payload
