"""Full experiment sequence and GPU diagnostics.

``run-all`` is the command that matters for an unattended session: it runs every
experiment in order, gives each one a fresh device state and a deadline, verifies
cleanup between them, and checkpoints so an interrupted run resumes where it
stopped rather than at the beginning.

``gpu-audit`` answers "why did that fail?" by separating memory this process
holds from memory another process holds — a distinction the budget alone cannot
make, and a common reason an allocation fails while the budget says it should fit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from . import hardware
from .execution.runner import ExperimentRunner, ExperimentSpec
from .gpu.lifecycle import ModelRegistry
from .gpu.limits import plan_workload
from .gpu.memory import snapshot as gpu_snapshot
from .provenance import Timer, write_result

log = logging.getLogger(__name__)


def gpu_audit(cfg: Any) -> int:
    """Report device memory, tracked residency and workload feasibility."""
    from .commands import _provenance  # local import avoids a cycle

    hw = hardware.probe()
    snap = gpu_snapshot(cfg.vram.device, label="audit")

    print("=" * 68, flush=True)
    print("GPU memory audit")
    print("=" * 68, flush=True)
    print(f"\n[device]\n  {hw.describe()}")

    if not snap.available:
        print("\n  No CUDA device present. GPU experiments cannot run here.")
        print("  Planning below uses the configured budget instead of a measurement.")
    else:
        print(f"  name           : {snap.gpu_name}")
        print(f"  total          : {snap.total_mb:>9.0f} MB")
        print(f"  free (driver)  : {snap.free_mb:>9.0f} MB")
        print(f"  allocated      : {snap.allocated_mb:>9.0f} MB   live tensors")
        print(f"  reserved       : {snap.reserved_mb:>9.0f} MB   allocator pool")
        print(f"  peak allocated : {snap.peak_allocated_mb:>9.0f} MB")
        print(f"  other processes: {snap.used_by_others_mb:>9.0f} MB")
        if snap.used_by_others_mb > 256:
            print(
                "\n  Another process holds memory on this GPU. It is unavailable to\n"
                "  this run no matter what the configured budget says."
            )

    audit = ModelRegistry.instance().audit(cfg.vram.device)
    print("\n[tracked residency in this process]", flush=True)
    if audit["tracked_models"]:
        for m in audit["tracked_models"]:
            print(
                f"  {m['label']:<14}{m['checkpoint']:<42}{m['estimated_mb']:>7.0f} MB"
            )
    else:
        print("  none")
    print(f"  untracked allocated: {audit['untracked_allocated_mb']:.0f} MB", flush=True)

    total = cfg.vram.total_mb or (snap.total_mb if snap.available else 4096.0)
    usable = total * cfg.vram.usable_fraction

    print("\n[workload feasibility]", flush=True)
    for label, model_mb, batch in (
        (
            "sentiment",
            cfg.models.sentiment_estimated_vram_mb,
            cfg.models.sentiment_batch_size,
        ),
        ("summariser", cfg.models.summarizer_estimated_vram_mb, 1),
    ):
        plan = plan_workload(
            label,
            model_mb=model_mb,
            device=cfg.vram.device,
            usable_fraction=cfg.vram.usable_fraction,
            total_mb_override=total,
            requested_batch_size=batch,
        )
        print(
            f"  {label:<12} needs {plan.required_mb:>7.0f} MB  "
            f"-> {'fits' if plan.fits else 'DOES NOT FIT'}"
        )
        if plan.blocking_reason:
            print(f"     {plan.blocking_reason}")

    both = (
        cfg.models.sentiment_estimated_vram_mb + cfg.models.summarizer_estimated_vram_mb
    )
    print(f"\n  co-resident would need {both:.0f} MB of {usable:.0f} MB usable", flush=True)
    marker = "" if (both <= usable or cfg.vram.phase_serialised) else "   <-- REQUIRED"
    print(f"  phase serialisation : {'ON' if cfg.vram.phase_serialised else 'OFF'}{marker}", flush=True)
    print("\n" + "=" * 68)
    return 0


def _experiment_specs(
    cfg: Any, device: str, results_dir: Path
) -> list[ExperimentSpec]:
    """Build the ordered sequence. Cheap and CPU-capable work runs first.

    Ordering is deliberate: the experiments that can run anywhere come first, so
    an unattended session produces those results even if the GPU work later
    fails.
    """
    from . import commands

    def _chunk() -> dict[str, Any]:
        from .evaluation.chunk_filter_eval import run_chunk_filter_study

        hw = hardware.probe()
        with Timer() as t:
            payload = run_chunk_filter_study(cfg, n_documents=None)
        payload["wall_clock_s"] = round(t.elapsed_s, 2)
        prov = commands._provenance("chunk_filter_study", cfg, hw)
        prov.duration_s = round(t.elapsed_s, 2)
        out = write_result(
            results_dir / "chunk_filter_study.json", payload, provenance=prov
        )
        return {"result_path": str(out)}

    def _h3() -> dict[str, Any]:
        commands.h3_sequence(cfg, device)
        return {"result_path": str(results_dir / "h3_sentiment.json")}

    def _vram() -> dict[str, Any]:
        from .evaluation.vram_verify import run_vram_verification

        hw = hardware.probe()
        with Timer() as t:
            payload = run_vram_verification(cfg, dry_run=False)
        payload["wall_clock_s"] = round(t.elapsed_s, 2)
        prov = commands._provenance("vram_verification", cfg, hw)
        prov.duration_s = round(t.elapsed_s, 2)
        out = write_result(
            results_dir / "vram_verification.json", payload, provenance=prov
        )
        return {"result_path": str(out)}

    def _h2() -> dict[str, Any]:
        commands.h2_summarisation(cfg, device, 100, False)
        return {"result_path": str(results_dir / "h2_summarisation.json")}

    def _h1() -> dict[str, Any]:
        # "local", not "full": the distribution factor requires a Ray path that
        # is not implemented, and running those cells would report a
        # distribution effect that nothing distributed produced.
        ablation = "full" if cfg.execution.mode == "ray" else "local"
        commands.h1_latency(
            cfg, device, ["AAPL", "MSFT", "GOOGL"], cfg.execution.n_repeats, ablation
        )
        return {"result_path": str(results_dir / "h1_latency.json")}

    stage = cfg.execution.stage_timeout_s
    return [
        ExperimentSpec("chunk_filter", _chunk, timeout_s=stage * 4),
        ExperimentSpec(
            "h3_sentiment",
            _h3,
            timeout_s=stage * 4,
            estimated_vram_mb=cfg.models.sentiment_estimated_vram_mb,
        ),
        ExperimentSpec(
            "vram_verification",
            _vram,
            requires_gpu=True,
            timeout_s=cfg.execution.model_load_timeout_s * 2,
            estimated_vram_mb=cfg.models.summarizer_estimated_vram_mb,
        ),
        ExperimentSpec(
            "h2_summarisation",
            _h2,
            requires_gpu=True,
            timeout_s=stage * 12,
            estimated_vram_mb=cfg.models.summarizer_estimated_vram_mb,
        ),
        ExperimentSpec(
            "h1_latency",
            _h1,
            requires_gpu=True,
            timeout_s=stage * 16,
            estimated_vram_mb=cfg.models.summarizer_estimated_vram_mb,
        ),
    ]


def run_all(
    cfg: Any, device: str, *, resume: bool = True, only: list[str] | None = None
) -> int:
    """Run the whole sequence with isolation, deadlines and verified cleanup."""
    from .commands import _provenance

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = results_dir / "run_all_checkpoint.json" if resume else None

    specs = _experiment_specs(cfg, device, results_dir)
    known = {s.name for s in specs}
    if only:
        unknown = set(only) - known
        if unknown:
            print(f"unknown experiment(s): {sorted(unknown)}")
            print(f"available: {sorted(known)}")
            return 2
        specs = [s for s in specs if s.name in only]

    hw = hardware.probe()
    print(f"Experiment sequence | {hw.describe()} | device={device}", flush=True)
    print(f"  {len(specs)} experiment(s): {', '.join(s.name for s in specs)}")
    if checkpoint:
        print(f"  checkpoint: {checkpoint}")
    gpu_only = [s.name for s in specs if s.requires_gpu]
    if gpu_only and not hw.has_cuda:
        print(
            f"  note: {', '.join(gpu_only)} require CUDA and will be reported "
            f"BLOCKED, not failed. The rest still run."
        )
    print(flush=True)

    runner = ExperimentRunner(
        device=cfg.vram.device,
        checkpoint_path=checkpoint,
        residual_tolerance_mb=256.0,
    )
    runner.run_all(specs)
    runner.print_summary()

    summary = runner.summary()
    prov = _provenance(
        "experiment_sequence",
        cfg,
        hw,
        caveats=["Execution log, not an experimental result."],
    )
    out = write_result(results_dir / "run_all_log.json", summary, provenance=prov)
    print(f"\nwrote {out}", flush=True)

    if summary["any_dirty_cleanup"]:
        print(
            "\nAt least one experiment left GPU memory allocated. Restart the "
            "process before running more; the CUDA context cannot be reset in "
            "place.",
            flush=True,
        )
        return 1

    if summary["n_blocked"]:
        print(
            f"\n{summary['n_blocked']} experiment(s) deferred for missing hardware: "
            f"{', '.join(summary['blocked_experiments'])}.\n"
            "They are implemented and will run unchanged on a GPU machine; see "
            "docs/GPU_RUNBOOK.md.",
            flush=True,
        )

    # Exit non-zero only when something that *could* have run did not. Deferred
    # GPU work on a CPU-only machine is an expected outcome, not a failure, and
    # returning 1 for it would make a correct unattended run look broken.
    return 0 if summary["all_runnable_completed"] else 1
