"""Sequential experiment runner with guaranteed cleanup between runs.

Implements the lifecycle:

    initialise -> validate resources -> load -> execute -> save/checkpoint
              -> release -> verify cleanup -> next

The property that matters is **isolation**: one experiment failing — for any
reason, including an out-of-memory error or a timeout — must not prevent the next
from running. That requires cleanup in ``finally``, verification that the cleanup
worked, and a decision rule for when the environment is too damaged to continue.

Checkpointing exists because the full sequence takes hours on a T1000. A run
interrupted at experiment four should resume at experiment four, not at one.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..gpu.lifecycle import ModelRegistry
from ..gpu.memory import MemoryTracker, empty_cache, reset_peak, snapshot
from .timeouts import TimeoutError_, is_oom

log = logging.getLogger(__name__)


class Status(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    OOM = "OOM"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


@dataclass
class ExperimentSpec:
    """One unit of work with its resource requirements and deadline."""

    name: str
    run: Callable[[], dict[str, Any]]
    estimated_vram_mb: float = 0.0
    timeout_s: float = 3600.0
    requires_gpu: bool = False
    # A failure here stops the sequence rather than continuing to the next.
    critical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "estimated_vram_mb": self.estimated_vram_mb,
            "timeout_s": self.timeout_s,
            "requires_gpu": self.requires_gpu,
            "critical": self.critical,
        }


@dataclass
class ExperimentOutcome:
    """What happened, including the instrumentation required for the audit trail."""

    name: str
    status: Status
    started_utc: str
    duration_s: float
    gpu_name: str = "cpu"
    total_vram_mb: float = 0.0
    allocated_before_mb: float = 0.0
    allocated_after_mb: float = 0.0
    reserved_after_mb: float = 0.0
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    residual_mb: float = 0.0
    cleanup_clean: bool = True
    batch_size: int | None = None
    concurrency: int | None = None
    failure_reason: str | None = None
    failure_type: str | None = None
    traceback_text: str | None = None
    result_path: str | None = None
    memory_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.name,
            "status": self.status.value,
            "started_utc": self.started_utc,
            "duration_s": round(self.duration_s, 2),
            "gpu_name": self.gpu_name,
            "total_vram_mb": round(self.total_vram_mb, 1),
            "allocated_before_mb": round(self.allocated_before_mb, 1),
            "allocated_after_mb": round(self.allocated_after_mb, 1),
            "reserved_after_mb": round(self.reserved_after_mb, 1),
            "peak_allocated_mb": round(self.peak_allocated_mb, 1),
            "peak_reserved_mb": round(self.peak_reserved_mb, 1),
            "residual_mb": round(self.residual_mb, 1),
            "cleanup_clean": self.cleanup_clean,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "failure_reason": self.failure_reason,
            "failure_type": self.failure_type,
            "result_path": self.result_path,
            "memory_trace": self.memory_trace,
        }


class ExperimentRunner:
    """Runs experiments in sequence, isolating each from the last.

    ``residual_tolerance_mb`` is how much allocated memory may remain after an
    experiment before the environment is judged unsafe. Exceeding it repeatedly
    means something is retaining models, and continuing would produce an OOM
    attributed to the wrong experiment.
    """

    def __init__(
        self,
        *,
        device: int = 0,
        checkpoint_path: Path | None = None,
        residual_tolerance_mb: float = 256.0,
        max_consecutive_failures: int = 3,
        stop_on_dirty_environment: bool = True,
    ) -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.residual_tolerance_mb = residual_tolerance_mb
        self.max_consecutive_failures = max_consecutive_failures
        self.stop_on_dirty_environment = stop_on_dirty_environment
        self.outcomes: list[ExperimentOutcome] = []
        self._completed: set[str] = set()
        self._load_checkpoint()

    # -- checkpointing ---------------------------------------------------

    def _load_checkpoint(self) -> None:
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                self._completed = set(data.get("completed", []))
                if self._completed:
                    log.info(
                        "resuming: %d experiment(s) already completed (%s)",
                        len(self._completed),
                        ", ".join(sorted(self._completed)),
                    )
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("could not read checkpoint %s: %s", self.checkpoint_path, exc)

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_text(
                json.dumps(
                    {
                        "completed": sorted(self._completed),
                        "updated_utc": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        "outcomes": [o.to_dict() for o in self.outcomes],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("could not write checkpoint: %s", exc)

    # -- environment -----------------------------------------------------

    def _prepare_environment(self) -> None:
        """Bring the device to a known state before an experiment starts."""
        ModelRegistry.instance().release_all(self.device)
        empty_cache()
        reset_peak(self.device)

    def _verify_environment(self, label: str) -> tuple[bool, float]:
        """Confirm the device is clean enough for the next experiment."""
        registry = ModelRegistry.instance()
        leftover = registry.resident(self.device)
        if leftover:
            log.warning(
                "%s left %d model(s) registered: %s — releasing",
                label,
                len(leftover),
                ", ".join(h.label for h in leftover),
            )
            registry.release_all(self.device)

        empty_cache()
        snap = snapshot(self.device, label=f"{label}:verify")
        residual = snap.allocated_mb
        clean = residual <= self.residual_tolerance_mb
        if not clean:
            log.error(
                "after %s, %.0f MB is still allocated (tolerance %.0f MB). "
                "Something is retaining GPU memory across experiments.",
                label,
                residual,
                self.residual_tolerance_mb,
            )
        return clean, residual

    # -- execution -------------------------------------------------------

    def run_all(self, specs: list[ExperimentSpec]) -> list[ExperimentOutcome]:
        """Run every experiment, continuing past non-critical failures."""
        consecutive_failures = 0

        for spec in specs:
            if spec.name in self._completed:
                log.info("skipping %s (already completed in a previous run)", spec.name)
                self.outcomes.append(
                    ExperimentOutcome(
                        name=spec.name,
                        status=Status.SKIPPED,
                        started_utc=datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        duration_s=0.0,
                        failure_reason="already completed (resumed from checkpoint)",
                    )
                )
                continue

            outcome = self.run_one(spec)
            self.outcomes.append(outcome)

            if outcome.status is Status.COMPLETED:
                consecutive_failures = 0
                self._completed.add(spec.name)
            elif outcome.status is Status.BLOCKED:
                # BLOCKED means the hardware is absent, not that anything went
                # wrong. Counting it as a failure trips the circuit breaker and
                # aborts experiments that could still have run — which is what
                # happened on a CPU-only machine where the three GPU experiments
                # ended the sequence. It is also not recorded as completed, so a
                # later run on a GPU picks it up.
                log.info("%s blocked; not counted as a failure", spec.name)
            else:
                consecutive_failures += 1

            self._save_checkpoint()

            if not outcome.cleanup_clean and self.stop_on_dirty_environment:
                log.error(
                    "stopping: %.0f MB still allocated after %s. Continuing would "
                    "produce failures attributed to the wrong experiment. Restart "
                    "the process to reset the CUDA context.",
                    outcome.residual_mb,
                    spec.name,
                )
                break

            if spec.critical and outcome.status is not Status.COMPLETED:
                log.error("stopping: %s is critical and did not complete", spec.name)
                break

            if consecutive_failures >= self.max_consecutive_failures:
                log.error(
                    "stopping after %d consecutive failures", consecutive_failures
                )
                break

        return self.outcomes

    def run_one(self, spec: ExperimentSpec) -> ExperimentOutcome:
        """Run a single experiment with full lifecycle management."""
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.perf_counter()

        self._prepare_environment()
        tracker = MemoryTracker(self.device)
        before = tracker.mark(f"{spec.name}:before")

        outcome = ExperimentOutcome(
            name=spec.name,
            status=Status.RUNNING,
            started_utc=started,
            duration_s=0.0,
            gpu_name=before.gpu_name,
            total_vram_mb=before.total_mb,
            allocated_before_mb=before.allocated_mb,
        )

        if spec.requires_gpu and not before.available:
            outcome.status = Status.BLOCKED
            outcome.failure_reason = (
                "requires a CUDA device; none available. The experiment is "
                "implemented and will run unchanged on a GPU machine."
            )
            outcome.failure_type = "NoCUDADevice"
            outcome.duration_s = time.perf_counter() - t0
            log.warning("%s BLOCKED: no CUDA device", spec.name)
            return outcome

        log.info("=" * 60)
        log.info("running %s (timeout %.0fs)", spec.name, spec.timeout_s)

        from .timeouts import TimeoutGuard

        try:
            with TimeoutGuard(spec.name, spec.timeout_s, interrupt=True):
                payload = spec.run()
            outcome.status = Status.COMPLETED
            if isinstance(payload, dict):
                outcome.result_path = payload.get("result_path")
                outcome.batch_size = payload.get("batch_size")
                outcome.concurrency = payload.get("concurrency")

        except TimeoutError_ as exc:
            outcome.status = Status.TIMED_OUT
            outcome.failure_reason = str(exc)
            outcome.failure_type = "Timeout"
            log.error("%s timed out after %.0fs", spec.name, spec.timeout_s)

        except BaseException as exc:  # noqa: BLE001 - recorded, never swallowed
            outcome.status = Status.OOM if is_oom(exc) else Status.FAILED
            outcome.failure_reason = f"{type(exc).__name__}: {exc}"
            outcome.failure_type = type(exc).__name__
            outcome.traceback_text = traceback.format_exc()[-4000:]
            if outcome.status is Status.OOM:
                log.error(
                    "%s ran out of GPU memory. Not retrying with identical "
                    "parameters: it would fail identically. Reduce batch size, "
                    "enable phase serialisation, or use stronger quantisation.",
                    spec.name,
                )
            else:
                log.error("%s failed: %s", spec.name, outcome.failure_reason)

        finally:
            # Cleanup runs whatever happened, including on timeout and OOM.
            # The traceback holds frame locals — including models — so it is
            # dropped before verifying, or the verification measures the
            # traceback's references rather than a leak.
            try:
                del exc  # type: ignore[possibly-undefined]  # noqa: F821
            except (NameError, UnboundLocalError):
                pass

            tracker.mark(f"{spec.name}:after-execute")
            clean, residual = self._verify_environment(spec.name)
            after = tracker.mark(f"{spec.name}:after-cleanup")

            outcome.duration_s = time.perf_counter() - t0
            outcome.allocated_after_mb = after.allocated_mb
            outcome.reserved_after_mb = after.reserved_mb
            outcome.peak_allocated_mb = tracker.peak_allocated_mb
            outcome.peak_reserved_mb = tracker.peak_reserved_mb
            outcome.residual_mb = residual
            outcome.cleanup_clean = clean
            outcome.memory_trace = tracker.to_dict()

            log.info(
                "%s %s in %.1fs | peak %.0f MB | residual %.0f MB | cleanup %s",
                spec.name,
                outcome.status.value,
                outcome.duration_s,
                outcome.peak_allocated_mb,
                outcome.residual_mb,
                "clean" if clean else "DIRTY",
            )

        return outcome

    # -- reporting -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for o in self.outcomes:
            by_status[o.status.value] = by_status.get(o.status.value, 0) + 1
        runnable = [o for o in self.outcomes if o.status is not Status.BLOCKED]
        return {
            "n_experiments": len(self.outcomes),
            "by_status": by_status,
            # Everything that *could* run did. BLOCKED is missing hardware, not
            # a failure, so a CPU-only session that completes its CPU work is a
            # success with three experiments deferred.
            "all_runnable_completed": all(
                o.status in (Status.COMPLETED, Status.SKIPPED) for o in runnable
            ),
            "all_completed": all(
                o.status in (Status.COMPLETED, Status.SKIPPED) for o in self.outcomes
            ),
            "n_blocked": sum(1 for o in self.outcomes if o.status is Status.BLOCKED),
            "blocked_experiments": [
                o.name for o in self.outcomes if o.status is Status.BLOCKED
            ],
            "any_dirty_cleanup": any(not o.cleanup_clean for o in self.outcomes),
            "peak_allocated_mb": round(
                max((o.peak_allocated_mb for o in self.outcomes), default=0.0), 1
            ),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 72, flush=True)
        print("EXPERIMENT SEQUENCE SUMMARY")
        print("=" * 72, flush=True)
        print(
            f"  {'experiment':<28}{'status':<12}{'time':>8}"
            f"{'peak MB':>10}{'residual':>10}"
        )
        for o in self.outcomes:
            print(
                f"  {o.name:<28}{o.status.value:<12}{o.duration_s:>7.0f}s"
                f"{o.peak_allocated_mb:>10.0f}{o.residual_mb:>10.0f}"
                + ("" if o.cleanup_clean else "  DIRTY")
            )
        s = self.summary()
        print("-" * 72, flush=True)
        print(f"  {s['by_status']}")
        if s["any_dirty_cleanup"]:
            print("  WARNING: at least one experiment left GPU memory allocated.")
        for o in self.outcomes:
            if o.failure_reason and o.status is not Status.SKIPPED:
                print(f"\n  {o.name}: {o.failure_reason}")
        print("=" * 72, flush=True)
