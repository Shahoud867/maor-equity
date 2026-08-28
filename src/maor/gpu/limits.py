"""Proactive capacity planning.

Cleaning up after an out-of-memory error is not enough: by then the CUDA context
may be in a state where subsequent allocations fail for reasons unrelated to the
original cause, and a retry loop around it produces a run that looks hung. The
cheaper approach is to decide, before loading anything, whether the workload fits
— and to say clearly why not when it does not.

The important constraint on this module is research validity. It may adjust
*execution* parameters (batch size, worker concurrency) because those do not
change what is computed. It must not silently adjust parameters that change the
experiment — sample counts, sequence lengths, model choice, quantisation. Those
are the caller's decision, and :func:`plan_workload` reports the shortfall rather
than resolving it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .memory import snapshot

log = logging.getLogger(__name__)


class InsufficientVRAM(RuntimeError):
    """Raised when a workload provably cannot fit, before anything is loaded."""


@dataclass
class WorkloadPlan:
    """The decision about how (or whether) a workload can run."""

    label: str
    fits: bool
    device: int
    free_mb: float
    required_mb: float
    headroom_mb: float
    batch_size: int
    max_concurrent_models: int
    adjustments: list[str] = field(default_factory=list)
    blocking_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "fits": self.fits,
            "device": self.device,
            "free_mb": round(self.free_mb, 1),
            "required_mb": round(self.required_mb, 1),
            "headroom_mb": round(self.headroom_mb, 1),
            "batch_size": self.batch_size,
            "max_concurrent_models": self.max_concurrent_models,
            "execution_adjustments": self.adjustments,
            "blocking_reason": self.blocking_reason,
            "validity_note": (
                "Only execution parameters (batch size, concurrency) are adjusted "
                "automatically; these do not change what is computed. Parameters "
                "that would change the experiment are never adjusted silently."
            ),
        }

    def raise_if_blocked(self) -> None:
        if not self.fits:
            raise InsufficientVRAM(self.blocking_reason or f"{self.label} does not fit")


def plan_workload(
    label: str,
    *,
    model_mb: float,
    device: int = 0,
    usable_fraction: float = 0.85,
    total_mb_override: float | None = None,
    requested_batch_size: int = 8,
    min_batch_size: int = 1,
    per_sample_mb: float = 0.0,
    activation_overhead_mb: float = 0.0,
    already_resident_mb: float = 0.0,
) -> WorkloadPlan:
    """Decide whether a workload fits, and size execution parameters to what is free.

    ``per_sample_mb`` is the marginal activation cost of one extra item in a
    batch. When supplied, the batch size is reduced to fit rather than letting
    the first forward pass discover the limit. Batch size does not change what is
    computed, only how it is grouped, so adjusting it preserves validity — but
    the adjustment is recorded so it appears in the result.
    """
    snap = snapshot(device, label=f"plan:{label}")

    if not snap.available:
        # CPU execution: no VRAM constraint to enforce.
        return WorkloadPlan(
            label=label,
            fits=True,
            device=device,
            free_mb=0.0,
            required_mb=model_mb,
            headroom_mb=0.0,
            batch_size=requested_batch_size,
            max_concurrent_models=1,
            adjustments=["no CUDA device; VRAM planning skipped"],
        )

    total = total_mb_override if total_mb_override is not None else snap.total_mb
    budget_mb = total * usable_fraction
    # What is genuinely available: the budget less what is already held. Using
    # the driver's free figure alone ignores this process's own reservations;
    # using reservations alone ignores other processes.
    available = min(budget_mb - already_resident_mb, snap.free_mb)

    adjustments: list[str] = []
    batch_size = max(min_batch_size, requested_batch_size)

    fixed_required = model_mb + activation_overhead_mb
    if per_sample_mb > 0:
        while batch_size > min_batch_size and (
            fixed_required + batch_size * per_sample_mb > available
        ):
            batch_size //= 2
        if batch_size != requested_batch_size:
            adjustments.append(
                f"batch size reduced {requested_batch_size} -> {batch_size} to fit "
                f"{available:.0f} MB available (execution-only change)"
            )

    required = fixed_required + batch_size * per_sample_mb
    headroom = available - required
    fits = headroom >= 0

    blocking_reason = None
    if not fits:
        blocking_reason = (
            f"{label} needs {required:.0f} MB but only {available:.0f} MB is "
            f"available on device {device} "
            f"({total:.0f} MB total x {usable_fraction:.0%} budget = {budget_mb:.0f} MB, "
            f"{already_resident_mb:.0f} MB already resident, "
            f"{snap.free_mb:.0f} MB reported free by the driver"
            + (
                f", {snap.used_by_others_mb:.0f} MB used by other processes"
                if snap.used_by_others_mb > 64
                else ""
            )
            + "). "
            "Options, in order of preference: release a resident model first; "
            "enable phase serialisation; use stronger quantisation; use a larger "
            "GPU. Reducing sample count would change the experiment and is not "
            "done automatically."
        )

    max_concurrent = max(1, int(available // model_mb)) if model_mb > 0 else 1

    plan = WorkloadPlan(
        label=label,
        fits=fits,
        device=device,
        free_mb=available,
        required_mb=required,
        headroom_mb=headroom,
        batch_size=batch_size,
        max_concurrent_models=max_concurrent,
        adjustments=adjustments,
        blocking_reason=blocking_reason,
    )
    log.info(
        "plan[%s]: %s — need %.0f MB, %.0f MB available, batch=%d",
        label,
        "fits" if fits else "DOES NOT FIT",
        required,
        available,
        batch_size,
    )
    return plan


def recommend_worker_count(
    *,
    model_mb: float,
    device: int = 0,
    usable_fraction: float = 0.85,
    requested_workers: int = 1,
    cpu_workers_ok: bool = True,
) -> tuple[int, list[str]]:
    """How many GPU workers can hold their own model copy.

    Each Ray worker holding its own copy of a model multiplies residency by the
    worker count. On a single small card the answer is almost always one, and
    oversubscribing is a reliable way to produce an OOM that looks like a
    scheduling problem.
    """
    notes: list[str] = []
    snap = snapshot(device)
    if not snap.available:
        if cpu_workers_ok:
            notes.append("no CUDA device; worker count not constrained by VRAM")
            return requested_workers, notes
        return 0, ["no CUDA device and GPU workers required"]

    budget = snap.total_mb * usable_fraction
    capacity = max(1, int(budget // model_mb)) if model_mb > 0 else requested_workers
    workers = min(requested_workers, capacity)
    if workers < requested_workers:
        notes.append(
            f"GPU workers reduced {requested_workers} -> {workers}: each holds a "
            f"~{model_mb:.0f} MB model copy and the budget is {budget:.0f} MB. "
            f"Requesting more would oversubscribe the device."
        )
    if workers == 1 and requested_workers > 1:
        notes.append(
            "Only one worker fits. Data parallelism across workers is not "
            "available on this hardware; the map step will serialise."
        )
    return workers, notes
