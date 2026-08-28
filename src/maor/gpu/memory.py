"""GPU memory accounting.

Two numbers matter and only one of them was previously tracked.

``memory_allocated`` is what live tensors occupy. ``memory_reserved`` is what the
caching allocator holds from the driver, which is what actually limits the next
allocation and what the driver reports as used. A model can be dereferenced —
allocated drops to zero — while reserved stays high, so a check based on
``allocated`` alone will report a clean release that has not happened.

Everything here degrades to zeros without CUDA rather than raising, so the same
instrumentation runs on CPU and the recorded shape of a result is identical.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)

MB = 1024 * 1024


def _torch():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


@dataclass(frozen=True)
class MemorySnapshot:
    """A point-in-time view of one device's memory."""

    device: int
    available: bool
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    free_mb: float = 0.0
    total_mb: float = 0.0
    gpu_name: str = "cpu"
    label: str = ""

    @property
    def used_by_others_mb(self) -> float:
        """Memory in use on the device that this process's allocator does not hold.

        Non-zero means another process (or another CUDA context) is sharing the
        card, which is a common cause of an OOM that the local budget cannot
        explain.
        """
        if not self.available:
            return 0.0
        return max(0.0, self.total_mb - self.free_mb - self.reserved_mb)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["used_by_others_mb"] = round(self.used_by_others_mb, 1)
        return d

    def describe(self) -> str:
        if not self.available:
            return f"{self.label or 'snapshot'}: no CUDA device"
        return (
            f"{self.label or 'snapshot'}: allocated={self.allocated_mb:.0f} "
            f"reserved={self.reserved_mb:.0f} free={self.free_mb:.0f} "
            f"total={self.total_mb:.0f} MB"
        )


def snapshot(device: int = 0, *, label: str = "") -> MemorySnapshot:
    """Measure device memory. Never raises."""
    torch = _torch()
    if torch is None:
        return MemorySnapshot(device=device, available=False, label=label)

    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        props = torch.cuda.get_device_properties(device)
        return MemorySnapshot(
            device=device,
            available=True,
            allocated_mb=torch.cuda.memory_allocated(device) / MB,
            reserved_mb=torch.cuda.memory_reserved(device) / MB,
            peak_allocated_mb=torch.cuda.max_memory_allocated(device) / MB,
            peak_reserved_mb=torch.cuda.max_memory_reserved(device) / MB,
            free_mb=free_b / MB,
            total_mb=total_b / MB,
            gpu_name=props.name,
            label=label,
        )
    except Exception as exc:  # pragma: no cover - driver-dependent
        log.warning("could not snapshot device %d: %s", device, exc)
        return MemorySnapshot(device=device, available=False, label=label)


def reset_peak(device: int = 0) -> None:
    """Reset peak counters so the next measurement is scoped to what follows."""
    torch = _torch()
    if torch is None:
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def empty_cache() -> None:
    """Return cached-but-unused blocks to the driver.

    Only useful after references are dropped: it cannot free memory that live
    tensors still hold. Called as the last step of a release, never instead of
    dereferencing.
    """
    gc.collect()
    torch = _torch()
    if torch is None:
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


@dataclass
class ReleaseVerification:
    """Did a release actually return memory?"""

    label: str
    before: MemorySnapshot
    after: MemorySnapshot
    expected_mb: float | None = None
    tolerance_mb: float = 64.0

    @property
    def allocated_freed_mb(self) -> float:
        return self.before.allocated_mb - self.after.allocated_mb

    @property
    def reserved_freed_mb(self) -> float:
        return self.before.reserved_mb - self.after.reserved_mb

    @property
    def residual_mb(self) -> float:
        """Allocated memory still held after the release."""
        return self.after.allocated_mb

    @property
    def clean(self) -> bool:
        """True when the release returned what it should have.

        Judged on *allocated*: reserved memory legitimately stays high because
        the caching allocator keeps blocks for reuse, and forcing it to zero
        would only make the next load slower.
        """
        if not self.before.available:
            return True
        if self.expected_mb is not None:
            return self.allocated_freed_mb >= (self.expected_mb - self.tolerance_mb)
        return self.after.allocated_mb <= (self.before.allocated_mb + self.tolerance_mb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "allocated_freed_mb": round(self.allocated_freed_mb, 1),
            "reserved_freed_mb": round(self.reserved_freed_mb, 1),
            "residual_allocated_mb": round(self.residual_mb, 1),
            "expected_mb": self.expected_mb,
            "clean": self.clean,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }

    def warn_if_dirty(self) -> None:
        if self.clean:
            return
        log.warning(
            "release of %r did not return the expected memory: freed %.0f MB "
            "(expected ~%s MB), %.0f MB still allocated. Something still holds a "
            "reference — check for cached pipelines, retained outputs, or a "
            "traceback holding frame locals.",
            self.label,
            self.allocated_freed_mb,
            "?" if self.expected_mb is None else f"{self.expected_mb:.0f}",
            self.residual_mb,
        )


def verify_released(
    label: str,
    before: MemorySnapshot,
    *,
    device: int = 0,
    expected_mb: float | None = None,
    tolerance_mb: float = 64.0,
) -> ReleaseVerification:
    """Measure what a release actually returned, and warn when it did not."""
    after = snapshot(device, label=f"{label}:after")
    verification = ReleaseVerification(
        label=label,
        before=before,
        after=after,
        expected_mb=expected_mb,
        tolerance_mb=tolerance_mb,
    )
    verification.warn_if_dirty()
    return verification


class MemoryTracker:
    """Records memory at named points across an experiment.

    The trace is the evidence that memory is released rather than accumulating:
    a run whose post-release allocation climbs experiment after experiment has a
    leak, and that is visible here without instrumenting anything else.
    """

    def __init__(self, device: int = 0, *, reset_peak_on_start: bool = True) -> None:
        self.device = device
        self.marks: list[MemorySnapshot] = []
        self.verifications: list[ReleaseVerification] = []
        if reset_peak_on_start:
            reset_peak(device)
        self.mark("start")

    def mark(self, label: str) -> MemorySnapshot:
        snap = snapshot(self.device, label=label)
        self.marks.append(snap)
        log.debug(snap.describe())
        return snap

    def record_release(self, verification: ReleaseVerification) -> None:
        self.verifications.append(verification)

    @property
    def peak_allocated_mb(self) -> float:
        return max((m.peak_allocated_mb for m in self.marks), default=0.0)

    @property
    def peak_reserved_mb(self) -> float:
        return max((m.peak_reserved_mb for m in self.marks), default=0.0)

    def leaked_mb(self) -> float:
        """Allocated memory at the last mark above the first.

        Positive after every model has been released means something is being
        retained across experiments.
        """
        if len(self.marks) < 2:
            return 0.0
        return self.marks[-1].allocated_mb - self.marks[0].allocated_mb

    def to_dict(self) -> dict[str, Any]:
        first = self.marks[0] if self.marks else snapshot(self.device)
        return {
            "device": self.device,
            "gpu_name": first.gpu_name,
            "total_mb": round(first.total_mb, 1),
            "cuda_available": first.available,
            "peak_allocated_mb": round(self.peak_allocated_mb, 1),
            "peak_reserved_mb": round(self.peak_reserved_mb, 1),
            "leaked_allocated_mb": round(self.leaked_mb(), 1),
            "marks": [m.to_dict() for m in self.marks],
            "release_verifications": [v.to_dict() for v in self.verifications],
            "all_releases_clean": all(v.clean for v in self.verifications),
            "note": (
                "leaked_allocated_mb is the allocation at the final mark minus the "
                "first. After every model has been released it should be near zero; "
                "a positive value that grows across experiments is a retained "
                "reference, not allocator caching (which shows up in reserved)."
            ),
        }
