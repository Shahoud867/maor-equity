"""Hardware probing and VRAM budget enforcement.

The original system treated the 4 GB T1000 budget as a comment in the README and a
number in a paper table. Runs stalled when that budget was exceeded. This module
makes the budget a runtime object that is checked before weights are loaded and
enforced while they are resident.

Design rules, in priority order:

1. **Nothing is hard-coded.** Total VRAM is probed from the device. The usable
   budget is a *fraction* of what is actually present, so the same code runs on a
   4 GB T1000, an 8 GB laptop card, or a 16 GB T4 without edits.
2. **Fail before allocating, not during.** :meth:`VRAMBudget.reserve` refuses a
   model that cannot fit alongside what is already resident, raising immediately
   rather than letting CUDA thrash and hang.
3. **Fail loudly, never silently degrade.** An over-budget request raises
   :class:`VRAMBudgetExceeded`. It does not quietly reduce batch size, because a
   silent reduction changes the experiment without changing the record of it.
4. **Registry prevents duplicate residency.** Audit finding: the shared
   ``Phi3ModelActor`` existed precisely to avoid paying 2,736 MB twice. The
   registry makes double-loading an error instead of an OOM.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

log = logging.getLogger(__name__)

MB = 1024 * 1024


class VRAMBudgetExceeded(RuntimeError):
    """Raised before allocation when a request would exceed the configured budget."""


class ModelAlreadyResident(RuntimeError):
    """Raised when a model would be loaded twice into the same device."""


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_mb: float
    free_mb: float | None = None
    compute_capability: str | None = None


@dataclass(frozen=True)
class HardwareInfo:
    """A snapshot of the machine a measurement was taken on."""

    has_cuda: bool
    gpus: tuple[GPUInfo, ...]
    cpu_count: int
    total_ram_mb: float
    platform: str
    torch_version: str | None = None
    cuda_version: str | None = None

    @property
    def primary_gpu(self) -> GPUInfo | None:
        return self.gpus[0] if self.gpus else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_cuda": self.has_cuda,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_mb": g.total_mb,
                    "compute_capability": g.compute_capability,
                }
                for g in self.gpus
            ],
            "cpu_count": self.cpu_count,
            "total_ram_mb": round(self.total_ram_mb, 1),
            "platform": self.platform,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
        }

    def describe(self) -> str:
        if not self.has_cuda:
            return f"CPU-only ({self.cpu_count} cores, {self.total_ram_mb/1024:.1f} GB RAM)"
        g = self.primary_gpu
        assert g is not None
        return f"{g.name} ({g.total_mb:.0f} MB VRAM), {self.cpu_count} CPU cores"


def _probe_via_torch() -> tuple[bool, list[GPUInfo], str | None, str | None]:
    try:
        import torch
    except ImportError:
        return False, [], None, None

    torch_version = torch.__version__
    cuda_version = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        return False, [], torch_version, cuda_version

    gpus: list[GPUInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        try:
            free_b, _total_b = torch.cuda.mem_get_info(i)
            free_mb = free_b / MB
        except Exception:
            free_mb = None
        gpus.append(
            GPUInfo(
                index=i,
                name=props.name,
                total_mb=props.total_memory / MB,
                free_mb=free_mb,
                compute_capability=f"{props.major}.{props.minor}",
            )
        )
    return True, gpus, torch_version, cuda_version


def _probe_via_nvidia_smi() -> list[GPUInfo]:
    """Fallback when torch is absent or CPU-only but a GPU may still exist."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    gpus: list[GPUInfo] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mb=float(parts[2]),
                    free_mb=float(parts[3]),
                )
            )
        except ValueError:
            continue
    return gpus


def _total_ram_mb() -> float:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / MB
    except (ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return stat.ullTotalPhys / MB
    except Exception:
        return 0.0


def probe() -> HardwareInfo:
    """Detect the machine this process is running on. Never raises."""
    import platform as _platform

    has_cuda, gpus, torch_version, cuda_version = _probe_via_torch()
    if not gpus:
        smi_gpus = _probe_via_nvidia_smi()
        if smi_gpus:
            # A GPU exists but torch cannot use it — almost always a CPU-only
            # torch build. Surface it rather than silently reporting no GPU.
            log.warning(
                "nvidia-smi reports %d GPU(s) but torch cannot use CUDA. "
                "You likely have a CPU-only torch build installed.",
                len(smi_gpus),
            )
            gpus = smi_gpus

    return HardwareInfo(
        has_cuda=has_cuda,
        gpus=tuple(gpus),
        cpu_count=os.cpu_count() or 1,
        total_ram_mb=_total_ram_mb(),
        platform=_platform.platform(),
        torch_version=torch_version,
        cuda_version=cuda_version,
    )


def current_vram_mb(device: int = 0) -> float:
    """Currently allocated VRAM in MB, or 0.0 when CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(device) / MB
    except Exception:
        return 0.0


def peak_vram_mb(device: int = 0) -> float:
    """Peak allocated VRAM since the last reset, in MB."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(device) / MB
    except Exception:
        return 0.0


def reset_peak_vram(device: int = 0) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def release_vram() -> None:
    """Free cached VRAM between experiments.

    ``empty_cache`` only returns memory the caching allocator holds but is not
    using, so this is called after dropping references, not instead of it.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


@dataclass
class VRAMBudget:
    """A checked allowance of GPU memory.

    ``total_mb`` is probed from the device unless explicitly supplied.
    ``usable_fraction`` reserves headroom for the CUDA context, fragmentation and
    KV cache growth, which are not visible to ``memory_allocated``.
    """

    total_mb: float
    usable_fraction: float = 0.85
    device: int = 0
    _reservations: dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def from_hardware(
        cls,
        hardware: HardwareInfo | None = None,
        *,
        usable_fraction: float = 0.85,
        override_total_mb: float | None = None,
        device: int = 0,
    ) -> "VRAMBudget":
        if override_total_mb is not None:
            total = float(override_total_mb)
        else:
            hardware = hardware or probe()
            gpu = hardware.primary_gpu
            total = gpu.total_mb if gpu else 0.0
        return cls(total_mb=total, usable_fraction=usable_fraction, device=device)

    @property
    def usable_mb(self) -> float:
        return self.total_mb * self.usable_fraction

    @property
    def reserved_mb(self) -> float:
        return sum(self._reservations.values())

    @property
    def available_mb(self) -> float:
        return self.usable_mb - self.reserved_mb

    def check(self, requested_mb: float, *, label: str = "allocation") -> None:
        """Raise if ``requested_mb`` would not fit. Call before loading weights."""
        if self.total_mb <= 0:
            return  # No GPU: CPU execution path enforces nothing here.
        if requested_mb > self.available_mb:
            raise VRAMBudgetExceeded(
                f"{label} needs {requested_mb:.0f} MB but only {self.available_mb:.0f} MB "
                f"of the {self.usable_mb:.0f} MB budget is free "
                f"(total {self.total_mb:.0f} MB x {self.usable_fraction:.0%}; "
                f"currently reserved: {self._format_reservations()}). "
                f"Reduce model size, lower quantisation precision, or run stages in "
                f"separate phases with release_vram() between them."
            )

    def reserve(self, label: str, estimated_mb: float) -> None:
        """Record that ``label`` now occupies ``estimated_mb``.

        Raises :class:`ModelAlreadyResident` on a duplicate label, which is the
        cheap guard against the model-replication failure mode.
        """
        if label in self._reservations:
            raise ModelAlreadyResident(
                f"'{label}' is already resident on device {self.device} "
                f"({self._reservations[label]:.0f} MB). Loading it twice is what the "
                f"shared model actor exists to prevent — pass a handle to the existing "
                f"instance instead."
            )
        self.check(estimated_mb, label=label)
        self._reservations[label] = estimated_mb
        log.info(
            "VRAM reserve: %s +%.0f MB (%.0f/%.0f MB used)",
            label,
            estimated_mb,
            self.reserved_mb,
            self.usable_mb,
        )

    def release(self, label: str) -> None:
        freed = self._reservations.pop(label, 0.0)
        if freed:
            log.info(
                "VRAM release: %s -%.0f MB (%.0f/%.0f MB used)",
                label,
                freed,
                self.reserved_mb,
                self.usable_mb,
            )

    def _format_reservations(self) -> str:
        if not self._reservations:
            return "none"
        return ", ".join(f"{k}={v:.0f}MB" for k, v in self._reservations.items())

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_mb": round(self.total_mb, 1),
            "usable_fraction": self.usable_fraction,
            "usable_mb": round(self.usable_mb, 1),
            "reserved_mb": round(self.reserved_mb, 1),
            "available_mb": round(self.available_mb, 1),
            "reservations": {k: round(v, 1) for k, v in self._reservations.items()},
            "measured_allocated_mb": round(current_vram_mb(self.device), 1),
            "measured_peak_mb": round(peak_vram_mb(self.device), 1),
        }

    @contextmanager
    def phase(self, label: str, estimated_mb: float) -> Iterator["VRAMBudget"]:
        """Reserve for the duration of a block, then release and free the cache.

        This is the mechanism behind phase-serialised execution: FinBERT occupies
        the budget during Phase A, is released, and only then does Phi-3 load.
        """
        self.reserve(label, estimated_mb)
        try:
            yield self
        finally:
            self.release(label)
            release_vram()
