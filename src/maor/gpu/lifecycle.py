"""Model residency: load once, release completely, verify it happened.

Setting ``self._model = None`` is not a release. A HuggingFace ``pipeline`` holds
the model and tokenizer; a returned output tensor holds its graph; an exception
traceback holds every frame local including the model. Any one of those keeps
weights on the device after the owning attribute has been cleared, and the next
load then fails with an OOM that appears to come from nowhere.

:func:`release_torch_module` performs the full sequence — move to CPU, drop
submodule references, delete, collect, empty the cache — and measures the result.
:class:`ModelRegistry` tracks what is actually resident so that a second load of
the same checkpoint is refused rather than silently doubling residency.
"""

from __future__ import annotations

import gc
import logging
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .memory import (
    MemorySnapshot,
    ReleaseVerification,
    empty_cache,
    snapshot,
    verify_released,
)

log = logging.getLogger(__name__)


class ReleaseError(RuntimeError):
    """Raised when a release leaves memory behind and the caller demanded clean."""


class DuplicateResidencyError(RuntimeError):
    """Raised when a checkpoint would be loaded twice onto the same device."""


def _move_to_cpu(obj: Any) -> None:
    """Move a torch module off the device before dropping it.

    Moving first means the device memory is freed by the ``.to()`` call itself,
    at a point where the reference is still valid, rather than depending on the
    garbage collector to reach it later. For a module inside a reference cycle
    this is the difference between freeing now and freeing at the next full
    collection.
    """
    try:
        import torch

        if isinstance(obj, torch.nn.Module):
            obj.to("cpu")
            return
        # HF pipeline: unwrap and move the model it holds.
        inner = getattr(obj, "model", None)
        if inner is not None and isinstance(inner, torch.nn.Module):
            inner.to("cpu")
    except Exception as exc:
        log.debug("could not move object to CPU before release: %s", exc)


def release_torch_module(
    label: str,
    *objects: Any,
    device: int = 0,
    expected_mb: float | None = None,
    strict: bool = False,
    tolerance_mb: float = 64.0,
) -> ReleaseVerification:
    """Release GPU objects completely and verify the memory came back.

    Pass every reference the caller holds — model, tokenizer, pipeline, cached
    outputs. The caller must drop its own references after calling this;
    ``del`` here only removes this function's local ones.

    ``strict=True`` raises when the memory does not come back, which is
    appropriate between experiments where a leak would corrupt the next run.
    """
    before = snapshot(device, label=f"{label}:before-release")

    for obj in objects:
        if obj is None:
            continue
        _move_to_cpu(obj)

    # Drop this frame's references before collecting, or gc sees them as live.
    objects = ()  # type: ignore[assignment]
    gc.collect()
    empty_cache()

    verification = verify_released(
        label, before, device=device, expected_mb=expected_mb, tolerance_mb=tolerance_mb
    )
    if strict and not verification.clean:
        raise ReleaseError(
            f"releasing {label!r} left {verification.residual_mb:.0f} MB allocated "
            f"(freed {verification.allocated_freed_mb:.0f} MB"
            + (f", expected ~{expected_mb:.0f} MB" if expected_mb else "")
            + "). Something still holds a reference. Common causes: a cached "
            "HuggingFace pipeline, a retained output tensor, or an exception "
            "traceback holding frame locals."
        )
    return verification


@dataclass
class ModelHandle:
    """A resident model and the metadata needed to release and audit it."""

    label: str
    checkpoint: str
    device: int
    estimated_mb: float
    loaded_allocated_mb: float = 0.0
    releaser: Callable[[], None] | None = None
    _ref: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "estimated_mb": self.estimated_mb,
            "measured_residency_mb": round(self.loaded_allocated_mb, 1),
        }


class ModelRegistry:
    """Tracks what is resident on each device, process-wide.

    Two failure modes this closes:

    * **Duplicate residency.** The shared summariser exists so a ~2.7 GB model is
      paid for once. Loading it again — a second ``Pipeline``, a second worker,
      an evaluation harness that builds its own — doubles it. The registry
      refuses by checkpoint identity, not by variable name.
    * **Silent accumulation.** :meth:`resident_mb` is what is *actually* held,
      which a budget of scoped reservations cannot see once a model outlives its
      reservation scope.
    """

    _instance: "ModelRegistry | None" = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._handles: dict[tuple[int, str], ModelHandle] = {}

    @classmethod
    def instance(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the process-wide registry. For tests and worker teardown."""
        with cls._lock:
            cls._instance = None

    def key(self, checkpoint: str, device: int) -> tuple[int, str]:
        return (device, checkpoint)

    def is_resident(self, checkpoint: str, device: int = 0) -> bool:
        return self.key(checkpoint, device) in self._handles

    def get(self, checkpoint: str, device: int = 0) -> ModelHandle | None:
        return self._handles.get(self.key(checkpoint, device))

    def register(
        self,
        *,
        label: str,
        checkpoint: str,
        device: int,
        estimated_mb: float,
        obj: Any = None,
        releaser: Callable[[], None] | None = None,
        allow_shared: bool = False,
    ) -> ModelHandle:
        """Record a resident model, refusing a duplicate unless sharing is intended.

        ``allow_shared=True`` returns the existing handle instead of raising, for
        the legitimate case where two components deliberately use one instance.
        """
        with self._lock:
            key = self.key(checkpoint, device)
            existing = self._handles.get(key)
            if existing is not None:
                if allow_shared:
                    log.debug(
                        "%s reuses already-resident %s on device %d",
                        label,
                        checkpoint,
                        device,
                    )
                    return existing
                raise DuplicateResidencyError(
                    f"{checkpoint!r} is already resident on device {device} as "
                    f"{existing.label!r} ({existing.estimated_mb:.0f} MB). Loading "
                    f"it again for {label!r} would double that residency. Pass a "
                    f"handle to the existing instance, or release it first. If two "
                    f"components are meant to share one copy, register with "
                    f"allow_shared=True."
                )
            handle = ModelHandle(
                label=label,
                checkpoint=checkpoint,
                device=device,
                estimated_mb=estimated_mb,
                loaded_allocated_mb=snapshot(device).allocated_mb,
                releaser=releaser,
            )
            if obj is not None:
                try:
                    handle._ref = weakref.ref(obj)
                except TypeError:
                    handle._ref = None
            self._handles[key] = handle
            log.info(
                "resident: %s (%s) on device %d, ~%.0f MB | total tracked %.0f MB",
                label,
                checkpoint,
                device,
                estimated_mb,
                self.resident_mb(device),
            )
            return handle

    def unregister(self, checkpoint: str, device: int = 0) -> ModelHandle | None:
        with self._lock:
            handle = self._handles.pop(self.key(checkpoint, device), None)
            if handle is not None:
                log.info(
                    "released: %s (%s) | total tracked now %.0f MB",
                    handle.label,
                    checkpoint,
                    self.resident_mb(device),
                )
            return handle

    def resident_mb(self, device: int = 0) -> float:
        """Estimated VRAM held by tracked models on a device."""
        return sum(
            h.estimated_mb for (d, _), h in self._handles.items() if d == device
        )

    def resident(self, device: int | None = None) -> list[ModelHandle]:
        return [
            h
            for (d, _), h in sorted(self._handles.items())
            if device is None or d == device
        ]

    def release_all(self, device: int | None = None, *, strict: bool = False) -> list[str]:
        """Release every tracked model. Used for teardown and failure recovery."""
        released: list[str] = []
        for handle in list(self.resident(device)):
            try:
                if handle.releaser is not None:
                    handle.releaser()
            except Exception as exc:
                log.warning("releaser for %s raised: %s", handle.label, exc)
                if strict:
                    raise
            finally:
                self.unregister(handle.checkpoint, handle.device)
                released.append(handle.label)
        empty_cache()
        return released

    def audit(self, device: int = 0) -> dict[str, Any]:
        """Compare what is tracked against what the device reports.

        A large gap between tracked residency and measured allocation means
        something is holding memory that the registry does not know about — the
        situation in which a budget check passes and the allocation still fails.
        """
        snap = snapshot(device, label="registry-audit")
        tracked = self.resident_mb(device)
        return {
            "tracked_models": [h.to_dict() for h in self.resident(device)],
            "tracked_estimated_mb": round(tracked, 1),
            "measured_allocated_mb": round(snap.allocated_mb, 1),
            "measured_reserved_mb": round(snap.reserved_mb, 1),
            "untracked_allocated_mb": round(max(0.0, snap.allocated_mb - tracked), 1),
            "free_mb": round(snap.free_mb, 1),
            "used_by_other_processes_mb": round(snap.used_by_others_mb, 1),
        }


def resident_models(device: int | None = None) -> list[ModelHandle]:
    return ModelRegistry.instance().resident(device)


@contextmanager
def model_scope(
    label: str,
    *,
    checkpoint: str,
    device: int = 0,
    estimated_mb: float = 0.0,
    loader: Callable[[], Any],
    releaser: Callable[[Any], None] | None = None,
    strict_release: bool = False,
    allow_shared: bool = False,
) -> Iterator[Any]:
    """Load a model, hand it to the block, and guarantee release afterwards.

    Release runs in ``finally``, so an exception inside the block cannot leave
    weights resident — which is the failure that turns one bad experiment into a
    run of failed experiments.
    """
    registry = ModelRegistry.instance()
    obj: Any = None
    try:
        obj = loader()
        registry.register(
            label=label,
            checkpoint=checkpoint,
            device=device,
            estimated_mb=estimated_mb,
            obj=obj,
            allow_shared=allow_shared,
        )
        yield obj
    finally:
        try:
            if releaser is not None and obj is not None:
                releaser(obj)
        except Exception as exc:
            log.warning("releaser for %s raised during cleanup: %s", label, exc)
        finally:
            registry.unregister(checkpoint, device)
            local = obj
            obj = None
            release_torch_module(
                label,
                local,
                device=device,
                expected_mb=estimated_mb or None,
                strict=strict_release,
            )
