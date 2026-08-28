"""GPU resource management: memory accounting, model lifecycle, capacity limits.

The contract this package enforces is that GPU memory is owned, not borrowed:
every allocation has a named owner, every owner has a release path that runs even
on failure, and release is verified by measurement rather than assumed.
"""

from .memory import (
    MemorySnapshot,
    MemoryTracker,
    ReleaseVerification,
    reset_peak,
    snapshot,
    verify_released,
)
from .lifecycle import (
    ModelHandle,
    ModelRegistry,
    ReleaseError,
    release_torch_module,
    resident_models,
)

__all__ = [
    "MemorySnapshot",
    "MemoryTracker",
    "ReleaseVerification",
    "snapshot",
    "reset_peak",
    "verify_released",
    "ModelHandle",
    "ModelRegistry",
    "ReleaseError",
    "release_torch_module",
    "resident_models",
]
