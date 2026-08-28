"""Provenance and result integrity.

Audit finding C1 was that published numbers could not be traced to any run that
produced them, and that generated values sat in the same directory, with the same
naming convention, as measured ones. This module makes that failure mode
structurally impossible rather than merely discouraged.

Every result written through :func:`write_result` carries a provenance record: the
git commit, whether the tree was dirty, the host and hardware, the exact argv, the
config hash, timings, and — most importantly — an explicit ``evidence_class``.

There are only two evidence classes:

``MEASURED``
    Produced by executing the system under test on real data. Only these may be
    cited as experimental results.

``DERIVED``
    Computed analytically from other results (e.g. an Amdahl bound from measured
    stage timings). Legitimate, but never a substitute for measurement, and the
    inputs it was derived from must be named in ``derived_from``.

There is deliberately no third class for "estimated" or "projected" values. If a
number has not been measured, it does not get written to ``results/``; the paper
reports it as not-yet-measured. See ``docs/EVIDENCE_POLICY.md``.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


class EvidenceClass(str, Enum):
    """How a number came to exist. See module docstring."""

    MEASURED = "MEASURED"
    DERIVED = "DERIVED"


class ProvenanceError(RuntimeError):
    """Raised when a result cannot be written with honest provenance."""


def _run_git(*args: str) -> str | None:
    """Return trimmed stdout of a git command, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_commit() -> str | None:
    return _run_git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes to tracked files."""
    status = _run_git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None
    return bool(status.strip())


def config_hash(config: Any) -> str:
    """Stable short hash of a config object, for grouping runs of one setting."""
    if hasattr(config, "to_dict"):
        payload = config.to_dict()
    elif isinstance(config, dict):
        payload = config
    else:
        payload = asdict(config) if hasattr(config, "__dataclass_fields__") else repr(config)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class Provenance:
    """Everything needed to decide whether a number can be trusted and reproduced."""

    evidence_class: EvidenceClass
    experiment: str
    schema_version: str = SCHEMA_VERSION

    # Code identity
    git_commit: str | None = field(default_factory=git_commit)
    git_dirty: bool | None = field(default_factory=git_dirty)

    # Execution identity
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    hostname: str = field(default_factory=socket.gethostname)
    username: str = field(default_factory=lambda: _safe_user())
    command: str = field(default_factory=lambda: " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]))
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)

    # Scientific identity
    config_sha: str | None = None
    seed: int | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)

    # For DERIVED results only: which measured files this was computed from
    derived_from: list[str] = field(default_factory=list)

    # Free-form caveats that must travel with the number
    caveats: list[str] = field(default_factory=list)

    duration_s: float | None = None

    def validate(self) -> None:
        """Reject provenance that would let an untraceable number reach the paper."""
        if self.evidence_class == EvidenceClass.DERIVED and not self.derived_from:
            raise ProvenanceError(
                "DERIVED results must name the measured files they were computed from "
                "(derived_from=[...]). A derived number with no inputs is an estimate, "
                "and estimates are not written to results/."
            )
        if self.evidence_class == EvidenceClass.MEASURED and not self.hardware:
            raise ProvenanceError(
                "MEASURED results must record the hardware they were measured on. "
                "Call maor.hardware.probe() and pass the result."
            )


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USERNAME", "unknown")


def library_versions(*modules: str) -> dict[str, str]:
    """Record versions of the libraries that could change a numeric result."""
    import importlib

    versions: dict[str, str] = {}
    for name in modules:
        try:
            mod = importlib.import_module(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[name] = "not-installed"
    return versions


DEFAULT_TRACKED_LIBS = (
    "torch",
    "transformers",
    "numpy",
    "sklearn",
    "scipy",
    "rouge_score",
    "datasets",
    "ray",
)


def write_result(
    path: str | Path,
    payload: dict[str, Any],
    *,
    provenance: Provenance,
    overwrite: bool = True,
) -> Path:
    """Write a result file with its provenance attached.

    The provenance is embedded in the same file as the numbers so the two cannot
    be separated by copying, moving, or re-committing.
    """
    provenance.validate()
    if not provenance.library_versions:
        provenance.library_versions = library_versions(*DEFAULT_TRACKED_LIBS)

    path = Path(path)
    if path.exists() and not overwrite:
        raise ProvenanceError(f"{path} exists and overwrite=False")
    path.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "_provenance": asdict(provenance) | {"evidence_class": provenance.evidence_class.value},
        **payload,
    }
    path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    return path


def read_result(path: str | Path) -> tuple[dict[str, Any], Provenance]:
    """Load a result file and its provenance, rejecting files that lack it."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "_provenance" not in doc:
        raise ProvenanceError(
            f"{path} has no provenance record. It predates the evidence policy or was "
            f"written by hand; it must not be cited. See docs/EVIDENCE_POLICY.md."
        )
    raw = dict(doc["_provenance"])
    raw["evidence_class"] = EvidenceClass(raw["evidence_class"])
    known = set(Provenance.__dataclass_fields__)
    prov = Provenance(**{k: v for k, v in raw.items() if k in known})
    payload = {k: v for k, v in doc.items() if k != "_provenance"}
    return payload, prov


class Timer:
    """Context manager recording wall-clock duration in seconds.

    Uses ``perf_counter`` so it is monotonic and unaffected by clock adjustment.
    """

    def __init__(self) -> None:
        self.elapsed_s: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_s = time.perf_counter() - self._t0
