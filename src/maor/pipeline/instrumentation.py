"""Stage timing and communication-cost measurement.

Audit finding C5. The original instrumentation bracketed ``ray.get()`` calls and
labelled the result communication:

    t_transfer   = time around ray.get(finbert_ref)   # FinBERT GPU inference
    t_deserialize = time around ray.get(summary_ref)  # the whole Phi-3 map-reduce

In ``results/aapl.json`` this produced ``t_deserialize_ms = 549,504`` — 549
seconds of GPU generation recorded as deserialisation — and a ``t_comm_total_ms``
of 562,906 that equalled the entire parallel stage. The paper reported 250 ms.

The distinction this module enforces: ``ray.get()`` on a not-yet-computed
reference blocks on *computation*. Only the portion after the value is ready is
transfer. So communication is measured by:

* timing ``ray.put`` for serialisation, and
* fetching an *already-materialised* object reference for transfer, after
  ``ray.wait`` confirms the task is done.

That separates queueing and compute from the bytes moved, which is the quantity
the T_comm claim is about.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

log = logging.getLogger(__name__)


@dataclass
class StageTiming:
    """One pipeline stage, with what kind of work it represents."""

    name: str
    seconds: float
    kind: str  # "compute" | "communication" | "io" | "model_load" | "overhead"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "seconds": round(self.seconds, 4),
            "kind": self.kind,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class StageRecorder:
    """Accumulates stage timings, each tagged with the kind of work it is.

    The ``kind`` tag is what makes the critical-path breakdown honest: totals are
    reported per kind, so compute cannot be summed into a communication figure.
    """

    def __init__(self) -> None:
        self.stages: list[StageTiming] = []
        self._t_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str, kind: str = "compute", **metadata: Any) -> Iterator[dict[str, Any]]:
        if kind not in ("compute", "communication", "io", "model_load", "overhead"):
            raise ValueError(f"unknown stage kind {kind!r}")
        extra: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield extra
        finally:
            elapsed = time.perf_counter() - t0
            self.stages.append(
                StageTiming(name=name, seconds=elapsed, kind=kind, metadata={**metadata, **extra})
            )
            log.debug("stage %-28s %8.3f s (%s)", name, elapsed, kind)

    @property
    def total_s(self) -> float:
        return time.perf_counter() - self._t_start

    def by_kind(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for s in self.stages:
            totals[s.kind] = totals.get(s.kind, 0.0) + s.seconds
        return {k: round(v, 4) for k, v in sorted(totals.items())}

    def to_dict(self) -> dict[str, Any]:
        total = self.total_s
        by_kind = self.by_kind()
        return {
            "stages": [s.to_dict() for s in self.stages],
            "totals_by_kind_s": by_kind,
            "total_wall_clock_s": round(total, 4),
            "communication_fraction": (
                round(by_kind.get("communication", 0.0) / total, 6) if total > 0 else None
            ),
            "accounting_note": (
                "Stage kinds are disjoint. 'communication' covers serialisation and "
                "object transfer only; blocking waits on remote computation are "
                "'compute'. Summing a blocking ray.get() into communication is the "
                "error this accounting exists to prevent."
            ),
        }


@dataclass
class CommunicationMeasurement:
    """Serialisation and transfer cost for one payload, excluding compute."""

    payload_bytes: int
    serialise_s: float
    transfer_s: float
    deserialise_s: float
    compressed_bytes: int | None = None
    compression_applied: bool = False

    @property
    def total_s(self) -> float:
        return self.serialise_s + self.transfer_s + self.deserialise_s

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "payload_bytes": self.payload_bytes,
            "serialise_ms": round(self.serialise_s * 1000, 4),
            "transfer_ms": round(self.transfer_s * 1000, 4),
            "deserialise_ms": round(self.deserialise_s * 1000, 4),
            "total_comm_ms": round(self.total_s * 1000, 4),
            "compression_applied": self.compression_applied,
        }
        if self.compressed_bytes is not None:
            out["compressed_bytes"] = self.compressed_bytes
            out["compression_ratio"] = (
                round(self.compressed_bytes / self.payload_bytes, 4)
                if self.payload_bytes
                else None
            )
        return out


def measure_ray_communication(payload: Any, *, compress: bool = False) -> CommunicationMeasurement:
    """Measure put/get cost for one payload through the Ray object store.

    Transfer is measured by fetching an object reference that is already
    materialised, so no computation is in scope. This is a lower bound on
    cross-node cost when driver and worker share a node; run it from a driver on
    the other node to capture the network leg.

    ``compress`` genuinely compresses what is stored. The original code measured
    a compression ratio and then called ``ray.put`` on the *uncompressed* object,
    so the reported 80% bandwidth reduction was never realised on the wire. Here,
    if compression is on, the compressed bytes are what is transferred.
    """
    import pickle

    import ray

    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    payload_bytes = len(raw)
    compressed_bytes: int | None = None
    to_store: Any = payload

    if compress:
        import gzip

        t0 = time.perf_counter()
        blob = gzip.compress(raw, compresslevel=1)
        compressed_bytes = len(blob)
        to_store = blob  # what is actually stored and transferred
        serialise_s = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        _ = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        serialise_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = ray.put(to_store)
    put_s = time.perf_counter() - t0

    # The object is materialised in the store, so this get is transfer +
    # deserialisation only, with no computation pending.
    t0 = time.perf_counter()
    fetched = ray.get(ref)
    get_s = time.perf_counter() - t0

    deserialise_s = 0.0
    if compress:
        import gzip

        t0 = time.perf_counter()
        _ = pickle.loads(gzip.decompress(fetched))
        deserialise_s = time.perf_counter() - t0

    return CommunicationMeasurement(
        payload_bytes=payload_bytes,
        serialise_s=serialise_s,
        transfer_s=put_s + get_s,
        deserialise_s=deserialise_s,
        compressed_bytes=compressed_bytes,
        compression_applied=compress,
    )


def critical_path_analysis(recorder: StageRecorder) -> dict[str, Any]:
    """Where the time goes, and what parallelism could ever recover.

    Amdahl's law bounds speedup by the fraction of work that can run
    concurrently. The original analysis computed a parallel fraction of 0.038 —
    implying a two-node ceiling of 1.019x — and then multiplied past that ceiling
    by two hardcoded 1.3 factors to reach 1.72x. Warm start and input reduction
    are real effects, but they are not parallelism and do not belong in a speedup
    attributed to distribution. This function reports the bound only, and names
    the stages that constitute the serial remainder.
    """
    stages = recorder.stages
    total = sum(s.seconds for s in stages)
    if total <= 0:
        return {"error": "no stages recorded"}

    parallelisable = sum(
        s.seconds for s in stages if s.metadata.get("parallelisable") is True
    )
    fraction = parallelisable / total

    def amdahl(p: float, n: int) -> float:
        return 1.0 / ((1.0 - p) + p / n) if p > 0 else 1.0

    dominant = sorted(stages, key=lambda s: s.seconds, reverse=True)[:5]

    return {
        "total_s": round(total, 4),
        "parallelisable_s": round(parallelisable, 4),
        "parallel_fraction_p": round(fraction, 6),
        "serial_fraction": round(1 - fraction, 6),
        "amdahl_bound": {
            f"n={n}": round(amdahl(fraction, n), 4) for n in (2, 4, 8, 16)
        },
        "dominant_stages": [
            {
                "stage": s.name,
                "seconds": round(s.seconds, 3),
                "pct_of_total": round(100 * s.seconds / total, 2),
                "kind": s.kind,
                "parallelisable": bool(s.metadata.get("parallelisable", False)),
            }
            for s in dominant
        ],
        "interpretation": (
            "amdahl_bound is the ceiling on speedup from parallelism alone at this "
            "parallel fraction. Effects that are not parallelism — warm model "
            "residency, input reduction — must be reported as separate factors "
            "with their own controls, not multiplied into this bound."
        ),
    }
