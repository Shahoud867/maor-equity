"""H1: what does distribution actually buy, and what bounds it?

The previous H1 reported a 1.72x speedup that was never measured: the
"distributed" latencies were ``B1_total / 1.723``, where 1.723 was
``amdahl(p=0.038) x 1.30 x 1.30`` with both 1.30 factors hardcoded. Two
independent tickers consequently showed identical speedups to three significant
figures. The two real runs in ``results/`` gave 1.14x median, and on AAPL the
distributed pipeline was 1.69x *slower* than serial.

Three design corrections, each addressing a specific finding:

**C3 — the filter was double-counted.** B1 applied the same TF-IDF filter with
the same 12-chunk cap, so filtering cannot explain any difference between arms.
It is modelled here as an independent factor, giving a 2x2 design over
{serial, distributed} x {filter on, filter off}. That is the only arrangement in
which the filter's contribution is identifiable.

**C3b — the baseline was handicapped.** B1 cold-loaded models on every run while
the distributed arm kept them warm, handing the distributed arm ~90 s for free.
``warm_start`` is now an explicit factor, and the default comparison is
warm-versus-warm. Cold-start cost is reported separately as its own line item,
because amortising model load is a real effect that has nothing to do with
distribution.

**C5 — communication was measuring compute.** Stage timings are tagged by kind,
and only serialisation and transfer count as communication.
"""

from __future__ import annotations

import itertools
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..pipeline.instrumentation import StageRecorder, critical_path_analysis
from . import stats
from .metrics import HypothesisTest

log = logging.getLogger(__name__)


@dataclass
class Condition:
    """One cell of the factorial design."""

    distributed: bool
    filter_enabled: bool
    warm_start: bool

    @property
    def name(self) -> str:
        return (
            f"{'distributed' if self.distributed else 'serial'}"
            f"_{'filter' if self.filter_enabled else 'nofilter'}"
            f"_{'warm' if self.warm_start else 'cold'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "distributed": self.distributed,
            "filter_enabled": self.filter_enabled,
            "warm_start": self.warm_start,
            "name": self.name,
        }


@dataclass
class RunOutcome:
    """One measured execution."""

    condition: Condition
    ticker: str
    repeat: int
    total_s: float
    recorder: StageRecorder
    metadata: dict[str, Any] = field(default_factory=dict)


def local_conditions(*, include_cold: bool = True) -> list[Condition]:
    """Factors that are implemented single-node: chunk filter x warm start.

    This is the default. It excludes the distribution factor because the Ray
    execution path is not implemented, and a factor that does not change
    execution yields a contrast of ~0 that reads as a measured null result.
    """
    conditions = [
        Condition(distributed=False, filter_enabled=f, warm_start=True)
        for f in (True, False)
    ]
    if include_cold:
        conditions.append(
            Condition(distributed=False, filter_enabled=True, warm_start=False)
        )
    return conditions


def default_conditions(*, include_cold: bool = True) -> list[Condition]:
    """The full 2x2 over distribution and filtering, warm by default.

    Requires a working Ray path; :class:`~maor.pipeline.orchestrator.Pipeline`
    raises for ``distributed=True`` when ``execution.mode`` is not ``ray``.
    Cold-start conditions isolate the model-load cost the original baseline paid
    silently on every run.
    """
    conditions = [
        Condition(distributed=d, filter_enabled=f, warm_start=True)
        for d, f in itertools.product((False, True), (True, False))
    ]
    if include_cold:
        conditions.append(
            Condition(distributed=False, filter_enabled=True, warm_start=False)
        )
        conditions.append(
            Condition(distributed=True, filter_enabled=True, warm_start=False)
        )
    return conditions


def run_h1(
    *,
    execute: Callable[[Condition, str, int], RunOutcome],
    tickers: Sequence[str],
    conditions: Sequence[Condition] | None = None,
    n_repeats: int = 5,
    seed: int = 0,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    """Execute the factorial design and analyse it.

    ``execute`` is injected so the analysis is testable without a GPU: the
    pipeline supplies the real one, tests supply a deterministic stub.
    """
    conditions = list(conditions or default_conditions())
    outcomes: list[RunOutcome] = []

    for condition in conditions:
        for ticker in tickers:
            for repeat in range(n_repeats):
                log.info(
                    "H1: %s | %s | repeat %d/%d",
                    condition.name,
                    ticker,
                    repeat + 1,
                    n_repeats,
                )
                outcomes.append(execute(condition, ticker, repeat))

    by_condition: dict[str, list[RunOutcome]] = {}
    for o in outcomes:
        by_condition.setdefault(o.condition.name, []).append(o)

    condition_summaries: dict[str, Any] = {}
    for name, runs in by_condition.items():
        totals = [r.total_s for r in runs]
        kinds: dict[str, list[float]] = {}
        for r in runs:
            for kind, seconds in r.recorder.by_kind().items():
                kinds.setdefault(kind, []).append(seconds)

        # Present only for distributed runs (DistributedRunOutcome.node_hosts,
        # threaded through RunOutcome.metadata). Two distinct hostnames for a
        # role confirms the run actually spanned two machines; one hostname
        # repeated means every actor landed on the same node regardless of
        # what the cluster topology reported, and the timing here should be
        # read as actor overhead, not network cost.
        observed_hosts: dict[str, set[str]] = {}
        for r in runs:
            hosts = r.metadata.get("node_hosts") if r.metadata else None
            if not hosts:
                continue
            for role, host in hosts.items():
                observed_hosts.setdefault(role, set()).add(host)

        condition_summaries[name] = {
            "condition": runs[0].condition.to_dict(),
            "n_runs": len(runs),
            "total_s": stats.summarise(totals),
            "total_s_ci": stats.bootstrap_ci(
                totals, statistic=statistics.median, n_resamples=n_resamples, seed=seed
            ).to_dict(),
            "seconds_by_kind": {k: stats.summarise(v) for k, v in kinds.items()},
            "per_ticker_median_s": {
                t: round(
                    statistics.median([r.total_s for r in runs if r.ticker == t]), 4
                )
                for t in sorted({r.ticker for r in runs})
            },
            "observed_node_hosts": (
                {role: sorted(hosts) for role, hosts in observed_hosts.items()}
                if observed_hosts
                else None
            ),
            "genuinely_cross_machine": (
                len({h for hosts in observed_hosts.values() for h in hosts}) > 1
                if observed_hosts
                else None
            ),
        }

    contrasts = _contrasts(by_condition, seed=seed, n_resamples=n_resamples)

    # Critical path from the reference condition: distributed, filtered, warm.
    reference = by_condition.get("distributed_filter_warm") or next(
        iter(by_condition.values())
    )
    path = critical_path_analysis(reference[0].recorder)

    measured_speedup = contrasts.get("distribution_effect_warm_filtered", {}).get(
        "speedup"
    )
    amdahl_n2 = path.get("amdahl_bound", {}).get("n=2")

    tests: list[dict[str, Any]] = []
    if measured_speedup is not None:
        h1 = HypothesisTest(
            hypothesis_id="H1",
            claim=(
                "Distributing the pipeline across two nodes reduces median "
                "end-to-end latency by at least 30% versus an equally-configured "
                "serial baseline"
            ),
            metric_name="latency_reduction_pct",
            scale="percent",
            units="% reduction",
            observed=round((1 - 1 / measured_speedup) * 100, 3)
            if measured_speedup
            else 0.0,
            threshold=30.0,
            comparison=">=",
            falsifiable_range=(-1000.0, 100.0),
            notes=(
                "Compared warm-vs-warm with filtering held on in both arms, so "
                "neither model-load amortisation nor input reduction is credited "
                "to distribution."
            ),
        )
        tests.append(h1.to_dict())

    return {
        "experiment": "H1_latency_and_parallelism_ceiling",
        "design": {
            "type": "factorial",
            "factors": {
                "execution": ["serial", "distributed"],
                "chunk_filter": ["on", "off"],
                "model_residency": ["warm", "cold"],
            },
            "conditions": [c.to_dict() for c in conditions],
            "tickers": list(tickers),
            "n_repeats": n_repeats,
            "rationale": (
                "The previous comparison confounded three effects. B1 applied the "
                "same chunk filter as the distributed arm, so filtering could not "
                "explain their difference; and B1 cold-loaded models while the "
                "distributed arm stayed warm. Crossing the factors identifies each "
                "separately."
            ),
        },
        "conditions": condition_summaries,
        "contrasts": contrasts,
        "critical_path": path,
        "parallelism_ceiling": {
            "amdahl_bound_n2": amdahl_n2,
            "measured_speedup": measured_speedup,
            "exceeds_bound": (
                bool(measured_speedup and amdahl_n2 and measured_speedup > amdahl_n2)
            ),
            "interpretation": (
                "A measured speedup above the Amdahl bound is not evidence of "
                "better parallelism; it means an effect other than parallelism is "
                "present and must be attributed to its own factor. The previous "
                "analysis multiplied the bound by two hardcoded 1.3 factors to "
                "reach its headline number."
            ),
        },
        "statistics": {
            "sample_size_interpretation": stats.interpret_sample_size(
                len(tickers) * n_repeats
            ),
        },
        "hypothesis_tests": tests,
    }


def _contrasts(
    by_condition: dict[str, list[RunOutcome]], *, seed: int, n_resamples: int
) -> dict[str, Any]:
    """Isolate each factor's effect by holding the others fixed."""

    def totals(name: str) -> list[float] | None:
        runs = by_condition.get(name)
        return [r.total_s for r in runs] if runs else None

    def contrast(
        label: str, treatment: str, control: str, explanation: str
    ) -> dict[str, Any] | None:
        t, c = totals(treatment), totals(control)
        if not t or not c:
            return None
        t_med, c_med = statistics.median(t), statistics.median(c)
        ci = stats.paired_bootstrap_delta(
            t[: min(len(t), len(c))],
            c[: min(len(t), len(c))],
            statistic=statistics.mean,
            n_resamples=n_resamples,
            seed=seed,
        )
        return {
            "label": label,
            "treatment_condition": treatment,
            "control_condition": control,
            "treatment_median_s": round(t_med, 4),
            "control_median_s": round(c_med, 4),
            "delta_s": round(t_med - c_med, 4),
            "speedup": round(c_med / t_med, 4) if t_med > 0 else None,
            "delta_ci_s": ci.to_dict(),
            "explanation": explanation,
        }

    out: dict[str, Any] = {}
    for key, args in {
        "distribution_effect_warm_filtered": (
            "Effect of distribution alone",
            "distributed_filter_warm",
            "serial_filter_warm",
            "Both arms warm and both filtered, so the difference is distribution.",
        ),
        "distribution_effect_warm_unfiltered": (
            "Effect of distribution without filtering",
            "distributed_nofilter_warm",
            "serial_nofilter_warm",
            "Distribution effect when neither arm reduces its input.",
        ),
        "filter_effect_serial": (
            "Effect of the chunk filter on the serial arm",
            "serial_filter_warm",
            "serial_nofilter_warm",
            "The filter's own contribution, measurable because B1 also has it.",
        ),
        "filter_effect_distributed": (
            "Effect of the chunk filter on the distributed arm",
            "distributed_filter_warm",
            "distributed_nofilter_warm",
            "The filter's contribution under distribution.",
        ),
        "warm_start_effect_serial": (
            "Effect of warm model residency",
            "serial_filter_warm",
            "serial_filter_cold",
            "Model-load amortisation, which the old baseline gave the distributed "
            "arm for free by cold-loading B1 on every run.",
        ),
    }.items():
        result = contrast(*args)
        if result:
            out[key] = result

    return out
