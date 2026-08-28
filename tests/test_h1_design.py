"""Tests for the H1 factorial design and its analysis.

The executor is injected, so the whole analysis path — contrasts, Amdahl bound,
hypothesis test — is verifiable on CPU with a deterministic stub. This is what
lets the GPU run be a measurement rather than a debugging session.

The stub encodes a workload shaped like the real one: summarisation dominates,
and it is sequential on a single GPU actor.
"""

from __future__ import annotations

import pytest

from maor.evaluation.h1_latency import (
    Condition,
    RunOutcome,
    default_conditions,
    run_h1,
)
from maor.pipeline.instrumentation import StageRecorder


def make_executor(
    *,
    base_summarise_s: float = 100.0,
    sentiment_s: float = 8.0,
    ingestion_s: float = 3.0,
    model_load_s: float = 30.0,
    filter_speedup: float = 4.0,
    distribution_overhead_s: float = 1.0,
):
    """Build a deterministic executor with a realistic cost structure.

    Summarisation dominates and does not parallelise across two nodes, which is
    the structural fact that bounds any achievable speedup.
    """

    def execute(condition: Condition, ticker: str, repeat: int) -> RunOutcome:
        rec = StageRecorder()

        with rec.stage("ingestion", kind="io"):
            pass
        if not condition.warm_start:
            with rec.stage("model_load", kind="model_load"):
                pass
        with rec.stage("chunk_filter", kind="compute", parallelisable=False):
            pass
        with rec.stage("sentiment", kind="compute", parallelisable=True):
            pass
        with rec.stage("summarise", kind="compute", parallelisable=False):
            pass
        if condition.distributed:
            with rec.stage("put_get_payload", kind="communication"):
                pass

        summarise_s = base_summarise_s / (filter_speedup if condition.filter_enabled else 1.0)
        total = ingestion_s + summarise_s
        total += 0.0 if condition.warm_start else model_load_s
        # Sentiment overlaps with other work only when distributed.
        total += 0.0 if condition.distributed else sentiment_s
        total += distribution_overhead_s if condition.distributed else 0.0

        # Attribute synthetic durations onto the recorded stages so the
        # by-kind accounting is exercised.
        for stage in rec.stages:
            stage.seconds = {
                "ingestion": ingestion_s,
                "model_load": 0.0 if condition.warm_start else model_load_s,
                "chunk_filter": 0.1,
                "sentiment": 0.0 if condition.distributed else sentiment_s,
                "summarise": summarise_s,
                "put_get_payload": distribution_overhead_s,
            }.get(stage.name, stage.seconds)

        return RunOutcome(
            condition=condition, ticker=ticker, repeat=repeat, total_s=total, recorder=rec
        )

    return execute


class TestFactorialDesign:
    def test_default_design_crosses_execution_and_filtering(self):
        conditions = default_conditions(include_cold=False)
        names = {c.name for c in conditions}
        assert names == {
            "serial_filter_warm",
            "serial_nofilter_warm",
            "distributed_filter_warm",
            "distributed_nofilter_warm",
        }

    def test_cold_conditions_are_added_for_model_load_isolation(self):
        names = {c.name for c in default_conditions(include_cold=True)}
        assert "serial_filter_cold" in names


class TestContrasts:
    @pytest.fixture
    def analysis(self):
        return run_h1(
            execute=make_executor(),
            tickers=["AAPL", "MSFT"],
            n_repeats=3,
            n_resamples=500,
        )

    def test_filter_effect_is_measured_on_both_arms(self, analysis):
        """B1 also had the filter, so it must be measurable independently."""
        contrasts = analysis["contrasts"]
        assert "filter_effect_serial" in contrasts
        assert "filter_effect_distributed" in contrasts
        # Filtering cuts summarisation, so it must help both arms.
        assert contrasts["filter_effect_serial"]["speedup"] > 1.0
        assert contrasts["filter_effect_distributed"]["speedup"] > 1.0

    def test_distribution_effect_is_isolated_from_filtering_and_warmth(self, analysis):
        c = analysis["contrasts"]["distribution_effect_warm_filtered"]
        assert c["treatment_condition"] == "distributed_filter_warm"
        assert c["control_condition"] == "serial_filter_warm"
        # Only sentiment overlaps, so the gain is small — the honest shape.
        assert 1.0 < c["speedup"] < 1.3

    def test_warm_start_effect_is_its_own_factor(self, analysis):
        """Model-load amortisation must not be credited to distribution."""
        c = analysis["contrasts"]["warm_start_effect_serial"]
        assert c["speedup"] > 1.0
        assert "amortisation" in c["explanation"]

    def test_contrasts_carry_confidence_intervals(self, analysis):
        for contrast in analysis["contrasts"].values():
            assert "delta_ci_s" in contrast
            ci = contrast["delta_ci_s"]
            assert ci["lower"] <= ci["point"] <= ci["upper"]


class TestParallelismCeiling:
    def test_amdahl_bound_is_reported(self):
        analysis = run_h1(
            execute=make_executor(),
            tickers=["AAPL"],
            n_repeats=2,
            n_resamples=200,
        )
        ceiling = analysis["parallelism_ceiling"]
        assert ceiling["amdahl_bound_n2"] is not None
        assert ceiling["amdahl_bound_n2"] >= 1.0

    def test_speedup_exceeding_the_bound_is_flagged_not_absorbed(self):
        """The old analysis multiplied past the bound instead of flagging it."""
        analysis = run_h1(
            # Huge distribution benefit that no parallel fraction could justify.
            execute=make_executor(sentiment_s=500.0),
            tickers=["AAPL"],
            n_repeats=2,
            n_resamples=200,
        )
        ceiling = analysis["parallelism_ceiling"]
        assert ceiling["exceeds_bound"] is True
        assert "not evidence of" in ceiling["interpretation"]

    def test_dominant_stage_is_identified(self):
        analysis = run_h1(
            execute=make_executor(),
            tickers=["AAPL"],
            n_repeats=1,
            n_resamples=200,
        )
        dominant = analysis["critical_path"]["dominant_stages"][0]
        assert dominant["stage"] == "summarise"
        assert dominant["parallelisable"] is False


class TestHypothesisReporting:
    def test_h1_can_fail(self):
        """With a realistic workload the 30% target is not met, and that is reported."""
        analysis = run_h1(
            execute=make_executor(),
            tickers=["AAPL", "MSFT"],
            n_repeats=3,
            n_resamples=200,
        )
        test = analysis["hypothesis_tests"][0]
        assert test["hypothesis_id"] == "H1"
        assert test["result"] == "FAIL"
        assert test["sanity_warnings"] == []

    def test_communication_is_not_inflated_by_compute(self):
        analysis = run_h1(
            execute=make_executor(),
            tickers=["AAPL"],
            n_repeats=1,
            n_resamples=200,
        )
        dist = analysis["conditions"]["distributed_filter_warm"]
        by_kind = dist["seconds_by_kind"]
        assert by_kind["communication"]["median"] < by_kind["compute"]["median"]

    def test_sample_size_is_interpreted(self):
        analysis = run_h1(
            execute=make_executor(),
            tickers=["AAPL"],
            n_repeats=2,
            n_resamples=200,
        )
        interp = analysis["statistics"]["sample_size_interpretation"]
        assert interp["evidence_level"] in ("anecdote", "indicative", "weak", "adequate")


class TestDistributedFactorGuard:
    """A factor that does not change execution must not be measured.

    `Condition(distributed=True)` and `distributed=False` ran identical code
    because Pipeline never read the flag. The resulting contrast would be ~0 and
    would read as "distribution does not help" when nothing distributed had run.
    """

    def test_local_conditions_exclude_the_distribution_factor(self):
        from maor.evaluation.h1_latency import local_conditions

        conditions = local_conditions()
        assert all(not c.distributed for c in conditions)
        assert {c.name for c in conditions} == {
            "serial_filter_warm",
            "serial_nofilter_warm",
            "serial_filter_cold",
        }

    def test_local_conditions_still_cross_implemented_factors(self):
        from maor.evaluation.h1_latency import local_conditions

        conditions = local_conditions()
        assert {c.filter_enabled for c in conditions} == {True, False}
        assert {c.warm_start for c in conditions} == {True, False}

    def test_pipeline_refuses_distributed_without_a_ray_path(self):
        from maor.config import Config
        from maor.pipeline.orchestrator import Pipeline

        cfg = Config.load()
        assert cfg.execution.mode == "local"
        with pytest.raises(NotImplementedError, match="execution.mode='ray'"):
            Pipeline(cfg, device="cpu", distributed=True)

    def test_pipeline_allows_serial_construction(self):
        from maor.config import Config
        from maor.pipeline.orchestrator import Pipeline

        pipe = Pipeline(Config.load(), device="cpu", distributed=False)
        assert pipe.distributed is False
        pipe.close()

    def test_refusal_message_names_the_alternative(self):
        from maor.config import Config
        from maor.pipeline.orchestrator import Pipeline

        with pytest.raises(NotImplementedError) as exc:
            Pipeline(Config.load(), device="cpu", distributed=True)
        assert "--ablation local" in str(exc.value)


class TestCrossMachineVerification:
    """Confirms a 'distributed' condition actually touched two hosts.

    Complements the guard in test_distributed.py: that guard prevents
    distributed=True from silently running single-node code; this one confirms
    that when it does connect to a cluster, the result records enough evidence
    to tell whether the cluster had two distinct machines or one.
    """

    def _executor_with_hosts(self, hosts_by_condition: dict[str, dict[str, str]]):
        def execute(condition: Condition, ticker: str, repeat: int) -> RunOutcome:
            rec = StageRecorder()
            with rec.stage("work", kind="compute"):
                pass
            return RunOutcome(
                condition=condition,
                ticker=ticker,
                repeat=repeat,
                total_s=1.0,
                recorder=rec,
                metadata={"node_hosts": hosts_by_condition.get(condition.name)},
            )

        return execute

    def test_two_distinct_hosts_marked_genuinely_cross_machine(self):
        execute = self._executor_with_hosts(
            {
                "serial_filter_warm": {
                    "ingestion": "node-a", "sentiment": "node-a", "summariser": "node-a",
                },
                "distributed_filter_warm": {
                    "ingestion": "node-a", "sentiment": "node-b", "summariser": "node-b",
                },
            }
        )
        analysis = run_h1(
            execute=execute,
            tickers=["AAPL"],
            conditions=[
                Condition(distributed=False, filter_enabled=True, warm_start=True),
                Condition(distributed=True, filter_enabled=True, warm_start=True),
            ],
            n_repeats=1,
            n_resamples=200,
        )
        dist = analysis["conditions"]["distributed_filter_warm"]
        assert dist["genuinely_cross_machine"] is True
        assert dist["observed_node_hosts"]["sentiment"] == ["node-b"]

    def test_single_host_flagged_as_not_cross_machine(self):
        execute = self._executor_with_hosts(
            {
                "distributed_filter_warm": {
                    "ingestion": "node-a", "sentiment": "node-a", "summariser": "node-a",
                },
            }
        )
        analysis = run_h1(
            execute=execute,
            tickers=["AAPL"],
            conditions=[Condition(distributed=True, filter_enabled=True, warm_start=True)],
            n_repeats=1,
            n_resamples=200,
        )
        dist = analysis["conditions"]["distributed_filter_warm"]
        assert dist["genuinely_cross_machine"] is False

    def test_absent_node_hosts_does_not_crash_local_runs(self):
        """Local (non-distributed) runs never set node_hosts; must not error."""
        analysis = run_h1(
            execute=make_executor(),
            tickers=["AAPL"],
            conditions=[Condition(distributed=False, filter_enabled=True, warm_start=True)],
            n_repeats=1,
            n_resamples=200,
        )
        cond = analysis["conditions"]["serial_filter_warm"]
        assert cond["observed_node_hosts"] is None
        assert cond["genuinely_cross_machine"] is None
