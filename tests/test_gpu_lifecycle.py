"""Tests for GPU memory lifecycle, capacity planning and bounded execution.

All of these run on CPU. The memory functions degrade to zeros without CUDA, and
the registry, planner, timeout and runner logic are device-independent — which is
the point: the parts that decide whether a workload is safe to start are
verifiable before reaching a GPU.

What these tests cannot establish is that CUDA actually returns the memory. That
requires the target device and is listed as pending in RESULTS_STATUS.
"""

from __future__ import annotations

import time

import pytest

from maor.execution.runner import (
    ExperimentRunner,
    ExperimentSpec,
    Status,
)
from maor.execution.timeouts import (
    RetryPolicy,
    TimeoutError_,
    TimeoutGuard,
    is_oom,
    run_with_timeout,
)
from maor.gpu.lifecycle import (
    DuplicateResidencyError,
    ModelRegistry,
    model_scope,
    release_torch_module,
)
from maor.gpu.limits import InsufficientVRAM, plan_workload, recommend_worker_count
from maor.gpu.memory import MemoryTracker, ReleaseVerification, snapshot


@pytest.fixture(autouse=True)
def _clean_registry():
    ModelRegistry.reset_instance()
    yield
    ModelRegistry.reset_instance()


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------


class TestMemoryAccounting:
    def test_snapshot_never_raises_without_cuda(self):
        snap = snapshot(0, label="probe")
        assert snap.label == "probe"
        assert isinstance(snap.allocated_mb, float)

    def test_snapshot_records_reserved_not_just_allocated(self):
        """Reserved is what limits the next allocation; allocated alone misleads."""
        snap = snapshot(0)
        assert hasattr(snap, "reserved_mb")
        assert hasattr(snap, "peak_reserved_mb")

    def test_release_verification_judges_on_allocated_not_reserved(self):
        """The caching allocator legitimately keeps reserved memory high."""
        before = snapshot(0, label="b")
        after = snapshot(0, label="a")
        v = ReleaseVerification(label="m", before=before, after=after)
        assert v.clean is True

    def test_release_flagged_dirty_when_memory_not_returned(self):
        from maor.gpu.memory import MemorySnapshot

        before = MemorySnapshot(device=0, available=True, allocated_mb=3000.0)
        after = MemorySnapshot(device=0, available=True, allocated_mb=2900.0)
        v = ReleaseVerification(
            label="summariser", before=before, after=after, expected_mb=2800.0
        )
        assert v.clean is False
        assert v.residual_mb == 2900.0
        assert v.allocated_freed_mb == 100.0

    def test_tracker_detects_accumulation_across_experiments(self):
        from maor.gpu.memory import MemorySnapshot

        tracker = MemoryTracker(0)
        tracker.marks = [
            MemorySnapshot(device=0, available=True, allocated_mb=100.0, label="start"),
            MemorySnapshot(device=0, available=True, allocated_mb=900.0, label="end"),
        ]
        assert tracker.leaked_mb() == 800.0

    def test_used_by_others_surfaces_external_processes(self):
        from maor.gpu.memory import MemorySnapshot

        snap = MemorySnapshot(
            device=0, available=True, total_mb=4096.0, free_mb=1000.0, reserved_mb=1096.0
        )
        assert snap.used_by_others_mb == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# Model registry — duplicate residency
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_duplicate_load_of_same_checkpoint_is_refused(self):
        reg = ModelRegistry.instance()
        reg.register(label="summariser", checkpoint="phi-3", device=0, estimated_mb=2800)
        with pytest.raises(DuplicateResidencyError, match="already resident"):
            reg.register(
                label="guardrail-copy", checkpoint="phi-3", device=0, estimated_mb=2800
            )

    def test_deliberate_sharing_returns_the_existing_handle(self):
        reg = ModelRegistry.instance()
        first = reg.register(
            label="summariser", checkpoint="phi-3", device=0, estimated_mb=2800
        )
        second = reg.register(
            label="guardrail",
            checkpoint="phi-3",
            device=0,
            estimated_mb=2800,
            allow_shared=True,
        )
        assert first is second
        assert reg.resident_mb(0) == 2800, "shared model must be counted once"

    def test_same_checkpoint_on_different_devices_is_allowed(self):
        reg = ModelRegistry.instance()
        reg.register(label="a", checkpoint="phi-3", device=0, estimated_mb=2800)
        reg.register(label="b", checkpoint="phi-3", device=1, estimated_mb=2800)
        assert reg.resident_mb(0) == 2800
        assert reg.resident_mb(1) == 2800

    def test_release_all_clears_every_model(self):
        reg = ModelRegistry.instance()
        released_flags = []
        reg.register(
            label="a",
            checkpoint="m1",
            device=0,
            estimated_mb=500,
            releaser=lambda: released_flags.append("a"),
        )
        reg.register(
            label="b",
            checkpoint="m2",
            device=0,
            estimated_mb=500,
            releaser=lambda: released_flags.append("b"),
        )
        names = reg.release_all(0)
        assert set(names) == {"a", "b"}
        assert set(released_flags) == {"a", "b"}
        assert reg.resident_mb(0) == 0

    def test_release_all_continues_when_a_releaser_raises(self):
        """One broken teardown must not strand the other models."""
        reg = ModelRegistry.instance()

        def boom() -> None:
            raise RuntimeError("release failed")

        reg.register(label="bad", checkpoint="m1", device=0, estimated_mb=500, releaser=boom)
        reg.register(label="good", checkpoint="m2", device=0, estimated_mb=500)
        reg.release_all(0)
        assert reg.resident_mb(0) == 0

    def test_audit_reports_untracked_allocation(self):
        reg = ModelRegistry.instance()
        reg.register(label="a", checkpoint="m1", device=0, estimated_mb=500)
        audit = reg.audit(0)
        assert audit["tracked_estimated_mb"] == 500
        assert "untracked_allocated_mb" in audit
        assert "used_by_other_processes_mb" in audit


class TestModelScope:
    def test_scope_releases_on_normal_exit(self):
        reg = ModelRegistry.instance()
        with model_scope(
            "m", checkpoint="ckpt", estimated_mb=100, loader=lambda: object()
        ):
            assert reg.is_resident("ckpt", 0)
        assert not reg.is_resident("ckpt", 0)

    def test_scope_releases_when_the_block_raises(self):
        """A failed experiment must not leave weights resident."""
        reg = ModelRegistry.instance()
        with pytest.raises(ValueError):
            with model_scope(
                "m", checkpoint="ckpt", estimated_mb=100, loader=lambda: object()
            ):
                raise ValueError("experiment failed")
        assert not reg.is_resident("ckpt", 0)

    def test_scope_calls_the_custom_releaser(self):
        called = []
        with model_scope(
            "m",
            checkpoint="ckpt",
            estimated_mb=100,
            loader=lambda: "model",
            releaser=lambda obj: called.append(obj),
        ):
            pass
        assert called == ["model"]

    def test_release_of_nothing_is_safe(self):
        v = release_torch_module("empty", None, device=0)
        assert v.clean is True


# ---------------------------------------------------------------------------
# Capacity planning
# ---------------------------------------------------------------------------


class TestWorkloadPlanning:
    def test_plan_passes_through_on_cpu(self):
        plan = plan_workload("h2", model_mb=2800, requested_batch_size=8)
        assert plan.fits is True
        assert "no CUDA device" in " ".join(plan.adjustments)

    def test_oversized_model_is_blocked_with_actionable_reason(self):
        plan = plan_workload(
            "summariser",
            model_mb=9000,
            total_mb_override=4096,
            usable_fraction=0.78,
        )
        # On CPU the planner short-circuits; force the GPU branch explicitly.
        if plan.fits and "no CUDA device" in " ".join(plan.adjustments):
            pytest.skip("planner short-circuits without CUDA")
        assert plan.fits is False
        assert "quantisation" in (plan.blocking_reason or "")

    def test_batch_size_is_reduced_to_fit_and_recorded(self):
        from maor.gpu import limits
        from maor.gpu.memory import MemorySnapshot

        real = limits.snapshot
        limits.snapshot = lambda device=0, label="": MemorySnapshot(
            device=0, available=True, total_mb=4096.0, free_mb=3000.0
        )
        try:
            plan = plan_workload(
                "sentiment",
                model_mb=2800,
                usable_fraction=0.78,
                requested_batch_size=32,
                per_sample_mb=40.0,
            )
            assert plan.batch_size < 32
            assert any("batch size reduced" in a for a in plan.adjustments)
            assert "execution-only change" in " ".join(plan.adjustments)
        finally:
            limits.snapshot = real

    def test_plan_never_silently_reduces_sample_count(self):
        """Execution parameters may be adjusted; the experiment may not."""
        plan = plan_workload("h2", model_mb=100, requested_batch_size=4)
        assert "sample" not in " ".join(plan.adjustments).lower()
        assert "never adjusted silently" in plan.to_dict()["validity_note"]

    def test_blocked_plan_raises_on_demand(self):
        from maor.gpu import limits
        from maor.gpu.memory import MemorySnapshot

        real = limits.snapshot
        limits.snapshot = lambda device=0, label="": MemorySnapshot(
            device=0, available=True, total_mb=4096.0, free_mb=500.0
        )
        try:
            plan = plan_workload("big", model_mb=3000, usable_fraction=0.78)
            assert plan.fits is False
            with pytest.raises(InsufficientVRAM):
                plan.raise_if_blocked()
        finally:
            limits.snapshot = real

    def test_worker_count_is_capped_by_model_size(self):
        from maor.gpu import limits
        from maor.gpu.memory import MemorySnapshot

        real = limits.snapshot
        limits.snapshot = lambda device=0, label="": MemorySnapshot(
            device=0, available=True, total_mb=4096.0, free_mb=4000.0
        )
        try:
            workers, notes = recommend_worker_count(
                model_mb=2800, usable_fraction=0.78, requested_workers=4
            )
            assert workers == 1
            assert any("oversubscribe" in n for n in notes)
            assert any("serialise" in n for n in notes)
        finally:
            limits.snapshot = real


# ---------------------------------------------------------------------------
# Bounded execution
# ---------------------------------------------------------------------------


class TestTimeouts:
    def test_fast_call_returns_normally(self):
        assert run_with_timeout(lambda: 42, 5.0, label="quick") == 42

    def test_slow_call_raises_timeout(self):
        with pytest.raises(TimeoutError_, match="deadline"):
            run_with_timeout(lambda: time.sleep(3), 0.2, label="slow")

    def test_timeout_error_is_a_builtin_timeout_error(self):
        """So `except TimeoutError` in calling code still works."""
        assert issubclass(TimeoutError_, TimeoutError)

    def test_exception_propagates_with_its_type(self):
        def boom():
            raise ValueError("inner failure")

        with pytest.raises(ValueError, match="inner failure"):
            run_with_timeout(boom, 5.0)

    def test_zero_timeout_means_unbounded(self):
        assert run_with_timeout(lambda: "ok", 0) == "ok"

    def test_guard_does_not_fire_when_block_completes(self):
        with TimeoutGuard("fast", 5.0, interrupt=False) as guard:
            pass
        assert guard.timed_out is False

    def test_guard_fires_handler_on_overrun(self):
        fired = []
        guard = TimeoutGuard(
            "slow", 0.15, on_timeout=lambda: fired.append(True), interrupt=False
        )
        try:
            with guard:
                time.sleep(0.5)
        except TimeoutError_:
            pass
        assert fired == [True]
        assert guard.timed_out is True


class TestRetryPolicy:
    def test_retries_are_bounded(self):
        attempts = []

        def flaky():
            attempts.append(1)
            raise ConnectionError("network")

        policy = RetryPolicy(max_attempts=3, initial_delay_s=0.01)
        with pytest.raises(ConnectionError):
            policy.run(flaky, label="fetch")
        assert len(attempts) == 3, "must stop at max_attempts, not loop forever"

    def test_oom_is_not_retried_by_default(self):
        """Retrying an OOM with identical parameters fails identically."""
        attempts = []

        def oom():
            attempts.append(1)
            raise RuntimeError("CUDA out of memory")

        policy = RetryPolicy(max_attempts=5, initial_delay_s=0.01)
        with pytest.raises(RuntimeError):
            policy.run(oom)
        assert len(attempts) == 1

    def test_success_after_transient_failure(self):
        state = {"n": 0}

        def eventually():
            state["n"] += 1
            if state["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        policy = RetryPolicy(max_attempts=5, initial_delay_s=0.01)
        assert policy.run(eventually) == "ok"

    def test_max_attempts_must_be_positive(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_oom_detection_by_message(self):
        assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
        assert not is_oom(ValueError("something else"))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def test_completed_experiment_is_recorded_with_instrumentation(self):
        runner = ExperimentRunner()
        outcome = runner.run_one(
            ExperimentSpec(name="e1", run=lambda: {"result_path": "results/e1.json"})
        )
        assert outcome.status is Status.COMPLETED
        assert outcome.result_path == "results/e1.json"
        assert outcome.duration_s >= 0
        assert "marks" in outcome.memory_trace

    def test_failure_is_contained_and_the_next_experiment_still_runs(self):
        """The core isolation property."""
        ran = []

        def boom():
            raise RuntimeError("experiment blew up")

        def fine():
            ran.append("second")
            return {}

        runner = ExperimentRunner()
        outcomes = runner.run_all(
            [
                ExperimentSpec(name="bad", run=boom),
                ExperimentSpec(name="good", run=fine),
            ]
        )
        assert outcomes[0].status is Status.FAILED
        assert outcomes[1].status is Status.COMPLETED
        assert ran == ["second"]

    def test_oom_is_classified_separately_from_generic_failure(self):
        def oom():
            raise RuntimeError("CUDA out of memory")

        runner = ExperimentRunner()
        outcome = runner.run_one(ExperimentSpec(name="oom", run=oom))
        assert outcome.status is Status.OOM

    def test_timeout_is_classified_and_cleanup_still_runs(self):
        runner = ExperimentRunner()
        outcome = runner.run_one(
            ExperimentSpec(name="slow", run=lambda: time.sleep(2), timeout_s=0.2)
        )
        assert outcome.status is Status.TIMED_OUT
        assert outcome.memory_trace  # cleanup and instrumentation still ran

    def test_models_left_resident_by_a_failed_run_are_released(self):
        def leaky():
            ModelRegistry.instance().register(
                label="leaked", checkpoint="phi-3", device=0, estimated_mb=2800
            )
            raise RuntimeError("died holding a model")

        runner = ExperimentRunner()
        runner.run_one(ExperimentSpec(name="leaky", run=leaky))
        assert ModelRegistry.instance().resident_mb(0) == 0

    def test_critical_failure_stops_the_sequence(self):
        ran = []
        runner = ExperimentRunner()
        outcomes = runner.run_all(
            [
                ExperimentSpec(
                    name="critical",
                    run=lambda: (_ for _ in ()).throw(RuntimeError("x")),
                    critical=True,
                ),
                ExperimentSpec(name="later", run=lambda: ran.append("later") or {}),
            ]
        )
        assert len(outcomes) == 1
        assert ran == []

    def test_consecutive_failures_stop_the_sequence(self):
        def boom():
            raise RuntimeError("x")

        runner = ExperimentRunner(max_consecutive_failures=2)
        outcomes = runner.run_all(
            [ExperimentSpec(name=f"e{i}", run=boom) for i in range(5)]
        )
        assert len(outcomes) == 2

    def test_gpu_experiment_is_blocked_not_failed_without_cuda(self):
        runner = ExperimentRunner()
        outcome = runner.run_one(
            ExperimentSpec(name="gpu-only", run=lambda: {}, requires_gpu=True)
        )
        if snapshot(0).available:
            pytest.skip("CUDA present")
        assert outcome.status is Status.BLOCKED
        assert "implemented" in (outcome.failure_reason or "")

    def test_checkpoint_resume_skips_completed_experiments(self, tmp_path):
        checkpoint = tmp_path / "progress.json"
        ran = []

        specs = [
            ExperimentSpec(name="a", run=lambda: ran.append("a") or {}),
            ExperimentSpec(name="b", run=lambda: ran.append("b") or {}),
        ]

        ExperimentRunner(checkpoint_path=checkpoint).run_all(specs)
        assert ran == ["a", "b"]
        assert checkpoint.exists()

        ran.clear()
        outcomes = ExperimentRunner(checkpoint_path=checkpoint).run_all(specs)
        assert ran == [], "completed experiments must not re-run"
        assert all(o.status is Status.SKIPPED for o in outcomes)

    def test_summary_reports_status_counts(self):
        runner = ExperimentRunner()
        runner.run_all(
            [
                ExperimentSpec(name="ok", run=lambda: {}),
                ExperimentSpec(
                    name="bad", run=lambda: (_ for _ in ()).throw(RuntimeError("x"))
                ),
            ]
        )
        summary = runner.summary()
        assert summary["n_experiments"] == 2
        assert summary["by_status"]["COMPLETED"] == 1
        assert summary["by_status"]["FAILED"] == 1
        assert summary["all_completed"] is False


class TestBlockedSemantics:
    """BLOCKED means the hardware is absent, not that anything went wrong.

    Counting it as a failure trips the consecutive-failure circuit breaker and
    aborts experiments that could still have run — observed on a CPU-only
    machine where three GPU experiments ended the sequence early.
    """

    def _blocked_spec(self, name: str) -> ExperimentSpec:
        return ExperimentSpec(name=name, run=lambda: {}, requires_gpu=True)

    @pytest.fixture(autouse=True)
    def _needs_no_cuda(self):
        if snapshot(0).available:
            pytest.skip("CUDA present; BLOCKED cannot be provoked")

    def test_blocked_does_not_trip_the_failure_breaker(self):
        ran = []
        runner = ExperimentRunner(max_consecutive_failures=2)
        outcomes = runner.run_all(
            [
                self._blocked_spec("gpu_a"),
                self._blocked_spec("gpu_b"),
                self._blocked_spec("gpu_c"),
                ExperimentSpec(name="cpu_work", run=lambda: ran.append("ran") or {}),
            ]
        )
        assert len(outcomes) == 4, "blocked experiments must not abort the sequence"
        assert ran == ["ran"], "CPU work after blocked GPU work must still run"
        assert outcomes[-1].status is Status.COMPLETED

    def test_blocked_is_not_recorded_as_completed(self, tmp_path):
        """A GPU run later must pick these up rather than skipping them."""
        checkpoint = tmp_path / "progress.json"
        ExperimentRunner(checkpoint_path=checkpoint).run_all(
            [self._blocked_spec("gpu_only")]
        )
        import json

        completed = json.loads(checkpoint.read_text(encoding="utf-8"))["completed"]
        assert "gpu_only" not in completed

    def test_summary_separates_blocked_from_failed(self):
        runner = ExperimentRunner()
        runner.run_all(
            [
                ExperimentSpec(name="ok", run=lambda: {}),
                self._blocked_spec("gpu_only"),
            ]
        )
        summary = runner.summary()
        assert summary["n_blocked"] == 1
        assert summary["blocked_experiments"] == ["gpu_only"]
        assert summary["all_runnable_completed"] is True
        assert summary["all_completed"] is False

    def test_a_real_failure_still_fails_the_run(self):
        runner = ExperimentRunner()
        runner.run_all(
            [
                ExperimentSpec(
                    name="bad", run=lambda: (_ for _ in ()).throw(RuntimeError("x"))
                ),
                self._blocked_spec("gpu_only"),
            ]
        )
        summary = runner.summary()
        assert summary["all_runnable_completed"] is False
