"""Tests for the real Ray distributed pipeline.

These run against local, single-process Ray (``ray.init(num_cpus=..., num_gpus=0)``
with no ``address``) — a real Ray object store and real actor scheduling, but a
single machine. That validates the mechanism (actor construction, remote
dispatch, parallel fan-out, cluster topology detection, VRAM-aware model
lifecycle inside an actor) without requiring a second physical machine.

What it does not and cannot validate: cross-machine network latency, real
node-affinity placement across two hosts, or GPU actor scheduling (there is no
GPU here). Those require the actual two-node cluster and are listed as pending
in docs/RESULTS_STATUS.md. Tests that need model downloads are marked
``network`` and skip cleanly without one.
"""

from __future__ import annotations

import socket

import pytest

from maor.config import Config


def _network_available() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


class TestPipelineRefusesFakeDistribution:
    """The guard that prevents a repeat of the original H1 defect.

    Defined first in this module, before any test requests the ``ray_local``
    fixture below: the "no cluster reachable" case must be exercised while Ray
    is genuinely uninitialised, and ``ray_local`` is module-scoped, so once any
    other test activates it, Ray stays up for the rest of the module.
    """

    def test_pipeline_raises_without_ray_mode(self):
        from maor.pipeline.orchestrator import Pipeline

        cfg = Config.load()
        assert cfg.execution.mode == "local"
        with pytest.raises(NotImplementedError, match="execution.mode='ray'"):
            Pipeline(cfg, device="cpu", distributed=True)

    def test_pipeline_with_ray_mode_and_no_cluster_fails_rather_than_fabricating_one(
        self,
    ):
        """distributed=True + mode=ray must reach for a real cluster, not a stub.

        With no explicit address, Pipeline connects via ``ray.init(address="auto")``,
        which raises ``ConnectionError`` when no Ray instance is already running —
        it does not silently start a new local cluster. That matters here: a
        silent local fallback would let "distributed" mode quietly run
        single-process work again, the exact defect this guard exists to prevent.
        """
        import ray as ray_module

        if ray_module.is_initialized():
            pytest.skip("Ray already initialised earlier in this session")

        from maor.pipeline.orchestrator import Pipeline

        cfg = Config.load()
        cfg.execution.mode = "ray"
        with pytest.raises(Exception):
            Pipeline(cfg, device="cpu", distributed=True)

    def test_pipeline_connects_to_an_already_running_local_cluster(self, ray_local):
        """When a cluster genuinely is running, Pipeline uses it for real."""
        from maor.pipeline.orchestrator import Pipeline

        cfg = Config.load()
        cfg.execution.mode = "ray"
        pipe = Pipeline(cfg, device="cpu", distributed=True)
        try:
            assert pipe._distributed is not None
            assert pipe._distributed.topology.n_nodes >= 1
        finally:
            pipe.close()


@pytest.fixture(scope="module")
def ray_local():
    """A local, single-process Ray instance. Shut down after the module.

    Ray workers are separate OS processes and do not inherit the driver's
    PYTHONPATH, so without ``py_modules`` every actor construction here would
    fail with ``ModuleNotFoundError: No module named 'maor'`` — a failure that
    exercising only the driver process (e.g. calling functions directly, never
    dispatching to an actor) would never catch.
    """
    ray = pytest.importorskip("ray")
    from maor.pipeline.distributed import _MAOR_PACKAGE_DIR

    ray.init(
        num_cpus=4,
        num_gpus=0,
        ignore_reinit_error=True,
        logging_level=50,
        runtime_env={"py_modules": [str(_MAOR_PACKAGE_DIR)]},
    )
    yield ray
    ray.shutdown()


@pytest.mark.slow
class TestClusterTopology:
    def test_describe_cluster_reports_local_as_single_node(self, ray_local):
        from maor.pipeline.distributed import describe_cluster

        topo = describe_cluster()
        assert topo.n_nodes == 1
        assert topo.is_single_node is True
        assert topo.n_gpu_nodes == 0
        assert topo.total_cpus >= 1

    def test_topology_to_dict_warns_about_single_node_limits(self, ray_local):
        from maor.pipeline.distributed import describe_cluster

        d = describe_cluster().to_dict()
        assert "cannot demonstrate cross-machine" in d["note"]

    def test_describe_cluster_raises_without_a_connection(self):
        from maor.pipeline.distributed import ClusterError, describe_cluster

        ray = pytest.importorskip("ray")
        if ray.is_initialized():
            pytest.skip("Ray already initialised by another test in this session")
        with pytest.raises(Exception):
            describe_cluster()


@pytest.mark.slow
class TestIngestionActor:
    def test_ingestion_actor_chunks_and_filters(self, ray_local):
        from maor.pipeline.distributed import _make_ingestion_actor, describe_cluster

        cfg = Config.load()
        topo = describe_cluster()
        actor_cls = _make_ingestion_actor(topo)
        actor = actor_cls.remote(cfg.chunking)

        doc = "Revenue increased twelve percent this quarter. " * 100
        result = ray_local.get(actor.chunk_and_filter.remote(doc, "8-K", True))

        assert result["texts"], "must produce at least one chunk"
        assert result["chunk_info"]["n_before"] >= result["chunk_info"]["n_after"]

    def test_ingestion_actor_can_disable_filtering(self, ray_local):
        from maor.pipeline.distributed import _make_ingestion_actor, describe_cluster

        cfg = Config.load()
        cfg.chunking.max_chunks_8k = 2
        topo = describe_cluster()
        actor = _make_ingestion_actor(topo).remote(cfg.chunking)

        doc = "Revenue increased twelve percent this quarter. " * 200
        filtered = ray_local.get(actor.chunk_and_filter.remote(doc, "8-K", True))
        unfiltered = ray_local.get(actor.chunk_and_filter.remote(doc, "8-K", False))

        assert filtered["chunk_info"]["method"] != "disabled"
        assert unfiltered["chunk_info"]["method"] == "disabled"
        assert len(unfiltered["texts"]) >= len(filtered["texts"])

    def test_ingestion_actor_reports_its_own_host(self, ray_local):
        from maor.pipeline.distributed import _make_ingestion_actor, describe_cluster

        actor = _make_ingestion_actor(describe_cluster()).remote(Config.load().chunking)
        host = ray_local.get(actor.ping.remote())
        assert host  # any non-empty hostname


@pytest.mark.slow
class TestTechnicalActor:
    def test_technical_actor_computes_indicators(self, ray_local):
        if not _network_available():
            pytest.skip("no network access for yfinance")
        from maor.pipeline.distributed import _make_technical_actor, describe_cluster

        actor = _make_technical_actor(describe_cluster()).remote()
        result = ray_local.get(actor.compute.remote("AAPL"))
        assert result["status"] == "ok"
        assert 0 <= result["rsi"] <= 100

    def test_technical_actor_handles_bad_ticker_gracefully(self, ray_local):
        if not _network_available():
            pytest.skip("no network access for yfinance")
        from maor.pipeline.distributed import _make_technical_actor, describe_cluster

        actor = _make_technical_actor(describe_cluster()).remote()
        result = ray_local.get(actor.compute.remote("NOT_A_REAL_TICKER_XYZ123"))
        assert result["status"] == "error"
        assert "error" in result


@pytest.mark.slow
@pytest.mark.network
class TestSentimentActor:
    """Loads real 3-checkpoint models inside a Ray actor. Slow; needs network once."""

    def test_sentiment_actor_classifies_with_three_checkpoints(self, ray_local):
        if not _network_available():
            pytest.skip("no network access for model downloads")
        from maor.pipeline.distributed import _make_sentiment_actor

        cfg = Config.load()
        cfg.execution.device = "cpu"
        actor_cls = _make_sentiment_actor(cfg)
        actor = actor_cls.remote(None, 0.85)

        texts = [
            "Revenue grew twenty percent and we expect continued growth next year.",
            "The SEC opened an enforcement investigation into accounting practices.",
        ]
        result = ray_local.get(actor.classify.remote(texts))

        assert result["matrix"]["dimension_order"] == ["market", "regulatory", "temporal"]
        assert all(result["matrix"]["present"])
        labels = result["matrix"]["per_dimension_labels"]
        assert labels["market"] == ["positive", "neutral", "negative"]
        assert labels["temporal"] == ["not_fls", "nonspecific_fls", "specific_fls"]

        release = ray_local.get(actor.shutdown.remote())
        assert release["release"]["clean"] is True


@pytest.mark.slow
class TestParallelDispatch:
    """The actual PDC contribution: two remote calls in flight before either awaits."""

    def test_dispatch_returns_before_work_completes(self, ray_local):
        from maor.pipeline.distributed import (
            _make_ingestion_actor,
            _make_technical_actor,
            describe_cluster,
        )

        cfg = Config.load()
        topo = describe_cluster()
        ingestion = _make_ingestion_actor(topo).remote(cfg.chunking)
        technical = _make_technical_actor(topo).remote()

        import time

        doc = "Revenue increased twelve percent. " * 300
        t0 = time.perf_counter()
        ref_a = ingestion.chunk_and_filter.remote(doc, "8-K", True)
        ref_b = technical.compute.remote("AAPL" if _network_available() else "X")
        dispatch_s = time.perf_counter() - t0

        # Dispatch must return near-instantly: the work has not run yet, only
        # been submitted. This is what "parallel fan-out" means mechanically.
        assert dispatch_s < 0.5

        # Both results are still obtainable afterward.
        ray_local.get(ref_a)
        ray_local.get(ref_b)
