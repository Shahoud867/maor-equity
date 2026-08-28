"""Real two-node distributed execution over Ray.

This is the Ray path :class:`~maor.pipeline.orchestrator.Pipeline` refuses to
run under (it raises ``NotImplementedError`` for ``distributed=True`` unless
``execution.mode == "ray"``, precisely so a single-node run could never be
mislabelled as distributed — see ``docs/AUDIT_RESPONSE.md``, finding on the
original H1's fabricated distribution factor). This module is what makes the
label true: CPU-bound stages (ingestion, chunking, chunk filtering, technical
indicators) run as actors pinned to the head node (Node A); GPU-bound stages
(sentiment, summarisation, guardrail) run as actors that request a GPU
fraction, which Ray's scheduler can only satisfy on the node that advertises
one (Node B).

**Node placement.** Node A actors use :class:`NodeAffinitySchedulingStrategy`
with ``soft=True`` against the ID of whichever node reports zero GPUs — found
at connection time via ``ray.nodes()``, not hard-coded, so the same code runs
on a two-node cluster or (with a warning) on a single machine. GPU actors are
never pinned explicitly: requesting ``num_gpus > 0`` is sufficient, because a
CPU-only node cannot satisfy that request.

**Communication measurement.** Every cross-actor handoff goes through
:func:`maor.pipeline.instrumentation.measure_ray_communication`-style
accounting: a stage is tagged ``communication`` only for the serialise/transfer
window around an *already-dispatched* remote call, and the blocking wait for
the result is tagged ``compute``. This is the fix for the original project's
worst instrumentation defect (finding C5): ``t_deserialize_ms`` there was 549
seconds of GPU generation counted as communication because the wait for a
still-running remote call was timed as if it were a network operation.

**VRAM.** GPU actors are constructed with the same :class:`SentimentBundle` /
:class:`SummarisationModel` / :class:`GuardrailAgent` classes the single-node
pipeline uses, so the VRAM budget, phase serialisation, and release-verification
guarantees documented in ``docs/VRAM_LIFECYCLE.md`` are identical in both modes
— there is exactly one place model lifecycle is implemented, not two that could
drift apart.

**What this module does not claim.** It has been exercised with local
single-process Ray (``ray.init()`` with no ``address``), which validates actor
construction, remote dispatch, and the communication-accounting mechanism
against a real object store. It has not been run across two physical machines,
because that requires the second machine to be connected. That is recorded as
pending in ``docs/RESULTS_STATUS.md``, not asserted here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The project has no installed package (no pyproject.toml/pip install -e .);
# every entry point imports maor via PYTHONPATH set in the calling shell. Ray
# actors run in separate OS processes that do NOT inherit that PYTHONPATH, so
# `import maor` inside an actor raises ModuleNotFoundError even when the
# driver process (this one) can import it fine — a failure that would appear
# only once a real second machine was connected, and would look like a
# cluster/network problem rather than what it is. `py_modules` ships the
# package to every worker Ray starts, on this machine or a remote one, so the
# same code works locally and distributed without requiring a packaging step
# on Node B.
_MAOR_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # .../src/maor

from ..agents.guardrail import GuardrailAgent
from ..agents.sentiment import DimensionRouter, SentimentBundle, build_matrix
from ..agents.summarisation import MapReduceSummariser, SummarisationModel
from ..agents.technical import TechnicalAgent, TechnicalIndicators
from ..data.chunking import ChunkFilter, chunk_document
from ..gpu.lifecycle import ModelRegistry
from ..hardware import VRAMBudget
from .instrumentation import StageRecorder

log = logging.getLogger(__name__)


class ClusterError(RuntimeError):
    """Raised when the connected Ray cluster cannot satisfy what was asked of it."""


@dataclass
class ClusterTopology:
    """What :func:`describe_cluster` found when it connected."""

    n_nodes: int
    n_gpu_nodes: int
    n_cpu_only_nodes: int
    head_node_id: str | None
    gpu_node_ids: tuple[str, ...]
    total_cpus: float
    total_gpus: float
    is_single_node: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_gpu_nodes": self.n_gpu_nodes,
            "n_cpu_only_nodes": self.n_cpu_only_nodes,
            "total_cpus": self.total_cpus,
            "total_gpus": self.total_gpus,
            "is_single_node": self.is_single_node,
            "note": (
                "is_single_node=True means this run cannot demonstrate cross-"
                "machine communication even though it exercises the real Ray "
                "actor and object-store mechanism. Treat any communication "
                "measurement from a single-node run as a lower bound."
            ),
        }


def describe_cluster() -> ClusterTopology:
    """Inspect the connected cluster. Call after ``ray.init()``."""
    import ray

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise ClusterError("connected to Ray but no alive nodes were reported")

    gpu_nodes = [n for n in nodes if n.get("Resources", {}).get("GPU", 0) > 0]
    cpu_only_nodes = [n for n in nodes if n.get("Resources", {}).get("GPU", 0) == 0]

    resources = ray.cluster_resources()
    head = next((n for n in cpu_only_nodes), nodes[0])

    return ClusterTopology(
        n_nodes=len(nodes),
        n_gpu_nodes=len(gpu_nodes),
        n_cpu_only_nodes=len(cpu_only_nodes),
        head_node_id=head.get("NodeID"),
        gpu_node_ids=tuple(n.get("NodeID") for n in gpu_nodes),
        total_cpus=float(resources.get("CPU", 0)),
        total_gpus=float(resources.get("GPU", 0)),
        is_single_node=len(nodes) < 2,
    )


def _head_affinity(topology: ClusterTopology) -> Any:
    """Scheduling strategy that prefers the CPU-only node, falling back softly."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if topology.head_node_id is None:
        return "DEFAULT"
    return NodeAffinitySchedulingStrategy(node_id=topology.head_node_id, soft=True)


# ---------------------------------------------------------------------------
# Ray actor wrappers. Thin — all real logic lives in the plain classes these
# wrap, so single-node and distributed execution share one implementation of
# every model and every piece of business logic.
# ---------------------------------------------------------------------------


def _make_ingestion_actor(topology: ClusterTopology):
    import ray

    @ray.remote(num_cpus=1, scheduling_strategy=_head_affinity(topology))
    class IngestionActor:
        """Node A: chunking and TF-IDF filtering. No GPU, no model weights."""

        def __init__(self, chunking_cfg: Any) -> None:
            self._chunking_cfg = chunking_cfg
            self._filter = ChunkFilter(
                sim_threshold=chunking_cfg.sim_threshold,
                max_chunks_8k=chunking_cfg.max_chunks_8k,
                max_chunks_10k=chunking_cfg.max_chunks_10k,
                tfidf_max_features=chunking_cfg.tfidf_max_features,
            )

        def chunk_and_filter(
            self, document: str, filing_type: str, filter_enabled: bool
        ) -> dict[str, Any]:
            chunks = chunk_document(
                document,
                window_tokens=self._chunking_cfg.window_tokens,
                stride_tokens=self._chunking_cfg.stride_tokens,
            )
            if filter_enabled:
                result = self._filter.filter(chunks, filing_type=filing_type)
                selected = result.chunks
                info = result.to_dict()
            else:
                selected = chunks
                info = {
                    "n_before": len(chunks), "n_after": len(chunks),
                    "reduction_pct": 0.0, "elapsed_ms": 0.0, "method": "disabled",
                }
            return {"texts": [c.text for c in selected], "chunk_info": info}

        def ping(self) -> str:
            import socket

            return socket.gethostname()

    return IngestionActor


def _make_technical_actor(topology: ClusterTopology):
    import ray

    @ray.remote(num_cpus=1, scheduling_strategy=_head_affinity(topology))
    class TechnicalActor:
        """Node A: RSI/MACD/Bollinger/VWAP via yfinance. No GPU."""

        def __init__(self) -> None:
            self._agent = TechnicalAgent()

        def compute(self, ticker: str) -> dict[str, Any]:
            result: TechnicalIndicators = self._agent.compute_indicators(ticker)
            return result.to_dict()

    return TechnicalActor


def _make_sentiment_actor(config: Any):
    import ray

    gpu_frac = 0.0 if config.execution.device == "cpu" else 0.35

    @ray.remote(num_gpus=gpu_frac, max_restarts=3, max_task_retries=2)
    class SentimentActor:
        """Node B: three-checkpoint sentiment bundle, VRAM-budget aware."""

        def __init__(self, budget_total_mb: float | None, usable_fraction: float) -> None:
            self._budget = VRAMBudget.from_hardware(
                override_total_mb=budget_total_mb, usable_fraction=usable_fraction
            )
            device = "cuda" if gpu_frac > 0 else "cpu"
            self._bundle = SentimentBundle(
                market_checkpoint=config.models.sentiment_market,
                regulatory_checkpoint=config.models.sentiment_regulatory,
                temporal_checkpoint=config.models.sentiment_temporal,
                device=device,
                quantisation=config.models.sentiment_quantisation
                if device != "cpu"
                else "none",
                batch_size=config.models.sentiment_batch_size,
                max_length=config.models.sentiment_max_length,
            )
            with self._budget.phase(
                "sentiment", config.models.sentiment_estimated_vram_mb
            ):
                self._bundle.load()
            self._router = DimensionRouter()

        def classify(self, texts: list[str]) -> dict[str, Any]:
            routed = self._router.route(texts)
            results = self._bundle.classify_all(routed["routed"])
            matrix = build_matrix(results)
            return {
                "matrix": matrix.to_dict(),
                "dimensions": {k: v.to_dict() for k, v in results.items()},
                "routing_coverage": routed["coverage"],
            }

        def ping(self) -> str:
            import socket

            return socket.gethostname()

        def shutdown(self) -> dict[str, Any]:
            verification = self._bundle.unload()
            return {"release": verification.to_dict() if verification else None}

    return SentimentActor


def _make_summariser_actor(config: Any):
    import ray

    gpu_frac = 0.0 if config.execution.device == "cpu" else 0.2

    @ray.remote(num_gpus=gpu_frac, max_restarts=3, max_task_retries=2)
    class SummariserActor:
        """Node B: the shared Phi-3 model, used by both summarisation and the
        guardrail through :meth:`generate`. Loaded once — this actor handle is
        passed to :class:`GuardrailActor` rather than a second copy being
        constructed, which is what the shared-model VRAM saving depends on.
        """

        def __init__(self) -> None:
            device = "cuda" if gpu_frac > 0 else "cpu"
            self._model = SummarisationModel(
                checkpoint=config.models.summarizer,
                device=device,
                quantisation=config.models.summarizer_quantisation
                if device != "cpu"
                else "none",
                max_input_tokens=config.models.max_input_tokens,
                trust_remote_code=config.models.trust_remote_code,
                do_sample=config.models.do_sample,
                temperature=config.models.temperature,
                estimated_vram_mb=config.models.summarizer_estimated_vram_mb,
            )
            self._model.load()
            self._summariser = MapReduceSummariser(
                self._model,
                map_max_new_tokens=config.models.map_max_new_tokens,
                reduce_max_new_tokens=config.models.reduce_max_new_tokens,
            )

        def generate(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
            """Exposed so GuardrailActor can share this exact model instance."""
            result = self._model.generate(prompt, max_new_tokens=max_new_tokens)
            return {
                "text": result.text,
                "scaffolding_removed": result.scaffolding_removed,
                "n_input_tokens": result.n_input_tokens,
                "n_generated_tokens": result.n_generated_tokens,
                "truncated_input": result.truncated_input,
            }

        def summarise(self, chunk_texts: list[str]) -> dict[str, Any]:
            result = self._summariser.summarise(chunk_texts)
            return result.to_dict()

        def ping(self) -> str:
            import socket

            return socket.gethostname()

        def shutdown(self) -> dict[str, Any]:
            verification = self._model.unload()
            return {"release": verification.to_dict() if verification else None}

    return SummariserActor


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class DistributedRunOutcome:
    ticker: str
    recommendation: dict[str, Any]
    summary: str
    sentiment: dict[str, Any]
    technical: dict[str, Any]
    chunking: dict[str, Any]
    recorder: StageRecorder
    cluster: ClusterTopology
    node_hosts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "recommendation": self.recommendation,
            "summary": self.summary,
            "sentiment": self.sentiment,
            "technical": self.technical,
            "chunking": self.chunking,
            "timings": self.recorder.to_dict(),
            "cluster": self.cluster.to_dict(),
            "node_hosts": self.node_hosts,
            "warnings": self.warnings,
        }


class DistributedPipeline:
    """The Ray-backed counterpart to :class:`~maor.pipeline.orchestrator.Pipeline`.

    Same external contract (``run(ticker=..., document=...)``), different
    execution: every stage after chunking is a remote call to an actor placed
    by the cluster's actual topology rather than executed in-process.
    """

    def __init__(self, config: Any, *, ray_address: str | None = None) -> None:
        import ray

        self.config = config
        if not ray.is_initialized():
            ray.init(
                address=ray_address or "auto",
                ignore_reinit_error=True,
                runtime_env={"py_modules": [str(_MAOR_PACKAGE_DIR)]},
            )
        self.topology = describe_cluster()
        if self.topology.is_single_node:
            log.warning(
                "Ray reports a single node. Actor placement and the Ray object "
                "store are real, but no cross-machine communication is "
                "possible — see ClusterTopology.to_dict()."
            )

        self._ingestion = _make_ingestion_actor(self.topology).remote(config.chunking)
        self._technical = _make_technical_actor(self.topology).remote()
        self._sentiment = _make_sentiment_actor(config).remote(
            config.vram.total_mb, config.vram.usable_fraction
        )
        self._summariser = _make_summariser_actor(config).remote()
        self._guardrail = GuardrailAgentActorProxy(self._summariser, config)

    def close(self) -> dict[str, Any]:
        import ray

        release_info: dict[str, Any] = {}
        for name, handle in (
            ("sentiment", self._sentiment),
            ("summariser", self._summariser),
        ):
            try:
                release_info[name] = ray.get(handle.shutdown.remote(), timeout=60)
            except Exception as exc:
                log.warning("error shutting down %s actor: %s", name, exc)
        for handle in (self._ingestion, self._technical, self._sentiment, self._summariser):
            try:
                ray.kill(handle)
            except Exception:
                pass
        ModelRegistry.instance().release_all()
        return release_info

    def run(
        self,
        *,
        ticker: str,
        document: str,
        filing_type: str = "8-K",
        filter_enabled: bool = True,
    ) -> DistributedRunOutcome:
        import ray

        rec = StageRecorder()
        warnings: list[str] = []
        node_hosts: dict[str, str] = {}

        # ---- Stage 1: ingestion + chunk filter (Node A) --------------------
        with rec.stage("dispatch_ingestion", kind="communication"):
            ingestion_ref = self._ingestion.chunk_and_filter.remote(
                document, filing_type, filter_enabled
            )
        with rec.stage("await_ingestion", kind="compute"):
            ingestion_result = ray.get(ingestion_ref)
        texts = ingestion_result["texts"]
        chunk_info = ingestion_result["chunk_info"]
        if not texts:
            raise ValueError(f"{ticker}: document produced no chunks")

        # ---- Stage 2: parallel fan-out — sentiment (B) || technical (A) ---
        # Both remote calls are dispatched before either is awaited: this is
        # the actual task parallelism, not an artefact of instrumentation.
        with rec.stage(
            "dispatch_phase_a", kind="communication", parallelisable=True
        ):
            sentiment_ref = self._sentiment.classify.remote(texts)
            technical_ref = self._technical.compute.remote(ticker)

        with rec.stage(
            "await_phase_a", kind="compute", parallelisable=True
        ) as meta:
            ready, _ = ray.wait([sentiment_ref, technical_ref], num_returns=2)
            sentiment_result = ray.get(sentiment_ref)
            technical_result = ray.get(technical_ref)
            meta["n_ready"] = len(ready)

        # ---- Stage 3: summarisation (Node B, shared model) -----------------
        with rec.stage("dispatch_summarise", kind="communication"):
            summary_ref = self._summariser.summarise.remote(texts)
        with rec.stage("await_summarise", kind="compute", parallelisable=False):
            summary_result = ray.get(summary_ref)
        if summary_result.get("n_scaffolding_trimmed"):
            warnings.append(
                f"{summary_result['n_scaffolding_trimmed']} generation(s) "
                f"contained instruction scaffolding that was trimmed"
            )

        # ---- Stage 4: guardrail (Node B, same shared model) -----------------
        matrix_dict = sentiment_result["matrix"]
        with rec.stage("guardrail", kind="compute", parallelisable=False):
            verdict = self._guardrail.assess(
                summary_result["summary"], matrix_dict, technical_result
            )
        if verdict.get("recommendation") == "ASSESSMENT_FAILED":
            warnings.append(
                "guardrail could not parse one or both stances; no "
                "recommendation was produced"
            )

        try:
            node_hosts["ingestion"] = ray.get(self._ingestion.ping.remote(), timeout=5)
            node_hosts["sentiment"] = ray.get(self._sentiment.ping.remote(), timeout=5)
            node_hosts["summariser"] = ray.get(self._summariser.ping.remote(), timeout=5)
        except Exception as exc:
            warnings.append(f"could not collect node hostnames: {exc}")

        return DistributedRunOutcome(
            ticker=ticker,
            recommendation=verdict,
            summary=summary_result["summary"],
            sentiment=sentiment_result,
            technical=technical_result,
            chunking=chunk_info,
            recorder=rec,
            cluster=self.topology,
            node_hosts=node_hosts,
            warnings=warnings,
        )


class GuardrailAgentActorProxy:
    """Runs the guardrail's deterministic arbitration locally, generation remotely.

    The arbiter itself (:meth:`GuardrailAgent.arbitrate`) is pure Python with no
    model access — running it as a Ray actor would add a network hop for no
    benefit. Only the two LLM calls it needs (bull/bear stance generation) go
    through the shared summariser actor, which is where the GPU-resident model
    actually lives.
    """

    def __init__(self, summariser_actor_handle: Any, config: Any) -> None:
        self._handle = summariser_actor_handle
        self._max_new_tokens = config.models.guardrail_max_new_tokens
        self._agent = GuardrailAgent(model=None, max_new_tokens=self._max_new_tokens)

    def assess(
        self, summary: str, matrix_dict: dict[str, Any], technical: dict[str, Any]
    ) -> dict[str, Any]:
        import ray

        from ..agents.guardrail import BEAR_PROMPT, BULL_PROMPT, parse_stance

        context_lines = [f"Summary: {summary[:800]}"]
        for dim, present in zip(
            matrix_dict["dimension_order"], matrix_dict["present"]
        ):
            if not present:
                context_lines.append(f"{dim.capitalize()} sentiment: not present in document")
                continue
            idx = matrix_dict["dimension_order"].index(dim)
            row = matrix_dict["matrix"][idx]
            labels = matrix_dict["per_dimension_labels"][dim]
            detail = ", ".join(f"{lab}={v:.2f}" for lab, v in zip(labels, row))
            context_lines.append(f"{dim.capitalize()} ({'/'.join(labels)}): {detail}")
        rsi = technical.get("rsi")
        if rsi is not None:
            context_lines.append(
                f"RSI={rsi:.1f}, MACD bullish crossover="
                f"{technical.get('macd_crossover_bullish', False)}"
            )
        context = "\n".join(context_lines)

        bull_raw = ray.get(
            self._handle.generate.remote(
                BULL_PROMPT.format(context=context), self._max_new_tokens
            )
        )
        bear_raw = ray.get(
            self._handle.generate.remote(
                BEAR_PROMPT.format(context=context), self._max_new_tokens
            )
        )
        bull = parse_stance(bull_raw["text"])
        bear = parse_stance(bear_raw["text"])
        return self._agent.arbitrate(bull, bear, technical)
