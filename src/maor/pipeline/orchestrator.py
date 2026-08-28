"""Pipeline execution for both the serial and distributed arms.

One code path serves both, parameterised by :class:`~maor.evaluation.h1_latency.Condition`.
That is deliberate: in the original project the serial baseline was a separate
script that drifted from the pipeline it was supposed to be compared against
(different ingestion, different chunk handling, cold model loads). A comparison
between two different programs measures the programs, not the architecture.

Phase serialisation is enforced here rather than assumed. Phase A loads the
sentiment models inside a VRAM reservation and releases before Phase B loads the
summariser, so peak residency is ``max(A, B)`` rather than ``A + B``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..agents.guardrail import GuardrailAgent
from ..agents.sentiment import DimensionRouter, SentimentBundle, build_matrix
from ..agents.summarisation import MapReduceSummariser, SummarisationModel
from ..data.chunking import ChunkFilter, chunk_document
from ..execution.timeouts import run_with_timeout
from ..hardware import VRAMBudget, release_vram
from .instrumentation import StageRecorder

log = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    ticker: str
    recommendation: dict[str, Any]
    summary: str
    sentiment: dict[str, Any]
    technical: dict[str, Any]
    chunking: dict[str, Any]
    recorder: StageRecorder
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
            "warnings": self.warnings,
        }


class Pipeline:
    """Runs the five-agent DAG under an explicit VRAM budget.

    ``warm_start=True`` keeps models resident across documents. ``False``
    reloads them per document, which is what the original B1 baseline did on
    every run and what made its comparison unfair.
    """

    def __init__(
        self,
        config: Any,
        *,
        budget: VRAMBudget | None = None,
        device: str = "cpu",
        warm_start: bool = True,
        filter_enabled: bool = True,
        distributed: bool = False,
    ) -> None:
        self.config = config
        self.device = device
        self.warm_start = warm_start
        self.filter_enabled = filter_enabled
        self.distributed = distributed
        self.budget = budget or VRAMBudget(total_mb=0.0)

        # Refuse rather than silently run single-node work under a "distributed"
        # label. A factor that does not change execution produces a contrast of
        # approximately zero, which reads as "distribution does not help" when in
        # fact nothing distributed was ever run — a fabricated null result. This
        # is what happened to the original H1: Condition.distributed was never
        # read by anything.
        if distributed and config.execution.mode != "ray":
            raise NotImplementedError(
                "distributed=True requires execution.mode='ray'.\n"
                "Either connect a Ray cluster and set execution.mode=ray "
                "(see docs/GPU_RUNBOOK.md), or run H1 with --ablation local, "
                "which crosses the factors that are implemented single-node "
                "(chunk filter x warm start) and reports the distribution "
                "factor as not measured."
            )

        self._distributed: Any | None = None
        if distributed:
            # Delegate entirely to the Ray-backed implementation. The single-
            # node model/router/filter setup below is skipped: distributed
            # execution owns its own actors, placed by the cluster's actual
            # topology (maor.pipeline.distributed.describe_cluster), not by
            # this process's device string.
            from .distributed import DistributedPipeline

            self._distributed = DistributedPipeline(
                config, ray_address=config.execution.ray_address
            )
            return

        self.router = DimensionRouter()
        self.chunk_filter = ChunkFilter(
            sim_threshold=config.chunking.sim_threshold,
            max_chunks_8k=config.chunking.max_chunks_8k,
            max_chunks_10k=config.chunking.max_chunks_10k,
            tfidf_max_features=config.chunking.tfidf_max_features,
        )

        self._sentiment: SentimentBundle | None = None
        self._summariser: SummarisationModel | None = None
        # Reservations held for models that outlive their phase (warm start).
        self._held_reservations: set[str] = set()

    # -- model lifecycle -------------------------------------------------

    def _get_sentiment(self) -> SentimentBundle:
        if self._sentiment is not None:
            return self._sentiment
        bundle = SentimentBundle(
            market_checkpoint=self.config.models.sentiment_market,
            regulatory_checkpoint=self.config.models.sentiment_regulatory,
            temporal_checkpoint=self.config.models.sentiment_temporal,
            device=self.device,
            quantisation=self.config.models.sentiment_quantisation,
            batch_size=self.config.models.sentiment_batch_size,
            max_length=self.config.models.sentiment_max_length,
        )
        run_with_timeout(
            bundle.load,
            self.config.execution.model_load_timeout_s,
            label="sentiment model load",
        )
        if self.warm_start:
            self._sentiment = bundle
        return bundle

    def _get_summariser(self) -> SummarisationModel:
        if self._summariser is not None:
            return self._summariser
        model = SummarisationModel(
            checkpoint=self.config.models.summarizer,
            device=self.device,
            quantisation=self.config.models.summarizer_quantisation,
            max_input_tokens=self.config.models.max_input_tokens,
            trust_remote_code=self.config.models.trust_remote_code,
            do_sample=self.config.models.do_sample,
            temperature=self.config.models.temperature,
            estimated_vram_mb=self.config.models.summarizer_estimated_vram_mb,
        )
        run_with_timeout(
            model.load,
            self.config.execution.model_load_timeout_s,
            label="summariser model load",
        )
        if self.warm_start:
            self._summariser = model
        return model

    @contextmanager
    def _phase(self, label: str, estimated_mb: float, *, keep_resident: bool):
        """Reserve budget for a phase, holding the reservation while resident.

        The distinction that matters: with ``warm_start`` the model outlives the
        phase, so the reservation must outlive it too. Releasing a reservation
        for memory that is still occupied lets the next phase allocate against
        space that does not exist — a budget check that passes and an allocation
        that fails.
        """
        already_held = label in self._held_reservations
        if not already_held:
            self.budget.reserve(label, estimated_mb)
        try:
            yield
        finally:
            if keep_resident:
                self._held_reservations.add(label)
            else:
                self._held_reservations.discard(label)
                self.budget.release(label)
                release_vram()

    def close(self) -> dict[str, Any]:
        """Release every model and every reservation this pipeline holds.

        Safe to call twice, and safe to call after a failure — which is why the
        experiment runner can call it in ``finally`` without needing to know how
        far the run got.
        """
        if self._distributed is not None:
            info = self._distributed.close()
            self._distributed = None
            return {"release_verifications": [info]}

        verifications: list[Any] = []
        for attr in ("_sentiment", "_summariser"):
            model = getattr(self, attr, None)
            if model is None:
                continue
            try:
                verification = model.unload()
                if verification is not None:
                    verifications.append(verification.to_dict())
            except Exception as exc:
                log.warning("error unloading %s: %s", attr, exc)
            finally:
                setattr(self, attr, None)

        # Reservations outlive their phase under warm start, so they are only
        # returned here. Leaving them held would shrink the budget for the next
        # pipeline by the size of models that are no longer resident.
        for label in list(self._held_reservations):
            self.budget.release(label)
        self._held_reservations.clear()

        release_vram()
        return {"release_verifications": verifications}

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    # -- execution -------------------------------------------------------

    def run(
        self,
        *,
        ticker: str,
        document: str,
        technical: dict[str, Any] | None = None,
        filing_type: str = "8-K",
    ) -> PipelineOutput:
        if self._distributed is not None:
            # Distributed execution computes its own technical indicators via
            # the Node A actor; an externally supplied override is not
            # currently threaded through, since H1 does not use one.
            return self._distributed.run(
                ticker=ticker,
                document=document,
                filing_type=filing_type,
                filter_enabled=self.filter_enabled,
            )

        rec = StageRecorder()
        warnings: list[str] = []
        technical = technical or {"rsi": 50.0, "macd_crossover_bullish": False}

        # ---- chunking ----------------------------------------------------
        with rec.stage("chunk_document", kind="compute", parallelisable=False) as meta:
            chunks = chunk_document(
                document,
                window_tokens=self.config.chunking.window_tokens,
                stride_tokens=self.config.chunking.stride_tokens,
            )
            meta["n_chunks"] = len(chunks)

        if self.filter_enabled:
            with rec.stage("chunk_filter", kind="compute", parallelisable=False) as meta:
                filtered = self.chunk_filter.filter(chunks, filing_type=filing_type)
                meta.update({"n_before": filtered.n_before, "n_after": filtered.n_after})
            selected = filtered.chunks
            chunk_info = filtered.to_dict()
        else:
            selected = list(chunks)
            chunk_info = {
                "n_before": len(chunks),
                "n_after": len(chunks),
                "reduction_pct": 0.0,
                "elapsed_ms": 0.0,
                "method": "disabled",
            }

        if not selected:
            raise ValueError(f"{ticker}: document produced no chunks")

        # ---- Phase A: sentiment -----------------------------------------
        # Reserved and released before Phase B loads, so peak is max(A, B).
        with self._phase(
            "sentiment",
            self.config.models.sentiment_estimated_vram_mb,
            keep_resident=self.warm_start,
        ):
            with rec.stage("sentiment_load", kind="model_load"):
                bundle = self._get_sentiment()
            with rec.stage("route_dimensions", kind="compute", parallelisable=False) as meta:
                routed = self.router.route(selected)
                meta.update(routed["coverage"])
            with rec.stage("sentiment_classify", kind="compute", parallelisable=True) as meta:
                results = bundle.classify_all(routed["routed"])
                meta["n_distinct_checkpoints"] = bundle.n_distinct_checkpoints
            matrix = build_matrix(results)
            if matrix.n_present < len(matrix.dimension_order):
                absent = [
                    d for d, p in zip(matrix.dimension_order, matrix.present) if not p
                ]
                warnings.append(
                    f"dimensions absent from this document (not neutral): {absent}"
                )
            if not self.warm_start:
                bundle.unload()

        # ---- Phase B: summarisation + guardrail --------------------------
        with self._phase(
            "summariser",
            self.config.models.summarizer_estimated_vram_mb,
            keep_resident=self.warm_start,
        ):
            with rec.stage("summariser_load", kind="model_load"):
                model = self._get_summariser()

            summariser = MapReduceSummariser(
                model,
                map_max_new_tokens=self.config.models.map_max_new_tokens,
                reduce_max_new_tokens=self.config.models.reduce_max_new_tokens,
            )
            with rec.stage("summarise_map_reduce", kind="compute", parallelisable=False) as meta:
                summary = summariser.summarise([c.text for c in selected])
                meta.update(summary.to_dict())
            if summary.n_scaffolding_trimmed:
                warnings.append(
                    f"{summary.n_scaffolding_trimmed} generation(s) contained "
                    f"instruction scaffolding that was trimmed"
                )

            with rec.stage("guardrail", kind="compute", parallelisable=False) as meta:
                guardrail = GuardrailAgent(
                    model, max_new_tokens=self.config.models.guardrail_max_new_tokens
                )
                verdict = guardrail.assess(summary.summary, matrix, technical)
                meta["recommendation"] = verdict["recommendation"]
            if verdict["recommendation"] == "ASSESSMENT_FAILED":
                warnings.append(
                    "guardrail could not parse one or both stances; no "
                    "recommendation was produced"
                )

            if not self.warm_start:
                model.unload()
                self._summariser = None

        return PipelineOutput(
            ticker=ticker,
            recommendation=verdict,
            summary=summary.summary,
            sentiment={
                "matrix": matrix.to_dict(),
                "dimensions": {k: v.to_dict() for k, v in results.items()},
                "routing_coverage": routed["coverage"],
            },
            technical=technical,
            chunking=chunk_info,
            recorder=rec,
            warnings=warnings,
        )
