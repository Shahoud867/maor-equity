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
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..agents.guardrail import GuardrailAgent
from ..agents.sentiment import DimensionRouter, SentimentBundle, build_matrix
from ..agents.summarisation import MapReduceSummariser, SummarisationModel
from ..data.chunking import ChunkFilter, chunk_document
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
    ) -> None:
        self.config = config
        self.device = device
        self.warm_start = warm_start
        self.filter_enabled = filter_enabled
        self.budget = budget or VRAMBudget(total_mb=0.0)

        self.router = DimensionRouter()
        self.chunk_filter = ChunkFilter(
            sim_threshold=config.chunking.sim_threshold,
            max_chunks_8k=config.chunking.max_chunks_8k,
            max_chunks_10k=config.chunking.max_chunks_10k,
            tfidf_max_features=config.chunking.tfidf_max_features,
        )

        self._sentiment: SentimentBundle | None = None
        self._summariser: SummarisationModel | None = None

    # -- model lifecycle -------------------------------------------------

    def _get_sentiment(self) -> SentimentBundle:
        if self._sentiment is not None:
            return self._sentiment
        bundle = SentimentBundle(
            market_checkpoint=self.config.models.sentiment_market,
            regulatory_checkpoint=self.config.models.sentiment_regulatory,
            device=self.device,
            quantisation=self.config.models.sentiment_quantisation,
            batch_size=self.config.models.sentiment_batch_size,
            max_length=self.config.models.sentiment_max_length,
        ).load()
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
        ).load()
        if self.warm_start:
            self._summariser = model
        return model

    def close(self) -> None:
        if self._sentiment is not None:
            self._sentiment.unload()
            self._sentiment = None
        if self._summariser is not None:
            self._summariser.unload()
            self._summariser = None
        release_vram()

    # -- execution -------------------------------------------------------

    def run(
        self,
        *,
        ticker: str,
        document: str,
        technical: dict[str, Any] | None = None,
        filing_type: str = "8-K",
    ) -> PipelineOutput:
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
        with self.budget.phase(
            f"sentiment[{ticker}]", self.config.models.sentiment_estimated_vram_mb
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
        with self.budget.phase(
            f"summariser[{ticker}]", self.config.models.summarizer_estimated_vram_mb
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
