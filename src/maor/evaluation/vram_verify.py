"""VRAM verification against the configured budget.

Audit finding M4: the published VRAM trace (peak 3,261 MB) was written on
2026-04-21 01:14, two days *before* FinBERT was switched to 4-bit NF4 on
2026-04-23. Figure 4 and the headline VRAM claim therefore describe a build that
no longer exists, and the paper's text says FP16 where the code says NF4.

This module re-measures on whatever code is actually present, and records the
resolved model configuration next to the measurement so the two cannot drift
apart again.

It measures the sequence the pipeline actually executes:

    baseline -> load sentiment -> score -> release -> load summariser -> generate -> release

That ordering is the phase-serialisation mechanism. The interesting number is not
just the peak but whether the release between phases actually returns memory:
if it does not, the 4 GB budget is unachievable regardless of what the models
nominally weigh.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import hardware

log = logging.getLogger(__name__)


def _measure(device: int) -> dict[str, float]:
    return {
        "allocated_mb": round(hardware.current_vram_mb(device), 1),
        "peak_mb": round(hardware.peak_vram_mb(device), 1),
    }


def run_vram_verification(config: Any, *, dry_run: bool = False) -> dict[str, Any]:
    """Measure real VRAM at each pipeline phase. Requires CUDA."""
    hw = hardware.probe()
    if not hw.has_cuda:
        raise RuntimeError(
            "VRAM verification requires a CUDA device. "
            f"Detected: {hw.describe()}"
        )

    device = config.vram.device
    budget = hardware.VRAMBudget.from_hardware(
        hw,
        usable_fraction=config.vram.usable_fraction,
        override_total_mb=config.vram.total_mb,
        device=device,
    )

    trace: list[dict[str, Any]] = []

    def record(stage: str, **extra: Any) -> None:
        entry = {"stage": stage, **_measure(device), **extra}
        trace.append(entry)
        log.info(
            "vram[%s]: allocated=%.0f MB peak=%.0f MB",
            stage,
            entry["allocated_mb"],
            entry["peak_mb"],
        )

    if dry_run:
        return {
            "experiment": "vram_verification",
            "dry_run": True,
            "budget": budget.snapshot(),
            "planned_stages": [
                "baseline",
                "sentiment_loaded",
                "sentiment_scored",
                "sentiment_released",
                "summariser_loaded",
                "summariser_generated",
                "summariser_released",
            ],
            "note": "Wiring verified; no weights were loaded.",
        }

    hardware.reset_peak_vram(device)
    record("baseline")

    # ---- Phase A: sentiment ------------------------------------------------
    from ..agents.sentiment import DimensionRouter, SentimentBundle

    probe_texts = [
        "The company reported record quarterly revenue and raised full-year guidance.",
        "The SEC has opened an enforcement investigation into the company's accounting.",
        "Management expects margin pressure to continue into the next fiscal year.",
    ] * 8

    bundle = SentimentBundle(
        market_checkpoint=config.models.sentiment_market,
        regulatory_checkpoint=config.models.sentiment_regulatory,
        temporal_checkpoint=config.models.sentiment_temporal,
        device="cuda",
        quantisation=config.models.sentiment_quantisation,
        batch_size=config.models.sentiment_batch_size,
        max_length=config.models.sentiment_max_length,
    )
    with budget.phase("sentiment", config.models.sentiment_estimated_vram_mb):
        bundle.load()
        record(
            "sentiment_loaded",
            n_distinct_checkpoints=bundle.n_distinct_checkpoints,
            quantisation=config.models.sentiment_quantisation,
        )
        routed = DimensionRouter().route(probe_texts)["routed"]
        bundle.classify_all(routed)
        record("sentiment_scored")
        bundle.unload()
    record("sentiment_released")

    sentiment_resident = trace[1]["allocated_mb"] - trace[0]["allocated_mb"]
    released_back_to = trace[3]["allocated_mb"]

    # ---- Phase B: summariser ----------------------------------------------
    from ..agents.summarisation import SummarisationModel

    summariser = SummarisationModel(
        checkpoint=config.models.summarizer,
        device="cuda",
        quantisation=config.models.summarizer_quantisation,
        max_input_tokens=config.models.max_input_tokens,
        trust_remote_code=config.models.trust_remote_code,
    )
    with budget.phase("summariser", config.models.summarizer_estimated_vram_mb):
        summariser.load()
        record("summariser_loaded", quantisation=config.models.summarizer_quantisation)
        summariser.generate(
            "Summarise the following in one sentence: the company grew revenue 20%.",
            max_new_tokens=32,
        )
        record("summariser_generated")
        summariser.unload()
    record("summariser_released")

    summariser_resident = trace[5]["allocated_mb"] - trace[4]["allocated_mb"]
    peak = max(t["peak_mb"] for t in trace)

    within_budget = peak <= budget.usable_mb
    would_coreside = sentiment_resident + summariser_resident

    return {
        "experiment": "vram_verification",
        "budget": budget.snapshot(),
        "trace": trace,
        "measurements": {
            "peak_mb": round(peak, 1),
            "budget_usable_mb": round(budget.usable_mb, 1),
            "budget_total_mb": round(budget.total_mb, 1),
            "headroom_mb": round(budget.usable_mb - peak, 1),
            "within_budget": within_budget,
            "sentiment_resident_mb": round(sentiment_resident, 1),
            "summariser_resident_mb": round(summariser_resident, 1),
            "sum_if_coresident_mb": round(would_coreside, 1),
            "coresidence_would_fit": would_coreside <= budget.usable_mb,
            "memory_returned_after_phase_a_mb": round(
                trace[1]["allocated_mb"] - released_back_to, 1
            ),
        },
        "declared_vs_measured": {
            "sentiment_declared_mb": config.models.sentiment_estimated_vram_mb,
            "sentiment_measured_mb": round(sentiment_resident, 1),
            "sentiment_delta_mb": round(
                sentiment_resident - config.models.sentiment_estimated_vram_mb, 1
            ),
            "summariser_declared_mb": config.models.summarizer_estimated_vram_mb,
            "summariser_measured_mb": round(summariser_resident, 1),
            "summariser_delta_mb": round(
                summariser_resident - config.models.summarizer_estimated_vram_mb, 1
            ),
            "note": (
                "Declared values come from configs/*.yaml and are what the budget "
                "reserves. A large delta means the config should be corrected — "
                "not the measurement."
            ),
        },
        "resolved_model_configuration": {
            "sentiment_market": config.models.sentiment_market,
            "sentiment_regulatory": config.models.sentiment_regulatory,
            "sentiment_quantisation": config.models.sentiment_quantisation,
            "summarizer": config.models.summarizer,
            "summarizer_quantisation": config.models.summarizer_quantisation,
            "phase_serialised": config.vram.phase_serialised,
            "note": (
                "Recorded alongside the measurement so the paper cannot describe a "
                "different configuration than the one measured."
            ),
        },
        "interpretation": (
            "peak_mb is the maximum allocation observed across the phase-serialised "
            "sequence. coresidence_would_fit answers whether phase serialisation is "
            "actually required on this hardware, which is the design claim."
        ),
    }
