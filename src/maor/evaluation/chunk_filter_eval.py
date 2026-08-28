"""ChunkFilter cost and coverage study.

The paper called ChunkFilter "the single largest latency optimization" and "a
>4,000x return on computation", on the basis of an assumed 80 ms cost and an
assumed 15 s per discarded chunk. Neither was measured: the observed filter cost
in ``results/aapl.json`` was 12,341 ms, 154x the stated figure.

This study measures what can be measured without a GPU:

* the real wall-clock cost of filtering, as a function of document size;
* how much of the document survives, in vocabulary and position terms;
* how reduction and coverage trade off against the chunk cap.

What it deliberately does *not* do is convert chunk counts into saved GPU
seconds. That conversion needs a measured per-chunk generation cost from the GPU
node, and multiplying by an assumed 15 s is how the original overstatement
happened. The runner emits the coefficients so the arithmetic can be completed
once a real per-chunk cost exists.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

from ..data.chunking import ChunkFilter, chunk_document
from ..data.datasets import load_ectsum
from . import stats

log = logging.getLogger(__name__)

# Chunk caps to sweep. The pipeline default is 12 for 8-K and 20 for 10-K/10-Q.
DEFAULT_CAPS = (4, 8, 12, 16, 20, 32)


def run_chunk_filter_study(
    config: Any,
    *,
    n_documents: int | None = None,
    caps: tuple[int, ...] = DEFAULT_CAPS,
) -> dict[str, Any]:
    """Measure filter cost and coverage over real financial documents.

    Uses ECTSum earnings-call transcripts: real, long, already in the repository,
    and large enough that filtering actually engages.
    """
    data = load_ectsum(
        config.data.ectsum_path,
        n_samples=n_documents,
        seed=config.execution.seed,
    )
    log.info("chunk-filter study over %d documents", len(data))

    chunker_cfg = config.chunking
    per_cap: dict[int, list[dict[str, Any]]] = {cap: [] for cap in caps}
    doc_stats: list[dict[str, Any]] = []

    for doc_i, document in enumerate(data.documents):
        chunks = chunk_document(
            document,
            window_tokens=chunker_cfg.window_tokens,
            stride_tokens=chunker_cfg.stride_tokens,
        )
        doc_stats.append(
            {
                "document_index": doc_i,
                "n_words": len(document.split()),
                "n_chunks_raw": len(chunks),
            }
        )
        if not chunks:
            continue

        for cap in caps:
            filt = ChunkFilter(
                sim_threshold=chunker_cfg.sim_threshold,
                max_chunks_8k=cap,
                max_chunks_10k=cap,
                tfidf_max_features=chunker_cfg.tfidf_max_features,
            )
            result = filt.filter(chunks, filing_type="8-K", max_chunks=cap)
            per_cap[cap].append(
                {
                    "document_index": doc_i,
                    "n_before": result.n_before,
                    "n_after": result.n_after,
                    "reduction_pct": result.reduction_pct,
                    "elapsed_ms": result.elapsed_ms,
                    **result.coverage,
                }
            )

    # ---- aggregate per cap ------------------------------------------------
    curve: list[dict[str, Any]] = []
    for cap in caps:
        rows = per_cap[cap]
        if not rows:
            continue
        elapsed = [r["elapsed_ms"] for r in rows]
        reduction = [r["reduction_pct"] for r in rows]
        vocab = [r["vocabulary_retained_pct"] for r in rows if r["vocabulary_retained_pct"] is not None]
        positions = [
            r["document_positions_retained_pct"]
            for r in rows
            if r["document_positions_retained_pct"] is not None
        ]
        curve.append(
            {
                "max_chunks": cap,
                "n_documents": len(rows),
                "mean_chunks_after": round(statistics.mean(r["n_after"] for r in rows), 2),
                "reduction_pct": stats.summarise(reduction),
                "filter_cost_ms": stats.summarise(elapsed),
                "filter_cost_ms_ci": stats.bootstrap_ci(
                    elapsed, statistic=statistics.median, n_resamples=2000
                ).to_dict(),
                "vocabulary_retained_pct": stats.summarise(vocab),
                "document_positions_retained_pct": stats.summarise(positions),
            }
        )

    raw_counts = [d["n_chunks_raw"] for d in doc_stats if d["n_chunks_raw"] > 0]
    all_elapsed = [r["elapsed_ms"] for cap in caps for r in per_cap[cap]]

    summary = {
        "n_documents": len(data),
        "chunks_per_document": stats.summarise(raw_counts),
        "words_per_document": stats.summarise([d["n_words"] for d in doc_stats]),
        "filter_cost_ms_overall": stats.summarise(all_elapsed),
        "default_cap_row": next((c for c in curve if c["max_chunks"] == 12), None),
    }

    return {
        "experiment": "chunk_filter_cost_and_coverage",
        "dataset": data.source,
        "chunking": {
            "window_tokens": chunker_cfg.window_tokens,
            "stride_tokens": chunker_cfg.stride_tokens,
            "overlap_pct": round(
                100 * (chunker_cfg.window_tokens - chunker_cfg.stride_tokens)
                / chunker_cfg.window_tokens,
                1,
            ),
            "sim_threshold": chunker_cfg.sim_threshold,
            "tokenizer": "whitespace",
            "tokenizer_note": (
                "Whitespace tokenisation approximates subword boundaries. Token "
                "counts are therefore approximate; chunk counts and relative "
                "comparisons across caps are unaffected."
            ),
        },
        "cost_coverage_curve": curve,
        "summary": summary,
        "gpu_saving_arithmetic": {
            "status": "NOT COMPUTED — requires a measured per-chunk generation cost",
            "explanation": (
                "Saved GPU time = (chunks_removed) x (seconds per chunk). The "
                "second factor must come from a measured Phi-3-mini generation "
                "cost on the target GPU. The original analysis assumed 15 s/chunk "
                "and reported 690 s and 1,575 s saved as if measured."
            ),
            "coefficients_available": {
                "mean_chunks_removed_at_cap_12": next(
                    (
                        round(
                            statistics.mean(r["n_before"] - r["n_after"] for r in per_cap[12]),
                            2,
                        )
                        for _ in [0]
                        if per_cap.get(12)
                    ),
                    None,
                ),
            },
            "how_to_complete": (
                "Run `maor.cli h1-latency` on the GPU node to obtain measured "
                "seconds-per-chunk, then multiply."
            ),
        },
        "interpretation": {
            "filter_cost": (
                "Measured wall-clock cost of the filter itself, to replace the "
                "unmeasured '~80 ms' figure in the paper."
            ),
            "coverage": (
                "Vocabulary and document-position retention quantify what the "
                "filter discards. High retention supports the claim that removed "
                "chunks were near-duplicates created by the 87.5% window overlap; "
                "low retention would mean the filter is dropping content and the "
                "H2 quality comparison is confounded."
            ),
        },
    }
