"""H2: is chunked map-reduce summarisation non-inferior to single-pass?

Replaces the previous H2 entirely. Those numbers were literals in
``quantitative/h2_rouge_generator.py`` (``b2_rouge_1 = 0.28``; the map-reduce
scores were the baseline plus a hand-chosen delta). No ECTSum document was ever
scored, and the published B2 row had ROUGE-L 0.32 above ROUGE-1 0.28, which no
scorer can produce.

What this runs: both systems over the same ECTSum documents, scored with
``rouge_score`` on the 0-100 scale, with paired statistics because both arms see
identical inputs.

The non-inferiority test is stated correctly here. Non-inferiority means the
*lower* bound of the CI on (ours - baseline) sits above the margin, not that the
point estimate is close to zero. The original compared a 0-1 scale delta against
a tolerance of 1.0, which every possible result satisfies.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from ..agents.summarisation import (
    MapReduceSummariser,
    SummarisationModel,
    single_pass_summarise,
)
from ..data.chunking import ChunkFilter, chunk_document
from . import stats
from .metrics import (
    HypothesisTest,
    RougeScores,
    compute_rouge,
    compute_rouge_per_sample,
)

log = logging.getLogger(__name__)

# Non-inferiority margin in ROUGE-L points on the 0-100 scale. Chosen before
# running, matching the project's original stated intent of "within 1.0 point".
NON_INFERIORITY_MARGIN = -1.0


@dataclass
class H2Result:
    payload: dict[str, Any]
    tests: list[HypothesisTest]
    map_reduce_predictions: list[str]
    baseline_predictions: list[str]


def run_h2(
    *,
    model: SummarisationModel,
    documents: Sequence[str],
    references: Sequence[str],
    dataset_source: dict[str, Any],
    config: Any,
    compute_bertscore_metric: bool = True,
    seed: int = 0,
    n_resamples: int = 10_000,
) -> H2Result:
    """Run map-reduce and single-pass summarisation over the same documents."""
    if len(documents) != len(references):
        raise ValueError("documents and references must be the same length")
    if not documents:
        raise ValueError("cannot run H2 over zero documents")

    chunker = config.chunking
    chunk_filter = ChunkFilter(
        sim_threshold=chunker.sim_threshold,
        max_chunks_8k=chunker.max_chunks_8k,
        max_chunks_10k=chunker.max_chunks_10k,
        tfidf_max_features=chunker.tfidf_max_features,
    )
    summariser = MapReduceSummariser(
        model,
        map_max_new_tokens=config.models.map_max_new_tokens,
        reduce_max_new_tokens=config.models.reduce_max_new_tokens,
    )

    map_reduce_preds: list[str] = []
    baseline_preds: list[str] = []
    per_doc: list[dict[str, Any]] = []

    for i, document in enumerate(documents):
        log.info("H2: document %d/%d", i + 1, len(documents))

        chunks = chunk_document(
            document,
            window_tokens=chunker.window_tokens,
            stride_tokens=chunker.stride_tokens,
        )
        filtered = chunk_filter.filter(chunks, filing_type="8-K")

        mr = summariser.summarise([c.text for c in filtered.chunks])
        map_reduce_preds.append(mr.summary)

        b2 = single_pass_summarise(
            model, document, max_new_tokens=config.models.reduce_max_new_tokens
        )
        baseline_preds.append(b2.summary)

        per_doc.append(
            {
                "document_index": i,
                "n_chunks_raw": filtered.n_before,
                "n_chunks_kept": filtered.n_after,
                "chunk_filter_ms": round(filtered.elapsed_ms, 2),
                "coverage": filtered.coverage,
                "map_reduce": mr.to_dict(),
                "baseline_truncated_input": any(
                    s.get("truncated_input") for s in b2.generation_stats
                ),
                "baseline_scaffolding_trimmed": b2.n_scaffolding_trimmed,
            }
        )

    # ---- scoring ----------------------------------------------------------
    mr_rouge: RougeScores = compute_rouge(map_reduce_preds, references)
    b2_rouge: RougeScores = compute_rouge(baseline_preds, references)

    mr_per = compute_rouge_per_sample(map_reduce_preds, references)
    b2_per = compute_rouge_per_sample(baseline_preds, references)
    mr_rl = [d["rouge_l"] for d in mr_per]
    b2_rl = [d["rouge_l"] for d in b2_per]

    delta_ci = stats.paired_bootstrap_delta(
        mr_rl, b2_rl, statistic=statistics.mean, n_resamples=n_resamples, seed=seed
    )
    perm = stats.permutation_test(mr_rl, b2_rl, n_permutations=n_resamples, seed=seed)

    bertscore: dict[str, Any] = {"status": "not computed"}
    if compute_bertscore_metric:
        try:
            from .metrics import compute_bertscore

            mr_bs = compute_bertscore(map_reduce_preds, references)
            b2_bs = compute_bertscore(baseline_preds, references)
            bertscore = {
                "map_reduce": mr_bs,
                "baseline": b2_bs,
                "delta": round(
                    mr_bs["bertscore_f1"] - b2_bs["bertscore_f1"], 4
                ),
            }
        except ImportError:
            bertscore = {
                "status": "skipped — bert_score not installed",
                "remedy": "pip install bert-score",
            }

    # ---- hypothesis test --------------------------------------------------
    # Non-inferiority: the LOWER bound of the CI must clear the margin.
    non_inferior = delta_ci.lower >= NON_INFERIORITY_MARGIN
    h2 = HypothesisTest(
        hypothesis_id="H2",
        claim=(
            "Chunked map-reduce summarisation is non-inferior to single-pass "
            f"truncation on ROUGE-L (margin {NON_INFERIORITY_MARGIN} points)"
        ),
        metric_name="rouge_l_delta_lower_ci_bound",
        scale="0-100",
        units="ROUGE-L points",
        observed=round(delta_ci.lower, 4),
        threshold=NON_INFERIORITY_MARGIN,
        comparison=">=",
        falsifiable_range=(-100.0, 100.0),
        notes=(
            f"Point estimate {delta_ci.point:+.3f}, 95% CI "
            f"[{delta_ci.lower:+.3f}, {delta_ci.upper:+.3f}], permutation "
            f"p={perm['p_value']:.4f}. Non-inferiority is decided by the lower "
            f"bound, not the point estimate: a point estimate near zero with a "
            f"CI extending below the margin does not establish non-inferiority."
        ),
    )

    truncation_count = sum(1 for d in per_doc if d["baseline_truncated_input"])
    scaffolding_count = sum(
        d["map_reduce"]["n_scaffolding_trimmed"] for d in per_doc
    ) + sum(d["baseline_scaffolding_trimmed"] for d in per_doc)

    payload = {
        "experiment": "H2_summarisation_quality",
        "dataset": dataset_source,
        "n_documents": len(documents),
        "systems": {
            "map_reduce": {
                "description": "ChunkFilter -> per-chunk map -> recursive reduce",
                "rouge": mr_rouge.to_dict(),
            },
            "baseline_b2_single_pass": {
                "description": (
                    f"one generation over the document truncated to "
                    f"{config.models.max_input_tokens} tokens"
                ),
                "rouge": b2_rouge.to_dict(),
                "n_documents_truncated": truncation_count,
                "truncation_note": (
                    "Truncation is the baseline's defining limitation; the count "
                    "says how often it actually bound."
                ),
            },
        },
        "rouge_l_delta": {
            "point": round(delta_ci.point, 4),
            "ci_lower": round(delta_ci.lower, 4),
            "ci_upper": round(delta_ci.upper, 4),
            "confidence": delta_ci.confidence,
            "method": delta_ci.method,
            "permutation_test": perm,
        },
        "bertscore": bertscore,
        "generation_quality": {
            "n_generations_with_scaffolding_trimmed": scaffolding_count,
            "note": (
                "Instruction-tuning scaffolding leaking into output was visible in "
                "the previously published AAPL summary. A non-zero count here means "
                "prompts still need work even though the text is cleaned."
            ),
        },
        "per_document": per_doc,
        "statistics": {
            "sample_size_interpretation": stats.interpret_sample_size(len(documents)),
            "map_reduce_rouge_l": stats.summarise(mr_rl),
            "baseline_rouge_l": stats.summarise(b2_rl),
        },
        "hypothesis_tests": [h2.to_dict()],
        "validity_threats": [
            {
                "severity": "major",
                "threat": "domain transfer",
                "detail": (
                    "ECTSum is earnings-call transcripts; the deployed pipeline "
                    "summarises SEC 8-K filings. Quality on one is not evidence of "
                    "quality on the other."
                ),
                "remedy": "Report as a benchmark result, not a deployment estimate.",
            },
            {
                "severity": "major",
                "threat": "no published baseline reproduced",
                "detail": (
                    "Neither arm is a published ECTSum system, so these numbers "
                    "cannot be positioned against the state of the art without "
                    "also running one."
                ),
                "remedy": "Reproduce or cite a published ECTSum baseline for context.",
            },
        ],
    }

    return H2Result(
        payload=payload,
        tests=[h2],
        map_reduce_predictions=map_reduce_preds,
        baseline_predictions=baseline_preds,
    )
