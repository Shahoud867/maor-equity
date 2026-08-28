"""H3: does multi-dimensional sentiment change the recommendation, and is it better?

This replaces the previous H3 entirely. The original sampled categorical labels
from three hardcoded distributions with ``random.seed(i)`` and compared two
hand-written ``if`` statements; it never loaded a model, never read a sentence,
and its 48% "divergence" was an arithmetic property of the two rules. Because the
rules differ by construction, the hypothesis could not fail.

What runs here instead: real FinBERT checkpoints over real Financial PhraseBank
sentences with real gold labels, comparing a scalar baseline against the
multi-dimensional system, with the decision rules pre-registered below.

The design adds a second question the original never asked. Divergence alone is
not a virtue — a rule that flips answers at random diverges constantly and helps
nobody. So the primary outcome is **accuracy against gold labels**, and
divergence is reported as a descriptive statistic alongside it. If the
multi-dimensional system diverges often but is not more accurate, that is the
finding, and it is reported as such.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from ..agents.sentiment import LABELS, DimensionRouter, SentimentBundle
from . import stats
from .metrics import (
    HypothesisTest,
    accuracy,
    confusion_matrix,
    disagreement_rate,
    macro_f1,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Pre-registered decision rules.
#
# Written down before the run, in code, so the mapping cannot be adjusted after
# seeing results. Both systems map onto the same three directions so that
# accuracy against the gold sentiment label is directly comparable.
# --------------------------------------------------------------------------

DIRECTION_FROM_SENTIMENT = {
    "positive": "BUY",
    "neutral": "HOLD",
    "negative": "SELL",
}


def b3_direction(market_label: str) -> str:
    """B3 baseline: scalar market sentiment maps straight to a direction."""
    return DIRECTION_FROM_SENTIMENT[market_label]


def multidimensional_direction(
    market_label: str,
    regulatory_label: str | None,
    temporal_label: str | None,
) -> str:
    """The 3-D rule, with the regulatory veto the project claims as its mechanism.

    ``None`` means the dimension was not present for this text — not neutral.
    That distinction is the M2 fix: previously an absent dimension was scored
    from a placeholder sentence and came back confidently neutral.

    Rules, in order:
      1. Regulatory veto: negative regulatory signal caps upside at HOLD.
      2. Regulatory escalation: negative regulatory with non-positive market
         resolves to SELL.
      3. Temporal offset: negative market with positive forward-looking signal
         softens to HOLD.
      4. Otherwise fall through to the market signal.
    """
    if regulatory_label == "negative":
        if market_label == "positive":
            return "HOLD"  # rule 1
        return "SELL"  # rule 2

    if market_label == "negative" and temporal_label == "positive":
        return "HOLD"  # rule 3

    return DIRECTION_FROM_SENTIMENT[market_label]  # rule 4


@dataclass
class H3Result:
    payload: dict[str, Any]
    tests: list[HypothesisTest]


# Checkpoints whose published training data includes Financial PhraseBank.
# Evaluating them on PhraseBank measures memorisation, not generalisation.
PHRASEBANK_TRAINED_CHECKPOINTS = ("ProsusAI/finbert",)


def _validity_threats(
    *,
    market_checkpoint: str,
    dataset_source: dict[str, Any],
    n_regulatory: int,
    n_temporal: int,
    n_total: int,
) -> list[dict[str, str]]:
    """Threats that must travel with this result into the paper."""
    threats: list[dict[str, str]] = []

    dataset_name = str(dataset_source.get("dataset", "")).lower()
    if "phrasebank" in dataset_name and market_checkpoint in PHRASEBANK_TRAINED_CHECKPOINTS:
        threats.append(
            {
                "severity": "critical",
                "threat": "train-on-test contamination",
                "detail": (
                    f"{market_checkpoint} is fine-tuned on Financial PhraseBank "
                    f"(Araci 2019; see the model card). Evaluating it on PhraseBank "
                    f"measures memorisation of the training set, not generalisation. "
                    f"Absolute accuracy figures from this run are NOT valid estimates "
                    f"of field performance and must not be reported as such."
                ),
                "what_remains_valid": (
                    "The B3-vs-multidimensional comparison is a within-run contrast: "
                    "both arms use the same contaminated market model, so the "
                    "divergence rate and the accuracy delta between arms are still "
                    "interpretable as the effect of adding dimensions. The absolute "
                    "levels are not."
                ),
                "remedy": (
                    "For a generalisation claim, evaluate on a corpus disjoint from "
                    "PhraseBank — e.g. FiQA sentiment, SemEval-2017 Task 5, or "
                    "hand-labelled 8-K excerpts — or use a market checkpoint not "
                    "trained on PhraseBank."
                ),
            }
        )

    if n_regulatory == 0:
        threats.append(
            {
                "severity": "critical",
                "threat": "regulatory dimension never activated",
                "detail": (
                    f"The lexical router matched 0 of {n_total} texts as regulatory, "
                    f"so the regulatory veto — the mechanism the multi-dimensional "
                    f"design exists to provide — was never exercised. Any null result "
                    f"here is a statement about the router and the corpus, not about "
                    f"whether regulatory signal would help."
                ),
                "what_remains_valid": "Nothing about the regulatory dimension's value.",
                "remedy": (
                    "Evaluate on a corpus that contains regulatory language (SEC "
                    "filings, litigation disclosures) rather than short news "
                    "sentences, and report router coverage alongside the result."
                ),
            }
        )
    elif n_regulatory / max(n_total, 1) < 0.02:
        threats.append(
            {
                "severity": "major",
                "threat": "regulatory dimension rarely activated",
                "detail": (
                    f"Only {n_regulatory}/{n_total} texts "
                    f"({100 * n_regulatory / n_total:.1f}%) matched the regulatory "
                    f"router, so the veto rule affects too few cases to support a "
                    f"claim in either direction."
                ),
                "what_remains_valid": "The overall accuracy comparison, with wide uncertainty.",
                "remedy": "Use a corpus with more regulatory content, or stratify and report separately.",
            }
        )

    if n_temporal == 0:
        threats.append(
            {
                "severity": "major",
                "threat": "temporal dimension never activated",
                "detail": (
                    f"The router matched 0 of {n_total} texts as forward-looking, so "
                    f"the temporal offset rule was never exercised."
                ),
                "what_remains_valid": "Nothing about the temporal dimension's value.",
                "remedy": "Evaluate on documents containing guidance/outlook language.",
            }
        )

    threats.append(
        {
            "severity": "major",
            "threat": "sentence-level granularity mismatch",
            "detail": (
                "The deployed pipeline routes 512-token filing chunks; this "
                "evaluation routes single sentences. A sentence rarely carries both "
                "market and regulatory content, which structurally suppresses the "
                "interaction the multi-dimensional rule is designed to capture."
            ),
            "what_remains_valid": (
                "The comparison is valid at sentence granularity, which is the only "
                "granularity at which gold sentiment labels exist for this corpus."
            ),
            "remedy": (
                "Complement with a chunk-level evaluation on annotated filing "
                "excerpts, accepting that gold labels must then be produced."
            ),
        }
    )

    return threats


def _argmax_label(vector: Sequence[float]) -> str:
    return LABELS[max(range(len(vector)), key=lambda i: vector[i])]


def run_h3(
    *,
    bundle: SentimentBundle,
    router: DimensionRouter,
    texts: Sequence[str],
    gold_labels: Sequence[str],
    dataset_source: dict[str, Any],
    seed: int = 0,
    n_resamples: int = 10_000,
) -> H3Result:
    """Execute H3 over real texts. Returns results whether or not they support H3."""
    if len(texts) != len(gold_labels):
        raise ValueError("texts and gold_labels must be the same length")
    if not texts:
        raise ValueError("cannot run H3 over zero samples")

    # ---- Route every sentence, then batch by dimension --------------------
    # Routing is per-sentence: a sentence is always market content, and is also
    # regulatory and/or temporal content if it matches those lexical patterns.
    reg_mask: list[bool] = []
    tmp_mask: list[bool] = []
    for t in texts:
        routed = router.route([t])["routed"]
        reg_mask.append(bool(routed["regulatory"]))
        tmp_mask.append(bool(routed["temporal"]))

    reg_texts = [t for t, m in zip(texts, reg_mask) if m]
    tmp_texts = [t for t, m in zip(texts, tmp_mask) if m]

    log.info(
        "H3 routing: %d sentences, %d regulatory (%.1f%%), %d temporal (%.1f%%)",
        len(texts),
        len(reg_texts),
        100 * len(reg_texts) / len(texts),
        len(tmp_texts),
        100 * len(tmp_texts) / len(texts),
    )

    # ---- Score each dimension --------------------------------------------
    market_res = bundle.classify(list(texts), "market")
    reg_res = bundle.classify(reg_texts, "regulatory")
    tmp_res = bundle.classify(tmp_texts, "temporal")

    market_labels = [_argmax_label([s[l] for l in LABELS]) for s in market_res.scores]

    reg_iter = iter(
        [_argmax_label([s[l] for l in LABELS]) for s in reg_res.scores]
    )
    tmp_iter = iter(
        [_argmax_label([s[l] for l in LABELS]) for s in tmp_res.scores]
    )
    reg_labels: list[str | None] = [next(reg_iter) if m else None for m in reg_mask]
    tmp_labels: list[str | None] = [next(tmp_iter) if m else None for m in tmp_mask]

    # ---- Dimension redundancy check ---------------------------------------
    # The temporal dimension is the market checkpoint applied to a subset of the
    # market inputs. Whenever that subset contains the *identical* text (as it
    # does at sentence granularity), determinism forces temporal == market, and
    # any decision rule conditioning on the two disagreeing is unsatisfiable.
    # This is a structural property, not a property of this corpus, so it is
    # measured and reported rather than inferred.
    shared = tmp_res.shares_checkpoint_with == "market"
    tmp_matches_market = [
        (m, t) for m, t, mask in zip(market_labels, tmp_labels, tmp_mask) if mask
    ]
    n_temporal_identical = sum(1 for m, t in tmp_matches_market if m == t)
    temporal_redundant = (
        bool(tmp_matches_market) and n_temporal_identical == len(tmp_matches_market)
    )

    # ---- Apply the pre-registered decision rules --------------------------
    gold_dirs = [DIRECTION_FROM_SENTIMENT[g] for g in gold_labels]
    b3_dirs = [b3_direction(m) for m in market_labels]
    md_dirs = [
        multidimensional_direction(m, r, t)
        for m, r, t in zip(market_labels, reg_labels, tmp_labels)
    ]

    # ---- Metrics ----------------------------------------------------------
    acc_b3 = accuracy(b3_dirs, gold_dirs)
    acc_md = accuracy(md_dirs, gold_dirs)
    f1_b3 = macro_f1(b3_dirs, gold_dirs)
    f1_md = macro_f1(md_dirs, gold_dirs)
    divergence = disagreement_rate(b3_dirs, md_dirs)

    correct_b3 = [1.0 if p == g else 0.0 for p, g in zip(b3_dirs, gold_dirs)]
    correct_md = [1.0 if p == g else 0.0 for p, g in zip(md_dirs, gold_dirs)]

    acc_delta_ci = stats.paired_bootstrap_delta(
        correct_md, correct_b3, confidence=0.95, n_resamples=n_resamples, seed=seed
    )
    acc_test = stats.permutation_test(
        correct_md, correct_b3, n_permutations=n_resamples, seed=seed
    )

    # Where the two systems disagree, which one is right? This is the question
    # that decides whether divergence is useful or merely noisy.
    diverged = [i for i in range(len(texts)) if b3_dirs[i] != md_dirs[i]]
    md_right = sum(1 for i in diverged if md_dirs[i] == gold_dirs[i])
    b3_right = sum(1 for i in diverged if b3_dirs[i] == gold_dirs[i])
    both_wrong = len(diverged) - md_right - b3_right

    # ---- Hypothesis tests -------------------------------------------------
    # H3a is the original claim, kept for continuity. H3b is the claim that
    # actually matters and that the original never tested.
    h3a = HypothesisTest(
        hypothesis_id="H3a",
        claim=(
            "Multi-dimensional sentiment changes the directional recommendation "
            "in >10% of texts versus the scalar B3 baseline"
        ),
        metric_name="direction_divergence_rate",
        scale="percent",
        units="% of texts",
        observed=divergence,
        threshold=10.0,
        comparison=">",
        falsifiable_range=(0.0, 100.0),
        notes=(
            "Descriptive only. Divergence is not itself evidence of quality: a "
            "rule that flips answers at random would score highly here. H3b is "
            "the decision-relevant test."
        ),
    )
    h3b = HypothesisTest(
        hypothesis_id="H3b",
        claim=(
            "Multi-dimensional sentiment is more accurate than scalar B3 against "
            "Financial PhraseBank gold labels"
        ),
        metric_name="accuracy_delta_vs_b3",
        scale="percentage points",
        units="pp",
        observed=round(acc_md - acc_b3, 3),
        threshold=0.0,
        comparison=">",
        falsifiable_range=(-100.0, 100.0),
        notes=(
            f"Paired bootstrap 95% CI on the accuracy delta: "
            f"[{acc_delta_ci.lower * 100:.2f}, {acc_delta_ci.upper * 100:.2f}] pp; "
            f"paired permutation p={acc_test['p_value']:.4f}. "
            f"A positive point estimate whose CI spans zero is not evidence of "
            f"improvement."
        ),
    )

    payload: dict[str, Any] = {
        "experiment": "H3_sentiment_dimensionality",
        "dataset": dataset_source,
        "n_samples": len(texts),
        "decision_rules": {
            "b3": "direction = map(argmax(market_sentiment))",
            "multidimensional": (
                "regulatory==negative -> HOLD if market positive else SELL; "
                "market==negative and temporal==positive -> HOLD; "
                "else map(market)"
            ),
            "direction_mapping": DIRECTION_FROM_SENTIMENT,
            "pre_registered": True,
        },
        "routing": {
            "router_type": "lexical-regex",
            "n_regulatory_matched": len(reg_texts),
            "n_temporal_matched": len(tmp_texts),
            "regulatory_coverage_pct": round(100 * len(reg_texts) / len(texts), 2),
            "temporal_coverage_pct": round(100 * len(tmp_texts) / len(texts), 2),
        },
        "model_configuration": {
            "market_checkpoint": market_res.checkpoint,
            "regulatory_checkpoint": reg_res.checkpoint,
            "temporal_checkpoint": tmp_res.checkpoint,
            "temporal_shares_checkpoint_with": tmp_res.shares_checkpoint_with,
            "n_distinct_checkpoints_loaded": bundle.n_distinct_checkpoints,
            "quantisation": bundle.quantisation,
            "device": bundle.device,
            "note": (
                "The temporal dimension reuses the market checkpoint. There are "
                "two sets of weights, not three."
            ),
        },
        "results": {
            "accuracy_b3_pct": acc_b3,
            "accuracy_multidimensional_pct": acc_md,
            "accuracy_delta_pp": round(acc_md - acc_b3, 3),
            "macro_f1_b3_pct": f1_b3,
            "macro_f1_multidimensional_pct": f1_md,
            "macro_f1_delta_pp": round(f1_md - f1_b3, 3),
            "direction_divergence_pct": divergence,
            "n_diverged": len(diverged),
        },
        "divergence_analysis": {
            "n_diverged": len(diverged),
            "multidimensional_correct_when_diverged": md_right,
            "b3_correct_when_diverged": b3_right,
            "both_wrong_when_diverged": both_wrong,
            "multidimensional_win_rate_pct": (
                round(100 * md_right / len(diverged), 2) if diverged else None
            ),
            "interpretation": (
                "Of the cases where the two systems disagree, this is how often "
                "each was right. A win rate near or below 50% means divergence "
                "is not adding decision value."
            ),
        },
        "statistics": {
            "accuracy_delta_ci_pp": {
                "point": round(acc_delta_ci.point * 100, 4),
                "lower": round(acc_delta_ci.lower * 100, 4),
                "upper": round(acc_delta_ci.upper * 100, 4),
                "confidence": acc_delta_ci.confidence,
                "method": acc_delta_ci.method,
            },
            "accuracy_permutation_test": acc_test,
            "sample_size_interpretation": stats.interpret_sample_size(len(texts)),
        },
        "confusion_matrices": {
            "b3_vs_gold": confusion_matrix(b3_dirs, gold_dirs),
            "multidimensional_vs_gold": confusion_matrix(md_dirs, gold_dirs),
        },
        "validity_threats": _validity_threats(
            market_checkpoint=market_res.checkpoint,
            dataset_source=dataset_source,
            n_regulatory=len(reg_texts),
            n_temporal=len(tmp_texts),
            n_total=len(texts),
        ),
        "dimension_redundancy": {
            "temporal_shares_market_checkpoint": shared,
            "n_temporal_routed": len(tmp_matches_market),
            "n_temporal_label_identical_to_market": n_temporal_identical,
            "temporal_identical_pct": (
                round(100 * n_temporal_identical / len(tmp_matches_market), 2)
                if tmp_matches_market
                else None
            ),
            "temporal_fully_redundant": temporal_redundant,
            "unsatisfiable_rules": (
                ["market==negative and temporal==positive"] if temporal_redundant else []
            ),
            "explanation": (
                "The temporal dimension is scored by the market checkpoint. When it "
                "is routed the same text as market — which is always the case at "
                "sentence granularity — deterministic inference forces an identical "
                "label. Any rule requiring market and temporal to disagree is then "
                "unsatisfiable by construction, so the temporal dimension cannot "
                "change a recommendation no matter what the corpus contains. At "
                "chunk granularity the two receive different token spans, so the "
                "labels can differ; that case is untested here."
            ),
        },
        "low_confidence_counts": {
            "market": market_res.n_low_confidence,
            "regulatory": reg_res.n_low_confidence,
            "temporal": tmp_res.n_low_confidence,
        },
        "hypothesis_tests": [h3a.to_dict(), h3b.to_dict()],
    }

    return H3Result(payload=payload, tests=[h3a, h3b])


def sample_divergences(
    texts: Sequence[str],
    b3_dirs: Sequence[str],
    md_dirs: Sequence[str],
    gold_dirs: Sequence[str],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Concrete divergence cases for qualitative error analysis in the paper."""
    out: list[dict[str, Any]] = []
    for i, (t, b, m, g) in enumerate(zip(texts, b3_dirs, md_dirs, gold_dirs)):
        if b == m:
            continue
        out.append(
            {
                "index": i,
                "text": t[:400],
                "b3_direction": b,
                "multidimensional_direction": m,
                "gold_direction": g,
                "multidimensional_correct": m == g,
            }
        )
        if len(out) >= limit:
            break
    return out
