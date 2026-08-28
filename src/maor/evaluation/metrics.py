"""Evaluation metrics with explicit, falsifiable scales.

Audit finding C4: the published H2 test compared a delta on the 0-1 ROUGE scale
against a tolerance of 1.0 "ROUGE points". Since the maximum possible absolute
delta on that scale *is* 1.0, every conceivable result passed. The test could not
fail.

The fix is not a bigger number. It is to make the scale explicit and to make
hypothesis tests objects that state their own falsification condition, so a test
that cannot fail is visible as a bug rather than reported as a pass.

All ROUGE and BERTScore values in this module are on the **0-100 scale**. This is
the convention in the summarisation literature and the one the original
``rouge_eval.py`` already used; the generators used 0-1 and the mismatch is what
made the broken tolerance invisible.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# ROUGE / BERTScore
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RougeScores:
    """ROUGE F-measures on the 0-100 scale."""

    rouge_1: float
    rouge_2: float
    rouge_l: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        # ROUGE-L counts a longest common *subsequence* of tokens that ROUGE-1
        # already counts as unigram matches, so ROUGE-L <= ROUGE-1 always holds
        # for F-measures over the same pair. The published table had B2 at
        # R-L 0.32 > R-1 0.28, which no run of this scorer can produce.
        if self.rouge_l > self.rouge_1 + 1e-6:
            raise ValueError(
                f"ROUGE-L ({self.rouge_l:.3f}) > ROUGE-1 ({self.rouge_1:.3f}). "
                f"This is impossible for F-measures over the same pairs and "
                f"indicates the scores were not produced by a real scorer."
            )


def compute_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    use_stemmer: bool = True,
) -> RougeScores:
    """Mean ROUGE-1/2/L F-measure, 0-100 scale.

    Raises if inputs are misaligned or empty rather than returning a meaningless
    zero, because a silent zero here becomes a published number.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            f"must be the same length"
        )
    if not predictions:
        raise ValueError("cannot compute ROUGE over zero samples")

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
    )
    r1: list[float] = []
    r2: list[float] = []
    rl: list[float] = []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref or "", pred or "")
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    return RougeScores(
        rouge_1=round(statistics.mean(r1) * 100, 3),
        rouge_2=round(statistics.mean(r2) * 100, 3),
        rouge_l=round(statistics.mean(rl) * 100, 3),
        n=len(predictions),
    )


def compute_rouge_per_sample(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    use_stemmer: bool = True,
) -> list[dict[str, float]]:
    """Per-sample ROUGE, needed for paired tests and bootstrap CIs."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
    )
    out: list[dict[str, float]] = []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref or "", pred or "")
        out.append(
            {
                "rouge_1": s["rouge1"].fmeasure * 100,
                "rouge_2": s["rouge2"].fmeasure * 100,
                "rouge_l": s["rougeL"].fmeasure * 100,
            }
        )
    return out


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "distilbert-base-uncased",
    device: str | None = None,
) -> dict[str, float]:
    """BERTScore F1 on the 0-100 scale.

    Reported with ``model_type`` because BERTScore is not comparable across
    backbones; a number without its backbone is not interpretable.
    """
    from bert_score import score as bert_score_fn

    _p, _r, f1 = bert_score_fn(
        list(predictions),
        list(references),
        lang="en",
        model_type=model_type,
        verbose=False,
        device=device,
    )
    return {
        "bertscore_f1": round(float(f1.mean().item()) * 100, 3),
        "bertscore_model": model_type,
        "n": len(predictions),
    }


# ---------------------------------------------------------------------------
# Classification metrics (H3)
# ---------------------------------------------------------------------------


def accuracy(predictions: Sequence[str], labels: Sequence[str]) -> float:
    """Accuracy as a percentage."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must be the same length")
    if not predictions:
        raise ValueError("cannot compute accuracy over zero samples")
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return round(correct / len(predictions) * 100, 3)


def macro_f1(predictions: Sequence[str], labels: Sequence[str]) -> float:
    """Macro-averaged F1 as a percentage.

    Reported alongside accuracy because Financial PhraseBank is heavily skewed
    toward neutral; accuracy alone rewards a majority-class predictor.
    """
    classes = sorted(set(labels) | set(predictions))
    f1s: list[float] = []
    for c in classes:
        tp = sum(1 for p, l in zip(predictions, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(predictions, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(predictions, labels) if p != c and l == c)
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return round(statistics.mean(f1s) * 100, 3) if f1s else 0.0


def confusion_matrix(
    predictions: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    classes = sorted(set(labels) | set(predictions))
    matrix = {a: {b: 0 for b in classes} for a in classes}
    for p, l in zip(predictions, labels):
        matrix[l][p] += 1
    return matrix


def disagreement_rate(a: Sequence[str], b: Sequence[str]) -> float:
    """Percentage of positions where two label sequences differ."""
    if len(a) != len(b):
        raise ValueError("sequences must be the same length")
    if not a:
        raise ValueError("cannot compute disagreement over zero samples")
    return round(sum(1 for x, y in zip(a, b) if x != y) / len(a) * 100, 3)


# ---------------------------------------------------------------------------
# Hypothesis tests that can fail
# ---------------------------------------------------------------------------


@dataclass
class HypothesisTest:
    """A hypothesis with an explicit, checkable falsification condition.

    ``scale`` and ``units`` are mandatory. The bug in the original H2 test was a
    tolerance whose units did not match the quantity it was applied to, and
    recording both makes that mismatch checkable — see :meth:`sanity_check`.
    """

    hypothesis_id: str
    claim: str
    metric_name: str
    scale: str  # e.g. "0-100" or "percent" or "ratio"
    observed: float
    threshold: float
    comparison: str  # ">=", ">", "<=", "<"
    units: str
    falsifiable_range: tuple[float, float]
    notes: str = ""

    _OPS: dict[str, Callable[[float, float], bool]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        ops = {
            ">=": lambda a, b: a >= b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            "<": lambda a, b: a < b,
        }
        if self.comparison not in ops:
            raise ValueError(f"comparison must be one of {sorted(ops)}")
        object.__setattr__(self, "_OPS", ops)

    @property
    def passed(self) -> bool:
        return self._OPS[self.comparison](self.observed, self.threshold)

    def sanity_check(self) -> list[str]:
        """Return warnings if this test is degenerate.

        A test whose threshold sits outside the range the metric can occupy is
        not a test. This is the guard that would have caught the H2 defect.
        """
        warnings: list[str] = []
        lo, hi = self.falsifiable_range
        if not (lo <= self.threshold <= hi):
            warnings.append(
                f"{self.hypothesis_id}: threshold {self.threshold} lies outside the "
                f"achievable range of {self.metric_name} [{lo}, {hi}] on the "
                f"{self.scale} scale. This test cannot fail and is not evidence."
            )
        if self.comparison in (">=", ">") and self.threshold <= lo:
            warnings.append(
                f"{self.hypothesis_id}: threshold {self.threshold} is at or below the "
                f"metric floor {lo}; every possible outcome passes."
            )
        if self.comparison in ("<=", "<") and self.threshold >= hi:
            warnings.append(
                f"{self.hypothesis_id}: threshold {self.threshold} is at or above the "
                f"metric ceiling {hi}; every possible outcome passes."
            )
        return warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "metric_name": self.metric_name,
            "scale": self.scale,
            "units": self.units,
            "observed": self.observed,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "falsifiable_range": list(self.falsifiable_range),
            "result": "PASS" if self.passed else "FAIL",
            "sanity_warnings": self.sanity_check(),
            "notes": self.notes,
        }
