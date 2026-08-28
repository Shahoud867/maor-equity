"""Statistical validation.

Audit finding M5: the project reported point estimates from n=2, one run each, no
seeds, no intervals and no tests, then generalised them in the abstract. This
module supplies the uncertainty machinery so that a claim can state how much of
it is signal.

Everything here is non-parametric. Latency distributions are right-skewed and
small-n; ROUGE deltas are bounded and non-normal. Bootstrap and permutation tests
make no distributional assumption, which is the honest default at these sample
sizes.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    lower: float
    upper: float
    confidence: float
    n: int
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self, unit: str = "", precision: int = 2) -> str:
        return (
            f"{self.point:.{precision}f}{unit} "
            f"[{self.lower:.{precision}f}, {self.upper:.{precision}f}]"
            f" ({self.confidence:.0%} CI, n={self.n})"
        )


# Cap on elements held in one resampling array. n_resamples x n can reach 5x10^7
# at realistic sizes (10,000 resamples over 4,846 items), which is ~400 MB per
# temporary and several of them per call. Resampling is batched to stay bounded
# regardless of input size.
_RESAMPLE_BLOCK_ELEMENTS = 4_000_000


def _resample_blocks(n_resamples: int, n: int) -> list[int]:
    """Split n_resamples into chunks that keep each temporary array bounded."""
    per_block = max(1, _RESAMPLE_BLOCK_ELEMENTS // max(n, 1))
    blocks = []
    remaining = n_resamples
    while remaining > 0:
        take = min(per_block, remaining)
        blocks.append(take)
        remaining -= take
    return blocks


def _vectorised_statistic(statistic: Callable[[Sequence[float]], float]) -> str | None:
    """Name the numpy equivalent of a statistic, when one exists.

    Resampling in pure Python costs O(n_resamples x n) interpreter operations.
    At n=4,846 with 10,000 resamples that is ~10^8 operations per call and
    dominates the runtime of an otherwise cheap experiment. Mean and median have
    exact numpy equivalents and are vectorised; anything else uses the general
    path, which stays correct if slower.
    """
    if statistic is statistics.median:
        return "median"
    if statistic is statistics.mean:
        return "mean"
    return None


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.median,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for an arbitrary statistic.

    With n below roughly 5 the interval is reported but is nearly uninformative;
    :func:`interpret_sample_size` says so explicitly rather than letting a narrow
    interval imply precision that is not there.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    vals = list(values)
    point = float(statistic(vals))
    if len(vals) == 1:
        return Interval(point, point, point, confidence, 1, "degenerate-n1")

    n = len(vals)
    alpha = (1.0 - confidence) / 2.0
    kind = _vectorised_statistic(statistic)

    if kind is not None:
        import numpy as np

        rng = np.random.default_rng(seed)
        arr = np.asarray(vals, dtype=float)
        parts = []
        for block in _resample_blocks(n_resamples, n):
            samples = arr[rng.integers(0, n, size=(block, n))]
            parts.append(
                np.median(samples, axis=1) if kind == "median" else samples.mean(axis=1)
            )
        resampled_arr = np.concatenate(parts)
        lower = float(np.quantile(resampled_arr, alpha))
        upper = float(np.quantile(resampled_arr, 1.0 - alpha))
        method = f"percentile-bootstrap-{n_resamples}-vectorised"
    else:
        rng_py = random.Random(seed)
        resampled = [
            float(statistic([vals[rng_py.randrange(n)] for _ in range(n)]))
            for _ in range(n_resamples)
        ]
        resampled.sort()
        lower = resampled[int(alpha * n_resamples)]
        upper = resampled[min(int((1.0 - alpha) * n_resamples), n_resamples - 1)]
        method = f"percentile-bootstrap-{n_resamples}"

    return Interval(
        point=round(point, 4),
        lower=round(lower, 4),
        upper=round(upper, 4),
        confidence=confidence,
        n=n,
        method=method,
    )


def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.mean,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """CI for ``statistic(a) - statistic(b)`` on paired samples.

    Pairing matters: both systems see the same documents, so resampling documents
    rather than systems removes between-document variance, which dominates.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must be the same length")
    if not a:
        raise ValueError("cannot bootstrap an empty sample")

    point = float(statistic(list(a))) - float(statistic(list(b)))
    n = len(a)
    alpha = (1.0 - confidence) / 2.0
    kind = _vectorised_statistic(statistic)

    if kind is not None:
        import numpy as np

        rng = np.random.default_rng(seed)
        arr_a = np.asarray(a, dtype=float)
        arr_b = np.asarray(b, dtype=float)
        parts = []
        for block in _resample_blocks(n_resamples, n):
            idx = rng.integers(0, n, size=(block, n))
            sa, sb = arr_a[idx], arr_b[idx]
            if kind == "median":
                parts.append(np.median(sa, axis=1) - np.median(sb, axis=1))
            else:
                parts.append(sa.mean(axis=1) - sb.mean(axis=1))
        deltas_arr = np.concatenate(parts)
        lower = float(np.quantile(deltas_arr, alpha))
        upper = float(np.quantile(deltas_arr, 1.0 - alpha))
        method = f"paired-percentile-bootstrap-{n_resamples}-vectorised"
    else:
        rng_py = random.Random(seed)
        deltas: list[float] = []
        for _ in range(n_resamples):
            picks = [rng_py.randrange(n) for _ in range(n)]
            deltas.append(
                float(statistic([a[i] for i in picks]))
                - float(statistic([b[i] for i in picks]))
            )
        deltas.sort()
        lower = deltas[int(alpha * n_resamples)]
        upper = deltas[min(int((1 - alpha) * n_resamples), n_resamples - 1)]
        method = f"paired-percentile-bootstrap-{n_resamples}"

    return Interval(
        point=round(point, 4),
        lower=round(lower, 4),
        upper=round(upper, 4),
        confidence=confidence,
        n=n,
        method=method,
    )


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.mean,
    n_permutations: int = 10_000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Paired permutation test by sign-flipping the per-item differences.

    The exact test for "does the pairing carry signal", with no distributional
    assumption. Returns the p-value and the observed effect.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must be the same length")
    if not a:
        raise ValueError("cannot test an empty sample")

    diffs = [x - y for x, y in zip(a, b)]
    observed = float(statistic(diffs))
    kind = _vectorised_statistic(statistic)

    if kind is not None:
        import numpy as np

        rng = np.random.default_rng(seed)
        arr = np.asarray(diffs, dtype=float)
        count = 0
        for block in _resample_blocks(n_permutations, arr.size):
            # integers(0,2) then scaled beats choice() here: no lookup table and
            # no intermediate index array.
            signs = rng.integers(0, 2, size=(block, arr.size)) * 2.0 - 1.0
            flipped = signs * arr
            stats_arr = (
                np.median(flipped, axis=1) if kind == "median" else flipped.mean(axis=1)
            )
            if alternative == "two-sided":
                count += int(np.sum(np.abs(stats_arr) >= abs(observed)))
            elif alternative == "greater":
                count += int(np.sum(stats_arr >= observed))
            else:
                count += int(np.sum(stats_arr <= observed))
    else:
        rng_py = random.Random(seed)
        count = 0
        for _ in range(n_permutations):
            flipped_list = [d if rng_py.random() < 0.5 else -d for d in diffs]
            stat = float(statistic(flipped_list))
            if alternative == "two-sided":
                if abs(stat) >= abs(observed):
                    count += 1
            elif alternative == "greater":
                if stat >= observed:
                    count += 1
            else:
                if stat <= observed:
                    count += 1

    # Add-one correction: a p-value of exactly 0 is not attainable from a finite
    # number of permutations and should not be reported as such.
    p_value = (count + 1) / (n_permutations + 1)
    return {
        "observed_effect": round(observed, 4),
        "p_value": round(p_value, 5),
        "n_permutations": n_permutations,
        "alternative": alternative,
        "n_pairs": len(a),
        "method": "paired-sign-flip-permutation",
    }


def interpret_sample_size(n: int) -> dict[str, Any]:
    """State plainly what a sample of this size can support.

    Exists so that small-n results carry their own caveat into the result file
    instead of relying on a reader to remember it.
    """
    if n < 3:
        level, claim = "anecdote", (
            "Case study only. No population-level claim is supportable; report "
            "per-item values, never a median with an interval."
        )
    elif n < 10:
        level, claim = "indicative", (
            "Direction may be indicative; intervals will be very wide. Do not "
            "state a point estimate without its interval."
        )
    elif n < 30:
        level, claim = "weak", (
            "Supports a bounded claim with an explicit interval. Not sufficient "
            "for subgroup analysis."
        )
    else:
        level, claim = "adequate", (
            "Supports a population-level claim with bootstrap intervals for this "
            "population."
        )
    return {"n": n, "evidence_level": level, "supportable_claim": claim}


def summarise(values: Sequence[float]) -> dict[str, Any]:
    """Descriptive statistics reported together, never a bare mean."""
    vals = list(values)
    if not vals:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }
    if len(vals) > 1:
        out["stdev"] = round(statistics.stdev(vals), 4)
        out["cv"] = round(statistics.stdev(vals) / statistics.mean(vals), 4) if statistics.mean(vals) else None
    return out
