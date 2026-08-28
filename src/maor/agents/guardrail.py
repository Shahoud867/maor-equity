"""Bull/Bear guardrail with a deterministic arbiter.

Audit finding M6, root cause. The original ``_gen_json`` returned
``{"direction": "unknown", "confidence": 0.0, "signals": []}`` when JSON parsing
failed, and the arbiter then read ``bull.get("confidence", 0.5)``. Because the key
*existed* with value ``0.0``, the ``0.5`` default never applied: both scores
collapsed to zero, their difference was zero, and every run returned
``UNRESOLVED / LOW`` with ``bull_score=0.0, bear_score=0.0``. That is exactly what
both real runs in ``results/`` show. A parse failure was silently converted into a
confident-looking verdict.

The fix is to stop conflating "the model said zero" with "we could not read the
model". :class:`StanceParseResult` carries ``parsed`` explicitly, the arbiter
refuses to score unparsed stances, and the outcome is ``ASSESSMENT_FAILED`` —
distinguishable from a genuine ``UNRESOLVED``, which means the model was read and
the two sides were genuinely balanced.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class Recommendation:
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    UNRESOLVED = "UNRESOLVED"
    ASSESSMENT_FAILED = "ASSESSMENT_FAILED"


@dataclass
class StanceParseResult:
    """One side's argument, and whether we actually managed to read it."""

    parsed: bool
    confidence: float | None = None
    signals: list[str] = field(default_factory=list)
    raw_text: str = ""
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "confidence": self.confidence,
            "signals": self.signals,
            "parse_error": self.parse_error,
        }


def parse_stance(text: str) -> StanceParseResult:
    """Extract a stance JSON object from generated text.

    Tries strict JSON over the outermost braces, then a regex fallback for
    ``confidence``. Anything else is an explicit failure — never a default score.
    """
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        blob = text[start : end + 1]
        try:
            obj = json.loads(blob)
            conf = obj.get("confidence")
            if isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0:
                signals = obj.get("signals") or []
                return StanceParseResult(
                    parsed=True,
                    confidence=float(conf),
                    signals=[str(s) for s in signals][:5],
                    raw_text=text,
                )
            return StanceParseResult(
                parsed=False,
                raw_text=text,
                parse_error=f"'confidence' missing or out of range: {conf!r}",
            )
        except json.JSONDecodeError as exc:
            match = re.search(r'"confidence"\s*:\s*([01](?:\.\d+)?)', blob)
            if match:
                return StanceParseResult(
                    parsed=True,
                    confidence=float(match.group(1)),
                    signals=re.findall(r'"([^"]{4,80})"', blob)[:5],
                    raw_text=text,
                    parse_error=f"recovered by regex after JSONDecodeError: {exc.msg}",
                )
            return StanceParseResult(
                parsed=False, raw_text=text, parse_error=f"JSONDecodeError: {exc.msg}"
            )

    return StanceParseResult(
        parsed=False, raw_text=text, parse_error="no JSON object found in output"
    )


BULL_PROMPT = (
    "<|user|>\nMake the strongest BULL case for this equity, then rate your own "
    "confidence.\n\n{context}\n"
    'Reply with JSON only: {{"direction":"bullish","confidence":0.0,'
    '"signals":["s1","s2","s3"]}}\n<|end|>\n<|assistant|>\n'
)
BEAR_PROMPT = (
    "<|user|>\nMake the strongest BEAR case for this equity, then rate your own "
    "confidence.\n\n{context}\n"
    'Reply with JSON only: {{"direction":"bearish","confidence":0.0,'
    '"signals":["s1","s2","s3"]}}\n<|end|>\n<|assistant|>\n'
)


class GuardrailAgent:
    """Two opposing LLM stances resolved by a deterministic, LLM-free arbiter."""

    def __init__(
        self,
        model: Any,
        *,
        max_new_tokens: int = 200,
        technical_weight: float = 0.25,
        decisive_margin: float = 0.30,
        weak_margin: float = 0.10,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.technical_weight = technical_weight
        self.decisive_margin = decisive_margin
        self.weak_margin = weak_margin

    def _context(self, summary: str, sentiment: Any, technical: dict[str, Any]) -> str:
        parts = [f"Summary: {summary[:800]}"]

        if sentiment is not None:
            rows = getattr(sentiment, "rows", None)
            if rows is not None:
                order = sentiment.dimension_order
                present = sentiment.present
                for i, dim in enumerate(order):
                    if not present[i]:
                        # Absent dimensions are stated as absent, not as neutral.
                        parts.append(f"{dim.capitalize()} sentiment: not present in document")
                    else:
                        # Each dimension has its own label space (market/regulatory
                        # are polarity, temporal is forward-looking-statement
                        # specificity) — read per-row, not from one shared scheme.
                        row_labels = sentiment.label_order_for(dim)
                        row = rows[i]
                        detail = ", ".join(
                            f"{lab}={row[j]:.2f}" for j, lab in enumerate(row_labels)
                        )
                        parts.append(f"{dim.capitalize()} ({'/'.join(row_labels)}): {detail}")
            else:
                arr = np.asarray(sentiment, dtype=float)
                parts.append(f"Sentiment matrix: {np.array2string(arr, precision=2)}")

        rsi = technical.get("rsi")
        if rsi is not None:
            parts.append(
                f"RSI={rsi:.1f}, MACD bullish crossover="
                f"{technical.get('macd_crossover_bullish', False)}"
            )
        return "\n".join(parts)

    def assess(
        self, summary: str, sentiment: Any, technical: dict[str, Any]
    ) -> dict[str, Any]:
        context = self._context(summary, sentiment, technical)

        bull_raw = self.model.generate(
            BULL_PROMPT.format(context=context), max_new_tokens=self.max_new_tokens
        )
        bear_raw = self.model.generate(
            BEAR_PROMPT.format(context=context), max_new_tokens=self.max_new_tokens
        )
        bull = parse_stance(bull_raw.text)
        bear = parse_stance(bear_raw.text)
        return self.arbitrate(bull, bear, technical)

    def arbitrate(
        self,
        bull: StanceParseResult,
        bear: StanceParseResult,
        technical: dict[str, Any],
    ) -> dict[str, Any]:
        """Deterministic rule-based resolution. No LLM call.

        Refuses to produce a recommendation when either stance could not be read.
        """
        if not bull.parsed or not bear.parsed:
            failed = [
                name
                for name, r in (("bull", bull), ("bear", bear))
                if not r.parsed
            ]
            log.warning("guardrail: could not parse %s stance(s)", ", ".join(failed))
            return {
                "recommendation": Recommendation.ASSESSMENT_FAILED,
                "confidence": None,
                "bull_score": None,
                "bear_score": None,
                "unparsed_stances": failed,
                "parse_errors": {
                    "bull": bull.parse_error,
                    "bear": bear.parse_error,
                },
                "note": (
                    "One or both stances could not be parsed, so no recommendation "
                    "was produced. This is distinct from UNRESOLVED, which means "
                    "both stances were read and were genuinely balanced."
                ),
                "stances": {"bull": bull.to_dict(), "bear": bear.to_dict()},
            }

        rsi = float(technical.get("rsi", 50.0))
        strength = abs(rsi - 50.0) / 50.0
        if rsi < 30 or technical.get("macd_crossover_bullish"):
            direction = 1
        elif rsi > 70:
            direction = -1
        else:
            direction = 0

        bull_score = bull.confidence + (self.technical_weight * strength if direction > 0 else 0.0)
        bear_score = bear.confidence + (self.technical_weight * strength if direction < 0 else 0.0)
        margin = abs(bull_score - bear_score)

        if margin <= self.weak_margin:
            return {
                "recommendation": Recommendation.UNRESOLVED,
                "confidence": "LOW",
                "bull_score": round(bull_score, 4),
                "bear_score": round(bear_score, 4),
                "margin": round(margin, 4),
                "conflict": True,
                "conflict_signals": (bull.signals + bear.signals)[:6],
                "rsi": rsi,
                "note": "Both stances read; scores within the indecision band.",
                "stances": {"bull": bull.to_dict(), "bear": bear.to_dict()},
            }

        winner = bull if bull_score > bear_score else bear
        confidence = (
            "HIGH"
            if margin > self.decisive_margin
            else ("MEDIUM" if margin > (self.decisive_margin + self.weak_margin) / 2 else "LOW")
        )
        return {
            "recommendation": (
                Recommendation.BULLISH if bull_score > bear_score else Recommendation.BEARISH
            ),
            "confidence": confidence,
            "bull_score": round(bull_score, 4),
            "bear_score": round(bear_score, 4),
            "margin": round(margin, 4),
            "conflict": False,
            "winning_signals": winner.signals,
            "rsi": rsi,
            "technical_direction": direction,
            "stances": {"bull": bull.to_dict(), "bear": bear.to_dict()},
        }
