"""Technical price indicators: RSI, MACD, Bollinger Bands, VWAP.

Pure CPU, no model loading, no VRAM concerns — this is Node A's work in the
distributed pipeline. Kept as a plain class (not a Ray actor) so it is testable
and usable identically in local and distributed execution; the Ray wrapper lives
in :mod:`maor.pipeline.distributed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TechnicalIndicators:
    ticker: str
    status: str
    rsi: float | None = None
    rsi_signal: str | None = None
    macd_crossover_bullish: bool | None = None
    price_vs_upper_band: float | None = None
    vwap: float | None = None
    current_price: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None or k == "status"}


class TechnicalAgent:
    """Computes RSI(14), MACD(12,26,9), Bollinger(20, 2sigma) and VWAP via yfinance."""

    def compute_indicators(self, ticker: str, period: str = "3mo") -> TechnicalIndicators:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required: pip install yfinance") from exc

        try:
            hist = yf.Ticker(ticker).history(period=period)
        except Exception as exc:
            return TechnicalIndicators(ticker=ticker, status="error", error=str(exc))

        if hist.empty:
            return TechnicalIndicators(
                ticker=ticker, status="error", error=f"no price data for {ticker}"
            )

        try:
            close, vol = hist["Close"], hist["Volume"]

            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = float(
                (100 - 100 / (1 + gain / loss.replace(0, float("inf")))).iloc[-1]
            )

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_bull = bool(
                macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
            )

            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            price_vs_upper = float(close.iloc[-1] - (sma20 + 2 * std20).iloc[-1])

            vwap = float((close * vol).sum() / vol.sum())

            return TechnicalIndicators(
                ticker=ticker,
                status="ok",
                rsi=rsi,
                rsi_signal=(
                    "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
                ),
                macd_crossover_bullish=macd_bull,
                price_vs_upper_band=price_vs_upper,
                vwap=vwap,
                current_price=float(close.iloc[-1]),
            )
        except Exception as exc:
            return TechnicalIndicators(ticker=ticker, status="error", error=str(exc))
