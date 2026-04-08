"""
technical_agent.py  —  Node A (CPU)
RSI · MACD · Bollinger Bands · VWAP via yfinance + pandas/numpy.
"""
import ray
import numpy as np
import yfinance as yf


@ray.remote(num_cpus=1)
class TechnicalAnalysisAgent:

    def compute_indicators(self, ticker: str, period: str = "3mo") -> dict:
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if hist.empty:
                return {"error": f"No price data for {ticker}"}

            close, vol = hist["Close"], hist["Volume"]

            # RSI 14
            d    = close.diff()
            gain = d.clip(lower=0).rolling(14).mean()
            loss = (-d.clip(upper=0)).rolling(14).mean()
            rsi  = float((100 - 100 / (1 + gain / loss.replace(0, float("inf")))).iloc[-1])

            # MACD (12, 26, 9)
            e12  = close.ewm(span=12, adjust=False).mean()
            e26  = close.ewm(span=26, adjust=False).mean()
            macd = e12 - e26
            sig  = macd.ewm(span=9, adjust=False).mean()
            macd_bull = bool(macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-2] <= sig.iloc[-2])

            # Bollinger Bands 20
            sma20  = close.rolling(20).mean()
            std20  = close.rolling(20).std()
            pvs_ub = float(close.iloc[-1] - (sma20 + 2 * std20).iloc[-1])

            # VWAP (period)
            vwap = float((close * vol).sum() / vol.sum())

            return {
                "ticker": ticker, "rsi": rsi,
                "rsi_signal": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral",
                "macd_crossover_bullish": macd_bull,
                "price_vs_upper_band": pvs_ub,
                "vwap": vwap, "current_price": float(close.iloc[-1]),
                "status": "ok",
            }
        except Exception as exc:
            return {"error": str(exc), "ticker": ticker}