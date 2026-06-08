"""Polars-native technical indicator pipeline and signal engine.

``with_indicators`` adds indicator columns to an OHLCV DataFrame.
``compute_signals`` derives a BUY/SELL/HOLD trading signal from those
columns. Both functions operate via vectorised Polars expressions — no
row-level Python loops, no explicit iteration over prices.

Scoring model
-------------
The signal is scored on the interval ``[-1.0, +1.0]``:

+0.40  SMA-20 > SMA-50 (short-term trend above long-term)
−0.40  SMA-20 < SMA-50
+0.30  RSI < 30 (oversold — potential reversal)
−0.30  RSI > 75 (strongly overbought)
−0.10  RSI in [70, 75] (mildly elevated)
+0.20  MACD > 0 (net momentum positive)
−0.20  MACD < 0
+0.10  MACD bullish crossover this bar
−0.10  MACD bearish crossover this bar
+0.10  Price below lower Bollinger Band (mean-reversion opportunity)
−0.10  Price above upper Bollinger Band

BUY  → score >  0.20
SELL → score < −0.20
HOLD → otherwise
"""
from __future__ import annotations

from typing import Any

from yggdrasil.lazy_imports import polars as pl

__all__ = ["with_indicators", "compute_signals"]


def with_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Return *df* with SMA-20/50, RSI-14, MACD (12/26/9), and Bollinger-20 columns added.

    Input *df* must have at least a ``close`` column (Float64).  The
    returned frame always has all seven indicator columns; values before
    the required warm-up period are ``null``.
    """
    close = df["close"]
    n = len(df)

    # SMA
    sma20 = close.rolling_mean(20)
    sma50 = close.rolling_mean(min(50, n))

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower_bound=0.0).rolling_mean(14)
    loss = (-delta).clip(lower_bound=0.0).rolling_mean(14)
    rs = gain / (loss + 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # MACD (12, 26, 9 EMA)
    ema12 = close.ewm_mean(span=12, adjust=False)
    ema26 = close.ewm_mean(span=26, adjust=False)
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm_mean(span=9, adjust=False)

    # Bollinger Bands (20 bars, 2σ)
    roll_mean = close.rolling_mean(20)
    roll_std = close.rolling_std(20)
    bb_upper = roll_mean + 2.0 * roll_std
    bb_lower = roll_mean - 2.0 * roll_std

    return df.with_columns([
        sma20.alias("sma20"),
        sma50.alias("sma50"),
        rsi.alias("rsi"),
        macd_line.alias("macd"),
        macd_signal.alias("macd_signal"),
        bb_upper.alias("bb_upper"),
        bb_lower.alias("bb_lower"),
    ])


def compute_signals(df: pl.DataFrame) -> dict[str, Any]:
    """Compute a BUY/SELL/HOLD signal from an OHLCV Polars DataFrame.

    *df* must be sorted ascending by ``ts`` and have columns
    ``ts, open, high, low, close, volume``.

    Returns a dict with keys: ``signal``, ``strength``, ``score``,
    ``reasons``, ``indicators``, ``prices``.
    """
    _EMPTY = {"signal": "HOLD", "strength": 0.0, "score": 0.0, "reasons": [], "indicators": {}, "prices": []}

    if len(df) < 20:
        return _EMPTY

    enriched = with_indicators(df)
    last = enriched.row(-1, named=True)
    prev = enriched.row(-2, named=True) if len(enriched) >= 2 else last

    score = 0.0
    reasons: list[str] = []

    # -- SMA crossover (primary trend driver) --------------------------------
    s20, s50 = last.get("sma20"), last.get("sma50")
    if s20 is not None and s50 is not None:
        if s20 > s50:
            score += 0.40
            reasons.append("SMA20>SMA50 (bullish)")
        else:
            score -= 0.40
            reasons.append("SMA20<SMA50 (bearish)")

    # -- RSI (momentum) -------------------------------------------------------
    rsi_val = last.get("rsi")
    if rsi_val is not None:
        if rsi_val < 30:
            score += 0.30
            reasons.append(f"RSI={rsi_val:.1f} oversold")
        elif rsi_val > 75:
            score -= 0.30
            reasons.append(f"RSI={rsi_val:.1f} overbought")
        elif rsi_val > 70:
            score -= 0.10
            reasons.append(f"RSI={rsi_val:.1f} elevated")

    # -- MACD (momentum sign + crossover) ------------------------------------
    macd_v = last.get("macd")
    macd_sig = last.get("macd_signal")
    prev_macd = prev.get("macd")
    prev_sig = prev.get("macd_signal")

    if macd_v is not None:
        # Primary: is MACD itself positive or negative?
        if macd_v > 0:
            score += 0.20
            reasons.append("MACD positive (net bullish momentum)")
        else:
            score -= 0.20
            reasons.append("MACD negative (net bearish momentum)")

        # Secondary: crossover event this bar
        if macd_sig is not None and prev_macd is not None and prev_sig is not None:
            crossed_up = prev_macd <= prev_sig and macd_v > macd_sig
            crossed_dn = prev_macd >= prev_sig and macd_v < macd_sig
            if crossed_up:
                score += 0.10
                reasons.append("MACD bullish crossover")
            elif crossed_dn:
                score -= 0.10
                reasons.append("MACD bearish crossover")

    # -- Bollinger Bands (mean-reversion signal) ------------------------------
    price = last["close"]
    bb_u, bb_l = last.get("bb_upper"), last.get("bb_lower")
    if bb_u is not None and bb_l is not None:
        if price < bb_l:
            score += 0.10
            reasons.append("Price below lower Bollinger Band")
        elif price > bb_u:
            score -= 0.10
            reasons.append("Price above upper Bollinger Band")

    signal = "BUY" if score > 0.20 else ("SELL" if score < -0.20 else "HOLD")
    strength = round(min(abs(score), 1.0), 3)

    def _r(v: float | None, d: int = 4) -> float | None:
        return round(v, d) if v is not None else None

    recent = (
        enriched
        .tail(20)
        .select(["ts", "open", "high", "low", "close", "volume"])
        .to_dicts()
    )

    return {
        "signal": signal,
        "strength": strength,
        "score": round(score, 3),
        "reasons": reasons,
        "indicators": {
            "sma20": _r(s20),
            "sma50": _r(s50),
            "rsi": _r(rsi_val, 2),
            "macd": _r(macd_v),
            "macd_signal": _r(macd_sig),
            "bb_upper": _r(bb_u),
            "bb_lower": _r(bb_l),
        },
        "prices": recent,
    }
