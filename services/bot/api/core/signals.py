from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..models.market import OHLCV
from ..models.signal import Indicator, Signal, SignalDirection

# Signal cache: (symbol, timeframe) → (signal, cached_at)
_cache: dict[tuple[str, str], tuple[Signal, float]] = {}
_CACHE_TTL = 60.0


def _direction(value: float, buy_above: float, sell_below: float) -> SignalDirection:
    if value >= buy_above:
        return SignalDirection.BUY
    if value <= sell_below:
        return SignalDirection.SELL
    return SignalDirection.NEUTRAL


def compute_rsi(closes: np.ndarray | list[float], period: int = 14) -> float:
    """Wilder RSI — pure Python list operations (fastest for N<1000)."""
    if isinstance(closes, np.ndarray):
        closes = closes.tolist()
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    inv = period - 1
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * inv + g) / period
        avg_loss = (avg_loss * inv + l) / period
    if avg_loss == 0.0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def compute_macd(
    closes: np.ndarray | list[float],
    fast: int = 12, slow: int = 26, signal_period: int = 9,
) -> tuple[float, float, float]:
    """MACD — pure Python EMA (fastest for N<1000)."""
    if isinstance(closes, np.ndarray):
        closes = closes.tolist()
    if len(closes) < slow + signal_period:
        return 0.0, 0.0, 0.0

    def ema(data: list[float], n: int) -> list[float]:
        k = 2.0 / (n + 1)
        r = [data[0]]
        for v in data[1:]:
            r.append(v * k + r[-1] * (1 - k))
        return r

    fast_ema  = ema(closes, fast)
    slow_ema  = ema(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    sig_line  = ema(macd_line, signal_period)
    mv, sv    = macd_line[-1], sig_line[-1]
    return mv, sv, mv - sv


def compute_bollinger(
    closes: np.ndarray | list[float], period: int = 20, std_dev: float = 2.0,
) -> tuple[float, float, float]:
    """Bollinger Bands."""
    if isinstance(closes, np.ndarray):
        window = closes[-period:] if len(closes) >= period else closes
        mean   = float(window.mean())
        sigma  = float(window.std(ddof=0))
    else:
        window = closes[-period:]
        if len(window) < period:
            c = closes[-1] if closes else 0.0
            return c, c, c
        mean  = sum(window) / period
        var   = sum((x - mean) ** 2 for x in window) / period
        sigma = var ** 0.5
    return mean + std_dev * sigma, mean, mean - std_dev * sigma


def compute_sma(closes: np.ndarray | list[float], period: int) -> float:
    if isinstance(closes, np.ndarray):
        if len(closes) < period:
            return float(closes[-1]) if len(closes) else 0.0
        return float(closes[-period:].mean())
    if len(closes) < period:
        return closes[-1] if closes else 0.0
    return sum(closes[-period:]) / period


_SCORE_MAP = {
    SignalDirection.STRONG_BUY: 1.0,
    SignalDirection.BUY: 0.5,
    SignalDirection.NEUTRAL: 0.0,
    SignalDirection.SELL: -0.5,
    SignalDirection.STRONG_SELL: -1.0,
}


def generate_signals(symbol: str, bars: list[OHLCV], timeframe: str = "1d") -> Signal:
    """Compute technical indicators and aggregate a trading signal."""
    cache_key = (symbol, timeframe)
    now = time.monotonic()
    cached = _cache.get(cache_key)
    if cached and now - cached[1] < _CACHE_TTL:
        return cached[0]

    if not bars:
        sig = Signal(symbol=symbol, direction=SignalDirection.NEUTRAL, confidence=0.0, timeframe=timeframe)
        _cache[cache_key] = (sig, now)
        return sig

    closes = np.array([b.close for b in bars], dtype=np.float64)
    current = float(closes[-1])
    indicators: list[Indicator] = []
    scores: list[float] = []

    # RSI
    rsi = compute_rsi(closes)
    if rsi < 30:
        rsi_dir = SignalDirection.STRONG_BUY
    elif rsi > 70:
        rsi_dir = SignalDirection.STRONG_SELL
    elif rsi >= 50:
        rsi_dir = SignalDirection.BUY
    else:
        rsi_dir = SignalDirection.SELL
    indicators.append(Indicator(name="RSI(14)", value=round(rsi, 2), signal=rsi_dir,
                                description=f"RSI={rsi:.1f} — {'oversold' if rsi<30 else 'overbought' if rsi>70 else 'neutral'}"))
    scores.append(_SCORE_MAP[rsi_dir])

    # MACD
    macd_val, macd_sig, macd_hist = compute_macd(closes)
    macd_dir = SignalDirection.BUY if macd_val > macd_sig else SignalDirection.SELL
    indicators.append(Indicator(name="MACD(12,26,9)", value=round(macd_hist, 4), signal=macd_dir,
                                description=f"Histogram={macd_hist:+.4f}"))
    scores.append(0.5 if macd_dir == SignalDirection.BUY else -0.5)

    # Bollinger
    upper, mid, lower = compute_bollinger(closes)
    bb_pct = (current - lower) / (upper - lower) if upper != lower else 0.5
    if bb_pct < 0.2:
        bb_dir = SignalDirection.BUY
    elif bb_pct > 0.8:
        bb_dir = SignalDirection.SELL
    else:
        bb_dir = SignalDirection.NEUTRAL
    indicators.append(Indicator(name="BB(20,2)", value=round(bb_pct, 3), signal=bb_dir,
                                description=f"%B={bb_pct:.1%}"))
    scores.append({SignalDirection.BUY: 0.5, SignalDirection.NEUTRAL: 0.0, SignalDirection.SELL: -0.5}[bb_dir])

    # SMA crossover
    sma20 = compute_sma(closes, 20)
    sma50 = compute_sma(closes, 50)
    cross_dir = SignalDirection.BUY if sma20 > sma50 else SignalDirection.SELL
    indicators.append(Indicator(name="SMA(20/50)", value=round(sma20 - sma50, 4), signal=cross_dir,
                                description=f"SMA20={sma20:.2f} vs SMA50={sma50:.2f}"))
    scores.append(0.4 if cross_dir == SignalDirection.BUY else -0.4)

    avg_score = sum(scores) / len(scores)
    if avg_score >= 0.6:
        direction = SignalDirection.STRONG_BUY
    elif avg_score >= 0.2:
        direction = SignalDirection.BUY
    elif avg_score <= -0.6:
        direction = SignalDirection.STRONG_SELL
    elif avg_score <= -0.2:
        direction = SignalDirection.SELL
    else:
        direction = SignalDirection.NEUTRAL

    confidence = min(abs(avg_score) + 0.3, 1.0)

    # ATR for price targets
    hi = np.array([b.high for b in bars[-14:]])
    lo = np.array([b.low  for b in bars[-14:]])
    atr = float((hi - lo).mean())
    price_target = current + atr * 2 if avg_score > 0 else current - atr * 2
    stop_loss    = current - atr * 1.5 if avg_score > 0 else current + atr * 1.5

    sig = Signal(
        symbol=symbol,
        direction=direction,
        confidence=round(confidence, 3),
        price_target=round(price_target, 4),
        stop_loss=round(stop_loss, 4),
        indicators=indicators,
        timeframe=timeframe,
    )
    _cache[cache_key] = (sig, now)
    return sig
