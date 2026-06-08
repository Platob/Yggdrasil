"""Tests for ``yggdrasil.node.signals`` — the Polars indicator pipeline."""
from __future__ import annotations

import math

import pytest

from yggdrasil.lazy_imports import polars as pl
from yggdrasil.node.signals import compute_signals, with_indicators


def _bars(closes: list[float]) -> pl.DataFrame:
    n = len(closes)
    return pl.DataFrame(
        {
            "ts": [i * 86_400_000 for i in range(n)],
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
        }
    )


class TestIndicators:
    def test_with_indicators_adds_all_columns(self):
        df = with_indicators(_bars([100.0 + i for i in range(60)]))
        for col in ("sma20", "sma50", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower"):
            assert col in df.columns

    def test_sma20_matches_manual_mean(self):
        closes = [float(i) for i in range(1, 61)]
        df = with_indicators(_bars(closes))
        last_sma20 = df["sma20"][-1]
        assert last_sma20 == pytest.approx(sum(closes[-20:]) / 20)

    def test_rsi_in_zero_to_hundred(self):
        df = with_indicators(_bars([100.0 + math.sin(i * 0.4) * 5 for i in range(80)]))
        rsi = [v for v in df["rsi"].to_list() if v is not None and v == v]
        assert all(0.0 <= v <= 100.0 for v in rsi)


class TestComputeSignals:
    def test_short_series_is_flat_hold(self):
        out = compute_signals(_bars([100.0, 101.0, 102.0]))
        assert out["signal"] == "HOLD"
        assert out["strength"] == 0.0
        assert out["indicators"] == {}

    def test_oversold_pullback_in_uptrend_is_buy(self):
        # Long uptrend (SMA20>SMA50) with a sharp recent dip → oversold RSI +
        # price below the lower Bollinger band → strong BUY.
        closes = [100.0 + i * 0.6 for i in range(110)] + [166.0 - j * 3.0 for j in range(1, 11)]
        out = compute_signals(_bars(closes))
        assert out["signal"] == "BUY"
        assert out["score"] > 0.2
        assert out["indicators"]["rsi"] < 30

    def test_overbought_bounce_in_downtrend_is_sell(self):
        closes = [200.0 - i * 0.6 for i in range(110)] + [134.0 + j * 3.0 for j in range(1, 11)]
        out = compute_signals(_bars(closes))
        assert out["signal"] == "SELL"
        assert out["score"] < -0.2
        assert out["indicators"]["rsi"] > 70

    def test_indicator_values_present_on_long_series(self):
        out = compute_signals(_bars([100.0 + math.sin(i * 0.2) * 8 for i in range(120)]))
        ind = out["indicators"]
        assert ind["sma20"] is not None
        assert ind["rsi"] is not None
        assert ind["bb_upper"] > ind["bb_lower"]

    def test_prices_tail_capped_at_20(self):
        out = compute_signals(_bars([100.0 + i for i in range(120)]))
        assert len(out["prices"]) == 20
        assert set(out["prices"][0]) >= {"ts", "open", "high", "low", "close", "volume"}
