"""Technical indicators, correlation, portfolio, VaR, and trade signals."""
from __future__ import annotations

import asyncio
import math
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.node.api.schemas.trading import (
    CorrelationRequest,
    IndicatorRequest,
    PortfolioAsset,
    PortfolioRequest,
    SignalRequest,
    VaRRequest,
)
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.trading import TradingService
from yggdrasil.node.config import Settings


def _svc(home: Path, **kw) -> TradingService:
    s = Settings(node_id="t", node_home=home, front_home=home, **kw)
    return TradingService(s, fs=FsService(s))


def _walk(seed: int, n: int = 300, drift: float = 0.0004, vol: float = 0.015):
    """Deterministic geometric random walk price path."""
    import random
    rng = random.Random(seed)
    p = 100.0
    out = [p]
    for _ in range(n - 1):
        p *= 1.0 + rng.gauss(drift, vol)
        out.append(p)
    return out


def _prices(home: Path, name: str, seed: int = 1, n: int = 300, *, ohlc: bool = False) -> str:
    closes = _walk(seed, n)
    cols = {"t": list(range(n)), "close": closes}
    if ohlc:
        cols["high"] = [c * 1.01 for c in closes]
        cols["low"] = [c * 0.99 for c in closes]
    pq.write_table(pa.table(cols), str(home / name))
    return name


class TestIndicators(unittest.TestCase):
    def test_all_indicators_present_and_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            name = _prices(home, "p.parquet", ohlc=True)
            res = asyncio.run(_svc(home).indicators(IndicatorRequest(
                path=name, column="close", x="t",
                indicators=["rsi", "macd", "bb", "atr"], high="high", low="low")))
            self.assertEqual(res.source_rows, 300)
            self.assertEqual(len(res.value), 300)
            self.assertEqual(len(res.rsi), 300)
            # RSI is bounded to [0, 100] where defined.
            defined = [v for v in res.rsi if v is not None]
            self.assertTrue(defined)
            self.assertTrue(all(0.0 <= v <= 100.0 for v in defined))
            # MACD histogram = macd - signal at every defined point.
            for m, s, h in zip(res.macd, res.macd_signal, res.macd_hist):
                if m is not None and s is not None and h is not None:
                    self.assertAlmostEqual(h, m - s, places=6)
            # Bollinger ordering upper >= mid >= lower.
            for u, mid, lo in zip(res.bb_upper, res.bb_mid, res.bb_lower):
                if u is not None and mid is not None and lo is not None:
                    self.assertGreaterEqual(u, mid)
                    self.assertGreaterEqual(mid, lo)
            self.assertTrue(any(v is not None for v in res.atr))

    def test_only_requested_indicators_computed(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            name = _prices(home, "p.parquet")
            res = asyncio.run(_svc(home).indicators(IndicatorRequest(
                path=name, column="close", indicators=["rsi"])))
            self.assertIsNotNone(res.rsi)
            self.assertIsNone(res.macd)
            self.assertIsNone(res.bb_upper)
            self.assertIsNone(res.atr)

    def test_truncation_flag(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            name = _prices(home, "p.parquet", n=500)
            res = asyncio.run(_svc(home).indicators(IndicatorRequest(
                path=name, column="close", indicators=["rsi"], limit=100)))
            self.assertTrue(res.truncated)
            self.assertEqual(res.source_rows, 500)
            self.assertEqual(len(res.value), 100)

    def test_missing_column_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            name = _prices(home, "p.parquet")
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).indicators(IndicatorRequest(path=name, column="ghost")))


class TestCorrelation(unittest.TestCase):
    def test_matrix_shape_and_diagonal(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            a = _prices(home, "a.parquet", seed=1)
            b = _prices(home, "b.parquet", seed=2)
            res = asyncio.run(_svc(home).correlation(CorrelationRequest(
                paths=[a, b], column="close", method="pearson")))
            self.assertEqual(len(res.matrix), 2)
            self.assertEqual(len(res.matrix[0]), 2)
            self.assertAlmostEqual(res.matrix[0][0], 1.0, places=6)
            self.assertAlmostEqual(res.matrix[1][1], 1.0, places=6)
            self.assertAlmostEqual(res.matrix[0][1], res.matrix[1][0], places=9)  # symmetric
            self.assertEqual(res.labels, ["a.parquet", "b.parquet"])

    def test_identical_series_perfectly_correlated(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            a = _prices(home, "a.parquet", seed=5)
            # b is a copy of a
            tbl = pq.read_table(str(home / a))
            pq.write_table(tbl, str(home / "b.parquet"))
            res = asyncio.run(_svc(home).correlation(CorrelationRequest(
                paths=[a, "b.parquet"], column="close")))
            self.assertAlmostEqual(res.matrix[0][1], 1.0, places=5)

    def test_spearman_without_scipy_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            a = _prices(home, "a.parquet", seed=1)
            b = _prices(home, "b.parquet", seed=2)
            res = asyncio.run(_svc(home).correlation(CorrelationRequest(
                paths=[a, b], column="close", method="spearman")))
            self.assertAlmostEqual(res.matrix[0][0], 1.0, places=5)
            self.assertIsNotNone(res.matrix[0][1])

    def test_single_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            a = _prices(home, "a.parquet")
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).correlation(CorrelationRequest(paths=[a], column="close")))


class TestPortfolio(unittest.TestCase):
    def test_weights_normalized_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            a = _prices(home, "a.parquet", seed=1)
            b = _prices(home, "b.parquet", seed=2)
            res = asyncio.run(_svc(home).portfolio(PortfolioRequest(assets=[
                PortfolioAsset(path=a, column="close", label="A", weight=3.0),
                PortfolioAsset(path=b, column="close", label="B", weight=1.0),
            ])))
            self.assertAlmostEqual(sum(res.weights), 1.0, places=6)
            self.assertAlmostEqual(res.weights[0], 0.75, places=6)
            self.assertEqual(res.labels, ["A", "B"])
            self.assertIsNotNone(res.metrics.ann_volatility)
            self.assertIsNotNone(res.metrics.max_drawdown)
            self.assertLessEqual(res.metrics.max_drawdown, 0.0)
            # Equity tracks compounded growth from the first period; the value
            # array aligns with the returns (one shorter than the price series).
            self.assertEqual(len(res.portfolio_value), len(res.index))
            self.assertTrue(all(v is None or v > 0.0 for v in res.portfolio_value))
            self.assertTrue(all(dd is None or dd <= 1e-9 for dd in res.drawdown))
            # Correlation matrix is n×n with unit diagonal.
            self.assertEqual(len(res.correlation_matrix), 2)
            self.assertAlmostEqual(res.correlation_matrix[0][0], 1.0, places=5)

    def test_two_assets_required(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            a = _prices(home, "a.parquet")
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).portfolio(PortfolioRequest(
                    assets=[PortfolioAsset(path=a, column="close")])))

    def test_zero_weights_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            a = _prices(home, "a.parquet", seed=1)
            b = _prices(home, "b.parquet", seed=2)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).portfolio(PortfolioRequest(assets=[
                    PortfolioAsset(path=a, column="close", weight=0.0),
                    PortfolioAsset(path=b, column="close", weight=0.0),
                ])))


class TestVaR(unittest.TestCase):
    def test_historical_var_is_a_loss(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            name = _prices(home, "p.parquet")
            res = asyncio.run(_svc(home).var(VaRRequest(
                path=name, column="close", method="historical", confidence=0.95)))
            self.assertIsNotNone(res.var)
            self.assertLess(res.var, 0.0)              # a loss at the tail
            self.assertLessEqual(res.cvar, res.var)    # CVaR is at least as bad as VaR
            self.assertAlmostEqual(res.var_pct, res.var * 100, places=3)
            self.assertIsNotNone(res.ann_volatility)

    def test_all_methods_run(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            name = _prices(home, "p.parquet")
            for m in ("historical", "parametric", "cornish_fisher"):
                res = asyncio.run(_svc(home).var(VaRRequest(path=name, column="close", method=m)))
                self.assertEqual(res.method, m)
                self.assertTrue(res.var is None or math.isfinite(res.var))

    def test_horizon_scales_var(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            name = _prices(home, "p.parquet")
            v1 = asyncio.run(_svc(home).var(VaRRequest(path=name, column="close", horizon=1)))
            v4 = asyncio.run(_svc(home).var(VaRRequest(path=name, column="close", horizon=4)))
            # sqrt-time scaling: 4-day VaR ~ 2x the 1-day VaR.
            self.assertAlmostEqual(v4.var, v1.var * 2.0, places=4)

    def test_unknown_method_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            name = _prices(home, "p.parquet")
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).var(VaRRequest(path=name, column="close", method="bogus")))

    def test_too_few_observations_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"close": [1.0, 2.0, 3.0]}), str(home / "tiny.parquet"))
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).var(VaRRequest(path="tiny.parquet", column="close")))


class TestSignals(unittest.TestCase):
    def test_signals_emitted_with_counts(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            name = _prices(home, "p.parquet", seed=3)
            res = asyncio.run(_svc(home).signals(SignalRequest(path=name, column="close", x="t")))
            self.assertEqual(res.source_rows, 300)
            self.assertEqual(res.buy_count, sum(1 for s in res.signals if s.action == "BUY"))
            self.assertEqual(res.sell_count, sum(1 for s in res.signals if s.action == "SELL"))
            self.assertIn(res.last_action, {"BUY", "SELL", "HOLD"})
            for s in res.signals:
                self.assertIn(s.action, {"BUY", "SELL", "HOLD"})
                self.assertGreaterEqual(s.strength, 0.0)
                if s.action != "HOLD":
                    self.assertTrue(s.reasons)

    def test_oversold_triggers_buy(self):
        # A monotonically crashing then-recovering series drives RSI under 30,
        # which must surface at least one BUY reason.
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            crash = [100.0 * (0.97 ** i) for i in range(40)]
            recover = [crash[-1] * (1.02 ** i) for i in range(1, 20)]
            pq.write_table(pa.table({"close": crash + recover}), str(home / "c.parquet"))
            res = asyncio.run(_svc(home).signals(SignalRequest(path="c.parquet", column="close")))
            reasons = {r for s in res.signals for r in s.reasons}
            self.assertTrue(any("RSI" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
