"""Technical indicators (RSI/MACD/BB/SMA/EMA/ATR/VWAP/OBV/Stoch) over price files."""
from __future__ import annotations

import asyncio
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.node.api.schemas.technical import IndicatorSpec, TechnicalRequest
from yggdrasil.node.api.services.technical import TechnicalService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def _svc(home: Path, **kw) -> TechnicalService:
    s = Settings(node_id="t", node_home=home, front_home=home, **kw)
    return TechnicalService(s, fs=FsService(s))


def _prices(home: Path, n: int = 120, name: str = "ohlcv.parquet") -> None:
    rng = np.random.default_rng(7)
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    volume = rng.uniform(1e5, 1e6, n)
    ts = np.arange(n)
    pq.write_table(
        pa.table({"close": close, "high": high, "low": low, "volume": volume, "t": ts}),
        str(home / name),
    )


def _series(res, name):
    return next(i.series for i in res.indicators if i.name == name)


class TestSingleIndicators(unittest.TestCase):
    def test_rsi_bounded_and_length(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            res = asyncio.run(_svc(home).compute(TechnicalRequest(
                path="ohlcv.parquet", close="close",
                indicators=[IndicatorSpec(type="rsi", period=14)])))
            self.assertEqual(res.source_rows, 120)
            rsi = _series(res, "RSI(14)")
            self.assertEqual(len(rsi), 120)
            # First `period` values are warmup -> None.
            self.assertIsNone(rsi[0])
            for v in rsi:
                if v is not None:
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 100.0)

    def test_sma_matches_manual_window(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            svc = _svc(home)
            res = asyncio.run(svc.compute(TechnicalRequest(
                path="ohlcv.parquet", close="close",
                indicators=[IndicatorSpec(type="sma", period=5)])))
            sma = _series(res, "SMA(5)")
            close = pq.read_table(str(home / "ohlcv.parquet")).column("close").to_numpy()
            self.assertAlmostEqual(sma[4], float(np.mean(close[:5])), places=9)
            self.assertIsNone(sma[3])  # not enough rows yet

    def test_macd_emits_three_series(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            res = asyncio.run(_svc(home).compute(TechnicalRequest(
                path="ohlcv.parquet", close="close",
                indicators=[IndicatorSpec(type="macd", fast=12, slow=26, signal=9)])))
            names = {i.name for i in res.indicators}
            self.assertEqual(names, {"MACD(12,26)", "MACD_signal(9)", "MACD_hist"})

    def test_bb_bands_ordered(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            res = asyncio.run(_svc(home).compute(TechnicalRequest(
                path="ohlcv.parquet", close="close",
                indicators=[IndicatorSpec(type="bb", period=20, std_dev=2.0)])))
            up = _series(res, "BB_upper(20)")
            mid = _series(res, "BB_mid(20)")
            lo = _series(res, "BB_lower(20)")
            for u, m, l in zip(up, mid, lo):
                if None not in (u, m, l):
                    self.assertGreaterEqual(u, m)
                    self.assertGreaterEqual(m, l)

    def test_stoch_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            res = asyncio.run(_svc(home).compute(TechnicalRequest(
                path="ohlcv.parquet", close="close", high="high", low="low",
                indicators=[IndicatorSpec(type="stoch", period=14, d_period=3)])))
            k = _series(res, "Stoch_K(14)")
            for v in k:
                if v is not None:
                    self.assertGreaterEqual(v, 0.0)
                    self.assertLessEqual(v, 100.0)


class TestColumnHandling(unittest.TestCase):
    def test_index_column_sorts_and_is_returned(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            res = asyncio.run(_svc(home).compute(TechnicalRequest(
                path="ohlcv.parquet", close="close", x="t",
                indicators=[IndicatorSpec(type="sma", period=5)])))
            self.assertEqual(res.x, list(range(120)))

    def test_default_index_is_row_number(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            res = asyncio.run(_svc(home).compute(TechnicalRequest(
                path="ohlcv.parquet", close="close",
                indicators=[IndicatorSpec(type="ema", period=10)])))
            self.assertEqual(res.x, list(range(120)))

    def test_only_needed_columns_required(self):
        # A close-only RSI must not fail just because high/low were named but absent.
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}), str(home / "c.parquet"))
            res = asyncio.run(_svc(home).compute(TechnicalRequest(
                path="c.parquet", close="close", high="high", low="low",
                indicators=[IndicatorSpec(type="sma", period=2)])))
            self.assertEqual(res.source_rows, 5)


class TestErrors(unittest.TestCase):
    def test_missing_close_column(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).compute(TechnicalRequest(
                    path="ohlcv.parquet", close="nope",
                    indicators=[IndicatorSpec(type="rsi")])))

    def test_atr_without_high_low(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).compute(TechnicalRequest(
                    path="ohlcv.parquet", close="close",
                    indicators=[IndicatorSpec(type="atr", period=14)])))

    def test_vwap_without_volume(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).compute(TechnicalRequest(
                    path="ohlcv.parquet", close="close", high="high", low="low",
                    indicators=[IndicatorSpec(type="vwap")])))

    def test_unknown_indicator(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).compute(TechnicalRequest(
                    path="ohlcv.parquet", close="close",
                    indicators=[IndicatorSpec(type="frobnicate")])))

    def test_missing_file(self):
        from yggdrasil.node.exceptions import NotFoundError
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            with self.assertRaises(NotFoundError):
                asyncio.run(_svc(home).compute(TechnicalRequest(
                    path="absent.parquet", close="close",
                    indicators=[IndicatorSpec(type="rsi")])))


class TestFullStack(unittest.TestCase):
    def test_all_indicators_finite_or_none(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _prices(home)
            res = asyncio.run(_svc(home).compute(TechnicalRequest(
                path="ohlcv.parquet", close="close", high="high", low="low", volume="volume",
                indicators=[
                    IndicatorSpec(type="rsi"), IndicatorSpec(type="macd"),
                    IndicatorSpec(type="bb"), IndicatorSpec(type="atr"),
                    IndicatorSpec(type="vwap"), IndicatorSpec(type="obv"),
                    IndicatorSpec(type="stoch"),
                ])))
            for ind in res.indicators:
                self.assertEqual(len(ind.series), 120)
                for v in ind.series:
                    if v is not None:
                        self.assertTrue(math.isfinite(v))


if __name__ == "__main__":
    unittest.main()
