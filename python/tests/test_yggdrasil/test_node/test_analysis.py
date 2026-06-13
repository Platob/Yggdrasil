"""Analysis service: aggregate, series, ohlc, pivot, finance, forecast."""
from __future__ import annotations

import asyncio
import math
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.analysis import (
    AggMeasure,
    AggregateRequest,
    FinanceRequest,
    ForecastRequest,
    OhlcRequest,
    PivotRequest,
    SeriesRequest,
)
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def run(coro):
    return asyncio.run(coro)


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.home = Path(self.td.name)
        n = 2000
        cols = {
            "sector": [["Tech", "Energy", "Fin"][i % 3] for i in range(n)],
            "region": [["NA", "EU"][i % 2] for i in range(n)],
            "price": [100.0 + (i % 100) * 0.5 for i in range(n)],
        }
        pq.write_table(pa.table(cols), str(self.home / "d.parquet"))
        s = Settings(node_home=self.home, front_home=self.home)
        self.svc = AnalysisService(s, FsService(s))

    def tearDown(self):
        self.td.cleanup()

    def test_aggregate(self):
        res = run(self.svc.aggregate(AggregateRequest(
            path="d.parquet", group_by=["sector"],
            measures=[AggMeasure(column="price", agg="mean")])))
        self.assertEqual(res.group_count, 3)
        self.assertIn("price_mean", res.rows[0])

    def test_series_downsample(self):
        res = run(self.svc.series(SeriesRequest(path="d.parquet", column="price", points=100)))
        self.assertEqual(len(res.x), 100)
        self.assertEqual(len(res.y), 100)

    def test_series_no_downsample_when_small(self):
        res = run(self.svc.series(SeriesRequest(path="d.parquet", column="price", points=5000)))
        self.assertEqual(len(res.y), 2000)

    def test_ohlc(self):
        res = run(self.svc.ohlc(OhlcRequest(path="d.parquet", column="price", buckets=50)))
        self.assertEqual(res.bars, 50)
        self.assertEqual(len(res.opens), 50)
        for o, h, l, c in zip(res.opens, res.highs, res.lows, res.closes):
            self.assertLessEqual(l, o)
            self.assertGreaterEqual(h, o)

    def test_pivot(self):
        res = run(self.svc.pivot(PivotRequest(
            path="d.parquet", rows=["sector"], columns=["region"],
            measures=[AggMeasure(column="price", agg="sum")])))
        self.assertEqual(res.row_count, 3)
        self.assertEqual(res.col_count, 2)

    def test_finance(self):
        # geometric upward drift price series
        n = 500
        price = [100.0 * (1.0 + 0.001) ** i + math.sin(i / 7) for i in range(n)]
        pq.write_table(pa.table({"price": price}), str(self.home / "p.parquet"))
        res = run(self.svc.finance(FinanceRequest(path="p.parquet", column="price")))
        self.assertGreater(res.metrics.total_return, 0)
        self.assertLessEqual(res.metrics.max_drawdown, 0)
        self.assertEqual(len(res.ema), n)
        self.assertEqual(len(res.drawdown), n)
        self.assertEqual(len(res.cum_return), n)

    def test_forecast_ridge(self):
        n = 1000
        ts = list(range(n))
        val = [100.0 + 0.01 * i + 5 * math.sin(2 * math.pi * i / 24) for i in range(n)]
        pq.write_table(pa.table({"ts": ts, "value": val}), str(self.home / "ts.parquet"))
        res = run(self.svc.forecast(ForecastRequest(
            path="ts.parquet", column="value", x="ts", horizon=12, model="ridge", period=24)))
        self.assertEqual(res.model_used, "ridge")
        self.assertEqual(len(res.series), 1)
        self.assertEqual(len(res.series[0].forecast), 12)


if __name__ == "__main__":
    unittest.main()
