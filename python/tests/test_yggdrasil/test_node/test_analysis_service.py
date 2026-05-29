"""Aggregate/pivot, describe, and finance analytics over Arrow via polars."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.node.api.schemas.analysis import AggMeasure, AggregateRequest, FinanceRequest
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def _svc(home: Path, **kw) -> AnalysisService:
    s = Settings(node_id="t", node_home=home, front_home=home, **kw)
    return AnalysisService(s, fs=FsService(s))


def _trades(home: Path, name="t.parquet") -> None:
    pq.write_table(pa.table({
        "sector": ["Tech", "Energy", "Tech", "Energy", "Tech"],
        "price": [100.0, 50.0, 200.0, 60.0, 300.0],
        "volume": [10, 5, 20, 6, 30],
    }), str(home / name))


class TestAggregate(unittest.TestCase):
    def test_group_by_sum_and_mean(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _trades(home)
            req = AggregateRequest(path="t.parquet", group_by=["sector"],
                                   measures=[AggMeasure(column="price", agg="sum"),
                                             AggMeasure(column="volume", agg="mean")])
            res = asyncio.run(_svc(home).aggregate(req))
            self.assertEqual(set(res.columns), {"sector", "price_sum", "volume_mean"})
            by = {r[0]: r for r in res.rows}
            # Tech: 100+200+300=600 ; Energy: 50+60=110 ; sorted desc by price_sum
            self.assertEqual(by["Tech"][res.columns.index("price_sum")], 600.0)
            self.assertEqual(by["Energy"][res.columns.index("price_sum")], 110.0)
            self.assertEqual(res.rows[0][0], "Tech")     # sorted desc
            self.assertEqual(res.group_count, 2)

    def test_global_aggregate_no_groups(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _trades(home)
            res = asyncio.run(_svc(home).aggregate(
                AggregateRequest(path="t.parquet", measures=[AggMeasure(column="volume", agg="sum")])))
            self.assertEqual(res.rows, [[71]])           # 10+5+20+6+30

    def test_bad_column_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _trades(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).aggregate(
                    AggregateRequest(path="t.parquet", measures=[AggMeasure(column="ghost", agg="sum")])))

    def test_truncated_flag(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"g": ["a"] * 50, "v": list(range(50))}), str(home / "big.parquet"))
            res = asyncio.run(_svc(home, analysis_max_rows=10).aggregate(
                AggregateRequest(path="big.parquet", group_by=["g"], measures=[AggMeasure(column="v", agg="sum")])))
            self.assertTrue(res.truncated)
            self.assertEqual(res.source_rows, 10)        # capped


class TestDescribe(unittest.TestCase):
    def test_numeric_stats(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _trades(home)
            res = asyncio.run(_svc(home).describe("t.parquet"))
            self.assertIn("mean", res.statistics)
            self.assertEqual(set(res.columns), {"price", "volume"})   # only numeric
            self.assertEqual(len(res.rows), len(res.statistics))


class TestFinance(unittest.TestCase):
    def test_returns_and_rolling(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"price": [100.0, 110.0, 121.0, 133.1]}), str(home / "p.parquet"))
            res = asyncio.run(_svc(home).finance(FinanceRequest(path="p.parquet", column="price", window=2)))
            self.assertEqual(len(res.value), 4)
            self.assertIsNone(res.pct_change[0])                       # first return undefined
            self.assertAlmostEqual(res.pct_change[1], 0.1, places=6)   # 110/100-1
            # +10% each step compounds: cum_return last ≈ 0.331
            self.assertAlmostEqual(res.cum_return[-1], 0.331, places=3)

    def test_missing_column(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _trades(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).finance(FinanceRequest(path="t.parquet", column="ghost")))


if __name__ == "__main__":
    unittest.main()
