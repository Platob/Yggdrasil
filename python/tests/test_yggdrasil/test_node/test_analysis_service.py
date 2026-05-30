"""Aggregate/pivot, describe, and finance analytics over Arrow via polars."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.node.api.schemas.analysis import (
    AggMeasure, AggregateRequest, CastSpec, ExportRequest, FilterSpec,
    FinanceRequest, IndicatorRequest, IndicatorSpec, OhlcRequest, SeriesRequest,
    Transform,
)
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

    def test_streams_full_file_for_correct_totals(self):
        # Reductions stream the whole parquet (lazy), so the sum is correct over
        # all rows even past the analysis cap — not a capped partial.
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"g": ["a"] * 50, "v": list(range(50))}), str(home / "big.parquet"))
            res = asyncio.run(_svc(home, analysis_max_rows=10).aggregate(
                AggregateRequest(path="big.parquet", group_by=["g"], measures=[AggMeasure(column="v", agg="sum")])))
            self.assertFalse(res.truncated)
            self.assertEqual(res.source_rows, 50)
            self.assertEqual(res.rows[0][res.columns.index("v_sum")], sum(range(50)))  # 1225, all rows


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


class TestSeriesDownsample(unittest.TestCase):
    def test_downsamples_to_points(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"t": list(range(1000)), "v": [float(i) for i in range(1000)]}),
                           str(home / "s.parquet"))
            res = asyncio.run(_svc(home).series(SeriesRequest(path="s.parquet", column="v", x="t", points=100)))
            self.assertTrue(res.sampled)
            self.assertEqual(res.source_rows, 1000)
            self.assertLessEqual(len(res.x), 110)               # ~100 buckets
            self.assertEqual(len(res.y), len(res.y_min))         # envelope aligned

    def test_zoom_reads_only_range(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"t": list(range(1000)), "v": [float(i) for i in range(1000)]}),
                           str(home / "s.parquet"))
            res = asyncio.run(_svc(home).series(
                SeriesRequest(path="s.parquet", column="v", x="t", points=100, x_min=0, x_max=99)))
            self.assertEqual(res.source_rows, 100)               # predicate pushdown
            self.assertFalse(res.sampled)                        # 100 <= points, returned raw

    def test_small_series_not_sampled(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"v": [1.0, 2.0, 3.0]}), str(home / "s.parquet"))
            res = asyncio.run(_svc(home).series(SeriesRequest(path="s.parquet", column="v", points=800)))
            self.assertFalse(res.sampled)
            self.assertEqual(res.y, [1.0, 2.0, 3.0])


class TestOhlc(unittest.TestCase):
    def test_resamples_to_bars(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            # 100 rows, prices ramp; 10 bars
            pq.write_table(pa.table({"t": list(range(100)), "price": [float(i) for i in range(100)],
                                     "vol": [1.0] * 100}), str(home / "p.parquet"))
            res = asyncio.run(_svc(home).ohlc(OhlcRequest(path="p.parquet", column="price", x="t", volume="vol", buckets=10)))
            self.assertEqual(res.bars, 10)
            self.assertEqual(res.open[0], 0.0)                   # first price of bucket 0
            self.assertEqual(res.close[0], 9.0)                  # last price of bucket 0
            self.assertGreaterEqual(res.high[0], res.low[0])
            self.assertEqual(res.volume[0], 10.0)                # 10 rows * vol 1


class TestFiltersAndExport(unittest.TestCase):
    def test_aggregate_filter_pushdown(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _trades(home)
            res = asyncio.run(_svc(home).aggregate(AggregateRequest(
                path="t.parquet", group_by=["sector"],
                measures=[AggMeasure(column="price", agg="sum")],
                filters=[FilterSpec(column="price", op=">=", value=100)])))
            self.assertEqual(res.source_rows, 3)            # only price>=100 rows
            by = {r[0]: r[res.columns.index("price_sum")] for r in res.rows}
            self.assertEqual(by["Tech"], 600.0)             # 100+200+300
            self.assertNotIn("Energy", by)                  # 50,60 filtered out

    def test_export_csv_with_filter_and_cast(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _trades(home)
            tmp, name = asyncio.run(_svc(home).export(ExportRequest(
                path="t.parquet", fmt="csv",
                transform=Transform(filters=[FilterSpec(column="sector", op="==", value="Tech")],
                                    casts=[CastSpec(column="price", dtype="int")],
                                    columns=["sector", "price"]))))
            try:
                text = tmp.read_text()
            finally:
                tmp.unlink(missing_ok=True)
            self.assertTrue(name.endswith(".csv"))
            self.assertIn("Tech", text)
            self.assertNotIn("Energy", text)               # filtered out
            self.assertEqual(text.count("\n"), 4)           # header + 3 Tech rows (+trailing)

    def test_export_timezone_to_utc_and_convert(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"ts": ["2024-01-01 12:00:00"], "v": [1]}), str(home / "z.parquet"))
            tmp, _ = asyncio.run(_svc(home).export(ExportRequest(
                path="z.parquet", fmt="csv",
                transform=Transform(casts=[CastSpec(column="ts", dtype="datetime", tz="America/New_York")]))))
            try:
                text = tmp.read_text()
            finally:
                tmp.unlink(missing_ok=True)
            self.assertIn("07:00:00", text)                 # 12:00 UTC -> 07:00 EST
            self.assertIn("-0500", text)

    def test_export_parquet_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _trades(home)
            tmp, name = asyncio.run(_svc(home).export(ExportRequest(path="t.parquet", fmt="parquet")))
            try:
                self.assertTrue(name.endswith(".parquet"))
                self.assertEqual(pq.read_table(str(tmp)).num_rows, 5)
            finally:
                tmp.unlink(missing_ok=True)


class TestIndicators(unittest.TestCase):
    def _prices(self, home: Path, n=60) -> None:
        import math
        pq.write_table(pa.table({
            "price": [100 + 10 * math.sin(i / 8) for i in range(n)],
            "ts": list(range(n)),
        }), str(home / "p.parquet"))

    def test_rsi_bounded_and_aligned(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); self._prices(home)
            res = asyncio.run(_svc(home).indicators(
                IndicatorRequest(path="p.parquet", column="price", x="ts",
                                 indicators=[IndicatorSpec(type="rsi", params={"period": 14})])))
            self.assertEqual(res.source_rows, 60)
            self.assertEqual(len(res.x), 60)
            ind = res.indicators[0]
            self.assertEqual(ind.type, "rsi")
            self.assertEqual(ind.name, "RSI(14)")
            self.assertEqual(len(ind.values["rsi"]), 60)
            vals = [v for v in ind.values["rsi"] if v is not None]
            self.assertTrue(all(0.0 <= v <= 100.0 for v in vals))

    def test_macd_emits_three_series(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); self._prices(home)
            res = asyncio.run(_svc(home).indicators(
                IndicatorRequest(path="p.parquet", column="price", x="ts",
                                 indicators=[IndicatorSpec(type="macd")])))
            ind = res.indicators[0]
            self.assertEqual(set(ind.values), {"macd", "signal", "histogram"})
            # histogram == macd - signal at every defined point
            for m, s, h in zip(ind.values["macd"], ind.values["signal"], ind.values["histogram"]):
                if m is not None and s is not None and h is not None:
                    self.assertAlmostEqual(h, m - s, places=6)

    def test_bollinger_band_ordering(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); self._prices(home)
            res = asyncio.run(_svc(home).indicators(
                IndicatorRequest(path="p.parquet", column="price", x="ts",
                                 indicators=[IndicatorSpec(type="bb", params={"period": 10})])))
            ind = res.indicators[0]
            self.assertEqual(set(ind.values), {"middle", "upper", "lower"})
            for lo, mid, hi in zip(ind.values["lower"], ind.values["middle"], ind.values["upper"]):
                if lo is not None and mid is not None and hi is not None:
                    self.assertLessEqual(lo, mid)
                    self.assertLessEqual(mid, hi)

    def test_multiple_indicators_in_one_call(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); self._prices(home)
            res = asyncio.run(_svc(home).indicators(
                IndicatorRequest(path="p.parquet", column="price", x="ts",
                                 indicators=[IndicatorSpec(type="rsi"), IndicatorSpec(type="ema"),
                                             IndicatorSpec(type="macd"), IndicatorSpec(type="bb")])))
            self.assertEqual([i.type for i in res.indicators], ["rsi", "ema", "macd", "bb"])

    def test_caps_at_5000_points(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"price": [float(i) for i in range(6000)],
                                     "ts": list(range(6000))}), str(home / "big.parquet"))
            res = asyncio.run(_svc(home).indicators(
                IndicatorRequest(path="big.parquet", column="price", x="ts",
                                 indicators=[IndicatorSpec(type="ema")])))
            self.assertEqual(len(res.x), 5000)            # capped
            self.assertEqual(res.source_rows, 6000)       # but reports true total

    def test_no_x_uses_row_index(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); self._prices(home)
            res = asyncio.run(_svc(home).indicators(
                IndicatorRequest(path="p.parquet", column="price",
                                 indicators=[IndicatorSpec(type="ema")])))
            self.assertEqual(res.x, list(range(60)))

    def test_missing_column(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); self._prices(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).indicators(
                    IndicatorRequest(path="p.parquet", column="ghost",
                                     indicators=[IndicatorSpec(type="rsi")])))

    def test_unknown_indicator_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); self._prices(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).indicators(
                    IndicatorRequest(path="p.parquet", column="price",
                                     indicators=[IndicatorSpec(type="ichimoku")])))

    def test_empty_indicators_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); self._prices(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).indicators(
                    IndicatorRequest(path="p.parquet", column="price", indicators=[])))


if __name__ == "__main__":
    unittest.main()
