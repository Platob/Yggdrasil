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
    FinanceRequest, ForecastRequest, IndicatorsRequest, OhlcRequest,
    PivotRequest, PortfolioRequest, SeriesRequest, Transform,
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


def _sales(home: Path, name="s.parquet") -> None:
    pq.write_table(pa.table({
        "region":  ["NA", "NA", "EU", "EU", "NA", "EU", "APAC", "APAC"],
        "product": ["A", "B", "A", "B", "A", "A", "B", "B"],
        "qty":     [10, 5, 7, 3, 2, 8, 4, 6],
        "rev":     [100.0, 50.0, 70.0, 30.0, 20.0, 80.0, 40.0, 60.0],
    }), str(home / name))


class TestPivot(unittest.TestCase):
    def test_cross_tab_rows_by_columns(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            res = asyncio.run(_svc(home).pivot(PivotRequest(
                path="s.parquet", rows=["region"], columns=["product"],
                measures=[AggMeasure(column="rev", agg="sum")], totals=False)))
            self.assertEqual(res.columns, ["region", "A", "B"])  # one col per product value
            by = {r[0]: r for r in res.rows}
            self.assertEqual(by["EU"], ["EU", 150.0, 30.0])      # A=70+80, B=30
            self.assertEqual(by["NA"], ["NA", 120.0, 50.0])
            self.assertIsNone(by["APAC"][1])                     # APAC has no product A
            self.assertEqual(by["APAC"][2], 100.0)               # B=40+60
            self.assertEqual(res.col_count, 2)
            self.assertEqual(res.source_rows, 8)
            self.assertFalse(res.truncated)

    def test_multi_measure_flattened_headers(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            res = asyncio.run(_svc(home).pivot(PivotRequest(
                path="s.parquet", rows=["region"], columns=["product"],
                measures=[AggMeasure(column="rev", agg="sum"),
                          AggMeasure(column="qty", agg="mean")], totals=False)))
            self.assertEqual(res.columns,
                             ["region", "A · rev_sum", "A · qty_mean", "B · rev_sum", "B · qty_mean"])

    def test_rows_only_is_group_by(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            res = asyncio.run(_svc(home).pivot(PivotRequest(
                path="s.parquet", rows=["region"], columns=[],
                measures=[AggMeasure(column="rev", agg="sum")], totals=False)))
            self.assertEqual(res.columns, ["region", "rev_sum"])
            self.assertEqual({r[0] for r in res.rows}, {"NA", "EU", "APAC"})

    def test_grand_total_no_fields(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            res = asyncio.run(_svc(home).pivot(PivotRequest(
                path="s.parquet", measures=[AggMeasure(column="rev", agg="sum")])))
            self.assertEqual(res.rows, [[450.0]])

    def test_col_limit_keeps_top_n_and_flags_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            res = asyncio.run(_svc(home).pivot(PivotRequest(
                path="s.parquet", rows=["region"], columns=["product"],
                measures=[AggMeasure(column="rev", agg="sum")], col_limit=1, totals=False)))
            # B (40+60+50+30=180) outweighs A (70+80+20+100? no: A=70+80+20=170) -> keep B
            self.assertEqual(res.col_count, 2)
            self.assertTrue(res.truncated)
            self.assertEqual(len(res.columns), 2)  # region + 1 kept column

    def test_high_cardinality_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home, pivot_max_groups=2).pivot(PivotRequest(
                    path="s.parquet", rows=["region"], columns=["product"],
                    measures=[AggMeasure(column="rev", agg="sum")])))

    def test_filter_pushdown(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            res = asyncio.run(_svc(home).pivot(PivotRequest(
                path="s.parquet", rows=["region"], columns=["product"],
                measures=[AggMeasure(column="rev", agg="sum")],
                filters=[FilterSpec(column="region", op="==", value="EU")], totals=False)))
            self.assertEqual([r[0] for r in res.rows], ["EU"])
            self.assertEqual(res.source_rows, 3)

    def test_grand_totals_row_and_column(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            res = asyncio.run(_svc(home).pivot(PivotRequest(
                path="s.parquet", rows=["region"], columns=["product"],
                measures=[AggMeasure(column="rev", agg="sum")], totals=True)))
            self.assertEqual(res.columns, ["region", "A", "B", "Total"])
            self.assertEqual(res.total_columns, 1)
            self.assertTrue(res.has_total_row)
            by = {r[0]: r for r in res.rows}
            self.assertEqual(by["EU"][3], 180.0)        # row total A(150)+B(30)
            self.assertEqual(by["Total"], ["Total", 270.0, 180.0, 450.0])  # col + grand

    def test_totals_reaggregate_not_sum_of_cells(self):
        # The total of a *mean* is the mean over the group's source rows — not
        # the mean/sum of the displayed cells.
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            res = asyncio.run(_svc(home).pivot(PivotRequest(
                path="s.parquet", rows=["region"], columns=["product"],
                measures=[AggMeasure(column="rev", agg="mean")], totals=True)))
            by = {r[0]: r for r in res.rows}
            self.assertEqual(by["EU"][3], 60.0)         # mean(70,80,30) over EU rows
            self.assertEqual(by["Total"][3], 56.25)     # grand mean of all 8 rows

    def test_totals_off(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _sales(home)
            res = asyncio.run(_svc(home).pivot(PivotRequest(
                path="s.parquet", rows=["region"], columns=["product"],
                measures=[AggMeasure(column="rev", agg="sum")], totals=False)))
            self.assertEqual(res.columns, ["region", "A", "B"])
            self.assertEqual(res.total_columns, 0)
            self.assertFalse(res.has_total_row)
            self.assertNotIn("Total", [r[0] for r in res.rows])


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

    def test_column_equals_x_no_duplicate_error(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"t": list(range(10))}), str(home / "s.parquet"))
            res = asyncio.run(_svc(home).series(SeriesRequest(path="s.parquet", column="t", x="t", points=100)))
            self.assertEqual(res.source_rows, 10)


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

    def test_column_equals_x_no_duplicate_error(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"t": [float(i) for i in range(20)]}), str(home / "p.parquet"))
            res = asyncio.run(_svc(home).ohlc(OhlcRequest(path="p.parquet", column="t", x="t", buckets=4)))
            self.assertEqual(res.bars, 4)


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


def _series(home: Path, name="s.parquet", n=80) -> None:
    import math
    ts = list(range(n))
    val = [100.0 + 0.7 * t + 8.0 * math.sin(2 * math.pi * t / 12) for t in ts]
    grp = ["a" if t % 2 == 0 else "b" for t in ts]
    pq.write_table(pa.table({"ts": ts, "value": val, "grp": grp}), str(home / name))


class TestForecast(unittest.TestCase):
    def test_single_series_history_and_band(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _series(home)
            res = asyncio.run(_svc(home).forecast(ForecastRequest(
                path="s.parquet", column="value", x="ts", horizon=10, model="ridge")))
            self.assertEqual(len(res.series), 1)
            s = res.series[0]
            self.assertEqual(len(s.forecast_x), 10)
            self.assertEqual(len(s.forecast_y), 10)
            # band brackets the point forecast and widens with the horizon
            self.assertTrue(all(lo <= p <= up for lo, p, up in zip(s.lower, s.forecast_y, s.upper)))
            self.assertGreaterEqual(s.upper[-1] - s.lower[-1], s.upper[0] - s.lower[0])
            # future x extrapolates beyond the last observed step
            self.assertGreater(s.forecast_x[0], 79)
            self.assertEqual(res.model_used, "ridge")

    def test_per_group_forecast(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _series(home)
            res = asyncio.run(_svc(home).forecast(ForecastRequest(
                path="s.parquet", column="value", x="ts", group="grp",
                horizon=5, model="ridge", max_groups=4)))
            keys = {s.key for s in res.series}
            self.assertEqual(keys, {"a", "b"})
            self.assertTrue(all(len(s.forecast_y) == 5 for s in res.series))

    def test_short_series_naive_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"v": [1.0, 2.0, 3.0]}), str(home / "tiny.parquet"))
            res = asyncio.run(_svc(home).forecast(ForecastRequest(
                path="tiny.parquet", column="v", horizon=4)))
            self.assertEqual(res.model_used, "naive")
            self.assertEqual(res.series[0].forecast_y, [3.0, 3.0, 3.0, 3.0])

    def test_value_equals_x_column_no_crash(self):
        # forecasting a column that shares the x/group name must not trip
        # polars' duplicate-column error (aliased aggregation).
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _series(home)
            res = asyncio.run(_svc(home).forecast(ForecastRequest(
                path="s.parquet", column="ts", x="ts", horizon=3, model="ridge")))
            self.assertEqual(len(res.series[0].forecast_y), 3)

    def test_bad_column_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _series(home)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).forecast(ForecastRequest(path="s.parquet", column="nope")))


def _price_walk(home: Path, name: str, n=120, seed=0, drift=0.001) -> None:
    import numpy as np
    rng = np.random.default_rng(seed)
    px = 100.0 * np.cumprod(1.0 + rng.normal(drift, 0.01, n))
    pq.write_table(pa.table({"t": list(range(n)), "price": px.tolist()}), str(home / name))


class TestIndicators(unittest.TestCase):
    def test_all_indicators_present_and_named(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _price_walk(home, "p.parquet")
            res = asyncio.run(_svc(home).indicators(IndicatorsRequest(
                path="p.parquet", column="price", order_by="t",
                indicators=["rsi", "macd", "bb", "atr", "stoch"], window=14)))
            names = [s.name for s in res.indicators]
            self.assertEqual(names, [
                "rsi", "macd_line", "macd_signal", "macd_hist",
                "bb_upper", "bb_mid", "bb_lower", "atr", "stoch_k", "stoch_d"])
            self.assertEqual(len(res.price), 120)
            self.assertEqual(len(res.index), 120)
            self.assertFalse(res.truncated)
            for s in res.indicators:
                self.assertEqual(len(s.values), 120)

    def test_rsi_bounded_0_100(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _price_walk(home, "p.parquet")
            res = asyncio.run(_svc(home).indicators(IndicatorsRequest(
                path="p.parquet", column="price", indicators=["rsi"], window=14)))
            rsi = res.indicators[0].values
            self.assertIsNone(rsi[0])                          # warmup window null
            self.assertTrue(all(v is None or 0.0 <= v <= 100.0 for v in rsi))

    def test_bb_bands_ordered(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _price_walk(home, "p.parquet")
            res = asyncio.run(_svc(home).indicators(IndicatorsRequest(
                path="p.parquet", column="price", indicators=["bb"], window=20)))
            up = {s.name: s.values for s in res.indicators}
            for u, m, lo in zip(up["bb_upper"], up["bb_mid"], up["bb_lower"]):
                if None not in (u, m, lo):
                    self.assertGreaterEqual(u, m)
                    self.assertGreaterEqual(m, lo)

    def test_unknown_indicator_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _price_walk(home, "p.parquet")
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).indicators(IndicatorsRequest(
                    path="p.parquet", column="price", indicators=["bogus"])))

    def test_missing_column_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _price_walk(home, "p.parquet")
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).indicators(IndicatorsRequest(
                    path="p.parquet", column="ghost")))


class TestPortfolio(unittest.TestCase):
    def test_multi_asset_metrics_and_correlation(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _price_walk(home, "a.parquet", seed=1)
            _price_walk(home, "b.parquet", seed=2)
            _price_walk(home, "c.parquet", seed=3)
            res = asyncio.run(_svc(home).portfolio(PortfolioRequest(
                paths=["a.parquet", "b.parquet", "c.parquet"],
                columns=["price", "price", "price"], order_by="t")))
            self.assertEqual(res.labels, ["a", "b", "c"])
            self.assertEqual(len(res.correlation), 3)
            self.assertEqual(len(res.correlation[0]), 3)
            for i in range(3):
                self.assertAlmostEqual(res.correlation[i][i], 1.0, places=6)
            self.assertEqual(len(res.assets), 3)
            for a in res.assets:
                self.assertIsNotNone(a.sharpe)
                self.assertIsNotNone(a.beta)
                self.assertIsNotNone(a.max_drawdown)

    def test_cvar_at_or_below_var(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _price_walk(home, "a.parquet", seed=1)
            _price_walk(home, "b.parquet", seed=2)
            res = asyncio.run(_svc(home).portfolio(PortfolioRequest(
                paths=["a.parquet", "b.parquet"], columns=["price", "price"],
                order_by="t", confidence=0.95)))
            self.assertIsNotNone(res.var_95)
            self.assertLessEqual(res.cvar_95, res.var_95)

    def test_aligns_to_shortest_series(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _price_walk(home, "a.parquet", n=120, seed=1)
            _price_walk(home, "b.parquet", n=80, seed=2)
            res = asyncio.run(_svc(home).portfolio(PortfolioRequest(
                paths=["a.parquet", "b.parquet"], columns=["price", "price"], order_by="t")))
            self.assertEqual(len(res.index), 80)              # min length
            self.assertEqual(len(res.prices[0]), 80)
            self.assertEqual(len(res.prices[1]), 80)

    def test_default_labels_from_file_stem(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _price_walk(home, "tsla.parquet", seed=1)
            _price_walk(home, "aapl.parquet", seed=2)
            res = asyncio.run(_svc(home).portfolio(PortfolioRequest(
                paths=["tsla.parquet", "aapl.parquet"], columns=["price", "price"],
                labels=["TSLA"])))                            # second falls back to stem
            self.assertEqual(res.labels, ["TSLA", "aapl"])

    def test_length_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d); _price_walk(home, "a.parquet")
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).portfolio(PortfolioRequest(
                    paths=["a.parquet"], columns=["price", "price"])))

    def test_too_many_assets_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            paths = [f"p{i}.parquet" for i in range(9)]
            for i, p in enumerate(paths):
                _price_walk(home, p, seed=i)
            with self.assertRaises(BadRequestError):
                asyncio.run(_svc(home).portfolio(PortfolioRequest(
                    paths=paths, columns=["price"] * 9)))


if __name__ == "__main__":
    unittest.main()
