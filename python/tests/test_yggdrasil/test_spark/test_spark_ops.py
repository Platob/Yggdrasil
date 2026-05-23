"""Tests for :mod:`yggdrasil.spark.ops` — dedup / resample / fill on SparkDataFrame.

Mirrors the arrow ops coverage: the spark versions share the
contract (same parameter names, same vocabulary for
``fill_strategy``) so a regression in either path is caught here.

Local-only ``SparkSession`` fixture — skips cleanly when pyspark
is missing or the JVM mode is unreachable.
"""
from __future__ import annotations

import datetime as dt
import os
import unittest


def _local_spark():
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise unittest.SkipTest("pyspark not installed")

    os.environ.pop("DATABRICKS_HOST", None)
    jvm_opens = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.misc=ALL-UNNAMED"
    )
    try:
        return (
            SparkSession.builder
            .master("local[2]")
            .appName("ygg-test-spark-ops")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .config("spark.driver.extraJavaOptions", jvm_opens)
            .config("spark.executor.extraJavaOptions", jvm_opens)
            .getOrCreate()
        )
    except RuntimeError as exc:
        raise unittest.SkipTest(f"local SparkSession unavailable: {exc}")


class TestSparkOps(unittest.TestCase):
    """Dedup / resample / fill on a real local SparkSession."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = _local_spark()
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        # Don't stop — other test modules share the JVM.
        pass

    def _ts_frame(self, rows):
        import pyspark.sql.types as T
        schema = T.StructType([
            T.StructField("sym", T.StringType()),
            T.StructField("ts", T.TimestampType()),
            T.StructField("v", T.LongType()),
        ])
        return self.spark.createDataFrame(rows, schema=schema)

    def test_dedup_keeps_first_occurrence(self) -> None:
        from yggdrasil.spark.ops import dedup_spark_dataframe
        import pyspark.sql.types as T

        schema = T.StructType([
            T.StructField("id", T.LongType()),
            T.StructField("v", T.StringType()),
        ])
        df = self.spark.createDataFrame(
            [(1, "a"), (2, "b"), (1, "c"), (3, "d"), (2, "e")],
            schema=schema,
        )
        out = dedup_spark_dataframe(df, ["id"]).orderBy("id").collect()
        assert [(r.id, r.v) for r in out] == [(1, "a"), (2, "b"), (3, "d")]

    def test_dedup_missing_key_raises(self) -> None:
        from yggdrasil.spark.ops import dedup_spark_dataframe
        df = self.spark.createDataFrame([(1,)], schema="id long")
        with self.assertRaisesRegex(ValueError, "missing"):
            dedup_spark_dataframe(df, ["nope"])

    def test_dedup_empty_keys_short_circuits(self) -> None:
        from yggdrasil.spark.ops import dedup_spark_dataframe
        df = self.spark.createDataFrame([(1, "a"), (1, "b")], schema="id long, v string")
        assert dedup_spark_dataframe(df, []) is df

    def test_resample_flat_with_ffill(self) -> None:
        from yggdrasil.spark.ops import resample_spark_dataframe
        import pyspark.sql.types as T

        ts_rows = [
            (dt.datetime(2024, 1, 1, h), (h + 1) * 10 if h in (0, 3) else None)
            for h in range(6)
        ]
        df = self.spark.createDataFrame(
            ts_rows,
            schema=T.StructType([
                T.StructField("ts", T.TimestampType()),
                T.StructField("v", T.LongType()),
            ]),
        )
        out = resample_spark_dataframe(df, time_column="ts", sampling_seconds=7200)
        rows = sorted(out.collect(), key=lambda r: r.ts)
        # 2h buckets pick first row per bucket: ts=00 → 10, ts=02 → None,
        # ts=04 → None. ffill carries 10 forward.
        assert [r.v for r in rows] == [10, 10, 10]

    def test_resample_partitioned_no_cross_partition_leak(self) -> None:
        from yggdrasil.spark.ops import resample_spark_dataframe

        rows = []
        for sym in ("A", "B"):
            for h in range(6):
                if sym == "A":
                    v = (h + 1) * 10 if h in (0, 3) else None
                else:
                    v = (h + 1) * 10 if h in (4,) else None
                rows.append((sym, dt.datetime(2024, 1, 1, h), v))
        df = self._ts_frame(rows)
        out = resample_spark_dataframe(
            df, time_column="ts", sampling_seconds=7200,
            partition_by=["sym"],
        )
        collected = sorted(out.collect(), key=lambda r: (r.sym, r.ts))
        per_sym = {"A": [], "B": []}
        for r in collected:
            per_sym[r.sym].append(r.v)
        # A's buckets: [10, None, None] → ffill → [10, 10, 10].
        # B's buckets: [None, None, 50] (50 lives in the 04-05 bucket) →
        # leading nulls have no prior non-null in B → stay null, third
        # bucket carries 50.
        assert per_sym["A"] == [10, 10, 10]
        assert per_sym["B"] == [None, None, 50]

    def test_resample_no_fill_keeps_bucket_nulls(self) -> None:
        from yggdrasil.spark.ops import resample_spark_dataframe

        rows = [
            ("A", dt.datetime(2024, 1, 1, h), 10 if h == 0 else None)
            for h in range(4)
        ]
        df = self._ts_frame(rows)
        out = resample_spark_dataframe(
            df, time_column="ts", sampling_seconds=7200,
            partition_by=["sym"], fill_strategy="none",
        )
        rows_out = sorted(out.collect(), key=lambda r: r.ts)
        # 2h buckets: ts=00 → 10 (from h=0), ts=02 → None (from h=2).
        assert [r.v for r in rows_out] == [10, None]

    def test_resample_short_circuits_when_disabled(self) -> None:
        from yggdrasil.spark.ops import resample_spark_dataframe
        df = self._ts_frame([("A", dt.datetime(2024, 1, 1), 1)])
        assert resample_spark_dataframe(df, time_column="ts", sampling_seconds=0) is df
        assert resample_spark_dataframe(df, time_column="missing", sampling_seconds=60) is df

    def test_dataset_unique_returns_dataset(self) -> None:
        """The cross-engine :meth:`Tabular.unique` routes via
        :meth:`_native_spark_frame` and returns a fresh ``Dataset``
        instead of collecting through Arrow."""
        from yggdrasil.spark.tabular import Dataset

        df = self.spark.createDataFrame(
            [(1, "a"), (2, "b"), (1, "c"), (3, "d")],
            schema="id long, v string",
        )
        ds = Dataset(frame=df)
        out = ds.unique("id")
        assert isinstance(out, Dataset)
        assert out is not ds  # fresh holder
        rows = sorted(out.frame.collect(), key=lambda r: r.id)
        assert [(r.id, r.v) for r in rows] == [(1, "a"), (2, "b"), (3, "d")]

    def test_dataset_resample_returns_dataset_with_ffill(self) -> None:
        from yggdrasil.spark.tabular import Dataset

        rows = [
            ("A", dt.datetime(2024, 1, 1, h), (h + 1) * 10 if h in (0, 3) else None)
            for h in range(6)
        ]
        ds = Dataset(frame=self._ts_frame(rows))
        out = ds.resample(
            on="ts",
            sampling=dt.timedelta(hours=2),
            partition_by="sym",
        )
        assert isinstance(out, Dataset)
        collected = sorted(out.frame.collect(), key=lambda r: r.ts)
        # 2h buckets, first per bucket: [10, None, None] → ffill → [10, 10, 10].
        assert [r.v for r in collected] == [10, 10, 10]

    def test_dataset_resample_accepts_iso_duration_string(self) -> None:
        from yggdrasil.spark.tabular import Dataset

        rows = [("A", dt.datetime(2024, 1, 1, h), h) for h in range(4)]
        ds = Dataset(frame=self._ts_frame(rows))
        out = ds.resample(on="ts", sampling="PT2H", partition_by=["sym"])
        collected = sorted(out.frame.collect(), key=lambda r: r.ts)
        assert [r.v for r in collected] == [0, 2]

    def test_dataset_select_returns_dataset(self) -> None:
        from yggdrasil.spark.tabular import Dataset

        df = self.spark.createDataFrame(
            [(1, "x", 10), (2, "y", 20), (3, "z", 30)],
            schema="a long, b string, c long",
        )
        out = Dataset(frame=df).select("a", "c")
        assert isinstance(out, Dataset)
        assert out.frame.columns == ["a", "c"]
        rows = sorted(out.frame.collect(), key=lambda r: r.a)
        assert [(r.a, r.c) for r in rows] == [(1, 10), (2, 20), (3, 30)]

    def test_dataset_drop_returns_dataset(self) -> None:
        from yggdrasil.spark.tabular import Dataset

        df = self.spark.createDataFrame(
            [(1, "x", 10), (2, "y", 20)],
            schema="a long, b string, c long",
        )
        out = Dataset(frame=df).drop("b")
        assert isinstance(out, Dataset)
        assert out.frame.columns == ["a", "c"]

    def test_dataset_drop_missing_column_is_no_op(self) -> None:
        from yggdrasil.spark.tabular import Dataset
        df = self.spark.createDataFrame([(1,)], schema="a long")
        out = Dataset(frame=df).drop("nope")
        assert out.frame.columns == ["a"]

    def test_dataset_filter_sql_string_native_path(self) -> None:
        from yggdrasil.spark.tabular import Dataset

        df = self.spark.createDataFrame(
            [(1,), (2,), (3,), (4,)], schema="a long",
        )
        out = Dataset(frame=df).filter("a > 2")
        assert isinstance(out, Dataset)
        rows = sorted(out.frame.collect(), key=lambda r: r.a)
        assert [r.a for r in rows] == [3, 4]

    def test_dataset_filter_yggdrasil_expression_native_path(self) -> None:
        from yggdrasil.execution.expr import col
        from yggdrasil.spark.tabular import Dataset

        df = self.spark.createDataFrame(
            [(1, "x"), (2, "y"), (3, "x")], schema="a long, b string",
        )
        out = Dataset(frame=df).filter(col("b") == "x")
        rows = sorted(out.frame.collect(), key=lambda r: r.a)
        assert [(r.a, r.b) for r in rows] == [(1, "x"), (3, "x")]

    def test_dataset_filter_callable_still_works(self) -> None:
        """Legacy callable filter path is preserved — ``Dataset.filter``
        dispatches by argument type."""
        from yggdrasil.spark.tabular import Dataset
        from yggdrasil.data import field, schema
        from yggdrasil.data.types.primitive import Int64Type

        out_schema = schema([field("a", Int64Type, nullable=False)])
        df = self.spark.createDataFrame([(1,), (2,), (3,)], schema=out_schema.to_spark_schema())
        out = Dataset(frame=df, schema=out_schema).filter(
            lambda r: r["a"] >= 2, schema=out_schema,
        )
        assert isinstance(out, Dataset)
        rows = sorted(out.frame.collect(), key=lambda r: r.a)
        assert [r.a for r in rows] == [2, 3]

    def test_fill_spark_dataframe_ffill_per_partition(self) -> None:
        from yggdrasil.spark.ops import fill_spark_dataframe

        rows = [
            ("A", dt.datetime(2024, 1, 1, 0), 1),
            ("A", dt.datetime(2024, 1, 1, 1), None),
            ("A", dt.datetime(2024, 1, 1, 2), None),
            ("B", dt.datetime(2024, 1, 1, 0), None),
            ("B", dt.datetime(2024, 1, 1, 1), 5),
            ("B", dt.datetime(2024, 1, 1, 2), None),
        ]
        df = self._ts_frame(rows)
        out = fill_spark_dataframe(
            df, sort_by="ts", partition_by=["sym"], fill_strategy="ffill",
        )
        collected = sorted(out.collect(), key=lambda r: (r.sym, r.ts))
        vals = [(r.sym, r.v) for r in collected]
        assert vals == [
            ("A", 1), ("A", 1), ("A", 1),
            ("B", None), ("B", 5), ("B", 5),
        ]
