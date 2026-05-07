"""Spark → Arrow conversion tests.

Exercises the Spark branch of :func:`yggdrasil.arrow.cast.any_to_arrow_table`
(which reaches :func:`_spark_to_arrow`) and the streaming wrapper that
prefers ``DataFrame.toArrowBatchIterator`` when available. Goal is to
confirm:

* DataFrame → ``pa.Table`` round-trips data and column names.
* In-engine cast (``CastOptions.cast_spark``) fuses into the Spark
  plan before ``toArrow()`` materializes anything.
* Target-field projection rewrites the plan via ``DataFrame.select``.
* The streaming entry point uses the Spark-native batch iterator when
  pyspark is recent enough, and otherwise falls back to the bulk path.

Multi-inherits :class:`SparkTestCase` and :class:`ArrowTestCase`. The
class is skipped cleanly when PySpark is not installed (which is the
default on this repo's local env — Spark integration tests run in CI).
"""
from __future__ import annotations

import unittest

from yggdrasil.arrow.cast import (
    any_to_arrow_batch_iterator,
    any_to_arrow_record_batch,
    any_to_arrow_record_batch_reader,
    any_to_arrow_table,
)
from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.spark.tests import SparkTestCase


class TestSparkDataFrameToArrow(SparkTestCase, ArrowTestCase):
    def test_basic_dataframe(self) -> None:
        df = self.df([(1, "x"), (2, "y"), (3, "z")], schema=["a", "b"])
        out = any_to_arrow_table(df)
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column_names), {"a", "b"})
        self.assertEqual(out["a"].to_pylist(), [1, 2, 3])
        self.assertEqual(out["b"].to_pylist(), ["x", "y", "z"])

    def test_empty_dataframe(self) -> None:
        from pyspark.sql.types import StructType, StructField, IntegerType, StringType

        schema = StructType([
            StructField("a", IntegerType(), True),
            StructField("b", StringType(), True),
        ])
        df = self.spark.createDataFrame([], schema=schema)
        out = any_to_arrow_table(df)
        self.assertEqual(out.num_rows, 0)
        self.assertEqual(set(out.column_names), {"a", "b"})

    def test_target_field_casts_int_to_int32(self) -> None:
        df = self.df([(1,), (2,), (3,)], schema=["a"])
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        out = any_to_arrow_table(df, CastOptions(target_field=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())
        self.assertEqual(out["a"].to_pylist(), [1, 2, 3])

    def test_target_field_projects_subset(self) -> None:
        df = self.df([(1, "x", True), (2, "y", False)], schema=["a", "b", "c"])
        target = Schema.from_arrow(self.pa.schema([
            self.pa.field("c", self.pa.bool_()),
            self.pa.field("a", self.pa.int64()),
        ]))
        out = any_to_arrow_table(df, CastOptions(target_field=target))
        self.assertEqual(out.column_names, ["c", "a"])

    def test_record_batch_path(self) -> None:
        df = self.df([(1, "x"), (2, "y")], schema=["a", "b"])
        rb = any_to_arrow_record_batch(df)
        self.assertEqual(rb.num_rows, 2)
        self.assertEqual(set(rb.schema.names), {"a", "b"})

    def test_record_batch_reader(self) -> None:
        df = self.df([(1,), (2,), (3,)], schema=["a"])
        rbr = any_to_arrow_record_batch_reader(df)
        self.assertEqual(rbr.read_all().num_rows, 3)

    def test_batch_iterator_round_trips(self) -> None:
        df = self.df([(i,) for i in range(8)], schema=["a"])
        batches = list(any_to_arrow_batch_iterator(df))
        self.assertEqual(sum(b.num_rows for b in batches), 8)

    def test_batch_iterator_with_target_cast(self) -> None:
        df = self.df([(i,) for i in range(5)], schema=["a"])
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        batches = list(any_to_arrow_batch_iterator(df, CastOptions(target_field=target)))
        self.assertEqual(sum(b.num_rows for b in batches), 5)
        for b in batches:
            self.assertEqual(b.schema.field("a").type, self.pa.int32())


if __name__ == "__main__":
    unittest.main()
