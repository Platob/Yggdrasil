"""Regression coverage for :func:`yggdrasil.spark.cast.any_to_spark_dataframe`.

Spark Connect's Arrow gRPC transport rejects ``LargeList`` /
``large_string`` / ``large_binary`` / ``list_view`` with
``[UNSUPPORTED_ARROWTYPE] Unsupported arrow type LargeList. SQLSTATE:
0A000``. The cast path projects the casted schema through
:meth:`Schema.as_spark` and downcasts the Arrow table to the
Spark-compatible counterparts before handing it to
``createDataFrame``.

Tests run against a real local SparkSession (via
:class:`SparkTestCase` — skipped cleanly when PySpark is not installed
or the session can't initialize).
"""

from __future__ import annotations

import pyarrow as pa
import pyspark.sql as pyspark_sql

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.cast import convert
from yggdrasil.spark.tests import SparkTestCase


class TestAnyToSparkLargeArrowTypes(SparkTestCase, ArrowTestCase):
    """Arrow tables with "large" / "view" flavors flow into Spark."""

    def test_large_list_of_large_string_downcasts(self) -> None:
        # The exact shape that surfaces ``UNSUPPORTED_ARROWTYPE`` from
        # Spark Connect when sent without the ``as_spark`` rewrite.
        table = pa.table({
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "tags": pa.array(
                [["a", "b"], ["c"], []],
                type=pa.large_list(pa.large_string()),
            ),
        })

        df = convert(table, pyspark_sql.DataFrame)

        self.assertIsInstance(df, pyspark_sql.DataFrame)
        rows = sorted(df.collect(), key=lambda r: r["id"])
        self.assertEqual([r["id"] for r in rows], [1, 2, 3])
        self.assertEqual([list(r["tags"]) for r in rows], [["a", "b"], ["c"], []])

    def test_large_binary_downcasts(self) -> None:
        table = pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "blob": pa.array([b"hello", b"world"], type=pa.large_binary()),
        })

        df = convert(table, pyspark_sql.DataFrame)

        rows = sorted(df.collect(), key=lambda r: r["id"])
        self.assertEqual([bytes(r["blob"]) for r in rows], [b"hello", b"world"])

    def test_plain_list_passes_through_unchanged(self) -> None:
        # Sanity: when the source is already Spark-compatible the
        # schema-equality short-circuit fires and the cast is a no-op.
        table = pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "tags": pa.array([["a"], ["b", "c"]], type=pa.list_(pa.string())),
        })

        df = convert(table, pyspark_sql.DataFrame)

        rows = sorted(df.collect(), key=lambda r: r["id"])
        self.assertEqual([list(r["tags"]) for r in rows], [["a"], ["b", "c"]])
