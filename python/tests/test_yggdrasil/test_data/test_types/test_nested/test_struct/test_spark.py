"""Spark-side casts targeting :class:`StructType`.

Each helper builds a Spark Column expression that reshapes the source
into the target struct. Tests live or die on a real ``SparkSession``,
so the whole module is skipped when pyspark isn't installed.

Three regressions get explicit coverage at the bottom: NULL source
rows must stay NULL (not become all-NULL structs), and short list
sources must back-fill missing positions instead of raising
out-of-bounds.
"""
from __future__ import annotations

import pytest

from yggdrasil.data import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested.struct import (
    cast_spark_list_column,
    cast_spark_map_column,
    cast_spark_struct_column,
)
from ._helpers import normalize_nested

pytest.importorskip("pyspark")

from yggdrasil.spark.tests import spark  # noqa: E402,F401
from yggdrasil.spark.cast import spark_dataframe_to_arrow  # noqa: E402


def _read_struct_column(frame, column: str) -> list:
    """Pull a struct column out of Spark as a list of normalized dicts."""
    return [
        normalize_nested(v) for v in frame.toPandas()[column].tolist()
    ]


# ---------------------------------------------------------------------------
# struct → struct
# ---------------------------------------------------------------------------


class TestCastStructColumn:
    def test_reorders_fields_and_fills_missing(
        self,
        spark,  # noqa: F811
        source_struct_field: Field,
        target_struct_field: Field,
    ) -> None:
        import pyspark.sql.functions as F

        frame = spark.createDataFrame(
            [({"a": 1, "b": "x"},), ({"a": 2, "b": "y"},), (None,)],
            schema="source_struct struct<a:bigint,b:string>",
        )

        options = CastOptions(
            source_field=source_struct_field,
            target_field=target_struct_field,
        )

        result = frame.select(
            cast_spark_struct_column(F.col("source_struct"), options).alias(
                "target_struct"
            )
        )

        assert _read_struct_column(result, "target_struct") == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            None,
        ]

    def test_preserves_null_source_rows(
        self,
        spark,  # noqa: F811
        source_struct_field: Field,
        target_struct_field: Field,
    ) -> None:
        """A NULL row must stay NULL — not become an all-null struct."""
        import pyspark.sql.functions as F

        frame = spark.createDataFrame(
            [(None,), ({"a": 1, "b": "x"},)],
            schema="source_struct struct<a:bigint,b:string>",
        )

        options = CastOptions(
            source_field=source_struct_field,
            target_field=target_struct_field,
        )

        result = frame.select(
            cast_spark_struct_column(F.col("source_struct"), options).alias(
                "target_struct"
            )
        )

        rows = _read_struct_column(result, "target_struct")
        assert rows[0] is None
        assert rows[1] == {"b": "x", "c": None, "a": 1}


# ---------------------------------------------------------------------------
# map → struct
# ---------------------------------------------------------------------------


class TestCastMapColumn:
    def test_extracts_named_keys(
        self,
        spark,  # noqa: F811
        source_map_field: Field,
        target_struct_field: Field,
    ) -> None:
        import pyspark.sql.functions as F

        frame = spark.createDataFrame(
            [({"a": 1, "b": 2},), ({"b": 3},), (None,)],
            schema="source_map map<string,bigint>",
        )

        options = CastOptions(
            source_field=source_map_field,
            target_field=target_struct_field,
        )

        result = frame.select(
            cast_spark_map_column(F.col("source_map"), options).alias(
                "target_struct"
            )
        )

        assert _read_struct_column(result, "target_struct") == [
            {"b": "2", "c": None, "a": 1},
            {"b": "3", "c": None, "a": None},
            None,
        ]


# ---------------------------------------------------------------------------
# list → struct
# ---------------------------------------------------------------------------


class TestCastListColumn:
    def test_maps_by_position_and_fills_missing(
        self,
        spark,  # noqa: F811
        source_list_field: Field,
        target_list_to_struct_field: Field,
    ) -> None:
        import pyspark.sql.functions as F

        frame = spark.createDataFrame(
            [([1, 2, 3],), ([4],), (None,)],
            schema="source_list array<bigint>",
        )

        options = CastOptions(
            source_field=source_list_field,
            target_field=target_list_to_struct_field,
        )

        result = frame.select(
            cast_spark_list_column(F.col("source_list"), options).alias(
                "target_struct"
            )
        )

        rows = [
            normalize_nested(v)
            for v in spark_dataframe_to_arrow(result)["target_struct"].to_pylist()
        ]
        assert rows == [
            {"first": 1, "second": "2", "third": 3},
            {"first": 4, "second": None, "third": None},
            None,
        ]

    def test_handles_short_lists_without_oob(
        self,
        spark,  # noqa: F811
        source_list_field: Field,
        target_list_to_struct_field: Field,
    ) -> None:
        """A list shorter than the target struct fills with NULLs."""
        import pyspark.sql.functions as F

        frame = spark.createDataFrame(
            [([1],), ([2, 3],), ([4, 5, 6],)],
            schema="source_list array<bigint>",
        )

        options = CastOptions(
            source_field=source_list_field,
            target_field=target_list_to_struct_field,
        )

        result = frame.select(
            cast_spark_list_column(F.col("source_list"), options).alias(
                "target_struct"
            )
        )

        rows = [
            normalize_nested(v)
            for v in spark_dataframe_to_arrow(result)["target_struct"].to_pylist()
        ]
        assert rows == [
            {"first": 1, "second": None, "third": None},
            {"first": 2, "second": "3", "third": None},
            {"first": 4, "second": "5", "third": 6},
        ]
