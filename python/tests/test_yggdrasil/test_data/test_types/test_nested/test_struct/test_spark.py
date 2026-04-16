"""PySpark integration tests for StructType casts.

These exercise the three `cast_spark_*_column` helpers as well as
null-propagation (a real bug the previous monolith surfaced).
"""
from __future__ import annotations

import pytest

from yggdrasil.data import CastOptions, Field
from yggdrasil.data.types.nested.struct import (
    cast_spark_list_column,
    cast_spark_map_column,
    cast_spark_struct_column,
)

from ._helpers import normalize_nested

pytest.importorskip("pyspark")

from yggdrasil.spark.tests import spark  # noqa: E402,F401


# ---------------------------------------------------------------------------
# struct -> struct
# ---------------------------------------------------------------------------


def test_cast_spark_struct_column_reorders_fields_and_fills_missing(
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

    assert [
        normalize_nested(v) for v in result.toPandas()["target_struct"].tolist()
    ] == [
        {"b": "x", "c": None, "a": 1},
        {"b": "y", "c": None, "a": 2},
        None,
    ]


# ---------------------------------------------------------------------------
# map -> struct
# ---------------------------------------------------------------------------


def test_cast_spark_map_column_extracts_named_keys_to_struct(
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
        cast_spark_map_column(F.col("source_map"), options).alias("target_struct")
    )

    assert [
        normalize_nested(v) for v in result.toPandas()["target_struct"].tolist()
    ] == [
        {"b": "2", "c": None, "a": 1},
        {"b": "3", "c": None, "a": None},
        None,
    ]


# ---------------------------------------------------------------------------
# list -> struct
# ---------------------------------------------------------------------------


def test_cast_spark_list_column_maps_by_position_and_fills_missing(
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
        cast_spark_list_column(F.col("source_list"), options).alias("target_struct")
    )

    assert [
        normalize_nested(v) for v in result.toArrow()["target_struct"].to_pylist()
    ] == [
        {"first": 1, "second": "2", "third": 3},
        {"first": 4, "second": None, "third": None},
        None,
    ]


# ---------------------------------------------------------------------------
# Regression tests for the null-propagation + out-of-bounds fixes (5d2d3a0).
# ---------------------------------------------------------------------------


def test_cast_spark_struct_column_preserves_null_source_rows(
    spark,  # noqa: F811
    source_struct_field: Field,
    target_struct_field: Field,
) -> None:
    """NULL input rows must stay NULL — not become all-null structs."""
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

    rows = result.toPandas()["target_struct"].tolist()
    assert rows[0] is None


def test_cast_spark_list_column_handles_short_lists_without_oob(
    spark,  # noqa: F811
    source_list_field: Field,
    target_list_to_struct_field: Field,
) -> None:
    """List shorter than target struct fills with NULLs instead of raising."""
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
        cast_spark_list_column(F.col("source_list"), options).alias("target_struct")
    )

    rows = [
        normalize_nested(v) for v in result.toArrow()["target_struct"].to_pylist()
    ]
    assert rows == [
        {"first": 1, "second": None, "third": None},
        {"first": 2, "second": "3", "third": None},
        {"first": 4, "second": "5", "third": 6},
    ]
