"""PySpark integration tests for ArrayType.

Exercises ``cast_spark_list_column`` and the Spark-side type helpers
against a live SparkSession.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested.array import (
    ArrayType,
    cast_spark_list_column,
)
from yggdrasil.data.types.primitive import IntegerType

pytest.importorskip("pyspark")

from yggdrasil.spark.tests import spark  # noqa: E402,F401  — session-scoped fixture


def test_to_spark_emits_array_type(spark) -> None:  # noqa: F811
    pst = spark.createDataFrame([], schema="a string").schema.__class__.__module__
    del pst  # keep pyspark imported lazily via fixture
    from pyspark.sql.types import ArrayType as SparkArrayType

    dtype = ArrayType(
        item_field=Field(
            name="item",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=True,
        ),
    )
    spark_dtype = dtype.to_spark()

    assert isinstance(spark_dtype, SparkArrayType)


def test_from_spark_type_preserves_item_dtype(spark) -> None:  # noqa: F811
    from pyspark.sql.types import ArrayType as SparkArrayType, LongType

    rebuilt = ArrayType.from_spark_type(SparkArrayType(LongType(), containsNull=False))

    assert isinstance(rebuilt, ArrayType)
    assert rebuilt.item_field.dtype.type_id == DataTypeId.INTEGER
    assert rebuilt.item_field.nullable is False


def test_handles_spark_type_matches_array_only(spark) -> None:  # noqa: F811
    from pyspark.sql.types import ArrayType as SparkArrayType, LongType

    assert ArrayType.handles_spark_type(SparkArrayType(LongType())) is True
    assert ArrayType.handles_spark_type(LongType()) is False


def test_from_spark_type_rejects_non_array(spark) -> None:  # noqa: F811
    from pyspark.sql.types import LongType

    with pytest.raises(TypeError, match="Unsupported Spark data type"):
        ArrayType.from_spark_type(LongType())


def test_cast_spark_list_column_changes_item_dtype(
    spark,  # noqa: F811
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    import pyspark.sql.functions as F

    frame = spark.createDataFrame(
        [([1, 2],), ([3, None],), (None,)],
        schema="source_array array<bigint>",
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = frame.select(
        cast_spark_list_column(F.col("source_array"), options).alias("target_array")
    )

    rows = result.toArrow()["target_array"].to_pylist()
    assert rows == [["1", "2"], ["3", None], None]


def test_cast_spark_list_column_returns_input_when_target_is_none(
    spark,  # noqa: F811
    source_array_field: Field,
) -> None:
    import pyspark.sql.functions as F

    column = F.col("source_array")
    options = CastOptions(source_field=source_array_field, target_field=None)

    assert cast_spark_list_column(column, options) is column


def test_cast_spark_list_column_raises_for_non_array_source(
    spark,  # noqa: F811
    source_map_field: Field,
    target_array_field: Field,
) -> None:
    import pyspark.sql.functions as F

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_array_field,
    )

    with pytest.raises(TypeError, match="Cannot cast"):
        cast_spark_list_column(F.col("anything"), options)
