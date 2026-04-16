"""PySpark integration tests for MapType casts."""
from __future__ import annotations

import pytest

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import MapType
from yggdrasil.data.types.nested.map import (
    cast_spark_list_column_to_map,
    cast_spark_map_column,
    cast_spark_struct_column_to_map,
)

from ._helpers import normalize_map_like

pytest.importorskip("pyspark")

from yggdrasil.spark.tests import spark  # noqa: E402,F401


def test_to_spark_emits_map_type(spark, int64_type, string_type) -> None:  # noqa: F811
    from pyspark.sql.types import MapType as SparkMapType

    dtype = MapType.from_key_value(string_type, int64_type)
    assert isinstance(dtype.to_spark(), SparkMapType)


def test_handles_spark_type_matches_map_only(spark) -> None:  # noqa: F811
    from pyspark.sql.types import MapType as SparkMapType, LongType, StringType

    assert MapType.handles_spark_type(SparkMapType(StringType(), LongType())) is True
    assert MapType.handles_spark_type(LongType()) is False


def test_from_spark_type_preserves_keys_and_values(spark) -> None:  # noqa: F811
    from pyspark.sql.types import MapType as SparkMapType, LongType, StringType

    rebuilt = MapType.from_spark_type(SparkMapType(StringType(), LongType()))

    assert isinstance(rebuilt, MapType)
    assert rebuilt.key_field.name == "key"
    assert rebuilt.value_field.name == "value"


def test_cast_spark_map_column_recasts_values(
    spark,  # noqa: F811
    source_map_field: Field,
    target_map_field: Field,
) -> None:
    import pyspark.sql.functions as F

    frame = spark.createDataFrame(
        [({"a": 1, "b": 2},), ({"x": 3},), (None,)],
        schema="source_map map<string,bigint>",
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_map_field,
    )

    result = frame.select(
        cast_spark_map_column(F.col("source_map"), options).alias("target_map")
    )

    assert [
        normalize_map_like(v) for v in result.toPandas()["target_map"].tolist()
    ] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_spark_list_column_to_map(
    spark,  # noqa: F811
    source_list_of_struct_field: Field,
    target_map_field: Field,
) -> None:
    import pyspark.sql.functions as F

    frame = spark.createDataFrame(
        [
            ([{"key": "a", "value": 1}, {"key": "b", "value": 2}],),
            ([{"key": "x", "value": 3}],),
            (None,),
        ],
        schema="source_entries array<struct<key:string,value:bigint>>",
    )

    options = CastOptions(
        source_field=source_list_of_struct_field,
        target_field=target_map_field,
    )

    result = frame.select(
        cast_spark_list_column_to_map(F.col("source_entries"), options).alias(
            "target_map"
        )
    )

    assert [
        normalize_map_like(v) for v in result.toArrow()["target_map"].to_pylist()
    ] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_spark_struct_column_to_map(
    spark,  # noqa: F811
    source_struct_to_map_field: Field,
    target_map_field: Field,
) -> None:
    import pyspark.sql.functions as F

    frame = spark.createDataFrame(
        [
            ({"a": 1, "b": 2, "c": 3},),
            ({"a": 4, "b": None, "c": 6},),
            (None,),
        ],
        schema="source_struct struct<a:bigint,b:bigint,c:bigint>",
    )

    options = CastOptions(
        source_field=source_struct_to_map_field,
        target_field=target_map_field,
    )

    result = frame.select(
        cast_spark_struct_column_to_map(F.col("source_struct"), options).alias(
            "target_map"
        )
    )

    assert [
        normalize_map_like(v) for v in result.toPandas()["target_map"].tolist()
    ] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": None, "c": "6"},
        None,
    ]
