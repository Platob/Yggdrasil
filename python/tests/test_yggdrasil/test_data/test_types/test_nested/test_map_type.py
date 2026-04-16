from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
import pytest
from yggdrasil.data import CastOptions, DataType, Field
from yggdrasil.data.types import ArrayType, IntegerType, MapType
from yggdrasil.data.types.nested.map import (
    cast_arrow_list_array_to_map,
    cast_arrow_map_array,
    cast_arrow_struct_array_to_map,
    cast_pandas_list_series_to_map,
    cast_pandas_map_series,
    cast_pandas_struct_series_to_map,
    cast_polars_list_series_to_map,
    cast_polars_map_series,
    cast_polars_struct_series_to_map,
    cast_spark_list_column_to_map,
    cast_spark_map_column,
    cast_spark_struct_column_to_map,
)
from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.io import SaveMode


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _normalize_scalar(value: Any) -> Any:
    if _is_nan(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _normalize_map_like(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            _normalize_scalar(k): _normalize_scalar(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        if all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return {
                _normalize_scalar(k): _normalize_scalar(v)
                for k, v in value
            }

        if all(isinstance(item, dict) and "key" in item and "value" in item for item in value):
            return {
                _normalize_scalar(item["key"]): _normalize_scalar(item["value"])
                for item in value
            }

    return value


@pytest.fixture
def int64_type() -> IntegerType:
    return DataType.from_arrow_type(pa.int64())


@pytest.fixture
def int32_type() -> IntegerType:
    return DataType.from_arrow_type(pa.int32())


@pytest.fixture
def string_type():
    return DataType.from_arrow_type(pa.string())


@pytest.fixture
def source_map_field(int64_type: IntegerType, string_type) -> Field:
    return Field(
        name="source_map",
        dtype=MapType.from_key_value(
            key_field=Field(name="key", dtype=string_type, nullable=False),
            value_field=Field(name="value", dtype=int64_type, nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def target_map_field(string_type) -> Field:
    return Field(
        name="target_map",
        dtype=MapType.from_key_value(
            key_field=Field(name="key", dtype=string_type, nullable=False),
            value_field=Field(name="value", dtype=string_type, nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def source_list_of_struct_field(int64_type: IntegerType, string_type) -> Field:
    entry_struct = StructType(
        fields=[
            Field(name="key", dtype=string_type, nullable=False),
            Field(name="value", dtype=int64_type, nullable=True),
        ]
    )
    return Field(
        name="source_entries",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=entry_struct, nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def source_struct_to_map_field(int64_type: IntegerType) -> Field:
    return Field(
        name="source_struct",
        dtype=StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=True),
                Field(name="b", dtype=int64_type, nullable=True),
                Field(name="c", dtype=int64_type, nullable=True),
            ]
        ),
        nullable=True,
    )


def test_cast_arrow_map_array_recasts_values(
    source_map_field: Field,
    target_map_field: Field,
) -> None:
    array = pa.array(
        [
            [("a", 1), ("b", 2)],
            [("x", 3)],
            None,
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_map_field,
    )

    result = cast_arrow_map_array(array, options)

    assert isinstance(result, pa.MapArray)
    assert result.type == pa.map_(pa.string(), pa.string())
    assert [_normalize_map_like(v) for v in result.to_pylist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_arrow_list_array_to_map_converts_entries(
    source_list_of_struct_field: Field,
    target_map_field: Field,
) -> None:
    entry_type = pa.struct(
        [
            pa.field("key", pa.string()),
            pa.field("value", pa.int64()),
        ]
    )
    array = pa.array(
        [
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            [{"key": "x", "value": 3}],
            None,
        ],
        type=pa.list_(entry_type),
    )

    options = CastOptions(
        source_field=source_list_of_struct_field,
        target_field=target_map_field,
    )

    result = cast_arrow_list_array_to_map(array, options)

    assert isinstance(result, pa.MapArray)
    assert result.type == pa.map_(pa.string(), pa.string())
    assert [_normalize_map_like(v) for v in result.to_pylist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_arrow_struct_array_to_map_uses_field_names_as_keys(
    source_struct_to_map_field: Field,
    target_map_field: Field,
) -> None:
    array = pa.array(
        [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": None, "c": 6},
            None,
        ],
        type=pa.struct(
            [
                pa.field("a", pa.int64()),
                pa.field("b", pa.int64()),
                pa.field("c", pa.int64()),
            ]
        ),
    )

    options = CastOptions(
        source_field=source_struct_to_map_field,
        target_field=target_map_field,
    )

    result = cast_arrow_struct_array_to_map(array, options)

    assert isinstance(result, pa.MapArray)
    assert result.type == pa.map_(pa.string(), pa.string())
    assert [_normalize_map_like(v) for v in result.to_pylist()] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": None, "c": "6"},
        None,
    ]


def test_cast_pandas_map_series_recasts_values(
    source_map_field: Field,
    target_map_field: Field,
) -> None:
    pd = pytest.importorskip("pandas")

    series = pd.Series(
        [
            {"a": 1, "b": 2},
            {"x": 3},
            None,
        ],
        name="source_map",
        dtype="object",
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_map_field,
    )

    result = cast_pandas_map_series(series, options)

    assert result.name == "target_map"
    assert [_normalize_map_like(v) for v in result.tolist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_pandas_list_series_to_map_converts_entries(
    source_list_of_struct_field: Field,
    target_map_field: Field,
) -> None:
    pd = pytest.importorskip("pandas")

    series = pd.Series(
        [
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            [{"key": "x", "value": 3}],
            None,
        ],
        name="source_entries",
        dtype="object",
    )

    options = CastOptions(
        source_field=source_list_of_struct_field,
        target_field=target_map_field,
    )

    result = cast_pandas_list_series_to_map(series, options)

    assert result.name == "target_map"
    assert [_normalize_map_like(v) for v in result.tolist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_pandas_struct_series_to_map_uses_field_names_as_keys(
    source_struct_to_map_field: Field,
    target_map_field: Field,
) -> None:
    pd = pytest.importorskip("pandas")

    series = pd.Series(
        [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": None, "c": 6},
            None,
        ],
        name="source_struct",
        dtype="object",
    )

    options = CastOptions(
        source_field=source_struct_to_map_field,
        target_field=target_map_field,
    )

    result = cast_pandas_struct_series_to_map(series, options)

    assert result.name == "target_map"
    assert [_normalize_map_like(v) for v in result.tolist()] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": None, "c": "6"},
        None,
    ]


def test_cast_polars_map_series_recasts_values(
    source_map_field: Field,
    target_map_field: Field,
) -> None:
    pl = pytest.importorskip("polars")

    series = pl.Series(
        "source_map",
        [
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            [{"key": "x", "value": 3}],
            None,
        ],
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_map_field,
    )

    result = cast_polars_map_series(series, options)

    assert result.name == "target_map"
    assert [_normalize_map_like(v) for v in result.to_list()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_polars_list_series_to_map_converts_entries(
    source_list_of_struct_field: Field,
    target_map_field: Field,
) -> None:
    pl = pytest.importorskip("polars")

    series = pl.Series(
        "source_entries",
        [
            [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
            [{"key": "x", "value": 3}],
            None,
        ],
    )

    options = CastOptions(
        source_field=source_list_of_struct_field,
        target_field=target_map_field,
    )

    result = cast_polars_list_series_to_map(series, options)

    assert result.name == "target_map"
    assert [_normalize_map_like(v) for v in result.to_list()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_polars_struct_series_to_map_uses_field_names_as_keys(
    source_struct_to_map_field: Field,
    target_map_field: Field,
) -> None:
    pl = pytest.importorskip("polars")

    series = pl.Series(
        "source_struct",
        [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": None, "c": 6},
            None,
        ],
    )

    options = CastOptions(
        source_field=source_struct_to_map_field,
        target_field=target_map_field,
    )

    result = cast_polars_struct_series_to_map(series, options)

    assert result.name == "target_map"
    assert [_normalize_map_like(v) for v in result.to_list()] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": None, "c": "6"},
        None,
    ]


@pytest.fixture(scope="session")
def spark():
    pyspark = pytest.importorskip("pyspark")
    spark = (
        pyspark.sql.SparkSession.builder
        .master("local[1]")
        .appName("map-type-tests")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_cast_spark_map_column_recasts_values(
    spark,
    source_map_field: Field,
    target_map_field: Field,
) -> None:
    pyspark = pytest.importorskip("pyspark")
    F = pyspark.sql.functions

    frame = spark.createDataFrame(
        [
            ({"a": 1, "b": 2},),
            ({"x": 3},),
            (None,),
        ],
        schema="source_map map<string,bigint>",
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_map_field,
    )

    result = frame.select(
        cast_spark_map_column(F.col("source_map"), options).alias("target_map")
    )

    assert [_normalize_map_like(v) for v in result.toPandas()["target_map"].tolist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_spark_list_column_to_map_converts_entries(
    spark,
    source_list_of_struct_field: Field,
    target_map_field: Field,
) -> None:
    pyspark = pytest.importorskip("pyspark")
    F = pyspark.sql.functions

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
        cast_spark_list_column_to_map(F.col("source_entries"), options).alias("target_map")
    )

    assert [_normalize_map_like(v) for v in result.toArrow()["target_map"].tolist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_spark_struct_column_to_map_uses_field_names_as_keys(
    spark,
    source_struct_to_map_field: Field,
    target_map_field: Field,
) -> None:
    pyspark = pytest.importorskip("pyspark")
    F = pyspark.sql.functions

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
        cast_spark_struct_column_to_map(F.col("source_struct"), options).alias("target_map")
    )

    assert [_normalize_map_like(v) for v in result.toPandas()["target_map"].tolist()] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": None, "c": "6"},
        None,
    ]


def test_map_type_merge_with_same_id_merges_key_and_value_fields(
    int64_type: IntegerType,
    int32_type: IntegerType,
    string_type,
) -> None:
    left = MapType.from_key_value(
        key_field=Field(name="key", dtype=string_type, nullable=False),
        value_field=Field(name="value", dtype=int64_type, nullable=False),
    )
    right = MapType.from_key_value(
        key_field=Field(name="key", dtype=string_type, nullable=False),
        value_field=Field(name="value", dtype=int32_type, nullable=True),
    )

    result = left._merge_with_same_id(
        right,
        upcast=True,
    )

    assert isinstance(result, MapType)
    assert result.key_field.arrow_type == pa.string()
    assert result.key_field.nullable is False
    assert result.value_field.arrow_type == pa.int64()
    assert result.value_field.nullable is True


@pytest.mark.parametrize(
    "left_sorted,right_sorted,expected",
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_map_type_merge_with_same_id_ors_keys_sorted(
    int64_type: IntegerType,
    string_type,
    left_sorted: bool,
    right_sorted: bool,
    expected: bool,
) -> None:
    left = MapType.from_key_value(
        key_field=Field(name="key", dtype=string_type, nullable=False),
        value_field=Field(name="value", dtype=int64_type, nullable=True),
        keys_sorted=left_sorted,
    )
    right = MapType.from_key_value(
        key_field=Field(name="key", dtype=string_type, nullable=False),
        value_field=Field(name="value", dtype=int64_type, nullable=True),
        keys_sorted=right_sorted,
    )

    result = left._merge_with_same_id(right)

    assert isinstance(result, MapType)
    assert result.keys_sorted is expected