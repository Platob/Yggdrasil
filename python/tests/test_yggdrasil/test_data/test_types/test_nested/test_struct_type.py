from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
import pytest
from yggdrasil.data import CastOptions, DataType, Field, Schema
from yggdrasil.data.types import IntegerType
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.nested.map import MapType
from yggdrasil.data.types.nested.struct import (
    StructType,
    cast_arrow_list_array,
    cast_arrow_map_array,
    cast_arrow_struct_array,
    cast_arrow_tabular,
    cast_pandas_list_series,
    cast_pandas_struct_series,
    cast_pandas_tabular,
    cast_polars_list_expr,
    cast_polars_list_series,
    cast_polars_map_expr,
    cast_polars_struct_expr,
    cast_polars_struct_series,
    cast_polars_tabular, cast_spark_struct_column, cast_spark_list_column,
    cast_spark_map_column,
)
from yggdrasil.io import SaveMode
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase


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


def _normalize_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_nested(v) for v in value]
    return _normalize_scalar(value)


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
def bool_type():
    return DataType.from_arrow_type(pa.bool_())


@pytest.fixture
def source_struct_field(int64_type: IntegerType, string_type) -> Field:
    return Field(
        name="source_struct",
        dtype=StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=True),
                Field(name="b", dtype=string_type, nullable=True),
            ]
        ),
        nullable=True,
    )


@pytest.fixture
def target_struct_field(int64_type: IntegerType, string_type) -> Field:
    return Field(
        name="target_struct",
        dtype=StructType(
            fields=[
                Field(name="b", dtype=string_type, nullable=True),
                Field(name="c", dtype=int64_type, nullable=True),
                Field(name="a", dtype=int64_type, nullable=True),
            ]
        ),
        nullable=True,
    )


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
def source_list_field(int64_type: IntegerType) -> Field:
    return Field(
        name="source_list",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=int64_type,
                nullable=True,
            )
        ),
        nullable=True,
    )


@pytest.fixture
def target_list_to_struct_field(int64_type: IntegerType, string_type) -> Field:
    return Field(
        name="target_struct",
        dtype=StructType(
            fields=[
                Field(name="first", dtype=int64_type, nullable=True),
                Field(name="second", dtype=string_type, nullable=True),
                Field(name="third", dtype=int64_type, nullable=True),
            ]
        ),
        nullable=True,
    )


def test_cast_arrow_struct_array_reorders_fields_and_fills_missing(
    source_struct_field: Field,
    target_struct_field: Field,
) -> None:
    array = pa.array(
        [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            None,
        ],
        type=pa.struct(
            [
                pa.field("a", pa.int64()),
                pa.field("b", pa.string()),
            ]
        ),
    )

    options = CastOptions(
        source_field=source_struct_field,
        target_field=target_struct_field,
    )

    result = cast_arrow_struct_array(array, options)

    assert isinstance(result, pa.StructArray)
    assert result.type == pa.struct(
        [
            pa.field("b", pa.string()),
            pa.field("c", pa.int64()),
            pa.field("a", pa.int64()),
        ]
    )
    assert result.to_pylist() == [
        {"b": "x", "c": None, "a": 1},
        {"b": "y", "c": None, "a": 2},
        None,
    ]


def test_cast_arrow_map_array_extracts_named_keys_to_struct(
    source_map_field: Field,
    target_struct_field: Field,
) -> None:
    array = pa.array(
        [
            [("a", 1), ("b", 2)],
            [("b", 3)],
            None,
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_struct_field,
    )

    result = cast_arrow_map_array(array, options)

    assert isinstance(result, pa.StructArray)
    assert result.type == pa.struct(
        [
            pa.field("b", pa.string()),
            pa.field("c", pa.int64()),
            pa.field("a", pa.int64()),
        ]
    )
    assert result.to_pylist() == [
        {"b": "2", "c": None, "a": 1},
        {"b": "3", "c": None, "a": None},
        None,
    ]


def test_cast_arrow_list_array_maps_by_position_and_fills_missing(
    source_list_field: Field,
    target_list_to_struct_field: Field,
) -> None:
    array = pa.array(
        [
            [1, 2, 3],
            [4],
            None,
        ],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_list_field,
        target_field=target_list_to_struct_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.StructArray)
    assert result.type == pa.struct(
        [
            pa.field("first", pa.int64()),
            pa.field("second", pa.string()),
            pa.field("third", pa.int64()),
        ]
    )
    assert result.to_pylist() == [
        {"first": 1, "second": "2", "third": 3},
        {"first": 4, "second": None, "third": None},
        None,
    ]


def test_cast_arrow_tabular_table_reorders_columns_and_adds_missing(
    int64_type: IntegerType,
    string_type,
) -> None:
    source_schema = Schema(
        inner_fields=[
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=string_type, nullable=True),
        ]
    )
    target_schema = Schema(
        inner_fields=[
            Field(name="b", dtype=string_type, nullable=True),
            Field(name="c", dtype=int64_type, nullable=True),
            Field(name="a", dtype=int64_type, nullable=True),
        ]
    )

    table = pa.table(
        {
            "a": pa.array([1, 2, None], type=pa.int64()),
            "b": pa.array(["x", "y", "z"], type=pa.string()),
        }
    )

    options = CastOptions(
        source_field=source_schema,
        target_field=target_schema,
    )

    result = cast_arrow_tabular(table, options)

    assert isinstance(result, pa.Table)
    assert result.schema == pa.schema(
        [
            pa.field("b", pa.string()),
            pa.field("c", pa.int64()),
            pa.field("a", pa.int64()),
        ]
    )
    assert result.to_pylist() == [
        {"b": "x", "c": None, "a": 1},
        {"b": "y", "c": None, "a": 2},
        {"b": "z", "c": None, "a": None},
    ]


def test_cast_arrow_tabular_record_batch_reorders_columns_and_adds_missing(
    int64_type: IntegerType,
    string_type,
) -> None:
    source_schema = Schema(
        inner_fields=[
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=string_type, nullable=True),
        ]
    )
    target_schema = Schema(
        inner_fields=[
            Field(name="b", dtype=string_type, nullable=True),
            Field(name="c", dtype=int64_type, nullable=True),
            Field(name="a", dtype=int64_type, nullable=True),
        ]
    )

    batch = pa.record_batch(
        [
            pa.array([1, 2], type=pa.int64()),
            pa.array(["x", "y"], type=pa.string()),
        ],
        names=["a", "b"],
    )

    options = CastOptions(
        source_field=source_schema,
        target_field=target_schema,
    )

    result = cast_arrow_tabular(batch, options)

    assert isinstance(result, pa.RecordBatch)
    assert result.schema == pa.schema(
        [
            pa.field("b", pa.string()),
            pa.field("c", pa.int64()),
            pa.field("a", pa.int64()),
        ]
    )
    assert result.to_pylist() == [
        {"b": "x", "c": None, "a": 1},
        {"b": "y", "c": None, "a": 2},
    ]


def test_cast_arrow_tabular_returns_original_when_target_schema_is_none() -> None:
    table = pa.table({"a": [1, 2, 3]})
    options = CastOptions(target_field=None)

    result = cast_arrow_tabular(table, options)

    assert result is table


def test_cast_arrow_tabular_raises_for_unsupported_input_type() -> None:
    options = CastOptions(
        source_field=Schema(inner_fields=[]),
        target_field=Schema(inner_fields=[]),
    )

    with pytest.raises(TypeError, match="Unsupported tabular type"):
        cast_arrow_tabular({"a": [1, 2, 3]}, options)


class _StructCastFieldsMixin:
    """Shared setUp for pandas/polars/spark struct cast test classes."""

    def _init_fields(self) -> None:
        int64_type = DataType.from_arrow_type(pa.int64())
        string_type = DataType.from_arrow_type(pa.string())

        self.int64_type = int64_type
        self.string_type = string_type
        self.source_struct_field = Field(
            name="source_struct",
            dtype=StructType(
                fields=[
                    Field(name="a", dtype=int64_type, nullable=True),
                    Field(name="b", dtype=string_type, nullable=True),
                ]
            ),
            nullable=True,
        )
        self.target_struct_field = Field(
            name="target_struct",
            dtype=StructType(
                fields=[
                    Field(name="b", dtype=string_type, nullable=True),
                    Field(name="c", dtype=int64_type, nullable=True),
                    Field(name="a", dtype=int64_type, nullable=True),
                ]
            ),
            nullable=True,
        )
        self.source_map_field = Field(
            name="source_map",
            dtype=MapType.from_key_value(
                key_field=Field(name="key", dtype=string_type, nullable=False),
                value_field=Field(name="value", dtype=int64_type, nullable=True),
            ),
            nullable=True,
        )
        self.source_list_field = Field(
            name="source_list",
            dtype=ArrayType(
                item_field=Field(name="item", dtype=int64_type, nullable=True)
            ),
            nullable=True,
        )
        self.target_list_to_struct_field = Field(
            name="target_struct",
            dtype=StructType(
                fields=[
                    Field(name="first", dtype=int64_type, nullable=True),
                    Field(name="second", dtype=string_type, nullable=True),
                    Field(name="third", dtype=int64_type, nullable=True),
                ]
            ),
            nullable=True,
        )


class TestCastPandasStructColumns(_StructCastFieldsMixin, PandasTestCase):
    """Pandas-backed cast tests for struct, list, and tabular conversions."""

    def setUp(self) -> None:
        super().setUp()
        self._init_fields()

    def test_cast_pandas_struct_series_reorders_fields_and_fills_missing(self) -> None:
        series = self.pd.Series(
            [
                {"a": 1, "b": "x"},
                {"a": 2, "b": "y"},
                None,
            ],
            name="source_struct",
            dtype="object",
        )

        options = CastOptions(
            source_field=self.source_struct_field,
            target_field=self.target_struct_field,
        )

        result = cast_pandas_struct_series(series, options)

        self.assertEqual(result.name, "target_struct")
        self.assertEqual(
            [_normalize_nested(v) for v in result.tolist()],
            [
                {"b": "x", "c": None, "a": 1},
                {"b": "y", "c": None, "a": 2},
                None,
            ],
        )

    def test_cast_pandas_list_series_maps_by_position_and_fills_missing(self) -> None:
        series = self.pd.Series(
            [
                [1, 2, 3],
                [4],
                None,
            ],
            name="source_list",
            dtype="object",
        )

        options = CastOptions(
            source_field=self.source_list_field,
            target_field=self.target_list_to_struct_field,
        )

        result = cast_pandas_list_series(series, options)

        self.assertEqual(result.name, "target_struct")
        self.assertEqual(
            [_normalize_nested(v) for v in result.tolist()],
            [
                {"first": 1, "second": "2", "third": 3},
                {"first": 4, "second": None, "third": None},
                None,
            ],
        )

    def test_cast_pandas_tabular_reorders_columns_and_adds_missing(self) -> None:
        source_schema = Schema(
            inner_fields=[
                Field(name="a", dtype=self.int64_type, nullable=True),
                Field(name="b", dtype=self.string_type, nullable=True),
            ]
        )
        target_schema = Schema(
            inner_fields=[
                Field(name="b", dtype=self.string_type, nullable=True),
                Field(name="c", dtype=self.int64_type, nullable=True),
                Field(name="a", dtype=self.int64_type, nullable=True),
            ]
        )

        frame = self.pd.DataFrame(
            {
                "a": [1, 2, None],
                "b": ["x", "y", "z"],
            }
        )

        options = CastOptions(
            source_field=source_schema,
            target_field=target_schema,
        )

        result = cast_pandas_tabular(frame, options)

        self.assertEqual(list(result.columns), ["b", "c", "a"])
        self.assertEqual(
            [_normalize_nested(v) for v in result.to_dict(orient="records")],
            [
                {"b": "x", "c": None, "a": 1},
                {"b": "y", "c": None, "a": 2},
                {"b": "z", "c": None, "a": None},
            ],
        )


class TestCastPolarsStructColumns(_StructCastFieldsMixin, PolarsTestCase):
    """Polars-backed cast tests for struct, list, map, and tabular conversions."""

    def setUp(self) -> None:
        super().setUp()
        self._init_fields()

    def test_cast_polars_struct_series_reorders_fields_and_fills_missing(self) -> None:
        pl = self.pl

        series = pl.Series(
            "source_struct",
            [
                {"a": 1, "b": "x"},
                {"a": 2, "b": "y"},
                None,
            ],
        )

        options = CastOptions(
            source_field=self.source_struct_field,
            target_field=self.target_struct_field,
        )

        result = cast_polars_struct_series(series, options)

        self.assertEqual(result.name, "target_struct")
        self.assertEqual(
            result.to_list(),
            [
                {"b": "x", "c": None, "a": 1},
                {"b": "y", "c": None, "a": 2},
                None,
            ],
        )

    def test_cast_polars_list_series_maps_by_position_and_fills_missing(self) -> None:
        pl = self.pl

        series = pl.Series(
            "source_list",
            [
                [1, 2, 3],
                [4],
                None,
            ],
        )

        options = CastOptions(
            source_field=self.source_list_field,
            target_field=self.target_list_to_struct_field,
        )

        result = cast_polars_list_series(series, options)

        self.assertEqual(result.name, "target_struct")
        self.assertEqual(
            result.to_list(),
            [
                {"first": 1, "second": "2", "third": 3},
                {"first": 4, "second": None, "third": None},
                None,
            ],
        )

    def test_cast_polars_struct_expr_reorders_fields_and_fills_missing(self) -> None:
        pl = self.pl

        frame = pl.DataFrame(
            {
                "source_struct": [
                    {"a": 1, "b": "x"},
                    {"a": 2, "b": "y"},
                    None,
                ]
            }
        )

        options = CastOptions(
            source_field=self.source_struct_field,
            target_field=self.target_struct_field,
        )

        result = frame.select(
            cast_polars_struct_expr(pl.col("source_struct"), options).alias("target_struct")
        )

        self.assertEqual(
            result["target_struct"].to_list(),
            [
                {"b": "x", "c": None, "a": 1},
                {"b": "y", "c": None, "a": 2},
                None,
            ],
        )

    def test_cast_polars_list_expr_maps_by_position_and_fills_missing(self) -> None:
        pl = self.pl

        frame = pl.DataFrame(
            {
                "source_list": [
                    [1, 2, 3],
                    [4],
                    None,
                ]
            }
        )

        options = CastOptions(
            source_field=self.source_list_field,
            target_field=self.target_list_to_struct_field,
        )

        result = frame.select(
            cast_polars_list_expr(pl.col("source_list"), options).alias("target_struct")
        )

        self.assertEqual(
            result["target_struct"].to_list(),
            [
                {"first": 1, "second": "2", "third": 3},
                {"first": 4, "second": None, "third": None},
                None,
            ],
        )

    def test_cast_polars_map_expr_extracts_named_keys_to_struct(self) -> None:
        pl = self.pl

        frame = pl.DataFrame(
            {
                "source_map": [
                    [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
                    [{"key": "b", "value": 3}],
                    None,
                ]
            }
        )

        options = CastOptions(
            source_field=self.source_map_field,
            target_field=self.target_struct_field,
        )

        result = frame.select(
            cast_polars_map_expr(pl.col("source_map"), options).alias("target_struct")
        )

        self.assertEqual(
            result["target_struct"].to_list(),
            [
                {"b": "2", "c": None, "a": 1},
                {"b": "3", "c": None, "a": None},
                None,
            ],
        )

    def test_cast_polars_tabular_reorders_columns_and_adds_missing(self) -> None:
        pl = self.pl

        source_schema = Schema(
            inner_fields=[
                Field(name="a", dtype=self.int64_type, nullable=True),
                Field(name="b", dtype=self.string_type, nullable=True),
            ]
        )
        target_schema = Schema(
            inner_fields=[
                Field(name="b", dtype=self.string_type, nullable=True),
                Field(name="c", dtype=self.int64_type, nullable=True),
                Field(name="a", dtype=self.int64_type, nullable=True),
            ]
        )

        frame = pl.DataFrame(
            {
                "a": [1, 2, None],
                "b": ["x", "y", "z"],
            }
        )

        options = CastOptions(
            source_field=source_schema,
            target_field=target_schema,
        )

        result = cast_polars_tabular(frame, options)

        self.assertEqual(result.columns, ["b", "c", "a"])
        self.assertEqual(
            result.to_dicts(),
            [
                {"b": "x", "c": None, "a": 1},
                {"b": "y", "c": None, "a": 2},
                {"b": "z", "c": None, "a": None},
            ],
        )

    def test_cast_polars_tabular_lazy_reorders_columns_and_adds_missing(self) -> None:
        pl = self.pl

        source_schema = Schema(
            inner_fields=[
                Field(name="a", dtype=self.int64_type, nullable=True),
                Field(name="b", dtype=self.string_type, nullable=True),
            ]
        )
        target_schema = Schema(
            inner_fields=[
                Field(name="b", dtype=self.string_type, nullable=True),
                Field(name="c", dtype=self.int64_type, nullable=True),
                Field(name="a", dtype=self.int64_type, nullable=True),
            ]
        )

        frame = pl.DataFrame(
            {
                "a": [1, 2, None],
                "b": ["x", "y", "z"],
            }
        ).lazy()

        options = CastOptions(
            source_field=source_schema,
            target_field=target_schema,
        )

        result = cast_polars_tabular(frame, options).collect()

        self.assertEqual(result.columns, ["b", "c", "a"])
        self.assertEqual(
            result.to_dicts(),
            [
                {"b": "x", "c": None, "a": 1},
                {"b": "y", "c": None, "a": 2},
                {"b": "z", "c": None, "a": None},
            ],
        )


class TestCastSparkStructColumns(SparkTestCase):
    """Spark-backed cast tests for struct, map, and list → struct conversions."""

    def setUp(self) -> None:
        int64_type = DataType.from_arrow_type(pa.int64())
        int32_type = DataType.from_arrow_type(pa.int32())  # noqa: F841
        string_type = DataType.from_arrow_type(pa.string())

        self.source_struct_field = Field(
            name="source_struct",
            dtype=StructType(
                fields=[
                    Field(name="a", dtype=int64_type, nullable=True),
                    Field(name="b", dtype=string_type, nullable=True),
                ]
            ),
            nullable=True,
        )
        self.target_struct_field = Field(
            name="target_struct",
            dtype=StructType(
                fields=[
                    Field(name="b", dtype=string_type, nullable=True),
                    Field(name="c", dtype=int64_type, nullable=True),
                    Field(name="a", dtype=int64_type, nullable=True),
                ]
            ),
            nullable=True,
        )
        self.source_map_field = Field(
            name="source_map",
            dtype=MapType.from_key_value(
                key_field=Field(name="key", dtype=string_type, nullable=False),
                value_field=Field(name="value", dtype=int64_type, nullable=True),
            ),
            nullable=True,
        )
        self.source_list_field = Field(
            name="source_list",
            dtype=ArrayType(
                item_field=Field(name="item", dtype=int64_type, nullable=True)
            ),
            nullable=True,
        )
        self.target_list_to_struct_field = Field(
            name="target_struct",
            dtype=StructType(
                fields=[
                    Field(name="first", dtype=int64_type, nullable=True),
                    Field(name="second", dtype=string_type, nullable=True),
                    Field(name="third", dtype=int64_type, nullable=True),
                ]
            ),
            nullable=True,
        )

    def test_cast_spark_struct_column_reorders_fields_and_fills_missing(self) -> None:
        import pyspark.sql.functions as F

        frame = self.spark.createDataFrame(
            [
                ({"a": 1, "b": "x"},),
                ({"a": 2, "b": "y"},),
                (None,),
            ],
            schema="source_struct struct<a:bigint,b:string>",
        )

        options = CastOptions(
            source_field=self.source_struct_field,
            target_field=self.target_struct_field,
        )

        result = frame.select(
            cast_spark_struct_column(F.col("source_struct"), options).alias("target_struct")
        )

        self.assertEqual(
            [_normalize_nested(v) for v in result.toPandas()["target_struct"].tolist()],
            [
                {"b": "x", "c": None, "a": 1},
                {"b": "y", "c": None, "a": 2},
                None,
            ],
        )

    def test_cast_spark_map_column_extracts_named_keys_to_struct(self) -> None:
        import pyspark.sql.functions as F

        frame = self.spark.createDataFrame(
            [
                ({"a": 1, "b": 2},),
                ({"b": 3},),
                (None,),
            ],
            schema="source_map map<string,bigint>",
        )

        options = CastOptions(
            source_field=self.source_map_field,
            target_field=self.target_struct_field,
        )

        result = frame.select(
            cast_spark_map_column(F.col("source_map"), options).alias("target_struct")
        )

        self.assertEqual(
            [_normalize_nested(v) for v in result.toPandas()["target_struct"].tolist()],
            [
                {"b": "2", "c": None, "a": 1},
                {"b": "3", "c": None, "a": None},
                None,
            ],
        )

    def test_cast_spark_list_column_maps_by_position_and_fills_missing(self) -> None:
        import pyspark.sql.functions as F

        frame = self.spark.createDataFrame(
            [
                ([1, 2, 3],),
                ([4],),
                (None,),
            ],
            schema="source_list array<bigint>",
        )

        options = CastOptions(
            source_field=self.source_list_field,
            target_field=self.target_list_to_struct_field,
        )

        result = frame.select(
            cast_spark_list_column(F.col("source_list"), options).alias("target_struct")
        )

        self.assertEqual(
            [_normalize_nested(v) for v in result.toArrow()["target_struct"].tolist()],
            [
                {"first": 1, "second": "2", "third": 3},
                {"first": 4, "second": None, "third": None},
                None,
            ],
        )


def test_struct_type_merge_with_same_id_merges_matching_fields_by_name(
    int64_type: IntegerType,
    string_type,
) -> None:
    left = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=False),
            Field(name="b", dtype=string_type, nullable=True),
        ]
    )
    right = StructType(
        fields=[
            Field(name="a", dtype=DataType.from_arrow_type(pa.int32()), nullable=True),
            Field(name="b", dtype=string_type, nullable=True, metadata={"comment": "rhs"}),
        ]
    )

    result = left._merge_with_same_id(
        right,
        upcast=True,
    )

    assert isinstance(result, StructType)
    assert [field.name for field in result.fields] == ["a", "b"]
    assert result.fields[0].nullable is True
    assert result.fields[0].arrow_type == pa.int64()
    assert result.fields[1].metadata == {b"comment": b"rhs"}


def test_struct_type_merge_with_same_id_appends_unmatched_fields_when_names_do_not_match(
    int64_type: IntegerType,
    string_type,
) -> None:
    left = StructType(
        fields=[
            Field(name="left_a", dtype=int64_type, nullable=False),
            Field(name="left_b", dtype=string_type, nullable=True),
        ]
    )
    right = StructType(
        fields=[
            Field(name="right_a", dtype=DataType.from_arrow_type(pa.int32()), nullable=True),
            Field(name="right_b", dtype=DataType.from_arrow_type(pa.large_string()), nullable=True),
        ]
    )

    result = left._merge_with_same_id(
        right,
        upcast=True,
    )

    assert isinstance(result, StructType)
    assert [field.name for field in result.fields] == [
        "left_a",
        "left_b",
        "right_a",
        "right_b",
    ]
    assert result.fields[0].arrow_type == pa.int64()
    assert result.fields[0].nullable is False
    assert result.fields[1].arrow_type == pa.string()
    assert result.fields[1].nullable is True
    assert result.fields[2].arrow_type == pa.int32()
    assert result.fields[2].nullable is True
    assert result.fields[3].arrow_type == pa.large_string()
    assert result.fields[3].nullable is True


def test_struct_type_merge_with_same_id_keeps_left_only_fields(
    int64_type: IntegerType,
    string_type,
) -> None:
    left_only = Field(name="left_only", dtype=int64_type, nullable=True)
    shared_left = Field(name="shared", dtype=string_type, nullable=True)

    left = StructType(fields=[left_only, shared_left])
    right = StructType(fields=[Field(name="shared", dtype=string_type, nullable=True)])

    result = left._merge_with_same_id(right)

    assert isinstance(result, StructType)
    assert [field.name for field in result.fields] == ["left_only", "shared"]
    assert result.fields[0] == left_only


@pytest.mark.parametrize(
    "mode",
    [
        None,
        SaveMode.APPEND,
        SaveMode.UPSERT,
        SaveMode.AUTO,
    ],
)
def test_struct_type_merge_with_same_id_appends_right_only_fields_for_append_like_modes(
    int64_type: IntegerType,
    string_type,
    mode: SaveMode | None,
) -> None:
    left = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=True),
        ]
    )
    right = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=string_type, nullable=True),
            Field(name="c", dtype=int64_type, nullable=True),
        ]
    )

    result = left._merge_with_same_id(
        right,
        mode=mode,
    )

    assert isinstance(result, StructType)
    assert [field.name for field in result.fields] == ["a", "b", "c"]


@pytest.mark.parametrize(
    "mode",
    [
        SaveMode.OVERWRITE,
    ],
)
def test_struct_type_merge_with_same_id_does_not_append_right_only_fields_for_non_append_modes(
    int64_type: IntegerType,
    string_type,
    mode: SaveMode,
) -> None:
    left = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=True),
        ]
    )
    right = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=string_type, nullable=True),
        ]
    )

    result = left._merge_with_same_id(
        right,
        mode=mode,
    )

    assert isinstance(result, StructType)
    assert [field.name for field in result.fields] == ["a"]


def test_struct_type_merge_with_same_id_preserves_left_field_order_and_appends_new_right_fields(
    int64_type: IntegerType,
    string_type,
    bool_type,
) -> None:
    left = StructType(
        fields=[
            Field(name="b", dtype=string_type, nullable=True),
            Field(name="a", dtype=int64_type, nullable=False),
        ]
    )
    right = StructType(
        fields=[
            Field(name="a", dtype=DataType.from_arrow_type(pa.int32()), nullable=True),
            Field(name="c", dtype=bool_type, nullable=True),
        ]
    )

    result = left._merge_with_same_id(
        right,
        mode=SaveMode.APPEND,
        upcast=True,
    )

    assert isinstance(result, StructType)
    assert [field.name for field in result.fields] == ["b", "a", "c"]
    assert result.fields[1].arrow_type == pa.int64()
    assert result.fields[1].nullable is True


def test_struct_type_merge_with_same_id_raises_for_non_struct_type(
    int64_type: IntegerType,
) -> None:
    left = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=True),
        ]
    )

    with pytest.raises(TypeError, match="Cannot merge StructType with"):
        left._merge_with_same_id(
            DataType.from_arrow_type(pa.int64()),
        )