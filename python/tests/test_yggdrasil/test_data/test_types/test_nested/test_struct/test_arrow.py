"""Arrow integration tests for StructType casts.

Exercises struct->struct, map->struct, list->struct array casts as
well as the tabular entry points (Table and RecordBatch).
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import CastOptions, Field, Schema
from yggdrasil.data.types import IntegerType
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.nested.struct import (
    StructType,
    cast_arrow_list_array,
    cast_arrow_map_array,
    cast_arrow_struct_array,
    cast_arrow_tabular,
)


# ---------------------------------------------------------------------------
# struct -> struct
# ---------------------------------------------------------------------------


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
            [pa.field("a", pa.int64()), pa.field("b", pa.string())]
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


def test_cast_arrow_struct_array_returns_input_when_target_is_none() -> None:
    arrow_struct = pa.struct([pa.field("a", pa.int64())])
    array = pa.array([{"a": 1}], type=arrow_struct)

    source_field = Field(
        name="src",
        dtype=StructType(
            fields=[
                Field(
                    name="a",
                    dtype=IntegerType(byte_size=8, signed=True),
                    nullable=True,
                )
            ]
        ),
        nullable=True,
    )

    options = CastOptions(source_field=source_field, target_field=None)

    assert cast_arrow_struct_array(array, options) is array


def test_cast_arrow_struct_array_rejects_non_struct_source() -> None:
    array = pa.array([[1, 2]], type=pa.list_(pa.int64()))

    source_field = Field(
        name="src",
        dtype=ArrayType.from_item_field(
            IntegerType(byte_size=8, signed=True).to_field(name="item"),
            safe=True,
        ),
        nullable=True,
    )
    target_field = Field(
        name="tgt",
        dtype=StructType(
            fields=[
                Field(
                    name="a",
                    dtype=IntegerType(byte_size=8, signed=True),
                    nullable=True,
                )
            ]
        ),
        nullable=True,
    )

    options = CastOptions(source_field=source_field, target_field=target_field)

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_struct_array(array, options)


# ---------------------------------------------------------------------------
# map -> struct (field names = keys)
# ---------------------------------------------------------------------------


def test_cast_arrow_map_array_extracts_named_keys_to_struct(
    source_map_field: Field,
    target_struct_field: Field,
) -> None:
    array = pa.array(
        [[("a", 1), ("b", 2)], [("b", 3)], None],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_struct_field,
    )

    result = cast_arrow_map_array(array, options)

    assert isinstance(result, pa.StructArray)
    assert result.to_pylist() == [
        {"b": "2", "c": None, "a": 1},
        {"b": "3", "c": None, "a": None},
        None,
    ]


# ---------------------------------------------------------------------------
# list -> struct (positional mapping)
# ---------------------------------------------------------------------------


def test_cast_arrow_list_array_maps_by_position_and_fills_missing(
    source_list_field: Field,
    target_list_to_struct_field: Field,
) -> None:
    array = pa.array(
        [[1, 2, 3], [4], None],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_list_field,
        target_field=target_list_to_struct_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.StructArray)
    assert result.to_pylist() == [
        {"first": 1, "second": "2", "third": 3},
        {"first": 4, "second": None, "third": None},
        None,
    ]


# ---------------------------------------------------------------------------
# Tabular entry points
# ---------------------------------------------------------------------------


def test_cast_arrow_tabular_table_reorders_columns_and_adds_missing(
    source_tabular_schema: Schema,
    target_tabular_schema: Schema,
) -> None:
    table = pa.table(
        {
            "a": pa.array([1, 2, None], type=pa.int64()),
            "b": pa.array(["x", "y", "z"], type=pa.string()),
        }
    )

    options = CastOptions(
        source_field=source_tabular_schema,
        target_field=target_tabular_schema,
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
    source_tabular_schema: Schema,
    target_tabular_schema: Schema,
) -> None:
    batch = pa.record_batch(
        [
            pa.array([1, 2], type=pa.int64()),
            pa.array(["x", "y"], type=pa.string()),
        ],
        names=["a", "b"],
    )

    options = CastOptions(
        source_field=source_tabular_schema,
        target_field=target_tabular_schema,
    )

    result = cast_arrow_tabular(batch, options)

    assert isinstance(result, pa.RecordBatch)
    assert result.to_pylist() == [
        {"b": "x", "c": None, "a": 1},
        {"b": "y", "c": None, "a": 2},
    ]


def test_cast_arrow_tabular_returns_input_when_target_schema_is_none() -> None:
    table = pa.table({"a": [1, 2, 3]})
    options = CastOptions(target_field=None)

    assert cast_arrow_tabular(table, options) is table


def test_cast_arrow_tabular_rejects_non_table_input() -> None:
    options = CastOptions(
        source_field=Schema(inner_fields=[]),
        target_field=Schema(inner_fields=[]),
    )

    with pytest.raises(TypeError, match="Unsupported tabular type"):
        cast_arrow_tabular({"a": [1, 2, 3]}, options)
