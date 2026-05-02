"""Arrow integration tests for StructType casts.

Exercises struct->struct, map->struct, list->struct array casts as
well as the tabular entry points (Table and RecordBatch).
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import Field, Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types import IntegerType
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.nested.struct import (
    StructType,
    cast_arrow_batch_iterator,
    cast_arrow_list_array,
    cast_arrow_map_array,
    cast_arrow_struct_array,
    cast_arrow_tabular,
    rechunk_arrow_batches_by_byte_size,
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
        dtype=ArrayType.from_item(
            IntegerType(byte_size=8, signed=True).to_field(name="item"),
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


# ---------------------------------------------------------------------------
# Streaming — iterator of pa.RecordBatch + byte_size rechunking
# ---------------------------------------------------------------------------


def _batch(rows: list[dict], schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows, schema=schema)


def test_cast_arrow_batch_iterator_casts_each_batch_against_target(
    source_tabular_schema: Schema,
    target_tabular_schema: Schema,
) -> None:
    source_arrow = source_tabular_schema.to_arrow_schema()
    batches = [
        _batch([{"a": 1, "b": "x"}], source_arrow),
        _batch([{"a": 2, "b": "y"}, {"a": 3, "b": "z"}], source_arrow),
    ]

    options = CastOptions(
        source_field=source_tabular_schema,
        target_field=target_tabular_schema,
    )

    result = list(cast_arrow_batch_iterator(iter(batches), options))

    assert len(result) == 2
    target_arrow = target_tabular_schema.to_arrow_schema()
    for r in result:
        assert r.schema == target_arrow
    assert [row for b in result for row in b.to_pylist()] == [
        {"b": "x", "c": None, "a": 1},
        {"b": "y", "c": None, "a": 2},
        {"b": "z", "c": None, "a": 3},
    ]


def test_cast_arrow_batch_iterator_passes_through_when_byte_size_unset(
    source_tabular_schema: Schema,
) -> None:
    """No byte_size → no rechunk; one input batch == one output batch."""
    schema = source_tabular_schema.to_arrow_schema()
    inputs = [
        _batch([{"a": i, "b": "x"}], schema)
        for i in range(5)
    ]

    options = CastOptions(
        source_field=source_tabular_schema,
        target_field=source_tabular_schema,
    )

    result = list(cast_arrow_batch_iterator(iter(inputs), options))

    assert len(result) == len(inputs)
    assert sum(r.num_rows for r in result) == len(inputs)


def test_cast_arrow_batch_iterator_rechunks_when_byte_size_set(
    source_tabular_schema: Schema,
) -> None:
    """byte_size set → small inputs coalesce into fewer larger batches."""
    schema = source_tabular_schema.to_arrow_schema()
    inputs = [_batch([{"a": i, "b": "x"}], schema) for i in range(20)]

    # Pick byte_size large enough to coalesce most inputs into 1-2 batches.
    one_row_bytes = inputs[0].nbytes
    target_bytes = one_row_bytes * 8

    options = CastOptions(
        source_field=source_tabular_schema,
        target_field=source_tabular_schema,
        byte_size=target_bytes,
    )

    result = list(cast_arrow_batch_iterator(iter(inputs), options))

    assert len(result) < len(inputs)
    assert sum(r.num_rows for r in result) == len(inputs)


def test_cast_arrow_batch_iterator_handles_empty_iterator(
    target_tabular_schema: Schema,
) -> None:
    options = CastOptions(target_field=target_tabular_schema)
    assert list(cast_arrow_batch_iterator(iter([]), options)) == []


def test_cast_arrow_batch_iterator_rejects_non_batch_items() -> None:
    options = CastOptions()
    with pytest.raises(TypeError, match="expected pa.RecordBatch"):
        list(cast_arrow_batch_iterator(iter([{"a": 1}]), options))


def test_rechunk_passthrough_when_byte_size_is_none(
    source_tabular_schema: Schema,
) -> None:
    schema = source_tabular_schema.to_arrow_schema()
    inputs = [_batch([{"a": i, "b": "x"}], schema) for i in range(3)]

    out = list(rechunk_arrow_batches_by_byte_size(iter(inputs), byte_size=0))

    assert out == inputs


def test_rechunk_slices_oversized_batch(
    source_tabular_schema: Schema,
) -> None:
    """A single batch larger than byte_size splits into multiple zero-copy slices."""
    schema = source_tabular_schema.to_arrow_schema()
    big = _batch(
        [{"a": i, "b": "x"} for i in range(100)],
        schema,
    )

    bytes_per_row = max(1, big.nbytes // big.num_rows)
    target = bytes_per_row * 10

    out = list(rechunk_arrow_batches_by_byte_size(iter([big]), byte_size=target))

    assert len(out) > 1
    assert sum(b.num_rows for b in out) == big.num_rows
    # Every output batch (except possibly the tail) is at most
    # target_rows ≈ target // bytes_per_row.
    for b in out[:-1]:
        assert b.nbytes <= target * 2  # loose upper bound, accounts for overhead


def test_rechunk_coalesces_small_batches(
    source_tabular_schema: Schema,
) -> None:
    """Many small batches buffer into fewer larger batches."""
    schema = source_tabular_schema.to_arrow_schema()
    inputs = [_batch([{"a": i, "b": "x"}], schema) for i in range(10)]
    bytes_per_row = inputs[0].nbytes
    target = bytes_per_row * 5

    out = list(rechunk_arrow_batches_by_byte_size(iter(inputs), byte_size=target))

    assert len(out) < len(inputs)
    assert sum(b.num_rows for b in out) == sum(b.num_rows for b in inputs)


def test_rechunk_drops_empty_input_batches(
    source_tabular_schema: Schema,
) -> None:
    schema = source_tabular_schema.to_arrow_schema()
    empty = _batch([], schema)
    one = _batch([{"a": 1, "b": "x"}], schema)

    out = list(
        rechunk_arrow_batches_by_byte_size(
            iter([empty, one, empty]),
            byte_size=one.nbytes * 100,
        )
    )

    assert sum(b.num_rows for b in out) == 1
