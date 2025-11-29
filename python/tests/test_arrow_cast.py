import decimal

import pytest

import pyarrow as pa
from yggdrasil.types.cast.arrow import (
    ArrowCastOptions,
    cast_arrow_array,
    cast_arrow_batch,
    cast_arrow_table,
)


def test_cast_arrow_array_struct_recursive():
    source = pa.array([
        {"a": 1, "b": {"c": 2}},
        {"a": 3, "b": {"c": 4}},
    ])

    target_type = pa.struct([
        pa.field("a", pa.float64()),
        pa.field("b", pa.struct([pa.field("c", pa.float64())])),
    ])

    result = cast_arrow_array(source, target_type)

    assert result.type == target_type
    assert result.field("a").type == pa.float64()
    assert result.field("b").field("c").type == pa.float64()


def test_cast_arrow_array_list_and_map_recursive():
    list_array = pa.array([[1, 2, 3], [4, 5]])
    list_result = cast_arrow_array(list_array, pa.list_(pa.float64()))
    assert list_result.type == pa.list_(pa.float64())
    assert list_result.values.type == pa.float64()

    map_array = pa.array(
        [
            {"x": 1, "y": 2},
            {"z": 3},
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )
    map_result = cast_arrow_array(map_array, pa.map_(pa.string(), pa.float64()))
    assert map_result.type == pa.map_(pa.string(), pa.float64())
    assert map_result.items.type == pa.float64()


def test_cast_arrow_array_map_to_struct():
    map_array = pa.array(
        [
            {"A": 1, "b": 2},
            {"a": 3},
            None,
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    target_type = pa.struct([
        pa.field("a", pa.int64()),
        pa.field("b", pa.float64(), nullable=False),
    ])

    result = cast_arrow_array(map_array, target_type)

    assert result.type == target_type
    assert result.field("a").to_pylist() == [1, 3, None]
    assert result.field("b").to_pylist() == [2.0, 0.0, 0.0]


def test_cast_arrow_array_struct_to_map():
    struct_array = pa.array([
        {"A": 1, "b": 2},
        {"A": 3, "b": None},
        None,
    ])

    target_type = pa.map_(pa.string(), pa.float64())

    result = cast_arrow_array(struct_array, target_type)

    assert result.type == target_type
    assert result.offsets.to_pylist() == [0, 2, 4, 4]
    assert result.keys.to_pylist() == ["A", "b", "A", "b"]
    assert result.items.to_pylist() == [1.0, 2.0, 3.0, None]


def test_cast_arrow_array_struct_case_insensitive_names():
    source = pa.array(
        [
            {"A": 1, "B": 2},
            {"A": 3, "B": 4},
        ],
        type=pa.struct([pa.field("A", pa.int64()), pa.field("B", pa.int64())]),
    )

    target_type = pa.struct([
        pa.field("a", pa.int64()),
        pa.field("b", pa.float64()),
    ])

    result = cast_arrow_array(source, target_type)

    assert result.field("a").to_pylist() == [1, 3]
    assert result.field("b").to_pylist() == [2.0, 4.0]

    strict_options = ArrowCastOptions(strict_match_names=True, add_missing_columns=False)
    with pytest.raises(pa.ArrowInvalid):
        cast_arrow_array(source, target_type, strict_options)


def test_cast_arrow_array_best_effort_unsafe():
    options = ArrowCastOptions(safe=True)
    data = pa.array([1.2, 3.4])

    try:
        cast_arrow_array(data, pa.int64(), options)
    except pa.ArrowInvalid:
        pass
    else:
        raise AssertionError("Expected ArrowInvalid when safe casting truncates values")

    unsafe_result = cast_arrow_array(data, pa.int64())
    assert unsafe_result.to_pylist() == [1, 3]


def test_cast_arrow_array_no_op_when_types_match():
    arr = pa.array([1, 2, 3], type=pa.int64())

    result = cast_arrow_array(arr, pa.int64())

    assert result is arr


def test_cast_arrow_array_chunked_round_trip_and_cast():
    chunked = pa.chunked_array([pa.array([1, 2]), pa.array([3])], type=pa.int64())

    same_type = cast_arrow_array(chunked, pa.int64())
    assert same_type is chunked

    casted = cast_arrow_array(chunked, pa.float64())
    assert isinstance(casted, pa.ChunkedArray)
    assert len(casted.chunks) == 2
    assert casted.type == pa.float64()
    assert casted.chunk(0).type == pa.float64()


def test_cast_arrow_table_fills_missing_columns():
    data = pa.table({"a": [1, 2]})
    schema = pa.schema([
        pa.field("a", pa.int64()),
        pa.field("b", pa.string()),
        pa.field("c", pa.int64(), nullable=False),
    ])

    result = cast_arrow_table(data, schema)

    assert result.schema == schema
    assert result["a"].type == pa.int64()
    assert result["b"].null_count == 2
    assert result["c"].to_pylist() == [0, 0]


def test_cast_arrow_table_case_insensitive_names():
    data = pa.table({"A": [1], "B": ["x"]})
    schema = pa.schema([
        pa.field("a", pa.float64()),
        pa.field("b", pa.large_string()),
    ])

    result = cast_arrow_table(data, schema)

    assert result.schema == schema
    assert result["a"].to_pylist() == [1.0]
    assert result["b"].to_pylist() == ["x"]

    with pytest.raises(pa.ArrowInvalid):
        cast_arrow_table(
            data,
            schema,
            ArrowCastOptions(strict_match_names=True, add_missing_columns=False),
        )


def test_cast_arrow_batch_matches_table_behavior():
    batch = pa.record_batch({"a": [1, 2]})
    schema = pa.schema([
        pa.field("a", pa.float64()),
        pa.field(
            "b",
            pa.struct([pa.field("c", pa.decimal128(4, 1), nullable=False)]),
            nullable=False,
        ),
    ])

    result = cast_arrow_batch(batch, schema)

    assert result.schema == schema
    assert result.column(0).type == pa.float64()
    assert result.column(1).type == schema.field("b").type
    assert result.column(1).field("c").to_pylist() == [decimal.Decimal(0), decimal.Decimal(0)]


def test_cast_arrow_batch_case_insensitive_names():
    batch = pa.record_batch({"A": [1], "B": [2]})
    schema = pa.schema([
        pa.field("a", pa.int32()),
        pa.field("b", pa.int32()),
    ])

    result = cast_arrow_batch(batch, schema)

    assert result.column(0).to_pylist() == [1]
    assert result.column(1).to_pylist() == [2]

    with pytest.raises(pa.ArrowInvalid):
        cast_arrow_batch(
            batch,
            schema,
            ArrowCastOptions(strict_match_names=True, add_missing_columns=False),
        )
