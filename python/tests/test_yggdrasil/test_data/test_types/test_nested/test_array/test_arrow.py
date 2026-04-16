"""Arrow integration tests for ArrayType casts.

These exercise cast_arrow_list_array and cast_arrow_map_array_to_list
across every list flavor (regular, large, fixed-size, chunked, view)
and validate the rejection paths that guard against incompatible
target shapes.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested.array import (
    cast_arrow_list_array,
    cast_arrow_map_array_to_list,
)


# ---------------------------------------------------------------------------
# list -> list cast
# ---------------------------------------------------------------------------


def test_cast_list_to_list_changes_item_dtype(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    array = pa.array(
        [[1, 2], [3, None], None],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.ListArray)
    assert result.type == pa.list_(pa.string())
    assert result.to_pylist() == [["1", "2"], ["3", None], None]


def test_cast_list_to_large_list_preserves_values(
    source_array_field: Field,
    target_large_array_field: Field,
) -> None:
    array = pa.array([[1, 2], [3, None], None], type=pa.list_(pa.int64()))

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_large_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.LargeListArray)
    assert result.type == pa.large_list(pa.string())
    assert result.to_pylist() == [["1", "2"], ["3", None], None]


def test_cast_list_to_fixed_size_list_drops_null_rows(
    source_array_field: Field,
    target_fixed_array_field: Field,
) -> None:
    array = pa.array([[1, 2], [3, None], None], type=pa.list_(pa.int64()))

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_fixed_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.FixedSizeListArray)
    assert result.type == pa.list_(pa.string(), 2)
    # FixedSizeListArray.from_arrays with a mask materialises null rows as
    # absent rather than as Python None, so we get 2 visible rows.
    assert result.to_pylist() == [["1", "2"], ["3", None]]


def test_cast_chunked_list_to_list_preserves_chunks(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    chunk_1 = pa.array([[1, 2], None], type=pa.list_(pa.int64()))
    chunk_2 = pa.array([[3]], type=pa.list_(pa.int64()))
    array = pa.chunked_array([chunk_1, chunk_2], type=pa.list_(pa.int64()))

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.ChunkedArray)
    assert result.type == pa.list_(pa.string())
    assert result.to_pylist() == [["1", "2"], None, ["3"]]


def test_cast_preserves_null_mask_for_null_and_nested_null(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    array = pa.array(
        [None, [], [1, None, 3]],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert result.to_pylist() == [None, [], ["1", None, "3"]]
    assert result.is_null().to_pylist() == [True, False, False]


def test_cast_list_returns_input_when_target_is_none(
    source_array_field: Field,
) -> None:
    array = pa.array([[1, 2]], type=pa.list_(pa.int64()))
    options = CastOptions(source_field=source_array_field, target_field=None)

    assert cast_arrow_list_array(array, options) is array


def test_cast_list_raises_for_non_array_source(
    source_map_field: Field,
    target_array_field: Field,
) -> None:
    array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_list_array(array, options)


def test_cast_list_raises_when_target_is_list_view(
    source_array_field: Field,
    target_view_array_field: Field,
) -> None:
    array = pa.array([[1, 2]], type=pa.list_(pa.int64()))

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_view_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_list_array(array, options)


# ---------------------------------------------------------------------------
# map -> list<struct<key,value>> cast
# ---------------------------------------------------------------------------


def test_cast_map_to_list_of_entries_materialises_key_value_struct(
    source_map_field: Field,
    target_entries_array_field: Field,
) -> None:
    array = pa.array(
        [
            [("a", 1), ("b", 2)],
            [("c", None)],
            None,
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_entries_array_field,
    )

    result = cast_arrow_map_array_to_list(array, options)

    assert isinstance(result, pa.ListArray)
    assert result.type == pa.list_(
        pa.struct(
            [
                pa.field("key", pa.string(), nullable=False),
                pa.field("value", pa.string()),
            ]
        )
    )
    assert result.to_pylist() == [
        [{"key": "a", "value": "1"}, {"key": "b", "value": "2"}],
        [{"key": "c", "value": None}],
        None,
    ]


def test_cast_map_to_large_list_of_entries(
    source_map_field: Field,
    target_entries_large_array_field: Field,
) -> None:
    array = pa.array(
        [[("a", 1), ("b", 2)], [("c", None)], None],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_entries_large_array_field,
    )

    result = cast_arrow_map_array_to_list(array, options)

    assert isinstance(result, pa.LargeListArray)


def test_cast_chunked_map_to_list_of_entries(
    source_map_field: Field,
    target_entries_array_field: Field,
) -> None:
    chunk_1 = pa.array(
        [[("a", 1)], None],
        type=pa.map_(pa.string(), pa.int64()),
    )
    chunk_2 = pa.array(
        [[("b", 2), ("c", 3)]],
        type=pa.map_(pa.string(), pa.int64()),
    )
    array = pa.chunked_array(
        [chunk_1, chunk_2],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_entries_array_field,
    )

    result = cast_arrow_map_array_to_list(array, options)

    assert isinstance(result, pa.ChunkedArray)
    assert result.to_pylist() == [
        [{"key": "a", "value": "1"}],
        None,
        [{"key": "b", "value": "2"}, {"key": "c", "value": "3"}],
    ]


def test_cast_map_returns_input_when_target_is_none(
    source_map_field: Field,
) -> None:
    array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))
    options = CastOptions(source_field=source_map_field, target_field=None)

    assert cast_arrow_map_array_to_list(array, options) is array


def test_cast_map_raises_for_non_map_source(
    source_array_field: Field,
    target_entries_array_field: Field,
) -> None:
    array = pa.array([[1, 2]], type=pa.list_(pa.int64()))

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_entries_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_map_array_to_list(array, options)


def test_cast_map_raises_when_target_item_is_not_a_struct(
    source_map_field: Field,
    invalid_target_entries_scalar_array_field: Field,
) -> None:
    array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))

    options = CastOptions(
        source_field=source_map_field,
        target_field=invalid_target_entries_scalar_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_map_array_to_list(array, options)


def test_cast_map_raises_when_target_struct_has_wrong_arity(
    source_map_field: Field,
    invalid_target_entries_struct_one_field_array_field: Field,
) -> None:
    array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))

    options = CastOptions(
        source_field=source_map_field,
        target_field=invalid_target_entries_struct_one_field_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_map_array_to_list(array, options)
