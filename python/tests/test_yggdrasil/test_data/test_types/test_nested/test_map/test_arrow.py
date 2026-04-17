"""Arrow integration tests for MapType casts."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import MapType
from yggdrasil.data.types.nested.map import (
    cast_arrow_list_array_to_map,
    cast_arrow_map_array,
    cast_arrow_struct_array_to_map,
)

from ._helpers import normalize_map_like


# ---------------------------------------------------------------------------
# map -> map value recast
# ---------------------------------------------------------------------------


def test_cast_map_to_map_changes_value_dtype(
    source_map_field: Field,
    target_map_field: Field,
) -> None:
    array = pa.array(
        [[("a", 1), ("b", 2)], [("x", 3)], None],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_map_field,
    )

    result = cast_arrow_map_array(array, options)

    assert isinstance(result, pa.MapArray)
    assert result.type == pa.map_(pa.string(), pa.string())
    assert [normalize_map_like(v) for v in result.to_pylist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_map_returns_input_when_target_is_none(
    source_map_field: Field,
) -> None:
    array = pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64()))
    options = CastOptions(source_field=source_map_field, target_field=None)

    assert cast_arrow_map_array(array, options) is array


def test_cast_map_rejects_non_map_source(
    source_list_of_struct_field: Field,
    target_map_field: Field,
) -> None:
    entry_struct = pa.struct(
        [
            pa.field("key", pa.string(), nullable=False),
            pa.field("value", pa.int64()),
        ]
    )
    array = pa.array([[{"key": "a", "value": 1}]], type=pa.list_(entry_struct))

    options = CastOptions(
        source_field=source_list_of_struct_field,
        target_field=target_map_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_map_array(array, options)


# ---------------------------------------------------------------------------
# list<struct<k, v>> -> map<k, v>
# ---------------------------------------------------------------------------


def test_cast_list_of_struct_to_map_collapses_entries(
    source_list_of_struct_field: Field,
    target_map_field: Field,
) -> None:
    entry_type = pa.struct(
        [pa.field("key", pa.string()), pa.field("value", pa.int64())]
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
    assert [normalize_map_like(v) for v in result.to_pylist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


# ---------------------------------------------------------------------------
# struct -> map (field names become keys)
# ---------------------------------------------------------------------------


def test_cast_struct_to_map_uses_field_names_as_keys(
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
    assert [normalize_map_like(v) for v in result.to_pylist()] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": None, "c": "6"},
        None,
    ]


# ---------------------------------------------------------------------------
# Identity / round-trips
# ---------------------------------------------------------------------------


def test_from_arrow_type_round_trip_preserves_keys_sorted() -> None:
    arrow_type = pa.map_(pa.string(), pa.int64(), keys_sorted=True)

    dtype = MapType.from_arrow_type(arrow_type)
    assert isinstance(dtype, MapType)
    assert dtype.keys_sorted is True

    produced = dtype.to_arrow()
    assert pa.types.is_map(produced)
    assert produced.keys_sorted is True
    assert produced.key_type == pa.string()
    assert produced.item_type == pa.int64()


def test_from_arrow_type_rejects_non_map() -> None:
    with pytest.raises(TypeError, match="Unsupported Arrow data type"):
        MapType.from_arrow_type(pa.list_(pa.int64()))
