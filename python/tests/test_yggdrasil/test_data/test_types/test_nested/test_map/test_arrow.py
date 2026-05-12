"""Arrow-side casts for :class:`MapType`.

Three cast surfaces:

* :func:`cast_arrow_map_array` — map<k, v1> → map<k, v2>
  (value-dtype recast).
* :func:`cast_arrow_list_array_to_map` — list<struct<key, value>> →
  map<key, value>; the entry struct must be 2-ary.
* :func:`cast_arrow_struct_array_to_map` — struct<a, b, c> →
  map<string, …>; field names become keys.

Plus the rejection paths (non-map source, non-list source) and
``keys_sorted`` round-trip.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested import MapType
from yggdrasil.data.types.nested.map import (
    cast_arrow_list_array_to_map,
    cast_arrow_map_array,
    cast_arrow_struct_array_to_map,
)

from ._helpers import normalize_map_like


# ---------------------------------------------------------------------------
# map → map (recast values)
# ---------------------------------------------------------------------------


class TestMapToMap:

    def test_changes_value_dtype(
        self,
        source_map_field: Field,
        target_map_field: Field,
    ) -> None:
        array = pa.array(
            [[("a", 1), ("b", 2)], [("x", 3)], None],
            type=pa.map_(pa.string(), pa.int64()),
        )

        result = cast_arrow_map_array(
            array,
            CastOptions(
                source=source_map_field,
                target=target_map_field,
            ),
        )

        assert isinstance(result, pa.MapArray)
        assert result.type == pa.map_(pa.string(), pa.string())
        assert [normalize_map_like(v) for v in result.to_pylist()] == [
            {"a": "1", "b": "2"},
            {"x": "3"},
            None,
        ]

    def test_target_none_returns_input(self, source_map_field: Field) -> None:
        array = pa.array(
            [[("a", 1)]], type=pa.map_(pa.string(), pa.int64())
        )

        out = cast_arrow_map_array(
            array,
            CastOptions(source=source_map_field, target=None),
        )

        assert out is array

    def test_non_map_source_raises(
        self,
        source_list_of_struct_field: Field,
        target_map_field: Field,
    ) -> None:
        entry_struct = pa.struct(
            [
                pa.field("key", pa.string(), nullable=False),
                pa.field("value", pa.int64()),
            ]
        )
        array = pa.array(
            [[{"key": "a", "value": 1}]], type=pa.list_(entry_struct)
        )

        with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
            cast_arrow_map_array(
                array,
                CastOptions(
                    source=source_list_of_struct_field,
                    target=target_map_field,
                ),
            )


# ---------------------------------------------------------------------------
# list<struct<k, v>> → map
# ---------------------------------------------------------------------------


class TestListToMap:

    def test_collapses_entries_into_map(
        self,
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

        result = cast_arrow_list_array_to_map(
            array,
            CastOptions(
                source=source_list_of_struct_field,
                target=target_map_field,
            ),
        )

        assert isinstance(result, pa.MapArray)
        assert [normalize_map_like(v) for v in result.to_pylist()] == [
            {"a": "1", "b": "2"},
            {"x": "3"},
            None,
        ]


# ---------------------------------------------------------------------------
# struct → map (field names become keys)
# ---------------------------------------------------------------------------


class TestStructToMap:

    def test_field_names_become_keys(
        self,
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

        result = cast_arrow_struct_array_to_map(
            array,
            CastOptions(
                source=source_struct_to_map_field,
                target=target_map_field,
            ),
        )

        assert isinstance(result, pa.MapArray)
        assert [normalize_map_like(v) for v in result.to_pylist()] == [
            {"a": "1", "b": "2", "c": "3"},
            {"a": "4", "b": None, "c": "6"},
            None,
        ]


# ---------------------------------------------------------------------------
# from_arrow_type — keys_sorted round-trip
# ---------------------------------------------------------------------------


class TestArrowTypeRoundTrip:

    def test_keys_sorted_round_trips(self) -> None:
        arrow_type = pa.map_(pa.string(), pa.int64(), keys_sorted=True)

        dtype = MapType.from_arrow_type(arrow_type)
        assert isinstance(dtype, MapType)
        assert dtype.keys_sorted is True

        produced = dtype.to_arrow()
        assert pa.types.is_map(produced)
        assert produced.keys_sorted is True
        assert produced.key_type == pa.string()
        assert produced.item_type == pa.int64()

    def test_rejects_non_map_arrow_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Arrow data type"):
            MapType.from_arrow_type(pa.list_(pa.int64()))
