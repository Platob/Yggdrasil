"""Pandas-side casts for :class:`MapType` (via the pyarrow bridge).

Three cast helpers — same shapes as Arrow:

* ``cast_pandas_map_series`` — map → map value recast.
* ``cast_pandas_list_series_to_map`` — list-of-struct → map.
* ``cast_pandas_struct_series_to_map`` — struct → map.

Each lands at pyarrow under the hood, so these tests pin the
end-to-end shape (name preservation + per-row content) rather than
duplicating Arrow assertions.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested.map import (
    cast_pandas_list_series_to_map,
    cast_pandas_map_series,
    cast_pandas_struct_series_to_map,
)

from ._helpers import normalize_map_like

pd = pytest.importorskip("pandas")


class TestMapSeries:

    def test_recasts_values(
        self,
        source_map_field: Field,
        target_map_field: Field,
    ) -> None:
        series = pd.Series(
            [{"a": 1, "b": 2}, {"x": 3}, None],
            name="source_map",
            dtype="object",
        )

        out = cast_pandas_map_series(
            series,
            CastOptions(
                source_field=source_map_field,
                target_field=target_map_field,
            ),
        )

        assert out.name == "target_map"
        assert [normalize_map_like(v) for v in out.tolist()] == [
            {"a": "1", "b": "2"},
            {"x": "3"},
            None,
        ]


class TestListSeriesToMap:

    def test_collapses_list_of_struct_into_map(
        self,
        source_list_of_struct_field: Field,
        target_map_field: Field,
    ) -> None:
        series = pd.Series(
            [
                [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
                [{"key": "x", "value": 3}],
                None,
            ],
            name="source_entries",
            dtype="object",
        )

        out = cast_pandas_list_series_to_map(
            series,
            CastOptions(
                source_field=source_list_of_struct_field,
                target_field=target_map_field,
            ),
        )

        assert out.name == "target_map"
        assert [normalize_map_like(v) for v in out.tolist()] == [
            {"a": "1", "b": "2"},
            {"x": "3"},
            None,
        ]


class TestStructSeriesToMap:

    def test_field_names_become_keys(
        self,
        source_struct_to_map_field: Field,
        target_map_field: Field,
    ) -> None:
        series = pd.Series(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": None, "c": 6},
                None,
            ],
            name="source_struct",
            dtype="object",
        )

        out = cast_pandas_struct_series_to_map(
            series,
            CastOptions(
                source_field=source_struct_to_map_field,
                target_field=target_map_field,
            ),
        )

        assert out.name == "target_map"
        assert [normalize_map_like(v) for v in out.tolist()] == [
            {"a": "1", "b": "2", "c": "3"},
            {"a": "4", "b": None, "c": "6"},
            None,
        ]
