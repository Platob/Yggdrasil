"""Pandas integration tests for MapType casts (via pyarrow under the hood)."""
from __future__ import annotations

import pytest

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested.map import (
    cast_pandas_list_series_to_map,
    cast_pandas_map_series,
    cast_pandas_struct_series_to_map,
)

from ._helpers import normalize_map_like

pd = pytest.importorskip("pandas")


def test_cast_pandas_map_series_recasts_values(
    source_map_field: Field,
    target_map_field: Field,
) -> None:
    series = pd.Series(
        [{"a": 1, "b": 2}, {"x": 3}, None],
        name="source_map",
        dtype="object",
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_map_field,
    )

    result = cast_pandas_map_series(series, options)

    assert result.name == "target_map"
    assert [normalize_map_like(v) for v in result.tolist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_pandas_list_series_to_map(
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

    options = CastOptions(
        source_field=source_list_of_struct_field,
        target_field=target_map_field,
    )

    result = cast_pandas_list_series_to_map(series, options)

    assert result.name == "target_map"
    assert [normalize_map_like(v) for v in result.tolist()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_pandas_struct_series_to_map(
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

    options = CastOptions(
        source_field=source_struct_to_map_field,
        target_field=target_map_field,
    )

    result = cast_pandas_struct_series_to_map(series, options)

    assert result.name == "target_map"
    assert [normalize_map_like(v) for v in result.tolist()] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": None, "c": "6"},
        None,
    ]
