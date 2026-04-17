"""Polars integration tests for MapType casts."""
from __future__ import annotations

import pytest

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import MapType
from yggdrasil.data.types.nested.map import (
    cast_polars_list_series_to_map,
    cast_polars_map_series,
    cast_polars_struct_series_to_map,
)

from ._helpers import normalize_map_like

polars = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# Dtype round-trip
# ---------------------------------------------------------------------------


def test_from_polars_type_round_trip(int64_type, string_type) -> None:
    dtype = MapType.from_key_value(string_type, int64_type)
    polars_type = dtype.to_polars()

    assert isinstance(polars_type, polars.List)
    assert isinstance(polars_type.inner, polars.Struct)

    rebuilt = MapType.from_polars_type(polars_type)
    assert isinstance(rebuilt, MapType)
    assert rebuilt.key_field.name == "key"
    assert rebuilt.value_field.name == "value"


def test_handles_polars_type_requires_list_of_struct() -> None:
    assert MapType.handles_polars_type(polars.Int64()) is False
    assert MapType.handles_polars_type(polars.List(polars.Int64())) is False
    assert (
        MapType.handles_polars_type(
            polars.List(polars.Struct({"key": polars.Utf8(), "value": polars.Int64()}))
        )
        is True
    )


def test_from_polars_type_rejects_wrong_shape() -> None:
    with pytest.raises(TypeError, match="Unsupported Polars data type"):
        MapType.from_polars_type(polars.List(polars.Int64()))


# ---------------------------------------------------------------------------
# Cast functions
# ---------------------------------------------------------------------------


def test_cast_polars_map_series_recasts_values(
    source_map_field: Field,
    target_map_field: Field,
) -> None:
    series = polars.Series(
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
    assert [normalize_map_like(v) for v in result.to_list()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_polars_list_series_to_map(
    source_list_of_struct_field: Field,
    target_map_field: Field,
) -> None:
    series = polars.Series(
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
    assert [normalize_map_like(v) for v in result.to_list()] == [
        {"a": "1", "b": "2"},
        {"x": "3"},
        None,
    ]


def test_cast_polars_struct_series_to_map(
    source_struct_to_map_field: Field,
    target_map_field: Field,
) -> None:
    series = polars.Series(
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
    assert [normalize_map_like(v) for v in result.to_list()] == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": None, "c": "6"},
        None,
    ]
