"""Polars-side casts for :class:`MapType`.

Polars has no first-class map dtype — yggdrasil represents map as
``list<struct<key, value>>`` on the Polars side. Tests cover:

* Dtype probes — ``to_polars`` / ``handles_polars_type`` /
  ``from_polars_type``.
* Cast helpers — value recast (map→map), list→map, struct→map.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested import MapType
from yggdrasil.data.types.nested.map import (
    cast_polars_list_series_to_map,
    cast_polars_map_series,
    cast_polars_struct_series_to_map,
)

from ._helpers import normalize_map_like

polars = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# Dtype probes
# ---------------------------------------------------------------------------


class TestPolarsDtype:

    def test_to_polars_emits_list_of_struct(
        self, int64_type, string_type
    ) -> None:
        polars_type = MapType.from_key_value(string_type, int64_type).to_polars()

        assert isinstance(polars_type, polars.List)
        assert isinstance(polars_type.inner, polars.Struct)

    def test_from_polars_round_trip(self, int64_type, string_type) -> None:
        original = MapType.from_key_value(string_type, int64_type)

        rebuilt = MapType.from_polars_type(original.to_polars())

        assert isinstance(rebuilt, MapType)
        assert rebuilt.key_field.name == "key"
        assert rebuilt.value_field.name == "value"

    def test_handles_polars_type_only_for_list_of_struct(self) -> None:
        assert MapType.handles_polars_type(polars.Int64()) is False
        assert MapType.handles_polars_type(polars.List(polars.Int64())) is False
        assert (
            MapType.handles_polars_type(
                polars.List(
                    polars.Struct(
                        {"key": polars.Utf8(), "value": polars.Int64()}
                    )
                )
            )
            is True
        )

    def test_from_polars_rejects_wrong_inner_shape(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Polars data type"):
            MapType.from_polars_type(polars.List(polars.Int64()))


# ---------------------------------------------------------------------------
# Cast helpers
# ---------------------------------------------------------------------------


class TestMapSeries:

    def test_recasts_values(
        self,
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

        out = cast_polars_map_series(
            series,
            CastOptions(
                source_field=source_map_field,
                target_field=target_map_field,
            ),
        )

        assert out.name == "target_map"
        assert [normalize_map_like(v) for v in out.to_list()] == [
            {"a": "1", "b": "2"},
            {"x": "3"},
            None,
        ]


class TestListSeriesToMap:

    def test_list_of_struct_collapses_into_map(
        self,
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

        out = cast_polars_list_series_to_map(
            series,
            CastOptions(
                source_field=source_list_of_struct_field,
                target_field=target_map_field,
            ),
        )

        assert out.name == "target_map"
        assert [normalize_map_like(v) for v in out.to_list()] == [
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
        series = polars.Series(
            "source_struct",
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": None, "c": 6},
                None,
            ],
        )

        out = cast_polars_struct_series_to_map(
            series,
            CastOptions(
                source_field=source_struct_to_map_field,
                target_field=target_map_field,
            ),
        )

        assert out.name == "target_map"
        assert [normalize_map_like(v) for v in out.to_list()] == [
            {"a": "1", "b": "2", "c": "3"},
            {"a": "4", "b": None, "c": "6"},
            None,
        ]
