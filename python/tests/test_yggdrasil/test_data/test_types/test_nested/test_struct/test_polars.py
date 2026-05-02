"""Polars-side casts targeting :class:`StructType`.

Two surfaces here: per-column expressions (``cast_polars_*_expr`` —
plan-level rewrite, fires on collect) and per-element materialized
casts (``cast_polars_*_series`` — eager, used when callers don't have
an expression context). Tabular cast covers both ``DataFrame`` and
``LazyFrame`` shapes.
"""
from __future__ import annotations

import pytest

from yggdrasil.data import Field, Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested.struct import (
    cast_polars_list_expr,
    cast_polars_list_series,
    cast_polars_map_expr,
    cast_polars_struct_expr,
    cast_polars_struct_series,
    cast_polars_tabular,
    cast_polars_list_series
)

polars = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# Eager Series casts
# ---------------------------------------------------------------------------


class TestStructSeries:
    def test_reorders_and_fills_missing_fields(
        self,
        source_struct_field: Field,
        target_struct_field: Field,
    ) -> None:
        series = polars.Series(
            "source_struct",
            [
                {"a": 1, "b": "x"},
                {"a": 2, "b": "y"},
                None,
            ],
        )

        options = CastOptions(
            source_field=source_struct_field,
            target_field=target_struct_field,
        )

        result = cast_polars_struct_series(series, options)

        assert result.name == "target_struct"
        assert result.to_list() == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            None,
        ]


class TestListSeries:
    def test_maps_by_position_and_fills_missing(
        self,
        source_list_field: Field,
        target_list_to_struct_field: Field,
    ) -> None:
        series = polars.Series("source_list", [[1, 2, 3], [4], None])

        options = CastOptions(
            source_field=source_list_field,
            target_field=target_list_to_struct_field,
        )

        result = cast_polars_list_series(series, options)

        assert result.name == "target_struct"
        assert result.to_list() == [
            {"first": 1, "second": "2", "third": 3},
            {"first": 4, "second": None, "third": None},
            None,
        ]


# ---------------------------------------------------------------------------
# Expression-level casts
# ---------------------------------------------------------------------------


class TestStructExpr:
    def test_reorders_fields_inside_a_select(
        self,
        source_struct_field: Field,
        target_struct_field: Field,
    ) -> None:
        pl = polars
        frame = pl.DataFrame(
            {
                "source_struct": [
                    {"a": 1, "b": "x"},
                    {"a": 2, "b": "y"},
                    None,
                ]
            }
        )

        options = CastOptions(
            source_field=source_struct_field,
            target_field=target_struct_field,
        )

        result = frame.select(
            cast_polars_struct_expr(pl.col("source_struct"), options).alias(
                "target_struct"
            )
        )

        assert result["target_struct"].to_list() == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            None,
        ]


class TestListExpr:
    def test_maps_by_position_inside_a_select(
        self,
        source_list_field: Field,
        target_list_to_struct_field: Field,
    ) -> None:
        pl = polars
        frame = pl.DataFrame({"source_list": [[1, 2, 3], [4], None]})

        options = CastOptions(
            source_field=source_list_field,
            target_field=target_list_to_struct_field,
        )

        result = frame.select(
            cast_polars_list_expr(pl.col("source_list"), options).alias(
                "target_struct"
            )
        )

        assert result["target_struct"].to_list() == [
            {"first": 1, "second": "2", "third": 3},
            {"first": 4, "second": None, "third": None},
            None,
        ]


class TestMapExpr:
    def test_extracts_named_keys_into_struct(
        self,
        source_map_field: Field,
        target_struct_field: Field,
    ) -> None:
        pl = polars
        frame = pl.DataFrame(
            {
                "source_map": [
                    [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
                    [{"key": "b", "value": 3}],
                    None,
                ]
            }
        )

        options = CastOptions(
            source_field=source_map_field,
            target_field=target_struct_field,
        )

        result = frame.select(
            cast_polars_map_expr(pl.col("source_map"), options).alias(
                "target_struct"
            )
        )

        assert result["target_struct"].to_list() == [
            {"b": "2", "c": None, "a": 1},
            {"b": "3", "c": None, "a": None},
            None,
        ]


# ---------------------------------------------------------------------------
# Tabular: DataFrame + LazyFrame
# ---------------------------------------------------------------------------


class TestTabular:
    def test_eager_reorders_and_adds_missing(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        pl = polars
        frame = pl.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=target_tabular_schema,
        )

        result = cast_polars_tabular(frame, options)

        assert result.columns == ["b", "c", "a"]
        assert result.to_dicts() == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            {"b": "z", "c": None, "a": None},
        ]

    def test_lazy_round_trips_via_collect(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        pl = polars
        frame = pl.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]}).lazy()

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=target_tabular_schema,
        )

        result = cast_polars_tabular(frame, options).collect()

        assert result.columns == ["b", "c", "a"]
        assert result.to_dicts() == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            {"b": "z", "c": None, "a": None},
        ]
