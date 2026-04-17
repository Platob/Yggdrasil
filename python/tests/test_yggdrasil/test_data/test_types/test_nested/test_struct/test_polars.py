"""Polars integration tests for StructType casts."""
from __future__ import annotations

import pytest

from yggdrasil.data import CastOptions, Field, Schema
from yggdrasil.data.types.nested.struct import (
    cast_polars_list_expr,
    cast_polars_list_series,
    cast_polars_map_expr,
    cast_polars_struct_expr,
    cast_polars_struct_series,
    cast_polars_tabular,
)

polars = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# Series casts
# ---------------------------------------------------------------------------


def test_cast_polars_struct_series_reorders_fields_and_fills_missing(
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


def test_cast_polars_list_series_maps_by_position_and_fills_missing(
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


def test_cast_polars_struct_expr_reorders_fields(
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
        cast_polars_struct_expr(pl.col("source_struct"), options).alias("target_struct")
    )

    assert result["target_struct"].to_list() == [
        {"b": "x", "c": None, "a": 1},
        {"b": "y", "c": None, "a": 2},
        None,
    ]


def test_cast_polars_list_expr_maps_by_position(
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
        cast_polars_list_expr(pl.col("source_list"), options).alias("target_struct")
    )

    assert result["target_struct"].to_list() == [
        {"first": 1, "second": "2", "third": 3},
        {"first": 4, "second": None, "third": None},
        None,
    ]


def test_cast_polars_map_expr_extracts_named_keys_to_struct(
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
        cast_polars_map_expr(pl.col("source_map"), options).alias("target_struct")
    )

    assert result["target_struct"].to_list() == [
        {"b": "2", "c": None, "a": 1},
        {"b": "3", "c": None, "a": None},
        None,
    ]


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------


def test_cast_polars_tabular_reorders_columns_and_adds_missing(
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


def test_cast_polars_tabular_lazy_frame_round_trips(
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
