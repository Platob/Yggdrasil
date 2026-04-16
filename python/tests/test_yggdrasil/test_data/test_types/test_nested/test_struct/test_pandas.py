"""Pandas integration tests for StructType casts."""
from __future__ import annotations

import pytest

from yggdrasil.data import CastOptions, Field, Schema
from yggdrasil.data.types.nested.struct import (
    cast_pandas_list_series,
    cast_pandas_struct_series,
    cast_pandas_tabular,
)

from ._helpers import normalize_nested

pd = pytest.importorskip("pandas")


def test_cast_pandas_struct_series_reorders_fields_and_fills_missing(
    source_struct_field: Field,
    target_struct_field: Field,
) -> None:
    series = pd.Series(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, None],
        name="source_struct",
        dtype="object",
    )

    options = CastOptions(
        source_field=source_struct_field,
        target_field=target_struct_field,
    )

    result = cast_pandas_struct_series(series, options)

    assert result.name == "target_struct"
    assert [normalize_nested(v) for v in result.tolist()] == [
        {"b": "x", "c": None, "a": 1},
        {"b": "y", "c": None, "a": 2},
        None,
    ]


def test_cast_pandas_list_series_maps_by_position_and_fills_missing(
    source_list_field: Field,
    target_list_to_struct_field: Field,
) -> None:
    series = pd.Series(
        [[1, 2, 3], [4], None], name="source_list", dtype="object"
    )

    options = CastOptions(
        source_field=source_list_field,
        target_field=target_list_to_struct_field,
    )

    result = cast_pandas_list_series(series, options)

    assert result.name == "target_struct"
    assert [normalize_nested(v) for v in result.tolist()] == [
        {"first": 1, "second": "2", "third": 3},
        {"first": 4, "second": None, "third": None},
        None,
    ]


def test_cast_pandas_tabular_reorders_columns_and_adds_missing(
    source_tabular_schema: Schema,
    target_tabular_schema: Schema,
) -> None:
    frame = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})

    options = CastOptions(
        source_field=source_tabular_schema,
        target_field=target_tabular_schema,
    )

    result = cast_pandas_tabular(frame, options)

    assert list(result.columns) == ["b", "c", "a"]
    assert [
        normalize_nested(v) for v in result.to_dict(orient="records")
    ] == [
        {"b": "x", "c": None, "a": 1},
        {"b": "y", "c": None, "a": 2},
        {"b": "z", "c": None, "a": None},
    ]
