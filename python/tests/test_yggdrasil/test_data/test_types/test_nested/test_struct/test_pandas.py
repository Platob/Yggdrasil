"""Pandas-side casts targeting :class:`StructType`.

Pandas has no native struct dtype, so each casting helper round-trips
through Python objects. The tests here lock in the same reorder + fill
contract the Arrow / Polars / Spark layers honour, plus a sanity test
on null propagation through pandas' ``object`` dtype.
"""
from __future__ import annotations

import pytest

from yggdrasil.data import Field, Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested.struct import (
    cast_pandas_list_series,
    cast_pandas_struct_series,
    cast_pandas_tabular,
)
from ._helpers import normalize_nested

pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Series: struct → struct
# ---------------------------------------------------------------------------


class TestCastStructSeries:
    def test_reorders_and_fills_missing_fields(
        self,
        source_struct_field: Field,
        target_struct_field: Field,
    ) -> None:
        series = pd.Series(
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, None],
            name="source_struct",
            dtype="object",
        )

        options = CastOptions(
            source=source_struct_field,
            target=target_struct_field,
        )

        result = cast_pandas_struct_series(series, options)

        assert result.name == "target_struct"
        assert [normalize_nested(v) for v in result.tolist()] == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            None,
        ]

    def test_preserves_null_rows(
        self,
        source_struct_field: Field,
        target_struct_field: Field,
    ) -> None:
        series = pd.Series(
            [None, {"a": 1, "b": "x"}, None],
            name="source_struct",
            dtype="object",
        )

        options = CastOptions(
            source=source_struct_field,
            target=target_struct_field,
        )

        result = cast_pandas_struct_series(series, options)

        rows = [normalize_nested(v) for v in result.tolist()]
        assert rows[0] is None
        assert rows[2] is None
        assert rows[1] == {"b": "x", "c": None, "a": 1}


# ---------------------------------------------------------------------------
# Series: list → struct (positional)
# ---------------------------------------------------------------------------


class TestCastListSeries:
    def test_maps_by_position_and_fills_missing(
        self,
        source_list_field: Field,
        target_list_to_struct_field: Field,
    ) -> None:
        series = pd.Series(
            [[1, 2, 3], [4], None], name="source_list", dtype="object"
        )

        options = CastOptions(
            source=source_list_field,
            target=target_list_to_struct_field,
        )

        result = cast_pandas_list_series(series, options)

        assert result.name == "target_struct"
        assert [normalize_nested(v) for v in result.tolist()] == [
            {"first": 1, "second": "2", "third": 3},
            {"first": 4, "second": None, "third": None},
            None,
        ]


# ---------------------------------------------------------------------------
# Tabular cast (DataFrame)
# ---------------------------------------------------------------------------


class TestCastTabular:
    def test_reorders_columns_and_adds_missing(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        frame = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})

        options = CastOptions(
            source=source_tabular_schema,
            target=target_tabular_schema,
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

    def test_empty_frame_keeps_target_columns(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        frame = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="object")})

        options = CastOptions(
            source=source_tabular_schema,
            target=target_tabular_schema,
        )

        result = cast_pandas_tabular(frame, options)

        assert list(result.columns) == ["b", "c", "a"]
        assert len(result) == 0
