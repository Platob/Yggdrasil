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
            source_field=source_struct_field,
            target_field=target_struct_field,
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

    def test_empty_frame_keeps_target_columns(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        frame = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="object")})

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=target_tabular_schema,
        )

        result = cast_pandas_tabular(frame, options)

        assert list(result.columns) == ["b", "c", "a"]
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tabular cast — tricky column dtypes
#
# Pandas has no native struct / decimal / tz-timestamp dtype the way
# Arrow does, so the rebuild dips through pyarrow on the canonical
# path. The assertions stay shape-only (column order, per-row values
# after a normalize pass) since exact pandas dtypes drift between
# pyarrow / numpy backends.
# ---------------------------------------------------------------------------


class TestCastTabularTrickyTypes:
    @staticmethod
    def _build_frame(source_schema: Schema) -> "pd.DataFrame":
        from datetime import datetime, timezone
        from decimal import Decimal

        # Build via pyarrow so the source pandas frame actually carries
        # decimal / tz-timestamp / struct cells (object dtype on the
        # pandas side) instead of pandas' default coercions.
        import pyarrow as pa

        arrow_schema = source_schema.to_arrow_schema()
        table = pa.Table.from_pylist(
            [
                {
                    "drop_me": 99,
                    "amount": Decimal("1.23"),
                    "ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    "nested": {"x": 10, "y": "a"},
                    "items": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}],
                    "name": "row-1",
                },
                {
                    "drop_me": 100,
                    "amount": Decimal("4.56"),
                    "ts": datetime(2024, 6, 30, 12, 30, tzinfo=timezone.utc),
                    "nested": {"x": 20, "y": "b"},
                    "items": [{"x": 3, "y": "c"}],
                    "name": "row-2",
                },
            ],
            schema=arrow_schema,
        )
        return table.to_pandas()

    def test_reorders_selects_and_preserves_tricky_dtypes(
        self,
        tricky_source_schema: Schema,
        tricky_target_schema: Schema,
    ) -> None:
        from decimal import Decimal

        frame = self._build_frame(tricky_source_schema)

        options = CastOptions(
            source_field=tricky_source_schema,
            target_field=tricky_target_schema,
        )

        result = cast_pandas_tabular(frame, options)

        # Column order matches target schema; ``drop_me`` is selected
        # out; ``missing`` is appended with null cells.
        assert list(result.columns) == [
            "ts",
            "amount",
            "items",
            "nested",
            "name",
            "missing",
        ]
        assert len(result) == 2

        first = {
            k: normalize_nested(v)
            for k, v in result.iloc[0].to_dict().items()
        }
        # Nested struct children swapped (y before x) and list<struct>
        # children swapped too.
        assert first["nested"] == {"y": "a", "x": 10}
        assert first["items"] == [{"y": "a", "x": 1}, {"y": "b", "x": 2}]
        assert first["amount"] == Decimal("1.23")
        assert first["missing"] is None

        second_nested = normalize_nested(result.iloc[1]["nested"])
        assert second_nested == {"y": "b", "x": 20}
        assert normalize_nested(result.iloc[1]["amount"]) == Decimal("4.56")

    def test_widens_integer_dtype_during_reorder(self, string_type) -> None:
        from yggdrasil.data.types import IntegerType

        int32 = IntegerType(byte_size=4, signed=True)
        int64 = IntegerType(byte_size=8, signed=True)

        source = Schema(
            inner_fields=[
                Field(name="small", dtype=int32, nullable=True),
                Field(name="label", dtype=string_type, nullable=True),
            ]
        )
        target = Schema(
            inner_fields=[
                Field(name="label", dtype=string_type, nullable=True),
                Field(name="small", dtype=int64, nullable=True),
            ]
        )

        frame = pd.DataFrame(
            {
                "small": pd.Series([1, 2, 3], dtype="int32"),
                "label": pd.Series(["a", "b", "c"], dtype="object"),
            }
        )

        result = cast_pandas_tabular(
            frame, CastOptions(source_field=source, target_field=target)
        )

        assert list(result.columns) == ["label", "small"]
        rows = [
            {k: normalize_nested(v) for k, v in row.items()}
            for row in result.to_dict(orient="records")
        ]
        assert rows == [
            {"label": "a", "small": 1},
            {"label": "b", "small": 2},
            {"label": "c", "small": 3},
        ]
