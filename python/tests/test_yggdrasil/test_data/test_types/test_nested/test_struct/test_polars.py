"""Polars-side tabular casts targeting :class:`StructType`.

The Polars path goes through ``cast_polars_tabular`` which assembles a
single ``select(...)`` expression in target field order. The tests
below lock in the same reorder + fill + select contract the Arrow /
Pandas / Spark layers honour, with a stress on dtypes that easily slip
past a "match by name" sweep: fixed-precision decimal, timestamp with
a non-naive timezone, and a list<struct> whose inner children also
swap order.
"""
from __future__ import annotations

import pytest

from yggdrasil.data import Field, Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested.struct import cast_polars_tabular

# Importing the polars cast module wires its converters into the
# registry; the actual cast helpers above already pull polars in
# lazily via the lib guard, but we still need polars present.
pl = pytest.importorskip("polars")

# Pull the cast module so register-on-import side effects fire even when
# the test is run in isolation.
import yggdrasil.polars.cast  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Basic reorder + fill (mirrors test_arrow.py::TestCastTabular)
# ---------------------------------------------------------------------------


class TestCastTabular:
    def test_dataframe_reorders_columns_and_adds_missing(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        frame = pl.DataFrame(
            {
                "a": [1, 2, None],
                "b": ["x", "y", "z"],
            },
            schema={"a": pl.Int64, "b": pl.Utf8},
        )

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=target_tabular_schema,
        )

        result = cast_polars_tabular(frame, options)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["b", "c", "a"]
        assert result.to_dicts() == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            {"b": "z", "c": None, "a": None},
        ]

    def test_lazyframe_round_trip_keeps_shape(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        frame = pl.DataFrame(
            {"a": [1, 2], "b": ["x", "y"]},
            schema={"a": pl.Int64, "b": pl.Utf8},
        ).lazy()

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=target_tabular_schema,
        )

        result = cast_polars_tabular(frame, options)

        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert collected.columns == ["b", "c", "a"]
        assert collected.to_dicts() == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
        ]

    def test_rejects_non_frame_input(self) -> None:
        options = CastOptions(
            source_field=Schema(inner_fields=[]),
            target_field=Schema(inner_fields=[]),
        )

        with pytest.raises(TypeError, match="Unsupported tabular type"):
            cast_polars_tabular({"a": [1, 2, 3]}, options)


# ---------------------------------------------------------------------------
# Tricky-type tabular casts
# ---------------------------------------------------------------------------


class TestCastTabularTrickyTypes:
    @staticmethod
    def _build_frame(source_schema: Schema) -> "pl.DataFrame":
        from datetime import datetime, timezone
        from decimal import Decimal

        # Build the DataFrame with explicit Polars dtypes so the source
        # schema sent to ``CastOptions`` actually matches the runtime
        # column dtypes — otherwise polars would auto-promote (e.g. to
        # Float64 for decimal cells).
        return pl.DataFrame(
            {
                "drop_me": [99, 100, None],
                "amount": [Decimal("1.23"), Decimal("4.56"), None],
                "ts": [
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 6, 30, 12, 30, tzinfo=timezone.utc),
                    None,
                ],
                "nested": [
                    {"x": 10, "y": "a"},
                    {"x": 20, "y": "b"},
                    None,
                ],
                "items": [
                    [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}],
                    [{"x": 3, "y": "c"}],
                    None,
                ],
                "name": ["row-1", "row-2", None],
            },
            schema={
                "drop_me": pl.Int64,
                "amount": pl.Decimal(10, 2),
                "ts": pl.Datetime("us", "UTC"),
                "nested": pl.Struct({"x": pl.Int64, "y": pl.Utf8}),
                "items": pl.List(pl.Struct({"x": pl.Int64, "y": pl.Utf8})),
                "name": pl.Utf8,
            },
        )

    def test_dataframe_reorders_selects_and_preserves_tricky_dtypes(
        self,
        tricky_source_schema: Schema,
        tricky_target_schema: Schema,
    ) -> None:
        frame = self._build_frame(tricky_source_schema)

        options = CastOptions(
            source_field=tricky_source_schema,
            target_field=tricky_target_schema,
        )

        result = cast_polars_tabular(frame, options)

        assert isinstance(result, pl.DataFrame)
        assert result.columns == [
            "ts",
            "amount",
            "items",
            "nested",
            "name",
            "missing",
        ]
        # Tricky dtypes survive the rebuild.
        assert result.schema["amount"] == pl.Decimal(10, 2)
        assert result.schema["ts"] == pl.Datetime("us", "UTC")
        # Nested children also reordered (y before x).
        assert result.schema["nested"] == pl.Struct(
            {"y": pl.Utf8, "x": pl.Int64}
        )
        assert result.schema["items"] == pl.List(
            pl.Struct({"y": pl.Utf8, "x": pl.Int64})
        )
        assert result.schema["missing"] == pl.Int64

        rows = result.to_dicts()
        assert rows[0]["nested"] == {"y": "a", "x": 10}
        assert rows[0]["items"] == [{"y": "a", "x": 1}, {"y": "b", "x": 2}]
        assert rows[0]["missing"] is None
        assert rows[1]["nested"] == {"y": "b", "x": 20}
        # Final row was all-null — every output column stays null.
        assert rows[2]["amount"] is None
        assert rows[2]["ts"] is None
        # Polars represents null struct rows as ``{...: None, ...: None}``
        # rather than a single ``None`` (no native null-struct), so we
        # just check the contents are all null.
        nested_last = rows[2]["nested"]
        assert nested_last is None or all(v is None for v in nested_last.values())

    def test_lazyframe_path_matches_dataframe_path(
        self,
        tricky_source_schema: Schema,
        tricky_target_schema: Schema,
    ) -> None:
        frame = self._build_frame(tricky_source_schema)
        options = CastOptions(
            source_field=tricky_source_schema,
            target_field=tricky_target_schema,
        )

        eager = cast_polars_tabular(frame, options)
        lazy = cast_polars_tabular(frame.lazy(), options).collect()

        assert isinstance(lazy, pl.DataFrame)
        assert lazy.columns == eager.columns
        assert lazy.schema == eager.schema
        assert lazy.to_dicts() == eager.to_dicts()

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

        frame = pl.DataFrame(
            {"small": [1, 2, 3], "label": ["a", "b", "c"]},
            schema={"small": pl.Int32, "label": pl.Utf8},
        )

        result = cast_polars_tabular(
            frame, CastOptions(source_field=source, target_field=target)
        )

        assert result.columns == ["label", "small"]
        assert result.schema["small"] == pl.Int64
        assert result.to_dicts() == [
            {"label": "a", "small": 1},
            {"label": "b", "small": 2},
            {"label": "c", "small": 3},
        ]
