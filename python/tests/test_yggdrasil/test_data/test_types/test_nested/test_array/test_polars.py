"""Polars-side casts for :class:`ArrayType`.

Two cast surfaces:

* ``cast_polars_list_series`` — eager Series cast.
* ``cast_polars_list_expr`` — expression-level cast that fires on
  ``DataFrame.select`` collect.

Plus the dtype probes (``handles_polars_type`` / ``from_polars_type``
/ ``to_polars``) that drive cross-engine schema inference.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested.array import (
    ArrayType,
    cast_polars_list_expr,
    cast_polars_list_series,
)
from yggdrasil.data.types.primitive import IntegerType

polars = pytest.importorskip("polars")


# ---------------------------------------------------------------------------
# Dtype probes
# ---------------------------------------------------------------------------


class TestPolarsDtype:

    def test_to_polars_emits_list(self) -> None:
        dtype = ArrayType(
            item_field=Field(
                name="item",
                dtype=IntegerType(byte_size=8, signed=True),
                nullable=True,
            ),
        )

        assert isinstance(dtype.to_polars(), polars.List)

    def test_from_polars_round_trip_preserves_inner_dtype(self) -> None:
        original = ArrayType(
            item_field=Field(
                name="item",
                dtype=IntegerType(byte_size=8, signed=True),
                nullable=True,
            ),
        )

        rebuilt = ArrayType.from_polars_type(original.to_polars())

        assert isinstance(rebuilt, ArrayType)
        assert rebuilt.item_field.dtype.type_id == DataTypeId.INT64

    def test_from_polars_rejects_non_list(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Polars data type"):
            ArrayType.from_polars_type(polars.Int64())

    def test_handles_polars_type_matches_list_only(self) -> None:
        assert ArrayType.handles_polars_type(polars.List(polars.Int64())) is True
        assert ArrayType.handles_polars_type(polars.Int64()) is False
        assert (
            ArrayType.handles_polars_type(
                polars.Struct({"a": polars.Int64()})
            )
            is False
        )


# ---------------------------------------------------------------------------
# Casts
# ---------------------------------------------------------------------------


class TestCastSeries:

    def test_changes_item_dtype(
        self,
        source_array_field: Field,
        target_array_field: Field,
    ) -> None:
        series = polars.Series(
            "source_array",
            [[1, 2], [3, None], None],
            dtype=polars.List(polars.Int64()),
        )

        out = cast_polars_list_series(
            series,
            CastOptions(
                source=source_array_field,
                target=target_array_field,
            ),
        )

        assert out.name == "target_array"
        assert isinstance(out.dtype, polars.List)
        assert out.to_list() == [["1", "2"], ["3", None], None]


class TestCastExpr:

    def test_select_integration(
        self,
        source_array_field: Field,
        target_array_field: Field,
    ) -> None:
        pl = polars
        frame = pl.DataFrame({"source_array": [[1, 2], [3, None], None]})

        out = frame.select(
            cast_polars_list_expr(
                pl.col("source_array"),
                CastOptions(
                    source=source_array_field,
                    target=target_array_field,
                ),
            ).alias("target_array")
        )["target_array"]

        assert out.to_list() == [["1", "2"], ["3", None], None]

    def test_target_none_returns_input_expression(
        self, source_array_field: Field
    ) -> None:
        pl = polars
        expr = pl.col("source_array")

        out = cast_polars_list_expr(
            expr,
            CastOptions(source=source_array_field, target=None),
        )

        assert out is expr
