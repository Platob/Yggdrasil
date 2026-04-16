"""Polars integration tests for ArrayType.

Covers dtype round-trips and engine-level casting via
``cast_polars_list_expr`` / ``cast_polars_list_series``.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested.array import (
    ArrayType,
    cast_polars_list_expr,
    cast_polars_list_series,
)
from yggdrasil.data.types.primitive import IntegerType

polars = pytest.importorskip("polars")


def test_to_polars_emits_list_dtype() -> None:
    dtype = ArrayType(
        item_field=Field(
            name="item",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=True,
        ),
    )

    polars_dtype = dtype.to_polars()

    assert isinstance(polars_dtype, polars.List)


def test_from_polars_type_round_trip_preserves_item_dtype() -> None:
    original = ArrayType(
        item_field=Field(
            name="item",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=True,
        ),
    )

    rebuilt = ArrayType.from_polars_type(original.to_polars())

    assert isinstance(rebuilt, ArrayType)
    assert rebuilt.item_field.dtype.type_id == DataTypeId.INTEGER


def test_from_polars_type_rejects_non_list() -> None:
    with pytest.raises(TypeError, match="Unsupported Polars data type"):
        ArrayType.from_polars_type(polars.Int64())


def test_handles_polars_type_is_true_only_for_list() -> None:
    assert ArrayType.handles_polars_type(polars.List(polars.Int64())) is True
    assert ArrayType.handles_polars_type(polars.Int64()) is False
    assert ArrayType.handles_polars_type(polars.Struct({"a": polars.Int64()})) is False


def test_cast_polars_list_series_changes_item_dtype(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    series = polars.Series(
        "source_array",
        [[1, 2], [3, None], None],
        dtype=polars.List(polars.Int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_polars_list_series(series, options)

    assert result.name == "target_array"
    assert isinstance(result.dtype, polars.List)
    assert result.to_list() == [["1", "2"], ["3", None], None]


def test_cast_polars_list_expr_integrates_with_select(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    pl = polars
    frame = pl.DataFrame({"source_array": [[1, 2], [3, None], None]})

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = frame.select(
        cast_polars_list_expr(pl.col("source_array"), options).alias("target_array")
    )["target_array"]

    assert result.to_list() == [["1", "2"], ["3", None], None]


def test_cast_polars_list_expr_returns_original_when_target_is_none(
    source_array_field: Field,
) -> None:
    pl = polars
    options = CastOptions(source_field=source_array_field, target_field=None)
    expr = pl.col("source_array")

    assert cast_polars_list_expr(expr, options) is expr
