"""Pandas integration tests for ArrayType.

Pandas' object dtype has no first-class list type, so these tests
route through pyarrow under the hood (``cast_pandas_list_series`` and
``_cast_pandas_via_arrow``).
"""
from __future__ import annotations

import pytest

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested.array import (
    ArrayType,
    cast_pandas_list_series,
)

pd = pytest.importorskip("pandas")


def test_cast_pandas_list_series_changes_item_dtype(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    series = pd.Series(
        [[1, 2], [3, None], None],
        name="source_array",
        dtype="object",
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_pandas_list_series(series, options)

    assert result.name == "target_array"
    assert result.tolist() == [["1", "2"], ["3", None], None]


def test_cast_pandas_list_series_preserves_index(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    series = pd.Series(
        [[1], [2], None],
        name="source_array",
        dtype="object",
        index=pd.Index([10, 20, 30], name="idx"),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_pandas_list_series(series, options)

    assert list(result.index) == [10, 20, 30]
    assert result.index.name == "idx"


def test_cast_pandas_list_series_handles_all_null_rows(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    series = pd.Series([None, None, None], name="source_array", dtype="object")

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_pandas_list_series(series, options)

    assert result.tolist() == [None, None, None]


def test_array_type_default_pandas_series_respects_nullable() -> None:
    dtype = ArrayType(
        item_field=Field(
            name="item",
            dtype=ArrayType.get_data_field_class().from_any("string").dtype,
            nullable=True,
        ),
    )

    series = dtype.default_pandas_series(size=3, nullable=True)

    assert len(series) == 3
    assert series.isna().all()
