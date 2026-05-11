"""Pandas-side casts for :class:`ArrayType`.

Pandas has no first-class list dtype, so list-shaped values flow
through pyarrow under the hood (``cast_pandas_list_series`` →
``_cast_pandas_via_arrow`` → cast_arrow_list_array). The tests here
lock in the surface contract: name + index preservation, the
all-null-rows fast path, and the :meth:`ArrayType.default_pandas_series`
shape.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested.array import (
    ArrayType,
    cast_pandas_list_series,
)
from yggdrasil.data.types.primitive import StringType

pd = pytest.importorskip("pandas")


class TestCastPandasListSeries:

    def test_changes_item_dtype(
        self,
        source_array_field: Field,
        target_array_field: Field,
    ) -> None:
        series = pd.Series(
            [[1, 2], [3, None], None],
            name="source_array",
            dtype="object",
        )

        out = cast_pandas_list_series(
            series,
            CastOptions(
                source_field=source_array_field,
                target_field=target_array_field,
            ),
        )

        assert out.name == "target_array"
        # ``casted.to_pandas()`` surfaces list cells as numpy object arrays
        # — the standard pyarrow → pandas mapping.  Normalise to Python
        # lists before comparing so this stays a value contract, not a
        # numpy-vs-list contract.
        materialised = [None if v is None else list(v) for v in out]
        assert materialised == [["1", "2"], ["3", None], None]

    def test_preserves_index_and_index_name(
        self,
        source_array_field: Field,
        target_array_field: Field,
    ) -> None:
        series = pd.Series(
            [[1], [2], None],
            name="source_array",
            dtype="object",
            index=pd.Index([10, 20, 30], name="idx"),
        )

        out = cast_pandas_list_series(
            series,
            CastOptions(
                source_field=source_array_field,
                target_field=target_array_field,
            ),
        )

        assert list(out.index) == [10, 20, 30]
        assert out.index.name == "idx"

    def test_all_null_rows_pass_through(
        self,
        source_array_field: Field,
        target_array_field: Field,
    ) -> None:
        series = pd.Series(
            [None, None, None], name="source_array", dtype="object"
        )

        out = cast_pandas_list_series(
            series,
            CastOptions(
                source_field=source_array_field,
                target_field=target_array_field,
            ),
        )

        assert out.tolist() == [None, None, None]


class TestDefaultPandasSeries:

    def test_nullable_size_returns_all_null_series(self) -> None:
        dtype = ArrayType(
            item_field=Field(name="item", dtype=StringType(), nullable=True),
        )

        series = dtype.default_pandas_series(size=3, nullable=True)

        assert len(series) == 3
        assert series.isna().all()
