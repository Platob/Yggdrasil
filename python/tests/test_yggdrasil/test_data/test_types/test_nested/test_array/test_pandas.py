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
                source=source_array_field,
                target=target_array_field,
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
                source=source_array_field,
                target=target_array_field,
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
                source=source_array_field,
                target=target_array_field,
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


class TestPandasListViewBridge:
    """Pandas accepts list_view via ``Array.to_pandas`` (it surfaces as
    an object Series of numpy arrays). The arrow-backed cast pipeline
    routes the same list_view → list normalisation we apply on the
    Arrow side, so the cast survives and the resulting Series is
    drop-in usable.
    """

    def test_list_view_struct_arrow_to_pandas_round_trip(self) -> None:
        import pyarrow as pa
        from yggdrasil.data.data_field import Field as F
        from yggdrasil.data.types.nested import StructType
        from yggdrasil.data.types.nested.array import cast_arrow_list_array
        from yggdrasil.data.types.primitive import IntegerType, StringType as ST

        items_per_row, rows = 12, 16
        struct_t = pa.struct([
            ("id", pa.int64()), ("name", pa.string()),
            ("amt", pa.float64()),
        ])
        payload = [
            None if (r % 5 == 0) else [
                {"id": r * items_per_row + k, "name": f"r{r}-k{k}", "amt": 1.0 * k}
                for k in range(items_per_row)
            ]
            for r in range(rows)
        ]
        src_view = pa.array(payload, type=pa.list_view(struct_t))

        # 1) Cast list_view -> list (so pandas can consume it).
        list_target = F(
            "rows",
            ArrayType.from_item(F(
                "item",
                StructType(fields=(
                    F("id", IntegerType(byte_size=4, signed=True)),
                    F("name", ST()),
                    F("amt", IntegerType(byte_size=8, signed=True)),
                )),
            )),
        )
        as_list = cast_arrow_list_array(
            src_view, CastOptions(target=list_target),
        )

        # 2) pandas materialises list cells as object/numpy values; we
        #    only need the cell shape + sentinel-null contract to hold.
        series = as_list.to_pandas()
        assert len(series) == rows
        assert series.iloc[0] is None
        cell = series.iloc[1]
        assert len(cell) == items_per_row
        assert cell[0]["id"] == 1 * items_per_row + 0
        assert cell[0]["name"] == "r1-k0"
