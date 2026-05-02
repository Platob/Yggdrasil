"""``Field`` runtime API — defaults, fills, casts, and Arrow metadata.

These tests cover the parts of :class:`Field` that don't fit cleanly
into ``construction``, ``merge``, ``arrow``, or ``equals`` — mainly
the per-engine default-array / fill-nulls helpers, plus a few smoke
tests on Arrow-side metadata round-trips and the ``from_str``
nullability-suffix shorthand.
"""
from __future__ import annotations

import pandas as pd
import polars as pl
import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.primitive import IntegerType, StringType


class TestFieldDefault:

    def test_with_default_updates_metadata_backed_value(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
        )
        f = f.with_default(7)

        assert f.has_default is True
        assert f.default == 7
        assert f.default_arrow_scalar.as_py() == 7


class TestFieldArrowDefaults:

    def test_default_arrow_array_uses_field_default(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            default=11,
        )

        assert f.default_arrow_array(size=3).to_pylist() == [11, 11, 11]

    def test_fill_arrow_array_nulls_uses_field_default(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            default=9,
        )
        arr = pa.array([1, None, 3], type=pa.int64())

        assert f.fill_arrow_array_nulls(arr).to_pylist() == [1, 9, 3]


class TestFieldPandasHelpers:

    def test_default_pandas_series_uses_name_and_default(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            default=5,
        )

        out = f.default_pandas_series(size=3)

        assert out.name == "qty"
        assert out.tolist() == [5, 5, 5]

    def test_fill_pandas_series_nulls_uses_field_default(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            default=5,
        )
        series = pd.Series([1, None, 3], name="qty")

        assert f.fill_pandas_series_nulls(series).tolist() == [1.0, 5.0, 3.0]

    def test_cast_pandas_series_routes_through_dtype(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
        )
        series = pd.Series([1, 2, 3], name="qty")

        out = f.cast_pandas_series(series)

        assert out.name == "qty"
        assert out.tolist() == [1, 2, 3]


class TestFieldPolarsHelpers:

    def test_default_polars_series_uses_name_and_default(self) -> None:
        f = Field(
            name="book_id",
            dtype=StringType(),
            nullable=False,
            default="NA",
        )

        out = f.default_polars_series(size=2)

        assert out.name == "book_id"
        assert out.to_list() == ["NA", "NA"]

    def test_fill_polars_series_nulls_uses_field_default(self) -> None:
        f = Field(
            name="book_id",
            dtype=StringType(),
            nullable=False,
            default="NA",
        )
        series = pl.Series("book_id", ["A", None, "C"])

        assert f.fill_polars_array_nulls(series).to_list() == ["A", "NA", "C"]

    def test_cast_polars_series_routes_through_dtype(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
        )
        series = pl.Series("qty", [1, 2, 3])

        out = f.cast_polars_series(series)

        assert out.name == "qty"
        assert out.to_list() == [1, 2, 3]


class TestFieldArrowMetadata:

    def test_to_arrow_field_keeps_user_metadata_and_attaches_type_json(self) -> None:
        f = Field(
            name="qty",
            dtype=IntegerType(byte_size=8, signed=True),
            nullable=False,
            metadata={b"comment": b"quantity"},
        )

        out = f.to_arrow_field()

        assert out.name == "qty"
        assert out.nullable is False
        assert out.metadata is not None
        assert b"comment" in out.metadata
        assert b"type_json" in out.metadata


class TestFieldFromStrNullability:

    def test_bang_suffix_marks_non_nullable(self) -> None:
        f = Field.from_str("qty!: int64")

        assert f.name == "qty"
        assert f.nullable is False
