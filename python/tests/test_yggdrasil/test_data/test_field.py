from __future__ import annotations

import pandas as pd
import polars as pl
import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.primitive import IntegerType, StringType


def test_with_default_updates_metadata_backed_default():
    field = Field(name="qty", dtype=IntegerType(byte_size=8, signed=True), nullable=False)

    field = field.with_default(7)

    assert field.has_default is True
    assert field.default == 7
    assert field.default_arrow_scalar.as_py() == 7


def test_default_arrow_array_uses_field_default():
    field = Field(
        name="qty",
        dtype=IntegerType(byte_size=8, signed=True),
        nullable=False,
        default=11,
    )

    arr = field.default_arrow_array(size=3)

    assert arr.to_pylist() == [11, 11, 11]


def test_fill_arrow_array_nulls_uses_field_default():
    field = Field(
        name="qty",
        dtype=IntegerType(byte_size=8, signed=True),
        nullable=False,
        default=9,
    )
    arr = pa.array([1, None, 3], type=pa.int64())

    out = field.fill_arrow_array_nulls(arr)

    assert out.to_pylist() == [1, 9, 3]


def test_default_pandas_series_uses_field_name_and_default():
    field = Field(
        name="qty",
        dtype=IntegerType(byte_size=8, signed=True),
        nullable=False,
        default=5,
    )

    out = field.default_pandas_series(size=3)

    assert out.name == "qty"
    assert out.tolist() == [5, 5, 5]


def test_fill_pandas_series_nulls_uses_field_default():
    field = Field(
        name="qty",
        dtype=IntegerType(byte_size=8, signed=True),
        nullable=False,
        default=5,
    )
    series = pd.Series([1, None, 3], name="qty")

    out = field.fill_pandas_series_nulls(series)

    assert out.tolist() == [1.0, 5.0, 3.0]


def test_cast_pandas_series_applies_target_field():
    field = Field(
        name="qty",
        dtype=IntegerType(byte_size=8, signed=True),
        nullable=False,
    )
    series = pd.Series([1, 2, 3], name="qty")

    out = field.cast_pandas_series(series)

    assert out.name == "qty"
    assert out.tolist() == [1, 2, 3]


def test_default_polars_series_uses_field_name_and_default():
    field = Field(
        name="book_id",
        dtype=StringType(),
        nullable=False,
        default="NA",
    )

    out = field.default_polars_series(size=2)

    assert out.name == "book_id"
    assert out.to_list() == ["NA", "NA"]


def test_fill_polars_series_nulls_uses_field_default():
    field = Field(
        name="book_id",
        dtype=StringType(),
        nullable=False,
        default="NA",
    )
    series = pl.Series("book_id", ["A", None, "C"])

    out = field.fill_polars_array_nulls(series)

    assert out.to_list() == ["A", "NA", "C"]


def test_cast_polars_series_applies_target_field():
    field = Field(
        name="qty",
        dtype=IntegerType(byte_size=8, signed=True),
        nullable=False,
    )
    series = pl.Series("qty", [1, 2, 3])

    out = field.cast_polars_series(series)

    assert out.name == "qty"
    assert out.to_list() == [1, 2, 3]


def test_to_arrow_field_includes_type_json_metadata():
    field = Field(
        name="qty",
        dtype=IntegerType(byte_size=8, signed=True),
        nullable=False,
        metadata={b"comment": b"quantity"},
    )

    out = field.to_arrow_field()

    assert out.name == "qty"
    assert out.nullable is False
    assert out.metadata is not None
    assert b"comment" in out.metadata
    assert b"to_json" in out.metadata


def test_from_str_nullable_suffix():
    field = Field.from_str("qty!: int64")

    assert field.name == "qty"
    assert field.nullable is False