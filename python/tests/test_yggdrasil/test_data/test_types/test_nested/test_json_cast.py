"""JSON-string cast coverage for nested types.

Exercises the vectorised JSON parsing path that ArrayType, MapType,
and StructType use when the source field is a string or binary.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import ArrayType, MapType, StructType
from yggdrasil.data.types.primitive import (
    BinaryType,
    IntegerType,
    StringType,
)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def _array_int_target() -> Field:
    return Field(
        name="x",
        dtype=ArrayType.from_item_field(
            Field(
                name="item",
                dtype=IntegerType(byte_size=8, signed=True),
                nullable=True,
            )
        ),
        nullable=True,
    )


def _struct_target() -> Field:
    return Field(
        name="x",
        dtype=StructType(
            fields=[
                Field(
                    name="a",
                    dtype=IntegerType(byte_size=8, signed=True),
                    nullable=True,
                ),
                Field(name="b", dtype=StringType(), nullable=True),
            ]
        ),
        nullable=True,
    )


def _map_target() -> Field:
    return Field(
        name="x",
        dtype=MapType.from_key_value(
            StringType(),
            IntegerType(byte_size=8, signed=True),
        ),
        nullable=True,
    )


# ---------------------------------------------------------------------------
# Arrow
# ---------------------------------------------------------------------------


class TestArrowJsonStringCasts:

    def test_string_to_array(self):
        target = _array_int_target()
        arr = pa.array(
            ["[1,2,3]", "[4,5]", None, "", "[]"],
            type=pa.string(),
        )

        out = target.cast_arrow_array(arr)

        assert out.type == pa.list_(pa.int64())
        assert out.to_pylist() == [[1, 2, 3], [4, 5], None, None, []]

    def test_string_to_struct(self):
        target = _struct_target()
        arr = pa.array(
            ['{"a":1,"b":"x"}', '{"a":2}', None, ""],
            type=pa.string(),
        )

        out = target.cast_arrow_array(arr)

        assert out.type == pa.struct(
            [pa.field("a", pa.int64()), pa.field("b", pa.string())]
        )
        assert out.to_pylist() == [
            {"a": 1, "b": "x"},
            {"a": 2, "b": None},
            None,
            None,
        ]

    def test_string_to_map(self):
        target = _map_target()
        arr = pa.array(
            ['{"a":1,"b":2}', "{}", None, ""],
            type=pa.string(),
        )

        out = target.cast_arrow_array(arr)

        assert out.type == pa.map_(pa.string(), pa.int64())
        assert out.to_pylist() == [
            [("a", 1), ("b", 2)],
            [],
            None,
            None,
        ]

    def test_binary_to_array(self):
        target = _array_int_target()
        arr = pa.array([b"[1,2]", b"[3]", None, b""], type=pa.binary())

        out = target.cast_arrow_array(arr)

        assert out.type == pa.list_(pa.int64())
        assert out.to_pylist() == [[1, 2], [3], None, None]

    def test_binary_to_struct(self):
        target = _struct_target()
        arr = pa.array(
            [b'{"a":1,"b":"x"}', b'{"a":2}', None],
            type=pa.binary(),
        )

        out = target.cast_arrow_array(arr)

        assert out.to_pylist() == [
            {"a": 1, "b": "x"},
            {"a": 2, "b": None},
            None,
        ]


# ---------------------------------------------------------------------------
# Polars
# ---------------------------------------------------------------------------


@pytest.fixture
def pl():
    import polars as polars_module

    return polars_module


class TestPolarsJsonStringCasts:

    def test_string_to_array(self, pl):
        target = _array_int_target()
        s = pl.Series("x", ["[1,2,3]", "[4,5]", None, "", "[]"])

        out = target.cast_polars_series(s)

        assert out.dtype == pl.List(pl.Int64)
        assert out.to_list() == [[1, 2, 3], [4, 5], None, None, []]

    def test_string_to_struct(self, pl):
        target = _struct_target()
        s = pl.Series("x", ['{"a":1,"b":"x"}', '{"a":2}', None])

        out = target.cast_polars_series(s)

        assert out.dtype == pl.Struct({"a": pl.Int64, "b": pl.String})
        assert out.to_list() == [
            {"a": 1, "b": "x"},
            {"a": 2, "b": None},
            None,
        ]

    def test_string_to_map(self, pl):
        target = _map_target()
        s = pl.Series("x", ['{"a":1,"b":2}', "{}", None])

        out = target.cast_polars_series(s)

        # polars represents a map as ``List<Struct<key, value>>`` — verify
        # the entries roundtrip through the Arrow map parser unchanged.
        assert out.dtype == pl.List(
            pl.Struct({"key": pl.String, "value": pl.Int64})
        )
        assert out.to_list()[0] == [
            {"key": "a", "value": 1},
            {"key": "b", "value": 2},
        ]
        assert out.to_list()[1] == []

    def test_binary_to_array(self, pl):
        target = _array_int_target()
        s = pl.Series("x", [b"[1,2]", b"[3]", None], dtype=pl.Binary)

        out = target.cast_polars_series(s)

        assert out.to_list() == [[1, 2], [3], None]


# ---------------------------------------------------------------------------
# Pandas
# ---------------------------------------------------------------------------


@pytest.fixture
def pd():
    import pandas as pandas_module

    return pandas_module


class TestPandasJsonStringCasts:

    def test_string_to_array(self, pd):
        target = _array_int_target()
        s = pd.Series(["[1,2,3]", "[4,5]", None])

        out = target.cast_pandas_series(s)

        assert out.tolist() == [[1, 2, 3], [4, 5], None]
        assert out.name == "x"

    def test_string_to_struct(self, pd):
        target = _struct_target()
        s = pd.Series(['{"a":1,"b":"x"}', None])

        out = target.cast_pandas_series(s)

        assert out.tolist() == [{"a": 1, "b": "x"}, None]

    def test_string_to_map(self, pd):
        target = _map_target()
        s = pd.Series(['{"a":1,"b":2}', "{}", None])

        out = target.cast_pandas_series(s)

        assert out.tolist() == [
            [("a", 1), ("b", 2)],
            [],
            None,
        ]
