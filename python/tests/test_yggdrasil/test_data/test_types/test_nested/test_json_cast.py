"""JSON-string cast coverage for nested types.

Exercises the vectorised JSON parsing path that ArrayType, MapType,
and StructType use when the source field is a string or binary, plus
the reverse JSON encoding when the target is a string or binary.
"""
from __future__ import annotations

import json

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


def _array_source() -> Field:
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


def _struct_source() -> Field:
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


def _map_source() -> Field:
    return Field(
        name="x",
        dtype=MapType.from_key_value(
            StringType(),
            IntegerType(byte_size=8, signed=True),
        ),
        nullable=True,
    )


def _string_target() -> Field:
    return Field(name="x", dtype=StringType(), nullable=True)


def _binary_target() -> Field:
    return Field(name="x", dtype=BinaryType(), nullable=True)


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


# ---------------------------------------------------------------------------
# JSON encoding: nested -> string / binary
# ---------------------------------------------------------------------------


class TestArrowJsonEncode:

    def test_array_to_string(self):
        source = _array_source()
        target = _string_target()
        arr = pa.array(
            [[1, 2, 3], [4, 5], None, []],
            type=pa.list_(pa.int64()),
        )

        out = target.cast_arrow_array(arr, source=source)

        assert out.type == pa.string()
        parsed = [json.loads(v) if v is not None else None for v in out.to_pylist()]
        assert parsed == [[1, 2, 3], [4, 5], None, []]

    def test_struct_to_string(self):
        source = _struct_source()
        target = _string_target()
        arr = pa.array(
            [{"a": 1, "b": "x"}, {"a": 2, "b": None}, None],
            type=pa.struct(
                [pa.field("a", pa.int64()), pa.field("b", pa.string())]
            ),
        )

        out = target.cast_arrow_array(arr, source=source)

        parsed = [json.loads(v) if v is not None else None for v in out.to_pylist()]
        assert parsed == [{"a": 1, "b": "x"}, {"a": 2, "b": None}, None]

    def test_map_to_string_emits_object_syntax(self):
        source = _map_source()
        target = _string_target()
        arr = pa.array(
            [{"a": 1, "b": 2}, {}, None],
            type=pa.map_(pa.string(), pa.int64()),
        )

        out = target.cast_arrow_array(arr, source=source)

        parsed = [json.loads(v) if v is not None else None for v in out.to_pylist()]
        assert parsed == [{"a": 1, "b": 2}, {}, None]

    def test_array_to_binary(self):
        source = _array_source()
        target = _binary_target()
        arr = pa.array([[1, 2], [3], None], type=pa.list_(pa.int64()))

        out = target.cast_arrow_array(arr, source=source)

        assert out.type == pa.binary()
        parsed = [
            json.loads(v.decode("utf-8")) if v is not None else None
            for v in out.to_pylist()
        ]
        assert parsed == [[1, 2], [3], None]

    def test_array_to_large_string(self):
        source = _array_source()
        target = Field(
            name="x",
            dtype=StringType(large=True),
            nullable=True,
        )
        arr = pa.array([[1]], type=pa.list_(pa.int64()))

        out = target.cast_arrow_array(arr, source=source)

        assert out.type == pa.large_string()

    def test_nested_to_string_roundtrips_with_json_parse(self):
        source = _array_source()
        target = _string_target()
        arr = pa.array([[1, 2], [3]], type=pa.list_(pa.int64()))

        out = target.cast_arrow_array(arr, source=source)

        # Reverse-parse: string -> array<int> should recover the values.
        parsed = source.cast_arrow_array(out)
        assert parsed.to_pylist() == [[1, 2], [3]]


class TestPolarsJsonEncode:

    def test_array_to_string(self, pl):
        source = _array_source()
        target = _string_target()
        s = pl.Series("x", [[1, 2, 3], [4, 5], None])

        out = target.cast_polars_series(s, source=source)

        assert out.dtype == pl.String
        assert [json.loads(v) if v is not None else None for v in out.to_list()] == [
            [1, 2, 3],
            [4, 5],
            None,
        ]

    def test_struct_to_string(self, pl):
        source = _struct_source()
        target = _string_target()
        s = pl.Series("x", [{"a": 1, "b": "x"}, None])

        out = target.cast_polars_series(s, source=source)

        assert out.dtype == pl.String
        parsed = [json.loads(v) if v is not None else None for v in out.to_list()]
        assert parsed == [{"a": 1, "b": "x"}, None]


class TestPandasJsonEncode:

    def test_array_to_string(self, pd):
        source = _array_source()
        target = _string_target()
        s = pd.Series([[1, 2, 3], [4, 5], None])

        out = target.cast_pandas_series(s, source=source)

        parsed = [
            json.loads(v) if v is not None and not (isinstance(v, float) and v != v) else None
            for v in out.tolist()
        ]
        assert parsed == [[1, 2, 3], [4, 5], None]
        assert out.name == "x"

    def test_struct_to_string(self, pd):
        source = _struct_source()
        target = _string_target()
        s = pd.Series([{"a": 1, "b": "x"}, None])

        out = target.cast_pandas_series(s, source=source)

        parsed = [
            json.loads(v) if v is not None and not (isinstance(v, float) and v != v) else None
            for v in out.tolist()
        ]
        assert parsed == [{"a": 1, "b": "x"}, None]
