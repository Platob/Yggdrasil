from __future__ import annotations

from collections import OrderedDict

import polars as pl
import pytest

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.polars import (
    PolarsDataFrameSerialized,
    PolarsDataTypeSerialized,
    PolarsExprSerialized,
    PolarsLazyFrameSerialized,
    PolarsSchemaSerialized,
    PolarsSeriesSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def _roundtrip(serialized: Serialized[object]) -> Serialized[object]:
    buf = BytesIO()
    serialized.write_to(buf)
    return Serialized.read_from(buf, pos=0)


def test_polars_dataframe_arrow_roundtrip():
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [1.5, None, 3.5],
        }
    )

    ser = PolarsDataFrameSerialized.from_value(df)
    out = _roundtrip(ser)

    assert isinstance(out, PolarsDataFrameSerialized)
    result = out.value
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)
    assert (out.metadata or {}).get(b"serialization_format") == b"arrow"


def test_polars_series_arrow_roundtrip():
    s = pl.Series("numbers", [1, 2, 3, None], dtype=pl.Int64)

    ser = PolarsSeriesSerialized.from_value(s)
    out = _roundtrip(ser)

    assert isinstance(out, PolarsSeriesSerialized)
    result = out.value
    assert isinstance(result, pl.Series)
    assert result.name == s.name
    assert result.to_list() == s.to_list()
    assert result.dtype == s.dtype
    assert (out.metadata or {}).get(b"serialization_format") == b"arrow"


def test_polars_lazyframe_roundtrip_collect():
    lf = (
        pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        .lazy()
        .with_columns((pl.col("a") + pl.col("b")).alias("c"))
        .filter(pl.col("c") > 15)
        .select(["a", "c"])
    )

    ser = PolarsLazyFrameSerialized.from_value(lf)
    out = _roundtrip(ser)

    assert isinstance(out, PolarsLazyFrameSerialized)
    result = out.value
    assert isinstance(result, pl.LazyFrame)
    assert result.collect().equals(lf.collect())
    assert (out.metadata or {}).get(b"serialization_format") == b"polars_binary"


def test_polars_expr_roundtrip_behavior():
    expr = (pl.col("a") * 2 + pl.lit(1)).alias("x")

    ser = PolarsExprSerialized.from_value(expr)
    out = _roundtrip(ser)

    assert isinstance(out, PolarsExprSerialized)
    result = out.value
    assert isinstance(result, pl.Expr)

    df = pl.DataFrame({"a": [1, 2, 3]})
    expected = df.select(expr)
    actual = df.select(result)
    assert actual.equals(expected)
    assert (out.metadata or {}).get(b"serialization_format") == b"polars_binary"


def test_polars_schema_roundtrip_from_schema():
    schema = pl.Schema(
        [
            ("a", pl.Int64),
            ("b", pl.String),
            ("c", pl.List(pl.Int32)),
        ]
    )

    ser = PolarsSchemaSerialized.from_value(schema)
    out = _roundtrip(ser)

    assert isinstance(out, PolarsSchemaSerialized)
    result = out.value
    assert isinstance(result, pl.Schema)
    assert list(result.items()) == list(schema.items())
    assert (out.metadata or {}).get(b"serialization_format") == b"python_serialized"


def test_polars_schema_roundtrip_from_mapping():
    schema = OrderedDict(
        [
            ("a", pl.Int64),
            ("b", pl.String),
        ]
    )

    ser = PolarsSchemaSerialized.from_value(schema)
    out = _roundtrip(ser)

    result = out.value
    assert list(result.items()) == list(schema.items())


@pytest.mark.parametrize(
    ("dtype_in", "expected"),
    [
        (pl.Int64, pl.Int64),
        (pl.String, pl.String),
        (pl.Boolean, pl.Boolean),
        (pl.Date, pl.Date),
        (pl.Datetime, pl.Datetime),
        (pl.List, pl.List),
        (pl.Struct, pl.Struct),
        (pl.Decimal, pl.Decimal),
        (pl.Enum, pl.Enum),
        (pl.Categorical, pl.Categorical),
    ],
)
def test_polars_dtype_roundtrip(dtype_in, expected):
    ser = PolarsDataTypeSerialized.from_value(dtype_in)
    out = _roundtrip(ser)

    assert isinstance(out, PolarsDataTypeSerialized)
    result = out.value
    assert result == expected
    assert (out.metadata or {}).get(b"serialization_format") == b"python_serialized"


def test_polars_dataframe_python_fallback(monkeypatch):
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "yggdrasil.pickle.ser.polars._dataframe_to_arrow_table",
        boom,
    )

    ser = PolarsDataFrameSerialized.from_value(df)
    out = _roundtrip(ser)

    assert (out.metadata or {}).get(b"serialization_format") == b"python_serialized"
    result = out.value
    assert result.equals(df)


def test_polars_series_python_fallback(monkeypatch):
    s = pl.Series("values", [1, 2, 3])

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "yggdrasil.pickle.ser.polars._series_to_arrow_table",
        boom,
    )

    ser = PolarsSeriesSerialized.from_value(s)
    out = _roundtrip(ser)

    assert (out.metadata or {}).get(b"serialization_format") == b"python_serialized"
    result = out.value
    assert result.name == s.name
    assert result.to_list() == s.to_list()


def test_polars_from_python_object_dispatch():
    df = pl.DataFrame({"a": [1]})
    s = pl.Series("a", [1])
    lf = df.lazy()
    expr = pl.col("a") + 1
    schema = pl.Schema([("a", pl.Int64)])
    dtype = pl.Int64

    assert isinstance(PolarsDataFrameSerialized.from_python_object(df), PolarsDataFrameSerialized)
    assert isinstance(PolarsSeriesSerialized.from_python_object(s), PolarsSeriesSerialized)
    assert isinstance(PolarsLazyFrameSerialized.from_python_object(lf), PolarsLazyFrameSerialized)
    assert isinstance(PolarsExprSerialized.from_python_object(expr), PolarsExprSerialized)
    assert isinstance(PolarsSchemaSerialized.from_python_object(schema), PolarsSchemaSerialized)
    assert isinstance(PolarsDataTypeSerialized.from_python_object(dtype), PolarsDataTypeSerialized)


def test_polars_schema_rejects_non_string_keys():
    with pytest.raises(TypeError):
        PolarsSchemaSerialized.from_value({1: pl.Int64})


def test_polars_dtype_rejects_unknown_object():
    with pytest.raises(TypeError):
        PolarsDataTypeSerialized.from_value(object())
