# tests/test_serde.py
from __future__ import annotations

import io
from dataclasses import dataclass

import pytest

# from yggdrasil.pyutils.serde import ObjectSerde, ObjectSerdeProtocol, ObjectSerdeFormat, ObjectSerdeCompression
from yggdrasil.pyutils.serde import (  # type: ignore
    ObjectSerde,
    ObjectSerdeCompression,
    ObjectSerdeFormat,
    ObjectSerdeProtocol,
)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


HAS_DILL = _has("dill")
HAS_PANDAS = _has("pandas")
HAS_POLARS = _has("polars")

DILL_REQUIRED = pytest.mark.skipif(not HAS_DILL, reason="requires dill installed")
PANDAS_REQUIRED = pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas installed")
POLARS_REQUIRED = pytest.mark.skipif(not HAS_POLARS, reason="requires polars installed")


# ---------------------------
# full_namespace()
# ---------------------------
@dataclass
class Foo:
    x: int


def test_full_namespace_class_and_instance():
    ns_cls = ObjectSerde.full_namespace(Foo)
    ns_obj = ObjectSerde.full_namespace(Foo(1))

    assert ns_cls.endswith(".Foo")
    assert ns_obj.endswith(".Foo")
    assert "." in ns_cls


def test_full_namespace_fallback_when_no_class():
    class Weird:
        __class__ = None  # type: ignore

    ns = ObjectSerde.full_namespace(Weird(), fallback="fallback.ns")
    assert ns == "fallback.ns"


# ---------------------------
# dill path: encode/decode
# ---------------------------
@DILL_REQUIRED
def test_dill_encode_decode_roundtrip_simple_object():
    obj = {"a": 1, "b": [1, 2, 3], "c": ("x", True)}
    s = ObjectSerde.dill_encode(obj)

    assert s.protocol == ObjectSerdeProtocol.DILL
    assert s.format == ObjectSerdeFormat.BINARY
    assert s.compression is None

    out = s.decode()
    assert out == obj


@DILL_REQUIRED
def test_encode_default_routes_to_dill_for_non_pandas_polars():
    obj = {"k": "v"}
    s = ObjectSerde.encode(obj)

    assert s.protocol == ObjectSerdeProtocol.DILL
    assert s.format == ObjectSerdeFormat.BINARY
    assert s.decode() == obj


@DILL_REQUIRED
def test_dill_encode_gzip_compresses_when_byte_limit_exceeded():
    # Big-ish payload so pickle bytes are clearly > small limit
    obj = {"blob": "x" * 200_000}
    s = ObjectSerde.dill_encode(obj, spill_bytes=1000)

    assert s.protocol == ObjectSerdeProtocol.DILL
    assert s.compression == ObjectSerdeCompression.GZIP

    # sanity: stored bytes are gzipped bytes
    s.io.seek(0)
    raw = s.io.read()
    assert raw[:2] == b"\x1f\x8b"  # gzip magic

    out = s.decode()
    assert out == obj


@DILL_REQUIRED
def test_decode_ignores_current_cursor_position():
    obj = {"a": 123}
    s = ObjectSerde.dill_encode(obj)

    # move cursor to end, decode should still work
    s.io.seek(0, io.SEEK_END)
    out = s.decode()
    assert out == obj


def test_decode_unsupported_compression_raises():
    # construct a serde manually
    s = ObjectSerde(
        protocol=ObjectSerdeProtocol.DILL,
        format=ObjectSerdeFormat.BINARY,
        namespace="x",
        io=io.BytesIO(b"whatever"),
        compression=ObjectSerdeCompression.ZSTD,  # not supported on decode for dill bytes
    )
    with pytest.raises(ValueError, match="Unsupported compression"):
        s.decode()


def test_decode_unsupported_combo_raises():
    s = ObjectSerde(
        protocol=ObjectSerdeProtocol.JSON,
        format=ObjectSerdeFormat.JSON,
        namespace="x",
        io=io.BytesIO(b'{"a":1}'),
        compression=None,
    )
    with pytest.raises(ValueError, match="Unsupported serde combo"):
        s.decode()


# ---------------------------
# pandas: DataFrame fast-path parquet
# ---------------------------
@PANDAS_REQUIRED
def test_pandas_encode_dataframe_parquet_roundtrip():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    s = ObjectSerde.pandas_encode(df)

    assert s.protocol == ObjectSerdeProtocol.PANDAS
    assert s.format == ObjectSerdeFormat.PARQUET
    assert s.compression == ObjectSerdeCompression.ZSTD

    out = s.decode()
    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_frame_equal(out, df)


@PANDAS_REQUIRED
def test_encode_routes_to_pandas_when_namespace_startswith_pandas():
    import pandas as pd

    df = pd.DataFrame({"a": [1]})
    # full_namespace(df) begins with "pandas."
    s = ObjectSerde.encode(df)
    assert s.protocol == ObjectSerdeProtocol.PANDAS
    assert s.format == ObjectSerdeFormat.PARQUET


@PANDAS_REQUIRED
@DILL_REQUIRED
def test_pandas_encode_non_dataframe_falls_back_to_dill():
    import pandas as pd

    idx = pd.Index([1, 2, 3])  # not a DataFrame
    s = ObjectSerde.pandas_encode(idx)

    assert s.protocol == ObjectSerdeProtocol.DILL
    assert s.format == ObjectSerdeFormat.BINARY

    out = s.decode()
    # dill should preserve type/value
    assert list(out) == [1, 2, 3]


# ---------------------------
# polars: DataFrame/LazyFrame parquet fast-path
# ---------------------------
@POLARS_REQUIRED
def test_polars_encode_dataframe_parquet_roundtrip():
    import polars as pl

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    s = ObjectSerde.polars_encode(df)

    assert s.protocol == ObjectSerdeProtocol.POLARS
    assert s.format == ObjectSerdeFormat.PARQUET
    assert s.compression == ObjectSerdeCompression.ZSTD

    out = s.decode()
    assert isinstance(out, pl.DataFrame)
    assert out.equals(df)


@POLARS_REQUIRED
def test_polars_encode_lazyframe_parquet_roundtrip():
    import polars as pl

    lf = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).lazy()
    s = ObjectSerde.polars_encode(lf)

    assert s.protocol == ObjectSerdeProtocol.POLARS
    assert s.format == ObjectSerdeFormat.PARQUET
    assert s.compression == ObjectSerdeCompression.ZSTD

    out = s.decode()
    assert isinstance(out, pl.DataFrame)
    assert out.equals(lf.collect())


@POLARS_REQUIRED
def test_encode_routes_to_polars_when_namespace_startswith_polars():
    import polars as pl

    df = pl.DataFrame({"a": [1]})
    s = ObjectSerde.encode(df)
    assert s.protocol == ObjectSerdeProtocol.POLARS
    assert s.format == ObjectSerdeFormat.PARQUET


@POLARS_REQUIRED
@DILL_REQUIRED
def test_polars_encode_series_falls_back_to_dill():
    import polars as pl

    s_in = pl.Series("a", [1, 2, 3])
    s = ObjectSerde.polars_encode(s_in)

    assert s.protocol == ObjectSerdeProtocol.DILL
    out = s.decode()
    assert list(out) == [1, 2, 3]
