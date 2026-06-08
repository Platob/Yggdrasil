"""Tests for ``yggdrasil.node.transport`` — the Arrow/pickle wire format."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.node import transport as T


def _table(n: int = 50) -> pa.Table:
    return pa.table({"id": pa.array(range(n), pa.int64()), "label": [f"r{i}" for i in range(n)]})


class TestPickle:
    @pytest.mark.parametrize(
        "obj",
        [42, "hello", b"\x00\xff", {"a": 1, "b": [1, 2, 3]}, [1, "x", {"k": "v"}], None],
    )
    def test_pickle_roundtrip(self, obj):
        data = T.serialize_pickle(obj)
        assert isinstance(data, bytes)
        assert T.deserialize_pickle(data) == obj


class TestArrowStream:
    def test_roundtrip_equals_original(self):
        table = _table(100)
        data = b"".join(T.write_arrow_stream(table))
        back = T.read_arrow_stream(data)
        assert back.equals(table)

    def test_chunked_matches_single_shot_data(self):
        table = _table(20_000)
        chunked = T.read_arrow_stream(b"".join(T.write_arrow_stream_chunked(table, max_chunksize=4096)))
        assert chunked.equals(table)


class TestTabularDetection:
    def test_arrow_table_is_tabular(self):
        assert T.is_tabular(_table())

    def test_scalars_not_tabular(self):
        assert not T.is_tabular(42)
        assert not T.is_tabular({"a": 1})
        assert not T.is_tabular(b"bytes")

    def test_to_arrow_table_passthrough(self):
        table = _table()
        assert T.to_arrow_table(table) is table

    def test_to_arrow_table_rejects_unknown(self):
        with pytest.raises(TypeError, match="cannot convert"):
            T.to_arrow_table(object())


class TestPolars:
    def test_polars_dataframe_is_tabular_and_converts(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        assert T.is_tabular(df)
        table = T.to_arrow_table(df)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 3


class TestAutoDispatch:
    def test_tabular_picks_arrow_stream(self):
        data, ct = T.serialize_result(_table())
        assert ct == T.CONTENT_TYPE_ARROW_STREAM
        assert T.deserialize_result(data, ct).num_rows == 50

    def test_scalar_picks_pickle(self):
        data, ct = T.serialize_result({"k": "v"})
        assert ct == T.CONTENT_TYPE_PICKLE
        assert T.deserialize_result(data, ct) == {"k": "v"}

    def test_unknown_content_type_falls_back_to_pickle(self):
        data, _ = T.serialize_result([1, 2, 3])
        assert T.deserialize_result(data, "application/octet-stream") == [1, 2, 3]
