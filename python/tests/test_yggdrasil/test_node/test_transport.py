"""Transport layer: Arrow IPC for tabular, pickle for scalars."""
from __future__ import annotations

import unittest

import pyarrow as pa

from yggdrasil.node.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_pickle,
    deserialize_result,
    is_tabular,
    read_arrow_stream,
    serialize_pickle,
    serialize_result,
    to_arrow_table,
    write_arrow_stream,
    write_arrow_stream_chunked,
)


class TestTransport(unittest.TestCase):
    def _table(self, n=100):
        return pa.table({"id": pa.array(range(n)), "v": pa.array([float(i) for i in range(n)])})

    def test_pickle_roundtrip_scalar(self):
        for obj in (42, "hello", {"a": [1, 2, 3]}, [1, 2, 3], b"\xff\x00"):
            self.assertEqual(deserialize_pickle(serialize_pickle(obj)), obj)

    def test_arrow_stream_roundtrip(self):
        t = self._table()
        back = read_arrow_stream(b"".join(write_arrow_stream(t)))
        self.assertEqual(back.num_rows, 100)
        self.assertEqual(back.schema, t.schema)

    def test_arrow_stream_chunked_equivalent(self):
        t = self._table(10_000)
        back = read_arrow_stream(b"".join(write_arrow_stream_chunked(t, max_chunksize=1000)))
        self.assertEqual(back.num_rows, 10_000)

    def test_is_tabular(self):
        self.assertTrue(is_tabular(self._table()))
        self.assertFalse(is_tabular({"a": 1}))
        self.assertFalse(is_tabular(42))

    def test_serialize_result_dispatch_arrow(self):
        data, ct = serialize_result(self._table())
        self.assertEqual(ct, CONTENT_TYPE_ARROW_STREAM)
        back = deserialize_result(data, ct)
        self.assertEqual(back.num_rows, 100)

    def test_serialize_result_dispatch_pickle(self):
        data, ct = serialize_result({"a": 1})
        self.assertEqual(ct, CONTENT_TYPE_PICKLE)
        self.assertEqual(deserialize_result(data, ct), {"a": 1})

    def test_polars_frame_to_arrow(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        df = pl.DataFrame({"x": [1, 2, 3]})
        self.assertTrue(is_tabular(df))
        t = to_arrow_table(df)
        self.assertEqual(t.num_rows, 3)
        data, ct = serialize_result(df)
        self.assertEqual(ct, CONTENT_TYPE_ARROW_STREAM)

    def test_to_arrow_rejects_unknown(self):
        with self.assertRaises(TypeError):
            to_arrow_table(42)


if __name__ == "__main__":
    unittest.main()
