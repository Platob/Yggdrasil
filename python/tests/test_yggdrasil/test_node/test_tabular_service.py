"""Tabular inspect / bounded preview / in-place edit over LazyTabular."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.node.api.schemas.tabular import TabularWriteRequest
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.tabular import TabularService
from yggdrasil.node.config import Settings


def _service(home: Path, **overrides) -> TabularService:
    settings = Settings(node_id="t", node_home=home, front_home=home, **overrides)
    return TabularService(settings, fs=FsService(settings))


def _write_parquet(home: Path, name: str, rows: int) -> None:
    pq.write_table(
        pa.table({"id": list(range(rows)), "name": [f"n{i}" for i in range(rows)]}),
        str(home / name),
    )


class TestInspect(unittest.TestCase):
    def test_typed_columns_and_editable_when_small(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _write_parquet(home, "data.parquet", 3)
            svc = _service(home)
            info = asyncio.run(svc.inspect("data.parquet"))
            self.assertTrue(info.is_tabular)
            self.assertEqual(info.column_count, 2)
            self.assertEqual(info.columns[0].name, "id")
            self.assertEqual(info.columns[0].type, "int64")
            self.assertEqual(info.row_count, 3)
            self.assertTrue(info.editable)
            self.assertTrue(info.schema_hash)
            self.assertTrue(info.source_url.startswith("file://"))

    def test_large_file_is_not_editable(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _write_parquet(home, "big.parquet", 50)
            svc = _service(home, tabular_preview_max_rows=10)
            info = asyncio.run(svc.inspect("big.parquet"))
            self.assertIsNone(info.row_count)   # too big to count cheaply
            self.assertFalse(info.editable)

    def test_non_tabular_extension(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "notes.txt").write_text("hi", encoding="utf-8")
            info = asyncio.run(_service(home).inspect("notes.txt"))
            self.assertFalse(info.is_tabular)
            self.assertFalse(info.editable)


class TestPreview(unittest.TestCase):
    def test_bounded_and_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _write_parquet(home, "data.parquet", 20)
            svc = _service(home)
            prev = asyncio.run(svc.preview("data.parquet", limit=5))
            self.assertEqual(len(prev.rows), 5)
            self.assertTrue(prev.truncated)
            self.assertEqual(prev.rows[0], [0, "n0"])

    def test_limit_clamped_to_cap(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _write_parquet(home, "data.parquet", 30)
            svc = _service(home, tabular_preview_max_rows=10)
            prev = asyncio.run(svc.preview("data.parquet", limit=10_000))
            self.assertLessEqual(len(prev.rows), 10)


class TestWrite(unittest.TestCase):
    def test_roundtrip_preserves_types(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _write_parquet(home, "data.parquet", 3)
            svc = _service(home)
            # String cells (as the grid editor produces) cast back to int64.
            req = TabularWriteRequest(path="data.parquet", columns=["id", "name"], rows=[["10", "x"], ["11", "y"]])
            res = asyncio.run(svc.write(req))
            self.assertEqual(res.rows, 2)
            after = asyncio.run(svc.preview("data.parquet", limit=10))
            self.assertEqual(after.rows, [[10, "x"], [11, "y"]])
            self.assertEqual(after.columns[0].type, "int64")

    def test_refuses_over_cap(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _write_parquet(home, "data.parquet", 1)
            svc = _service(home, tabular_preview_max_rows=2)
            req = TabularWriteRequest(
                path="data.parquet", columns=["id", "name"],
                rows=[[i, f"n{i}"] for i in range(5)],
            )
            with self.assertRaises(BadRequestError):
                asyncio.run(svc.write(req))


if __name__ == "__main__":
    unittest.main()
