"""Tests for the Excel-facing node service.

Covers the three capabilities the connector / add-in lean on — run
Python to a table, read a file as a typed table, write a table to a
file, and walk the filesystem — plus the parquet/arrow/json wire
encodings. Coroutines run under ``asyncio.run`` (no pytest-asyncio).
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.exceptions.api import BadRequestError, ForbiddenError
from yggdrasil.exceptions.api import TimeoutError as APITimeoutError
from yggdrasil.node.api.schemas.excel import ExcelQueryRequest
from yggdrasil.node.api.services.excel import ExcelService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.pyenv import PyEnvService
from yggdrasil.node.config import Settings
from yggdrasil.node import transport
from yggdrasil.node.exceptions import NotFoundError


def _service(home: Path) -> ExcelService:
    settings = Settings(node_id="test-node", node_home=home, front_home=home)
    fs = FsService(settings)
    pyenv = PyEnvService(settings)
    return ExcelService(settings, fs=fs, pyenv=pyenv)


class TestExcelInfo(unittest.TestCase):
    def test_info_card(self):
        with tempfile.TemporaryDirectory() as d:
            info = _service(Path(d)).info()
            self.assertEqual(info.node_id, "test-node")
            self.assertIn("parquet", info.table_formats)
            self.assertIn("python", info.capabilities)


class TestRunPython(unittest.TestCase):
    def test_snippet_to_table(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _service(Path(d))
            table = asyncio.run(svc.run_python(ExcelQueryRequest(
                code="df = {'x': [1, 2, 3], 'y': ['a', 'b', 'c']}",
            )))
            self.assertEqual(table.num_rows, 3)
            self.assertEqual(set(table.column_names), {"x", "y"})

    def test_max_rows_truncates(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _service(Path(d))
            table = asyncio.run(svc.run_python(ExcelQueryRequest(
                code="df = {'n': list(range(100))}", max_rows=10,
            )))
            self.assertEqual(table.num_rows, 10)

    def test_list_of_dicts(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _service(Path(d))
            table = asyncio.run(svc.run_python(ExcelQueryRequest(
                code="df = [{'a': 1}, {'a': 2}]",
            )))
            self.assertEqual(table.column("a").to_pylist(), [1, 2])

    def test_missing_df_name_raises(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _service(Path(d))
            with self.assertRaises(BadRequestError):
                asyncio.run(svc.run_python(ExcelQueryRequest(code="x = 1")))

    def test_empty_code_raises(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _service(Path(d))
            with self.assertRaises(BadRequestError):
                asyncio.run(svc.run_python(ExcelQueryRequest(code="   ")))

    def test_unknown_env_raises(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _service(Path(d))
            with self.assertRaises(NotFoundError):
                asyncio.run(svc.run_python(ExcelQueryRequest(
                    code="df = {'x': [1]}", env="nope",
                )))

    def test_timeout_raises_clean_408(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _service(Path(d))
            with self.assertRaises(APITimeoutError) as ctx:
                asyncio.run(svc.run_python(ExcelQueryRequest(
                    code="import time\nwhile True: time.sleep(1)\ndf = {'x': [1]}",
                    timeout=1,
                )))
            # Clean message, no leaked driver command line.
            self.assertIn("timeout", str(ctx.exception).lower())
            self.assertNotIn("runpy", str(ctx.exception))

    def test_env_vars_passed_to_snippet(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _service(Path(d))
            table = asyncio.run(svc.run_python(ExcelQueryRequest(
                code="import os\ndf = {'v': [os.environ.get('YGG_EXCEL_TEST', 'missing')]}",
                env_vars={"YGG_EXCEL_TEST": "hello-env"},
            )))
            self.assertEqual(table.column("v").to_pylist(), ["hello-env"])

    def test_empty_dataframe(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _service(Path(d))
            table = asyncio.run(svc.run_python(ExcelQueryRequest(
                code="import pandas as pd\ndf = pd.DataFrame({'a': []})",
            )))
            self.assertEqual(table.num_rows, 0)
            self.assertIn("a", table.column_names)


class TestSerializeFormats(unittest.TestCase):
    def setUp(self):
        self.table = pa.table({"x": [1, 2], "y": ["a", "b"]})

    def test_parquet_round_trips(self):
        body, ct = ExcelService.serialize_table(self.table, "parquet")
        self.assertIn("parquet", ct)
        self.assertEqual(transport.read_parquet_bytes(body), self.table)

    def test_arrow_round_trips(self):
        body, ct = ExcelService.serialize_table(self.table, "arrow")
        self.assertIn("arrow", ct)
        self.assertEqual(transport.read_arrow_stream(body), self.table)

    def test_json_records(self):
        import json
        body, ct = ExcelService.serialize_table(self.table, "json")
        self.assertEqual(ct, "application/json")
        self.assertEqual(
            json.loads(body), [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}],
        )

    def test_json_serializes_temporal_as_strings(self):
        import datetime as dt
        import json
        t = pa.table({
            "d": [dt.date(2026, 1, 2)],
            "ts": [dt.datetime(2026, 1, 2, 3, 4, 5)],
        })
        body, _ = ExcelService.serialize_table(t, "json")
        row = json.loads(body)[0]
        self.assertEqual(row["d"], "2026-01-02")
        self.assertTrue(row["ts"].startswith("2026-01-02"))

    def test_unknown_format_raises(self):
        with self.assertRaises(BadRequestError):
            ExcelService.serialize_table(self.table, "xlsx")


class TestFileTable(unittest.TestCase):
    def test_read_parquet_file_as_table(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pq.write_table(pa.table({"v": [10, 20]}), str(home / "data.parquet"))
            table = asyncio.run(_service(home).read_table("data.parquet"))
            self.assertEqual(table.column("v").to_pylist(), [10, 20])

    def test_read_csv_file_as_table(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "data.csv").write_text("a,b\n1,x\n2,y\n")
            table = asyncio.run(_service(home).read_table("data.csv"))
            self.assertEqual(table.column("a").to_pylist(), [1, 2])

    def test_read_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(NotFoundError):
                asyncio.run(_service(Path(d)).read_table("nope.parquet"))

    def test_read_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ForbiddenError):
                asyncio.run(_service(Path(d)).read_table("../../etc/passwd"))

    def test_write_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            body = transport.write_parquet_bytes(pa.table({"k": [1]}))
            with self.assertRaises(ForbiddenError):
                asyncio.run(_service(Path(d)).write_table(
                    "../escape.parquet", body, transport.CONTENT_TYPE_PARQUET,
                ))

    def test_write_parquet_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            svc = _service(home)
            body = transport.write_parquet_bytes(pa.table({"k": [1, 2, 3]}))
            resp = asyncio.run(svc.write_table(
                "out/result.parquet", body, transport.CONTENT_TYPE_PARQUET,
            ))
            self.assertEqual(resp.rows, 3)
            self.assertEqual(resp.columns, 1)
            written = pq.read_table(str(home / "out" / "result.parquet"))
            self.assertEqual(written.column("k").to_pylist(), [1, 2, 3])


class TestTreeWalk(unittest.TestCase):
    def test_walk_lists_dirs_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "sub").mkdir()
            (home / "sub" / "a.txt").write_text("hi")
            (home / "top.parquet").write_text("x")
            resp = asyncio.run(_service(home).tree("", depth=3))
            names = {n.name: n for n in resp.tree}
            self.assertIn("sub", names)
            self.assertTrue(names["sub"].is_dir)
            child_names = {c.name for c in names["sub"].children}
            self.assertIn("a.txt", child_names)

    def test_depth_limit(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "a" / "b").mkdir(parents=True)
            (home / "a" / "b" / "deep.txt").write_text("x")
            resp = asyncio.run(_service(home).tree("", depth=1))
            a = next(n for n in resp.tree if n.name == "a")
            # depth=1 → only the top level is expanded; "a" has no children
            self.assertEqual(a.children, [])


if __name__ == "__main__":
    unittest.main()
