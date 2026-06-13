"""Node services: fs, tabular, audit, messenger, functions, monitor."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.services.audit import AuditLog
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.tabular import TabularService
from yggdrasil.node.config import Settings
from yggdrasil.node.schemas.function import FunctionCreate
from yggdrasil.node.schemas.messenger import MessageSend
from yggdrasil.node.services.function import FunctionService
from yggdrasil.node.services.messenger import MessengerService
from yggdrasil.node.services.monitor import MonitorService


def run(coro):
    return asyncio.run(coro)


class TestSettings(unittest.TestCase):
    def test_string_paths_coerced(self):
        s = Settings(node_home="/tmp/x", front_home="/tmp/y", logs_root="/tmp/z")
        self.assertIsInstance(s.node_home, Path)
        self.assertIsInstance(s.front_home, Path)
        self.assertIsInstance(s.logs_root, Path)


class TestAuditLog(unittest.TestCase):
    def test_ring_buffer_evicts(self):
        audit = AuditLog(Settings(), max_entries=10)
        for i in range(50):
            audit.log("create", "pyfunc", i)
        entries = audit.entries(limit=0)
        self.assertEqual(len(entries), 10)
        # Most-recent-first when limited.
        recent = audit.entries(limit=3)
        self.assertEqual([e["resource_id"] for e in recent], [49, 48, 47])


class TestFsService(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.home = Path(self.td.name)
        self.svc = FsService(Settings(node_home=self.home, front_home=self.home))

    def tearDown(self):
        self.td.cleanup()

    def test_ls_dirs_first_and_paging(self):
        for i in range(20):
            (self.home / f"f{i:02d}.txt").write_text("x")
        (self.home / "adir").mkdir()
        full = run(self.svc.ls(""))
        self.assertEqual(full.total, 21)
        self.assertTrue(full.entries[0].is_dir)  # dirs sort first
        page = run(self.svc.ls("", offset=0, limit=5))
        self.assertEqual(len(page.entries), 5)
        self.assertEqual(page.total, 21)

    def test_read_bounded(self):
        svc = FsService(Settings(node_home=self.home, front_home=self.home, max_read_bytes=10))
        (self.home / "big.txt").write_text("a" * 1000)
        res = run(svc.read("big.txt"))
        self.assertEqual(len(res.content), 10)
        self.assertTrue(res.truncated)
        self.assertEqual(res.size, 1000)

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            run(self.svc.ls("../../etc"))

    def test_ls_missing(self):
        with self.assertRaises(FileNotFoundError):
            run(self.svc.ls("nope"))


class TestTabularService(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.home = Path(self.td.name)
        s = Settings(node_home=self.home, front_home=self.home)
        self.svc = TabularService(s, FsService(s))

    def tearDown(self):
        self.td.cleanup()

    def test_parquet_footer_inspect(self):
        pq.write_table(pa.table({"a": range(1000), "b": range(1000)}), str(self.home / "t.parquet"))
        res = run(self.svc.inspect("t.parquet"))
        self.assertEqual(res.row_count, 1000)
        self.assertEqual(res.col_count, 2)
        self.assertFalse(res.editable)
        self.assertEqual(res.format, "parquet")

    def test_csv_editable(self):
        (self.home / "t.csv").write_text("a,b\n1,2\n3,4\n")
        res = run(self.svc.inspect("t.csv"))
        self.assertEqual(res.row_count, 2)
        self.assertTrue(res.editable)

    def test_unsupported_format(self):
        (self.home / "t.bin").write_bytes(b"x")
        with self.assertRaises(ValueError):
            run(self.svc.inspect("t.bin"))

    def test_inspect_serializes_schema_key(self):
        pq.write_table(pa.table({"a": range(5)}), str(self.home / "t.parquet"))
        res = run(self.svc.inspect("t.parquet"))
        dumped = res.model_dump()
        self.assertIn("schema", dumped)
        self.assertNotIn("schema_", dumped)


class TestMessenger(unittest.TestCase):
    def test_send_and_channels(self):
        svc = MessengerService(Settings())
        m = run(svc.send_message(MessageSend(text="hi", sender="u")))
        self.assertEqual(m.channel, "general")
        run(svc.create_channel("trade"))
        run(svc.send_message(MessageSend(text="buy", sender="u", channel="trade")))
        chans = {c.name: c.message_count for c in run(svc.list_channels())}
        self.assertEqual(chans["general"], 1)
        self.assertEqual(chans["trade"], 1)
        msgs = run(svc.get_messages("general", limit=10))
        self.assertEqual(msgs[0].text, "hi")

    def test_unknown_channel(self):
        svc = MessengerService(Settings())
        with self.assertRaises(KeyError):
            run(svc.get_messages("nope"))


class TestFunctions(unittest.TestCase):
    def test_upsert_keeps_id(self):
        svc = FunctionService(Settings())
        r1 = run(svc.create(FunctionCreate(name="f", code="x=1")))
        r2 = run(svc.create(FunctionCreate(name="f", code="x=2")))
        self.assertEqual(r1.function.id, r2.function.id)
        self.assertEqual(run(svc.get(r1.function.id)).code, "x=2")

    def test_run_executes(self):
        svc = FunctionService(Settings())
        r = run(svc.create(FunctionCreate(name="p", code="print('hi')")))
        run_resp = run(svc.run(r.function.id))
        self.assertEqual(run_resp.status, "ok")
        self.assertIn("hi", run_resp.stdout)

    def test_get_missing(self):
        svc = FunctionService(Settings())
        with self.assertRaises(KeyError):
            run(svc.get(999))


class TestMonitor(unittest.TestCase):
    def test_snapshot_cached(self):
        svc = MonitorService(Settings())
        s1 = svc.snapshot()
        s2 = svc.snapshot()
        self.assertIs(s1, s2)  # cached within 1s
        self.assertGreaterEqual(s1.cpu_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
