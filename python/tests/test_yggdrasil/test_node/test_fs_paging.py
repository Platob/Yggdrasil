"""Range + paging: fs ls pages, fs read byte-ranges, NodePath windowed reads."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings
from yggdrasil.node.path import NodePath


def _svc(home: Path) -> FsService:
    return FsService(Settings(node_id="t", node_home=home, front_home=home))


class TestLsPaging(unittest.TestCase):
    def test_pages_and_total(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            for i in range(25):
                (home / f"f{i:02d}.txt").write_text("x", encoding="utf-8")
            svc = _svc(home)
            page = asyncio.run(svc.ls("", offset=0, limit=10))
            self.assertEqual(page.total, 25)
            self.assertEqual(len(page.entries), 10)
            self.assertEqual(page.offset, 0)
            page2 = asyncio.run(svc.ls("", offset=10, limit=10))
            self.assertEqual(len(page2.entries), 10)
            # pages are disjoint and ordered
            self.assertNotEqual(page.entries[0].name, page2.entries[0].name)
            last = asyncio.run(svc.ls("", offset=20, limit=10))
            self.assertEqual(len(last.entries), 5)   # only 5 left

    def test_no_limit_returns_all(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            for i in range(5):
                (home / f"f{i}.txt").write_text("x", encoding="utf-8")
            page = asyncio.run(_svc(home).ls(""))
            self.assertEqual(page.total, 5)
            self.assertEqual(len(page.entries), 5)


class TestReadRange(unittest.TestCase):
    def test_byte_offset_window(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "data.txt").write_text("ABCDEFGHIJ", encoding="utf-8")
            svc = _svc(home)
            head = asyncio.run(svc.read("data.txt", max_bytes=4, offset=0))
            self.assertEqual(head.content, "ABCD")
            self.assertEqual(head.offset, 0)
            self.assertTrue(head.truncated)          # more after this window
            mid = asyncio.run(svc.read("data.txt", max_bytes=4, offset=4))
            self.assertEqual(mid.content, "EFGH")
            self.assertEqual(mid.offset, 4)
            tail = asyncio.run(svc.read("data.txt", max_bytes=4, offset=8))
            self.assertEqual(tail.content, "IJ")
            self.assertFalse(tail.truncated)         # reached EOF
            self.assertEqual(tail.size, 10)          # full size still reported


class TestNodePathWindows(unittest.TestCase):
    def test_read_bytes_range_local(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "f.bin").write_bytes(b"0123456789")
            np = NodePath("f.bin", _root=home)
            self.assertEqual(np.read_bytes(), b"0123456789")
            self.assertEqual(np.read_bytes(offset=3, length=4), b"3456")

    def test_iterdir_paging_local(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            for i in range(12):
                (home / f"f{i:02d}").write_text("x", encoding="utf-8")
            np = NodePath("", _root=home)
            page = list(np.iterdir(offset=0, limit=5))
            self.assertEqual(len(page), 5)
            page2 = list(np.iterdir(offset=5, limit=5))
            self.assertEqual(len(page2), 5)
            self.assertNotEqual(page[0].name, page2[0].name)
            self.assertEqual(len(list(np.iterdir())), 12)


if __name__ == "__main__":
    unittest.main()
