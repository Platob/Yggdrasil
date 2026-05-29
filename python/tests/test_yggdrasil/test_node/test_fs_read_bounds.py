"""Bounded filesystem read/disk-usage guards for the v2 FS service.

These cover the memory-safety contract: ``/fs/read`` never pulls more than
``max_read_bytes`` into memory, flags ``truncated`` on bigger files, keeps a
clean truncation as text (not base64) even when it splits a multibyte char,
and base64-encodes genuinely binary files. ``/fs/du`` stops after a bounded
number of entries instead of walking an unbounded tree.
"""
from __future__ import annotations

import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

from yggdrasil.exceptions.api import ForbiddenError
from yggdrasil.node.api.routers.fs import disk_usage
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings
from yggdrasil.node.exceptions import NotFoundError


def _service(home: Path, **overrides) -> FsService:
    settings = Settings(node_id="test-node", node_home=home, front_home=home, **overrides)
    return FsService(settings)


class TestBoundedRead(unittest.TestCase):
    def test_small_file_reads_whole(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "hello.txt").write_text("hello world", encoding="utf-8")
            svc = _service(home)
            res = asyncio.run(svc.read("hello.txt"))
            self.assertEqual(res.content, "hello world")
            self.assertEqual(res.encoding, "utf-8")
            self.assertEqual(res.size, 11)
            self.assertFalse(res.truncated)

    def test_large_file_is_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "big.txt").write_text("a" * 5000, encoding="utf-8")
            svc = _service(home, max_read_bytes=1000)
            res = asyncio.run(svc.read("big.txt"))
            self.assertTrue(res.truncated)
            self.assertEqual(len(res.content), 1000)   # only the cap, not 5000
            self.assertEqual(res.size, 5000)           # full size still reported
            self.assertEqual(res.encoding, "utf-8")

    def test_truncation_splitting_multibyte_stays_text(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            # Each euro sign is 3 bytes; a cap of 8 lands mid-character.
            (home / "euros.txt").write_text("€" * 10, encoding="utf-8")
            svc = _service(home, max_read_bytes=8)
            res = asyncio.run(svc.read("euros.txt"))
            self.assertTrue(res.truncated)
            self.assertEqual(res.encoding, "utf-8")     # not base64
            self.assertEqual(res.content, "€€")         # split char dropped

    def test_binary_file_is_base64(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            raw = b"\xff\xfe\x00\x01\x02\x80\x81"
            (home / "blob.bin").write_bytes(raw)
            svc = _service(home)
            res = asyncio.run(svc.read("blob.bin"))
            self.assertEqual(res.encoding, "base64")
            self.assertFalse(res.truncated)
            self.assertEqual(base64.b64decode(res.content), raw)

    def test_caller_window_smaller_than_cap(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "big.txt").write_text("a" * 5000, encoding="utf-8")
            svc = _service(home, max_read_bytes=4096)
            res = asyncio.run(svc.read("big.txt", max_bytes=100))
            self.assertTrue(res.truncated)
            self.assertEqual(len(res.content), 100)

    def test_caller_window_cannot_exceed_server_cap(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "big.txt").write_text("a" * 5000, encoding="utf-8")
            svc = _service(home, max_read_bytes=500)
            # Asking for 100000 must still clamp to the 500-byte server cap.
            res = asyncio.run(svc.read("big.txt", max_bytes=100000))
            self.assertTrue(res.truncated)
            self.assertEqual(len(res.content), 500)

    def test_directory_and_missing_errors(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "sub").mkdir()
            svc = _service(home)
            with self.assertRaises(ForbiddenError):
                asyncio.run(svc.read("sub"))
            with self.assertRaises(NotFoundError):
                asyncio.run(svc.read("nope.txt"))


class TestBoundedDiskUsage(unittest.TestCase):
    def test_du_stops_at_entry_cap(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            for i in range(50):
                (home / f"f{i}.txt").write_text("x", encoding="utf-8")
            svc = _service(home, du_max_entries=10)
            res = asyncio.run(disk_usage(path="", service=svc))
            self.assertTrue(res["truncated"])
            self.assertEqual(res["file_count"] + res["dir_count"], 10)

    def test_du_full_walk_not_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            for i in range(5):
                (home / f"f{i}.txt").write_text("xyz", encoding="utf-8")
            svc = _service(home, du_max_entries=1000)
            res = asyncio.run(disk_usage(path="", service=svc))
            self.assertFalse(res["truncated"])
            self.assertEqual(res["file_count"], 5)
            self.assertEqual(res["total_size_bytes"], 15)


class TestBoundedGrep(unittest.TestCase):
    def test_grep_finds_and_reports_untruncated(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "a.txt").write_text("needle here\nother\n", encoding="utf-8")
            (home / "b.txt").write_text("nothing\n", encoding="utf-8")
            svc = _service(home)
            matches, truncated = svc.grep("", "needle")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["line_number"], 1)
            self.assertFalse(truncated)

    def test_grep_stops_at_scan_cap(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            # Many files, none matching — without a scan cap this walks them all.
            for i in range(40):
                (home / f"f{i}.txt").write_text("plain\n", encoding="utf-8")
            svc = _service(home, du_max_entries=10)
            matches, truncated = svc.grep("", "needle")
            self.assertEqual(matches, [])
            self.assertTrue(truncated)

    def test_grep_truncates_at_match_cap(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "big.txt").write_text("hit\n" * 50, encoding="utf-8")
            svc = _service(home)
            matches, truncated = svc.grep("", "hit", max_matches=5)
            self.assertEqual(len(matches), 5)
            self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
