"""Integration tests — real Databricks API calls.

These require a live Databricks workspace with auth configured.
Skipped automatically when DATABRICKS_HOST is not set.
"""
from __future__ import annotations

import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ._it_base import DatabricksIntegrationBase
from ..conftest import requires_databricks

pytestmark = [requires_databricks, pytest.mark.integration]


# ══════════════════════════════════════════════════════════════════════════
# DBFS
# ══════════════════════════════════════════════════════════════════════════

class TestDBFSIntegration(DatabricksIntegrationBase):

    def test_roundtrip_text(self):
        d = self.dbfs_base / "dir"
        f = d / "hello.txt"
        with f.open("wb") as out:
            out.write(b"hello from DBFS integration")
        self.assertTrue(f.exists())
        with f.open("r") as inp:
            got = inp.read()
        self.assertEqual(got, "hello from DBFS integration")
        files = list(d.ls())
        self.assertIn(f, files)

    def test_roundtrip_binary(self):
        d = self.dbfs_base / "dirbin"
        f = d / "data.bin"
        payload = b"\x00\xff\x10\x20"
        with f.open("wb") as out:
            out.write(payload)
        with f.open("rb") as inp:
            got = inp.read()
        self.assertEqual(got, payload)

    def test_mkdir_rmdir(self):
        d = self.dbfs_base / "subdir"
        d.mkdir()
        self.assertTrue(d.exists())
        d.mkdir()  # idempotent
        d.rmdir()
        self.assertFalse(d.exists())
        d.rmdir()  # idempotent

    def test_read_bytes_write_bytes(self):
        f = self.dbfs_base / "rw.bin"
        f.write_bytes(b"binary payload")
        self.assertEqual(f.read_bytes(), b"binary payload")

    def test_read_text_write_text(self):
        f = self.dbfs_base / "rw.txt"
        f.write_text("hello unicode: ñ")
        self.assertEqual(f.read_text(), "hello unicode: ñ")

    def test_stat(self):
        f = self.dbfs_base / "stat_test.txt"
        f.write_bytes(b"123")
        st = f.stat()
        self.assertEqual(st.st_size, 3)
        self.assertGreater(st.st_mtime, 0)

    def test_iterdir(self):
        d = self.dbfs_base / "iterdir"
        (d / "a.txt").write_bytes(b"a")
        (d / "b.txt").write_bytes(b"b")
        names = {c.name for c in d.iterdir()}
        self.assertIn("a.txt", names)
        self.assertIn("b.txt", names)

    def test_glob(self):
        d = self.dbfs_base / "globdir"
        (d / "data.csv").write_bytes(b"1,2")
        (d / "data.parquet").write_bytes(b"PAR1")
        (d / "sub" / "nested.csv").write_bytes(b"3,4")
        matches = {c.name for c in d.glob("*.csv")}
        self.assertIn("data.csv", matches)
        self.assertIn("nested.csv", matches)
        self.assertNotIn("data.parquet", matches)

    def test_rename(self):
        src = self.dbfs_base / "rename_src.txt"
        dst = self.dbfs_base / "rename_dst.txt"
        src.write_bytes(b"move me")
        result = src.rename(dst)
        self.assertEqual(result.full_path(), dst.full_path())
        self.assertTrue(dst.exists())
        self.assertFalse(src.exists())
        self.assertEqual(dst.read_bytes(), b"move me")

    def test_touch(self):
        f = self.dbfs_base / "touch.txt"
        f.touch()
        self.assertTrue(f.exists())

    def test_unlink(self):
        f = self.dbfs_base / "unlink.txt"
        f.write_bytes(b"x")
        f.unlink()
        self.assertFalse(f.exists())

    def test_pathlib_properties(self):
        """Verify pathlib-like properties work on a live path."""
        f = self.dbfs_base / "props" / "data.tar.gz"
        f.write_bytes(b"fake")
        self.assertEqual(f.suffix, ".gz")
        self.assertEqual(f.suffixes, [".tar", ".gz"])
        self.assertEqual(f.stem, "data.tar")
        self.assertTrue(f.match("*.gz"))
        self.assertTrue(f.is_relative_to(self.dbfs_base))


# ══════════════════════════════════════════════════════════════════════════
# Workspace
# ══════════════════════════════════════════════════════════════════════════

class TestWorkspaceIntegration(DatabricksIntegrationBase):

    def test_roundtrip_text(self):
        d = self.ws_base / "a" / "b"
        f = d / "c.py"
        with f.open("w") as out:
            out.write("print('hello from workspace integration')\n")
        self.assertTrue(f.exists())
        with f.open("r") as inp:
            got = inp.read()
        self.assertIn("hello from workspace integration", got)

    def test_roundtrip_binary(self):
        d = self.ws_base / "bin"
        f = d / "data.bin"
        payload = b"\x01\x02\x03\x04\x05"
        with f.open("wb") as out:
            out.write(payload)
        with f.open("rb") as inp:
            got = inp.read()
        self.assertEqual(got, payload)

    def test_mkdir_rmdir(self):
        d = self.ws_base / "dir1" / "dir2"
        d.mkdir()
        self.assertTrue(d.exists())
        d.mkdir()
        d.rmdir()
        self.assertFalse(d.exists())
        d.rmdir()

    def test_read_text_write_text(self):
        f = self.ws_base / "text.txt"
        f.write_text("workspace text")
        self.assertEqual(f.read_text(), "workspace text")


# ══════════════════════════════════════════════════════════════════════════
# Volumes (Unity Catalog)
# ══════════════════════════════════════════════════════════════════════════

class TestVolumeIntegration(DatabricksIntegrationBase):

    def test_roundtrip_text(self):
        d = self.vol_base / "nested"
        f = d / "hello.txt"
        with f.open("w") as out:
            out.write("hello from volumes integration")
        self.assertTrue(f.exists())
        with f.open("r") as inp:
            got = inp.read()
        self.assertEqual(got, "hello from volumes integration")
        files = list(d.ls())
        self.assertIn(f, files)
        d.rmdir()
        self.assertFalse(d.exists())

    def test_roundtrip_binary(self):
        d = self.vol_base / "nestedbin"
        f = d / "data.bin"
        payload = b"\xde\xad\xbe\xef"
        with f.open("wb") as out:
            out.write(payload)
        with f.open("rb") as inp:
            got = inp.read()
        self.assertEqual(got, payload)

    def test_mkdir_rmdir(self):
        d = self.vol_base / "dirA" / "dirB"
        d.mkdir()
        self.assertTrue(d.exists())
        d.mkdir()
        d.rmdir()
        self.assertFalse(d.exists())
        d.rmdir()

    def test_read_write_parquet(self):
        d = self.vol_base / "datafolder"
        f = d / "data.parquet"
        table = pa.table({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        with f.open("wb") as out:
            pq.write_table(table, out)
        with f.open("rb") as inp:
            read_table = pq.read_table(inp)
        self.assertTrue(table.equals(read_table))
        d.rmdir()

    def test_temporary_credentials(self):
        folder_path = self.vol_base / "tmp_path"
        folder_path.mkdir()
        creds = folder_path.temporary_credentials()
        self.assertTrue(creds)

    def test_read_text_write_text(self):
        f = self.vol_base / "text.txt"
        f.write_text("volume text ñ")
        self.assertEqual(f.read_text(), "volume text ñ")

    def test_stat(self):
        f = self.vol_base / "stat_test.dat"
        f.write_bytes(b"abcdef")
        st = f.stat()
        self.assertEqual(st.st_size, 6)
        self.assertGreater(st.st_mtime, 0)

    def test_iterdir(self):
        d = self.vol_base / "iterdir"
        (d / "x.txt").write_bytes(b"x")
        (d / "y.txt").write_bytes(b"y")
        names = {c.name for c in d.iterdir()}
        self.assertIn("x.txt", names)
        self.assertIn("y.txt", names)

    def test_glob(self):
        d = self.vol_base / "globdir"
        (d / "a.csv").write_bytes(b"1")
        (d / "b.json").write_bytes(b"{}")
        (d / "sub" / "c.csv").write_bytes(b"2")
        matches = {c.name for c in d.glob("*.csv")}
        self.assertIn("a.csv", matches)
        self.assertIn("c.csv", matches)
        self.assertNotIn("b.json", matches)

    def test_rename(self):
        src = self.vol_base / "mv_src.txt"
        dst = self.vol_base / "mv_dst.txt"
        src.write_bytes(b"move me")
        result = src.rename(dst)
        self.assertTrue(dst.exists())
        self.assertFalse(src.exists())
        self.assertEqual(dst.read_bytes(), b"move me")

    def test_copy_to(self):
        src = self.vol_base / "cp_src.txt"
        dst = self.vol_base / "cp_dst.txt"
        src.write_bytes(b"copy me")
        src.copy_to(dst)
        self.assertTrue(dst.exists())
        self.assertTrue(src.exists())  # source kept
        self.assertEqual(dst.read_bytes(), b"copy me")


if __name__ == "__main__":
    unittest.main(verbosity=2)

