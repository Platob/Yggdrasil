"""Dispatch tests for ``ygg databricks fs`` (mocked DatabricksPath)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


def _child(name: str, *, is_dir: bool, size: int = 0) -> MagicMock:
    c = MagicMock()
    c.name = name
    c.is_dir.return_value = is_dir
    c.size = size
    c.full_path.return_value = f"/Volumes/cat/sch/vol/{name}"
    return c


class TestFSHelp(unittest.TestCase):
    def test_fs_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["fs", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestFSDispatch(unittest.TestCase):
    def test_ls_lists_children(self):
        path = MagicMock()
        path.iterdir.return_value = [_child("a.parquet", is_dir=False, size=10), _child("sub", is_dir=True)]
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.return_value = path
            rc = main(["fs", "ls", "/Volumes/cat/sch/vol"])
        self.assertEqual(rc, 0)
        DP.from_.assert_called_once()
        path.iterdir.assert_called_once()

    def test_cat_reads_bytes(self):
        path = MagicMock()
        path.read_bytes.return_value = b"hello"
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.return_value = path
            rc = main(["fs", "cat", "/Volumes/cat/sch/vol/a.txt"])
        self.assertEqual(rc, 0)
        path.read_bytes.assert_called_once()

    def test_write_data_literal(self):
        path = MagicMock()
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.return_value = path
            rc = main(["fs", "write", "/Volumes/cat/sch/vol/a.txt", "--data", "hi"])
        self.assertEqual(rc, 0)
        path.write_bytes.assert_called_once_with(b"hi", overwrite=True)

    def test_cp_moves_bytes_across_surfaces(self):
        src, dst = MagicMock(), MagicMock()
        src.read_bytes.return_value = b"DATA"
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.side_effect = [src, dst]
            rc = main(["fs", "cp", "/Workspace/Shared/a.txt", "/Volumes/cat/sch/vol/a.txt"])
        self.assertEqual(rc, 0)
        src.read_bytes.assert_called_once()
        dst.write_bytes.assert_called_once_with(b"DATA", overwrite=True)

    def test_mv_copies_then_deletes_source(self):
        src, dst = MagicMock(), MagicMock()
        src.read_bytes.return_value = b"DATA"
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.side_effect = [src, dst]
            rc = main(["fs", "mv", "/dbfs/tmp/a", "/Volumes/cat/sch/vol/a"])
        self.assertEqual(rc, 0)
        dst.write_bytes.assert_called_once_with(b"DATA", overwrite=True)
        src.unlink.assert_called_once()

    def test_rm_recursive(self):
        path = MagicMock()
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.return_value = path
            rc = main(["fs", "rm", "/Volumes/cat/sch/vol/dir", "-r"])
        self.assertEqual(rc, 0)
        path.remove.assert_called_once_with(recursive=True)


if __name__ == "__main__":
    unittest.main()
