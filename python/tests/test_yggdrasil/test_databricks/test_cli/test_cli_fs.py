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

    def test_create_notebook_imports_source(self):
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath

        path = MagicMock(spec=WorkspacePath)
        path.full_path.return_value = "/Workspace/Users/me/nb"
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.return_value = path
            rc = main([
                "fs", "create-notebook", "/Workspace/Users/me/nb",
                "--language", "sql", "--data", "SELECT 1",
            ])
        self.assertEqual(rc, 0)
        path.create_notebook.assert_called_once_with(
            "sql", content="SELECT 1", overwrite=False,
        )

    def test_create_notebook_rejects_non_workspace(self):
        path = MagicMock()  # not a WorkspacePath
        path.full_path.return_value = "/Volumes/cat/sch/vol/nb"
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.return_value = path
            rc = main(["fs", "create-notebook", "/Volumes/cat/sch/vol/nb"])
        self.assertEqual(rc, 1)
        path.create_notebook.assert_not_called()

    def test_run_notebook_submits_with_params(self):
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath

        path = MagicMock(spec=WorkspacePath)
        path.full_path.return_value = "/Workspace/Shared/etl"
        run = MagicMock()
        run.run_id = 99
        run.is_succeeded = True
        path.run_notebook.return_value = run
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.return_value = path
            rc = main([
                "fs", "run-notebook", "/Workspace/Shared/etl",
                "--param", "date=2024-01-01", "--param", "n=5", "--no-wait",
            ])
        self.assertEqual(rc, 0)
        path.run_notebook.assert_called_once_with(
            {"date": "2024-01-01", "n": "5"},
            cluster=None, environment=None, wait=False, raise_error=False,
        )

    def test_run_notebook_rejects_non_workspace(self):
        path = MagicMock()  # not a WorkspacePath
        path.full_path.return_value = "/Volumes/cat/sch/vol/nb"
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.client.DatabricksClient"):
            DP.from_.return_value = path
            rc = main(["fs", "run-notebook", "/Volumes/cat/sch/vol/nb"])
        self.assertEqual(rc, 1)
        path.run_notebook.assert_not_called()

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
