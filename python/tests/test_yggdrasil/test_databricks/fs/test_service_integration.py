from __future__ import annotations

import pytest

from ._it_base import DatabricksIntegrationBase
from ..conftest import requires_databricks

pytestmark = [requires_databricks, pytest.mark.integration]


class TestFileSystemServiceIntegration(DatabricksIntegrationBase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.fs = cls.workspace.filesystem

    def test_dbfs_roundtrip(self):
        root = self.dbfs_base / "service"
        file_path = root / "hello.txt"

        self.fs.makedirs(root, exist_ok=True)
        self.fs.write_text(file_path, "hello from service")

        assert self.fs.exists(file_path)
        assert self.fs.isfile(file_path)
        assert not self.fs.isdir(file_path)
        assert self.fs.read_text(file_path) == "hello from service"
        assert "hello.txt" in self.fs.listdir(root)

    def test_rename_and_remove(self):
        root = self.dbfs_base / "service-rename"
        src = root / "src.txt"
        dst = root / "dst.txt"

        self.fs.makedirs(root, exist_ok=True)
        self.fs.write_bytes(src, b"payload")
        renamed = self.fs.rename(src, dst)

        assert renamed.full_path() == dst.full_path()
        assert self.fs.exists(dst)
        assert not self.fs.exists(src)
        assert self.fs.read_bytes(dst) == b"payload"

        self.fs.remove(dst)
        assert not self.fs.exists(dst)

    def test_walk_and_copytree(self):
        src_root = self.dbfs_base / "service-tree-src"
        dst_root = self.dbfs_base / "service-tree-dst"

        self.fs.write_text(src_root / "a.txt", "a")
        self.fs.write_text(src_root / "sub" / "b.txt", "b")
        copied = self.fs.copytree(src_root, dst_root)

        assert copied.full_path() == dst_root.full_path()
        assert sorted(self.fs.listdir(dst_root, recursive=True)) == ["a.txt", "b.txt"]

        walked = list(self.fs.walk(dst_root))
        assert walked
        assert walked[0][0].full_path() == dst_root.full_path()
