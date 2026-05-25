"""Tests for NodePath local filesystem operations."""
from __future__ import annotations

import shutil
import pytest
from pathlib import Path

from yggdrasil.node.config import Settings
from yggdrasil.node.path import NodePath


@pytest.fixture
def tmp_root(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    return root


class TestNodePathLocal:
    def test_write_and_read_text(self, tmp_root):
        p = NodePath("test.txt", _root=tmp_root)
        p.write_text("hello world")
        assert p.read_text() == "hello world"

    def test_write_and_read_bytes(self, tmp_root):
        p = NodePath("test.bin", _root=tmp_root)
        p.write_bytes(b"\x00\x01\x02")
        assert p.read_bytes() == b"\x00\x01\x02"

    def test_mkdir_and_iterdir(self, tmp_root):
        d = NodePath("subdir", _root=tmp_root)
        d.mkdir()
        (d / "a.txt").write_text("a")
        (d / "b.txt").write_text("b")
        children = list(d.iterdir())
        names = sorted(c.name for c in children)
        assert names == ["a.txt", "b.txt"]

    def test_is_dir_and_is_file(self, tmp_root):
        d = NodePath("mydir", _root=tmp_root)
        d.mkdir()
        (d / "file.txt").write_text("x")
        assert d.is_dir()
        assert not d.is_file()
        assert (d / "file.txt").is_file()

    def test_exists(self, tmp_root):
        p = NodePath("nope.txt", _root=tmp_root)
        assert not p.exists()
        p.write_text("now I exist")
        assert p.exists()

    def test_stat(self, tmp_root):
        p = NodePath("stat.txt", _root=tmp_root)
        p.write_text("data")
        info = p.stat()
        assert info["name"] == "stat.txt"
        assert info["size"] == 4
        assert not info["is_dir"]

    def test_unlink(self, tmp_root):
        p = NodePath("del.txt", _root=tmp_root)
        p.write_text("bye")
        assert p.exists()
        p.unlink()
        assert not p.exists()

    def test_unlink_dir(self, tmp_root):
        d = NodePath("deldir", _root=tmp_root)
        d.mkdir()
        (d / "inner.txt").write_text("x")
        d.unlink()
        assert not d.exists()

    def test_rename(self, tmp_root):
        p = NodePath("old.txt", _root=tmp_root)
        p.write_text("content")
        new = p.rename("new.txt")
        assert new.name == "new.txt"
        assert new.read_text() == "content"
        assert not p.exists()

    def test_path_traversal_blocked(self, tmp_root):
        p = NodePath("../../etc/passwd", _root=tmp_root)
        with pytest.raises(PermissionError):
            p._local_path()

    def test_parent(self, tmp_root):
        p = NodePath("a/b/c.txt", _root=tmp_root)
        assert str(p.parent) == "a/b"
        assert p.parent.parent.name == "a"

    def test_truediv(self, tmp_root):
        p = NodePath("base", _root=tmp_root)
        child = p / "sub" / "file.txt"
        assert str(child) == "base/sub/file.txt"

    def test_stream_read(self, tmp_root):
        p = NodePath("stream.bin", _root=tmp_root)
        data = b"x" * 10000
        p.write_bytes(data)
        chunks = list(p.stream_read(chunk_size=4096))
        assert b"".join(chunks) == data

    def test_copy_to(self, tmp_root):
        src = NodePath("src.txt", _root=tmp_root)
        src.write_text("copy me")
        dst = NodePath("dst.txt", _root=tmp_root)
        src.copy_to(dst)
        assert dst.read_text() == "copy me"

    def test_suffix_and_stem(self, tmp_root):
        p = NodePath("data/file.csv", _root=tmp_root)
        assert p.suffix == ".csv"
        assert p.stem == "file"
