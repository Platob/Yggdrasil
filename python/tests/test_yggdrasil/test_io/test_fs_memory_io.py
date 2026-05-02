"""Tests for yggdrasil.io.fs.memory_io.MemoryPath / MemoryIO."""

from __future__ import annotations

import pytest

from yggdrasil.io.fs.memory_io import MemoryIO, MemoryPath, REGISTRY
from yggdrasil.io.path_stat import PathKind
from yggdrasil.io.url import URL


@pytest.fixture(autouse=True)
def _clean_memory_registry():
    """Wipe the in-memory registry between tests so they don't leak state."""
    REGISTRY.clear()
    yield
    REGISTRY.clear()


def _mem(path: str) -> MemoryPath:
    return MemoryPath.from_url(URL.from_str(f"memory:{path}"))


# ---------------------------------------------------------------------------
# Path lifecycle
# ---------------------------------------------------------------------------


class TestMemoryPathLifecycle:
    def test_missing_initially(self):
        path = _mem("/data/x")
        assert path.stat().kind is PathKind.MISSING

    def test_write_then_read(self):
        path = _mem("/data/x")
        path.write_bytes(b"hello")
        assert path.read_bytes() == b"hello"

    def test_stat_size_after_write(self):
        path = _mem("/data/x")
        path.write_bytes(b"hello")
        assert path.stat().size == len(b"hello")
        assert path.stat().kind is PathKind.FILE

    def test_unlink_removes_entry(self):
        path = _mem("/data/x")
        path.write_bytes(b"x")
        path.unlink()
        assert not path.exists()

    def test_full_path_includes_scheme(self):
        path = _mem("/data/x")
        assert path.full_path() == "memory:/data/x"


# ---------------------------------------------------------------------------
# Directory semantics
# ---------------------------------------------------------------------------


class TestMemoryDirectories:
    def test_prefix_implies_directory(self):
        leaf = _mem("/data/x")
        leaf.write_bytes(b"x")
        parent = _mem("/data")
        assert parent.stat().kind is PathKind.DIRECTORY

    def test_listing_yields_children(self):
        _mem("/data/a").write_bytes(b"")
        _mem("/data/b").write_bytes(b"")
        _mem("/data/sub/c").write_bytes(b"")
        names = sorted(p.name for p in _mem("/data").iterdir())
        assert "a" in names
        assert "b" in names
        # Nested 'sub' shows up as an implicit directory
        assert any("sub" in n for n in names)


# ---------------------------------------------------------------------------
# Open modes
# ---------------------------------------------------------------------------


class TestOpenModes:
    def test_open_rb_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            with _mem("/ghost").open_io("rb"):
                pass

    def test_open_wb_truncates(self):
        path = _mem("/x")
        path.write_bytes(b"original")
        with path.open_io("wb") as fh:
            fh.write(b"new")
        assert path.read_bytes() == b"new"

    def test_open_xb_on_existing_raises(self):
        path = _mem("/x")
        path.write_bytes(b"data")
        with pytest.raises(FileExistsError):
            path.open_io("xb")

    def test_open_ab_appends(self):
        path = _mem("/x")
        path.write_bytes(b"orig")
        with path.open_io("ab") as fh:
            fh.write(b"-more")
        assert path.read_bytes() == b"orig-more"


# ---------------------------------------------------------------------------
# MemoryIO behavior
# ---------------------------------------------------------------------------


class TestMemoryIOSurface:
    def test_read_advances_cursor(self):
        path = _mem("/x")
        path.write_bytes(b"abcdef")
        with path.open_io("rb") as fh:
            assert fh.read(3) == b"abc"
            assert fh.tell() == 3

    def test_seek_set(self):
        path = _mem("/x")
        path.write_bytes(b"abcdef")
        with path.open_io("rb") as fh:
            fh.seek(2)
            assert fh.read(2) == b"cd"

    def test_independent_cursors(self):
        path = _mem("/x")
        path.write_bytes(b"abcdef")
        with path.open_io("rb") as fh1, path.open_io("rb") as fh2:
            fh1.read(3)
            assert fh2.tell() == 0

    def test_write_reflected_across_handles(self):
        path = _mem("/x")
        path.write_bytes(b"abc")
        with path.open_io("rb+") as writer, path.open_io("rb") as reader:
            writer.seek(0)
            writer.write(b"XYZ")
            assert reader.read() == b"XYZ"

    def test_write_to_read_only_raises(self):
        path = _mem("/x")
        path.write_bytes(b"abc")
        with path.open_io("rb") as fh:
            with pytest.raises(ValueError):
                fh.write(b"x")

    def test_path_property(self):
        path = _mem("/x")
        path.write_bytes(b"abc")
        with path.open_io("rb") as fh:
            assert isinstance(fh, MemoryIO)
            assert fh.path is path
