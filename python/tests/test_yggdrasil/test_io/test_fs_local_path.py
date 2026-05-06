"""Tests for yggdrasil.io.fs.local_path.LocalPath."""

from __future__ import annotations

import os

import pytest

from yggdrasil.io.path import LocalPath, Path
from yggdrasil.io.io_stats import IOKind


# ---------------------------------------------------------------------------
# Construction / dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_path_factory_returns_local_for_filesystem_path(self, tmp_path):
        p = Path.from_(tmp_path)
        assert isinstance(p, LocalPath)

    def test_local_path_is_local(self, tmp_path):
        assert LocalPath.from_(tmp_path).is_local is True

    def test_handles_str(self):
        assert LocalPath.handles("/tmp/x") is True

    def test_handles_pathlib(self, tmp_path):
        assert LocalPath.handles(tmp_path) is True

    def test_handles_remote_url(self):
        from yggdrasil.io.url import URL

        assert LocalPath.handles(URL.from_str("https://example.com/")) is False


# ---------------------------------------------------------------------------
# Stat / exists
# ---------------------------------------------------------------------------


class TestStat:
    def test_missing_kind(self, tmp_path):
        path = LocalPath.from_(tmp_path / "ghost")
        assert path.stat().kind is IOKind.MISSING

    def test_directory_kind(self, tmp_path):
        path = LocalPath.from_(tmp_path)
        assert path.stat().kind is IOKind.DIRECTORY

    def test_file_kind(self, tmp_path):
        target = tmp_path / "x.txt"
        target.write_bytes(b"hello")
        path = LocalPath.from_(target)
        assert path.stat().kind is IOKind.FILE
        assert path.stat().size == len(b"hello")

    def test_exists_true_for_directory(self, tmp_path):
        assert LocalPath.from_(tmp_path).exists() is True

    def test_exists_false_for_missing(self, tmp_path):
        assert LocalPath.from_(tmp_path / "ghost").exists() is False

    def test_is_file_is_dir(self, tmp_path):
        target = tmp_path / "x.txt"
        target.write_bytes(b"x")
        path = LocalPath.from_(target)
        assert path.is_file() is True
        assert path.is_dir() is False


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


class TestReadWrite:
    def test_write_bytes_creates_file(self, tmp_path):
        path = LocalPath.from_(tmp_path / "x.txt")
        path.write_bytes(b"hello")
        assert path.read_bytes() == b"hello"

    def test_write_bytes_creates_parents(self, tmp_path):
        path = LocalPath.from_(tmp_path / "nested" / "deep" / "x.txt")
        path.write_bytes(b"hello")
        assert path.read_bytes() == b"hello"

    def test_read_bytes_missing_raises(self, tmp_path):
        path = LocalPath.from_(tmp_path / "ghost")
        with pytest.raises(OSError):
            path.read_bytes()

    def test_read_bytes_missing_silent(self, tmp_path):
        path = LocalPath.from_(tmp_path / "ghost")
        assert path.read_bytes(raise_error=False) == b""


class TestPositionalIO:
    def test_pread_partial(self, tmp_path):
        target = tmp_path / "x.bin"
        target.write_bytes(b"abcdef")
        path = LocalPath.from_(target)
        assert path.pread(3, 2) == b"cde"

    def test_pread_to_end(self, tmp_path):
        target = tmp_path / "x.bin"
        target.write_bytes(b"abcdef")
        path = LocalPath.from_(target)
        assert path.pread(-1, 0) == b"abcdef"

    def test_pwrite_creates_and_grows(self, tmp_path):
        path = LocalPath.from_(tmp_path / "x.bin")
        path.pwrite(b"hello", 0)
        assert path.read_bytes() == b"hello"

    def test_pwrite_overwrites_segment(self, tmp_path):
        target = tmp_path / "x.bin"
        target.write_bytes(b"AAAAAAAAAA")
        path = LocalPath.from_(target)
        path.pwrite(b"XYZ", 2)
        assert path.read_bytes() == b"AAXYZAAAAA"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListing:
    def test_iterdir(self, tmp_path):
        (tmp_path / "a").write_bytes(b"")
        (tmp_path / "b").write_bytes(b"")
        path = LocalPath.from_(tmp_path)
        names = sorted(p.name for p in path.iterdir())
        assert names == ["a", "b"]

    def test_ls_recursive(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.txt").write_bytes(b"")
        path = LocalPath.from_(tmp_path)
        names = {p.name for p in path.ls(recursive=True)}
        assert "deep.txt" in names

    def test_ls_with_include_pattern(self, tmp_path):
        (tmp_path / "a.csv").write_bytes(b"")
        (tmp_path / "b.json").write_bytes(b"")
        path = LocalPath.from_(tmp_path)
        names = sorted(
            p.name for p in path.ls(include_patterns=["*.csv"])
        )
        assert names == ["a.csv"]

    def test_ls_missing_silent(self, tmp_path):
        path = LocalPath.from_(tmp_path / "ghost")
        assert list(path.ls(allow_not_found=True)) == []


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


class TestMutation:
    def test_mkdir(self, tmp_path):
        path = LocalPath.from_(tmp_path / "deep")
        path.mkdir()
        assert path.is_dir()

    def test_unlink(self, tmp_path):
        target = tmp_path / "x.txt"
        target.write_bytes(b"x")
        LocalPath.from_(target).unlink()
        assert not target.exists()

    def test_unlink_missing_ok(self, tmp_path):
        # Should not raise
        LocalPath.from_(tmp_path / "ghost").unlink(missing_ok=True)

    def test_unlink_directory_raises(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            LocalPath.from_(tmp_path).unlink()

    def test_rmdir_recursive(self, tmp_path):
        target = tmp_path / "deep"
        target.mkdir()
        (target / "x.txt").write_bytes(b"x")
        LocalPath.from_(target).rmdir(recursive=True)
        assert not target.exists()

    def test_touch(self, tmp_path):
        target = tmp_path / "x.txt"
        path = LocalPath.from_(target)
        path.touch()
        assert target.exists()


# ---------------------------------------------------------------------------
# Pure-path API
# ---------------------------------------------------------------------------


class TestPathProperties:
    def test_name_stem_suffix(self, tmp_path):
        path = LocalPath.from_(tmp_path / "data.csv.gz")
        assert path.name == "data.csv.gz"
        assert path.stem == "data.csv"
        assert path.suffix == ".gz"

    def test_extensions(self, tmp_path):
        path = LocalPath.from_(tmp_path / "data.csv.gz")
        assert path.extensions == ["csv", "gz"]

    def test_parent(self, tmp_path):
        path = LocalPath.from_(tmp_path / "deep" / "x.txt")
        assert path.parent.url.path == str(tmp_path / "deep")

    def test_truediv_operator(self, tmp_path):
        path = LocalPath.from_(tmp_path)
        assert (path / "x").name == "x"

    def test_fspath_protocol(self, tmp_path):
        path = LocalPath.from_(tmp_path)
        assert os.fspath(path).startswith("/")


# ---------------------------------------------------------------------------
# fd lifecycle — the user-visible contract
# ---------------------------------------------------------------------------


class TestFdLifecycle:
    def test_fileno_returns_open_fd_for_existing_file(self, tmp_path):
        target = tmp_path / "x.bin"
        target.write_bytes(b"hi")
        path = LocalPath.from_pathlib(target)
        try:
            fd = path.fileno()
            assert fd >= 0
            # Fd is real — fstat round-trips.
            assert os.fstat(fd).st_size == 2
        finally:
            path.close()

    def test_fileno_raises_after_close(self, tmp_path):
        target = tmp_path / "x.bin"
        target.write_bytes(b"hi")
        path = LocalPath.from_pathlib(target)
        path.fileno()  # open succeeds
        path.close()
        with pytest.raises(OSError):
            path.fileno()

    def test_construction_does_not_create_missing_file(self, tmp_path):
        target = tmp_path / "ghost.bin"
        # Default mode "rb+"; auto_open=True.
        path = LocalPath.from_pathlib(target)
        try:
            # Construction-time acquire is best-effort: missing target
            # leaves fd closed, fileno raises, file is NOT created.
            assert not target.exists()
            with pytest.raises(OSError):
                path.fileno()
        finally:
            path.close()

    def test_acquire_io_with_write_mode_opens_and_creates(self, tmp_path):
        target = tmp_path / "fresh.bin"
        path = LocalPath.from_pathlib(target)
        try:
            path.acquire_io("wb+")
            fd = path.fileno()
            assert fd >= 0
            assert target.exists()
        finally:
            path.close()

    def test_close_io_closes_fd_but_keeps_path_alive(self, tmp_path):
        target = tmp_path / "x.bin"
        target.write_bytes(b"hello")
        path = LocalPath.from_pathlib(target)
        try:
            path.fileno()  # open
            path.close_io()
            with pytest.raises(OSError):
                path.fileno()
            # Re-acquire works — path object is still alive.
            path.acquire_io("rb")
            assert path.fileno() >= 0
        finally:
            path.close()
