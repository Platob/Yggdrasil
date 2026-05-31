"""Tests for :class:`yggdrasil.path.local_path.LocalPath`."""

from __future__ import annotations

import os
import pathlib
import time

import pytest

from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.io_stats import IOKind, IOStats


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_from_string(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "a.bin"), singleton_ttl=False)
        assert p.os_path == os.path.normpath(str(tmp_path / "a.bin"))

    def test_from_pathlib(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(path=tmp_path / "b.bin", singleton_ttl=False)
        assert p.os_path == os.path.normpath(str(tmp_path / "b.bin"))

    def test_from_url(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "c.bin"
        p = LocalPath(url=f"file://{target}", singleton_ttl=False)
        assert p.os_path == os.path.normpath(str(target))

    def test_from_factory(self, tmp_path: pathlib.Path) -> None:
        target = str(tmp_path / "d.bin")
        p = LocalPath.from_(target)
        assert isinstance(p, LocalPath)
        assert p.os_path == os.path.normpath(target)

    def test_repr(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "r.bin"), singleton_ttl=False)
        r = repr(p)
        assert r.startswith("LocalPath(")
        assert "r.bin" in r


# ---------------------------------------------------------------------------
# TestFileIO
# ---------------------------------------------------------------------------


class TestFileIO:

    def test_write_then_read(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "io.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"hello world")
            assert p.read_bytes() == b"hello world"

    def test_pwrite_pread(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "pos.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"0123456789")
            p.pwrite(b"XX", 3)
            assert p.pread(2, 3) == b"XX"
            # surrounding bytes unaffected
            assert p.pread(3, 0) == b"012"
            assert p.pread(4, 5) == b"56789"[:4]

    def test_truncate(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "trunc.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"abcdefghij")
            p.truncate(5)
            assert p.size == 5
            assert p.read_bytes() == b"abcde"

    def test_reserve_negative_raises(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "res.bin"), singleton_ttl=False)
        with p:
            with pytest.raises(ValueError, match="reserve size must be >= 0"):
                p.reserve(-1)

    def test_reserve_noop(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "res2.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"abc")
            p.reserve(1024)
            # reserve is a no-op on local FS; size should not change
            assert p.size == 3

    def test_size_property(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "sz.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"abcde")
            assert p.size == 5

    def test_read_mv_write_mv(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "mv.bin"), singleton_ttl=False)
        with p:
            data = memoryview(b"memoryview-data")
            p.write_mv(data, 0)
            out = p.read_mv(len(data), 0)
            assert bytes(out) == b"memoryview-data"

    def test_empty_file_size_zero(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "empty.bin"
        target.write_bytes(b"")
        p = LocalPath(str(target), singleton_ttl=False)
        assert p.size == 0

    def test_write_then_overwrite(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "ow.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"first")
            assert p.read_bytes() == b"first"
            p.write_bytes(b"SECOND", overwrite=True)
            assert p.read_bytes() == b"SECOND"


# ---------------------------------------------------------------------------
# TestCursor
# ---------------------------------------------------------------------------


class TestCursor:

    def test_open_returns_cursor(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "cur.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"cursor-data")
            cursor = p.open()
            try:
                # cursor should have a parent pointing back at p
                assert cursor._parent is p
            finally:
                cursor.close()

    def test_cursor_read_seek_tell(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "cst.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"abcdef")
            cursor = p.open()
            try:
                assert cursor.tell() == 0
                chunk = cursor.read_bytes(3, cursor=True)
                assert chunk == b"abc"
                assert cursor.tell() == 3
                cursor.seek(1)
                assert cursor.tell() == 1
                chunk2 = cursor.read_bytes(2, cursor=True)
                assert chunk2 == b"bc"
            finally:
                cursor.close()

    def test_multiple_cursors_independent(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "multi.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"0123456789")
            c1 = p.open()
            c2 = p.open()
            try:
                c1.seek(2)
                c2.seek(7)
                assert c1.tell() == 2
                assert c2.tell() == 7
                # reading from one cursor shouldn't affect the other
                c1.read_bytes(3, cursor=True)
                assert c1.tell() == 5
                assert c2.tell() == 7
            finally:
                c1.close()
                c2.close()

    def test_cursor_close_does_not_close_path(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "cclose.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"data")
            cursor = p.open(owns_holder=False)
            cursor.close()
            # path should still be open and readable
            assert p.opened
            assert p.read_bytes() == b"data"

    def test_cursor_with_statement(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "cwith.bin"), singleton_ttl=False)
        with p:
            p.write_bytes(b"context")
            with p.open() as cursor:
                got = cursor.read_bytes(7, cursor=True)
                assert got == b"context"
            # path still usable after cursor exits
            assert p.read_bytes() == b"context"


# ---------------------------------------------------------------------------
# TestStat
# ---------------------------------------------------------------------------


class TestStat:

    def test_stat_returns_iostats(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "st.bin"
        target.write_bytes(b"hello")
        p = LocalPath(str(target), singleton_ttl=False)
        s = p.stat()
        assert isinstance(s, IOStats)
        assert s.size == 5
        assert s.kind == IOKind.FILE

    def test_mtime_updates_after_write(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "mt.bin"
        target.write_bytes(b"v1")
        p = LocalPath(str(target), singleton_ttl=False)
        mtime1 = p.stat().mtime

        # Ensure filesystem mtime granularity is exceeded
        time.sleep(0.05)
        target.write_bytes(b"v2-updated")
        p.invalidate_singleton(remove_global=False)
        mtime2 = p.stat().mtime
        assert mtime2 >= mtime1

    def test_exists_is_file_is_dir(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "exist.bin"
        f.write_bytes(b"x")
        d = tmp_path / "subdir"
        d.mkdir()

        pf = LocalPath(str(f), singleton_ttl=False)
        pd = LocalPath(str(d), singleton_ttl=False)
        pm = LocalPath(str(tmp_path / "nope"), singleton_ttl=False)

        assert pf.exists()
        assert pf.is_file()
        assert not pf.is_dir()

        assert pd.exists()
        assert pd.is_dir()
        assert not pd.is_file()

        assert not pm.exists()

    def test_stat_cache_invalidation(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "cache.bin"
        target.write_bytes(b"abc")
        p = LocalPath(str(target), singleton_ttl=False)

        s1 = p.stat()
        assert s1.size == 3

        # Modify the file externally
        target.write_bytes(b"abcdefgh")
        # Without invalidation, the LocalPath has no TTL-based cache
        # by default, so _stat always does a live probe.
        s2 = p.stat()
        assert s2.size == 8

        # Explicitly seed a cached stat, then invalidate
        p._persist_stat_cache(IOStats(size=999, kind=IOKind.FILE))
        p.invalidate_singleton(remove_global=False)
        s3 = p.stat()
        assert s3.size == 8  # live probe, not the 999


# ---------------------------------------------------------------------------
# TestDirectoryOps
# ---------------------------------------------------------------------------


class TestDirectoryOps:

    def test_mkdir(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "newdir"
        p = LocalPath(str(target), singleton_ttl=False)
        p.mkdir(parents=False, exist_ok=True)
        assert target.is_dir()

    def test_mkdir_parents(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        p = LocalPath(str(target), singleton_ttl=False)
        p.mkdir(parents=True, exist_ok=True)
        assert target.is_dir()

    def test_iterdir_lists_files(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        (tmp_path / "file3.txt").write_text("c")

        p = LocalPath(str(tmp_path), singleton_ttl=False)
        children = list(p.iterdir())
        names = sorted(c.name for c in children)
        assert names == ["file1.txt", "file2.txt", "file3.txt"]

    def test_iterdir_includes_dotfiles(self, tmp_path: pathlib.Path) -> None:
        # LocalPath._ls uses os.scandir which yields ALL entries
        # including dotfiles (no skip logic in the implementation).
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("public")

        p = LocalPath(str(tmp_path), singleton_ttl=False)
        children = list(p.iterdir())
        names = sorted(c.name for c in children)
        assert ".hidden" in names
        assert "visible.txt" in names

    def test_remove_file_and_remove_dir(self, tmp_path: pathlib.Path) -> None:
        # File removal
        f = tmp_path / "to_delete.bin"
        f.write_bytes(b"bye")
        pf = LocalPath(str(f), singleton_ttl=False)
        pf.remove(recursive=False, missing_ok=False)
        assert not f.exists()

        # Directory removal
        d = tmp_path / "dir_to_delete"
        d.mkdir()
        (d / "child.txt").write_text("x")
        pd = LocalPath(str(d), singleton_ttl=False)
        pd.remove(recursive=True, missing_ok=False)
        assert not d.exists()


# ---------------------------------------------------------------------------
# TestPathOps
# ---------------------------------------------------------------------------


class TestPathOps:

    def test_name_stem_suffix_suffixes(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "data.tar.gz"), singleton_ttl=False)
        assert p.name == "data.tar.gz"
        assert p.stem == "data.tar"
        assert p.suffix == ".gz"
        assert p.suffixes == [".tar", ".gz"]

    def test_parts(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "sub" / "file.txt"), singleton_ttl=False)
        parts = p.parts
        assert parts[-1] == "file.txt"
        assert parts[-2] == "sub"
        assert len(parts) >= 3  # at least /tmp/<something>/sub/file.txt

    def test_with_name_with_suffix(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "old.csv"), singleton_ttl=False)
        p2 = p.with_name("new.csv")
        assert p2.name == "new.csv"
        assert isinstance(p2, LocalPath)

        p3 = p.with_suffix(".parquet")
        assert p3.name == "old.parquet"
        assert isinstance(p3, LocalPath)

    def test_truediv_joins(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path), singleton_ttl=False)
        child = p / "subdir" / "file.bin"
        assert isinstance(child, LocalPath)
        expected = os.path.normpath(os.path.join(str(tmp_path), "subdir", "file.bin"))
        assert child.os_path == expected

    def test_is_absolute(self, tmp_path: pathlib.Path) -> None:
        # URL.is_absolute requires both scheme and host; file:// URLs
        # have an empty host, so local paths report False here — the
        # OS path is still absolute in the POSIX sense.
        p = LocalPath(str(tmp_path / "abs.bin"), singleton_ttl=False)
        assert not p.is_absolute  # file:// has no host
        assert os.path.isabs(p.os_path)


# ---------------------------------------------------------------------------
# TestSingleton
# ---------------------------------------------------------------------------


class TestSingleton:

    def test_same_path_same_instance(self, tmp_path: pathlib.Path) -> None:
        target = str(tmp_path / "single.bin")
        a = LocalPath(target, singleton_ttl=None)
        b = LocalPath(target, singleton_ttl=None)
        assert a is b

    def test_different_path_different_instance(self, tmp_path: pathlib.Path) -> None:
        a = LocalPath(str(tmp_path / "one.bin"), singleton_ttl=None)
        b = LocalPath(str(tmp_path / "two.bin"), singleton_ttl=None)
        assert a is not b

    def test_invalidate_singleton(self, tmp_path: pathlib.Path) -> None:
        target = str(tmp_path / "inv.bin")
        a = LocalPath(target, singleton_ttl=None)
        a.invalidate_singleton(remove_global=True)
        b = LocalPath(target, singleton_ttl=None)
        # After invalidation, a new instance should be created
        assert a is not b


# ---------------------------------------------------------------------------
# TestStaging
# ---------------------------------------------------------------------------


class TestStaging:

    def test_staging_path_creates_temp_file(self) -> None:
        p = LocalPath.staging_path()
        try:
            assert isinstance(p, LocalPath)
            # The staging path should be under the system temp directory
            assert "yggdrasil-staging" in p.os_path
            assert p.name.startswith("part-")
            assert p.temporary is True
        finally:
            try:
                p.close()
            except Exception:
                pass

    def test_staging_path_is_writable(self) -> None:
        p = LocalPath.staging_path()
        try:
            with p:
                p.write_bytes(b"staging data")
                assert p.read_bytes() == b"staging data"
                assert p.size == len(b"staging data")
        finally:
            try:
                p.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# TestAsMedia
# ---------------------------------------------------------------------------


class TestAsMedia:

    def test_as_media_ipc(self, tmp_path: pathlib.Path) -> None:
        from yggdrasil.io.arrow_ipc_file import ArrowIPCFile

        p = LocalPath(str(tmp_path / "data.ipc"), singleton_ttl=False)
        leaf = p.as_media()
        assert isinstance(leaf, ArrowIPCFile)

    def test_as_media_parquet(self, tmp_path: pathlib.Path) -> None:
        from yggdrasil.io.parquet_file import ParquetFile

        p = LocalPath(str(tmp_path / "data.parquet"), singleton_ttl=False)
        leaf = p.as_media()
        assert isinstance(leaf, ParquetFile)


# ---------------------------------------------------------------------------
# TestStaticValues
# ---------------------------------------------------------------------------


class TestStaticValues:

    def test_url_static_values_has_filepath(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "values.bin"), singleton_ttl=False)
        sv = p.url.static_values
        assert "$filepath" in sv

    def test_hive_partition_path_has_partition_values(
        self, tmp_path: pathlib.Path,
    ) -> None:
        hive_dir = tmp_path / "year=2024" / "month=01"
        hive_dir.mkdir(parents=True)
        target = hive_dir / "data.bin"
        target.write_bytes(b"x")
        p = LocalPath(str(target), singleton_ttl=False)
        sv = p.url.static_values
        assert sv.get("year") == "2024"
        assert sv.get("month") == "01"
        assert "$filepath" in sv
