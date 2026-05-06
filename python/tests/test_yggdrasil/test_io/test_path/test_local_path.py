"""Unit tests for :class:`yggdrasil.io.fs.local_path.LocalPath`.

Mirrors the structural surface of :class:`Memory`'s test suite:
construction shapes → lifecycle → five primitives → bounds → stat
→ inherited convenience surface → ``temporary`` flag → edge cases.

Every test uses ``tmp_path`` so nothing leaks across runs.
"""

from __future__ import annotations

import os
import pathlib
import time

import pytest

from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.io_stats import IOKind
from yggdrasil.io.url import URL


# ===========================================================================
# Helpers
# ===========================================================================


def _new(path, **kwargs) -> LocalPath:
    """Construct a :class:`LocalPath` and open it.

    Most tests want an opened holder ready for I/O; this trims the
    boilerplate and makes the rare "construct-but-don't-open" case
    explicit by *not* using the helper.
    """
    h = LocalPath(os.fspath(path), **kwargs)
    h.open()
    return h


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_from_string_path(self, tmp_path):
        p = tmp_path / "a.bin"
        h = LocalPath(os.fspath(p))
        assert h.os_path == os.fspath(p).replace("\\", "/")
        assert h.scheme == "file"

    def test_from_pathlib(self, tmp_path):
        p = tmp_path / "a.bin"
        h = LocalPath(pathlib.Path(p))
        assert h.os_path == os.fspath(p).replace("\\", "/")

    def test_from_url(self, tmp_path):
        p = tmp_path / "a.bin"
        url = URL.from_(p)
        h = LocalPath(url=url)
        assert h.os_path == os.fspath(p).replace("\\", "/")

    def test_does_not_create_file_until_open(self, tmp_path):
        """Pure construction is a navigation gesture; until you
        :meth:`open` we don't splat a ghost file onto the FS."""
        p = tmp_path / "ghost.bin"
        LocalPath(os.fspath(p))  # noqa: discarded
        assert not p.exists()

    def test_open_creates_missing_file(self, tmp_path):
        p = tmp_path / "fresh.bin"
        assert not p.exists()
        h = LocalPath(os.fspath(p))
        h.open()
        try:
            assert p.exists()
            assert h.size == 0
        finally:
            h.close()

    def test_open_creates_parent_directory(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "leaf.bin"
        h = LocalPath(os.fspath(p))
        h.open()
        try:
            assert p.parent.is_dir()
            assert p.exists()
        finally:
            h.close()

    def test_open_does_not_truncate_existing(self, tmp_path):
        p = tmp_path / "existing.bin"
        p.write_bytes(b"keep me")
        h = LocalPath(os.fspath(p))
        h.open()
        try:
            assert h.size == 7
            assert h.read_bytes() == b"keep me"
        finally:
            h.close()


# ===========================================================================
# Backing-shape predicates
# ===========================================================================


class TestPredicates:
    def test_predicates_are_local_path(self, tmp_path):
        h = LocalPath(os.fspath(tmp_path / "a"))
        assert h.is_local_path is True
        assert h.is_memory is False
        assert h.is_remote_path is False
        assert h.is_local is True
        assert h.is_remote is False


# ===========================================================================
# Lifecycle — fd open / close
# ===========================================================================


class TestLifecycle:
    def test_fd_negative_before_open(self, tmp_path):
        h = LocalPath(os.fspath(tmp_path / "a.bin"))
        assert h.fd == -1

    def test_fd_positive_after_open(self, tmp_path):
        h = LocalPath(os.fspath(tmp_path / "a.bin"))
        h.open()
        try:
            assert h.fd >= 0
        finally:
            h.close()

    def test_fd_negative_after_close(self, tmp_path):
        h = LocalPath(os.fspath(tmp_path / "a.bin"))
        h.open()
        h.close()
        assert h.fd == -1

    def test_with_block(self, tmp_path):
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            assert h.fd >= 0
            h.write_bytes(b"x")
        # Out of with: closed.
        assert h.fd == -1
        assert p.read_bytes() == b"x"

    def test_open_is_idempotent(self, tmp_path):
        h = LocalPath(os.fspath(tmp_path / "a.bin"))
        h.open()
        try:
            fd1 = h.fd
            h.open()
            fd2 = h.fd
            assert fd1 == fd2  # same fd, no leak
        finally:
            h.close()

    def test_read_when_closed_raises(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"abc")
        h = LocalPath(os.fspath(p))
        with pytest.raises(OSError):
            h._read_mv(3, 0)

    def test_write_when_closed_raises(self, tmp_path):
        h = LocalPath(os.fspath(tmp_path / "a.bin"))
        with pytest.raises(OSError):
            h._write_mv(memoryview(b"x"), 0)

# ===========================================================================
# Holder primitive — _read_mv
# ===========================================================================


class TestReadMv:
    def test_read_full(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"abcdef")
        with LocalPath(os.fspath(p)) as h:
            mv = h._read_mv(6, 0)
            assert bytes(mv) == b"abcdef"

    def test_read_partial(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"abcdef")
        with LocalPath(os.fspath(p)) as h:
            assert bytes(h._read_mv(3, 0)) == b"abc"
            assert bytes(h._read_mv(3, 3)) == b"def"

    def test_read_zero_returns_empty(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            mv = h._read_mv(0, 0)
            assert bytes(mv) == b""

    def test_read_via_public_pread(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"hello world")
        with LocalPath(os.fspath(p)) as h:
            assert h.pread(5, 6) == b"world"
            assert h.pread(-1, 0) == b"hello world"

    def test_read_via_read_mv_normalizes_pos_minus_1(self, tmp_path):
        """pos=-1 is the append sentinel; reading from it yields
        zero bytes (resolves to size, n=0). This goes through the
        base ``Holder.read_mv`` normalization."""
        p = tmp_path / "a.bin"
        p.write_bytes(b"abc")
        with LocalPath(os.fspath(p)) as h:
            assert bytes(h.read_mv(0, -1)) == b""


# ===========================================================================
# Holder primitive — _write_mv
# ===========================================================================


class TestWriteMv:
    def test_write_at_zero(self, tmp_path):
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            n = h._write_mv(memoryview(b"hello"), 0)
            assert n == 5
        assert p.read_bytes() == b"hello"

    def test_write_at_offset_overwrites_in_place(self, tmp_path):
        """Write at a position inside an already-grown file. Note
        ``_write_mv`` doesn't grow — that's :meth:`write_mv`'s job —
        so we pre-grow via :meth:`truncate`."""
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            h.truncate(8)
            h._write_mv(memoryview(b"BBBB"), 2)
        assert p.read_bytes() == b"\x00\x00BBBB\x00\x00"

    def test_write_zero_bytes_is_noop(self, tmp_path):
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            assert h._write_mv(memoryview(b""), 0) == 0
        assert p.read_bytes() == b""

    def test_write_via_public_write_bytes_grows_file(self, tmp_path):
        """Routed through the base ``Holder.write_mv``, which
        pre-grows via :meth:`resize` → :meth:`truncate`."""
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            h.write_bytes(b"hello")
            assert h.size == 5
            assert h.read_bytes() == b"hello"

    def test_write_at_minus_1_appends(self, tmp_path):
        """pos=-1 routes through ``Holder.write_mv``'s ``_resolve_pos``
        and lands at end-of-file."""
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            h.write_bytes(b"abc")
            h.write_bytes(b"def", pos=-1)
            assert h.read_bytes() == b"abcdef"

    def test_write_marks_dirty(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            h.write_bytes(b"x")
            assert h.is_dirty() is True


# ===========================================================================
# Holder primitive — reserve
# ===========================================================================


class TestReserve:
    def test_reserve_is_noop(self, tmp_path):
        """Local files have no useful capacity-vs-size distinction;
        reserve is a no-op for the contract but must not raise on
        legitimate inputs."""
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            h.reserve(0)
            h.reserve(1024 * 1024)
            assert h.size == 0  # No size change

    def test_reserve_negative_raises(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            with pytest.raises(ValueError):
                h.reserve(-1)


# ===========================================================================
# Holder primitive — truncate
# ===========================================================================


class TestTruncate:
    def test_truncate_grow_zero_pads(self, tmp_path):
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            h.write_bytes(b"abc")
            h.truncate(8)
            assert h.size == 8
            assert h.read_bytes() == b"abc\x00\x00\x00\x00\x00"

    def test_truncate_shrink_drops_tail(self, tmp_path):
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            h.write_bytes(b"abcdef")
            h.truncate(3)
            assert h.size == 3
            assert h.read_bytes() == b"abc"

    def test_truncate_to_zero(self, tmp_path):
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            h.write_bytes(b"hello")
            h.truncate(0)
            assert h.size == 0
            assert h.read_bytes() == b""

    def test_truncate_negative_raises(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            with pytest.raises(ValueError):
                h.truncate(-1)

    def test_truncate_returns_n(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            assert h.truncate(42) == 42


# ===========================================================================
# Holder primitive — clear
# ===========================================================================


class TestClear:
    def test_clear_unlinks_file(self, tmp_path):
        p = tmp_path / "a.bin"
        h = LocalPath(os.fspath(p))
        h.open()
        try:
            h.write_bytes(b"abc")
            assert p.exists()
            h.clear()
            assert not p.exists()
            assert h.size == 0
            assert h.fd == -1
            assert h.stat().kind == IOKind.MISSING
        finally:
            # h.fd is already -1 from clear(); close() just runs the
            # rest of the Disposable release path.
            h.close()

    def test_clear_idempotent_on_missing(self, tmp_path):
        h = LocalPath(os.fspath(tmp_path / "ghost.bin"))
        # Never opened; file doesn't exist; clear() should not raise.
        h.clear()
        assert h.size == 0

    def test_reopen_after_clear(self, tmp_path):
        """After clear(), the holder is closed but the URL is intact;
        next open() recreates the file fresh."""
        p = tmp_path / "a.bin"
        h = LocalPath(os.fspath(p))
        h.open()
        h.write_bytes(b"first")
        h.clear()
        h.open()
        try:
            assert h.size == 0
            h.write_bytes(b"second")
            assert h.read_bytes() == b"second"
        finally:
            h.close()
        assert p.read_bytes() == b"second"


# ===========================================================================
# Bounds — these come from the base, but exercise via this subclass
# ===========================================================================


class TestBounds:
    def test_read_past_eof_raises(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"abc")
        with LocalPath(os.fspath(p)) as h:
            with pytest.raises(ValueError):
                h.read_mv(10, 0)

    def test_read_at_pos_past_size_raises(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"abc")
        with LocalPath(os.fspath(p)) as h:
            with pytest.raises(ValueError):
                h.read_mv(0, 99)


# ===========================================================================
# Stat
# ===========================================================================


class TestStat:
    def test_stat_missing_before_open(self, tmp_path):
        h = LocalPath(os.fspath(tmp_path / "ghost.bin"))
        s = h.stat()
        assert s.kind == IOKind.MISSING
        assert s.size == 0

    def test_stat_kind_file_after_open(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            assert h.stat().kind == IOKind.FILE

    def test_stat_kind_directory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        h = LocalPath(os.fspath(sub))
        # Don't open — opening a dir as O_RDWR fails on most systems.
        # stat() falls back to os.stat in that case.
        assert h.stat().kind == IOKind.DIRECTORY

    def test_stat_size_tracks_writes(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            assert h.size == 0
            h.write_bytes(b"abc")
            assert h.size == 3
            h.write_bytes(b"defg", pos=-1)
            assert h.size == 7

    def test_stat_observes_external_writes(self, tmp_path):
        """File holders DON'T cache: another writer can grow the file
        and the next ``stat()`` call must see it."""
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            h.write_bytes(b"abc")
            assert h.size == 3
            # Sneaky external mutation through a different fd.
            with open(p, "ab") as fh:
                fh.write(b"DEF")
            assert h.size == 6

    def test_stat_returns_same_instance(self, tmp_path):
        """Caller can pin :meth:`stat` and observe live updates."""
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            s1 = h.stat()
            h.write_bytes(b"abc")
            s2 = h.stat()
            assert s1 is s2
            assert s1.size == 3

    def test_stat_mtime_advances_on_write(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            h.write_bytes(b"x")
            t1 = h.stat().mtime
            time.sleep(0.05)
            h.write_bytes(b"y", pos=-1)
            t2 = h.stat().mtime
            assert t2 >= t1


# ===========================================================================
# Inherited convenience surface
# ===========================================================================


class TestInheritedSurface:
    def test_read_text_write_text(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.txt")) as h:
            h.write_text("héllo")
            assert h.read_text() == "héllo"

    def test_memoryview(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"abcdef")
        with LocalPath(os.fspath(p)) as h:
            mv = h.memoryview()
            assert bytes(mv) == b"abcdef"

    def test_len(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            h.write_bytes(b"abcd")
            assert len(h) == 4

    def test_bytes(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h:
            h.write_bytes(b"abc")
            assert bytes(h) == b"abc"

    def test_eq_against_other_holder(self, tmp_path):
        with LocalPath(os.fspath(tmp_path / "a.bin")) as h1:
            with LocalPath(os.fspath(tmp_path / "b.bin")) as h2:
                h1.write_bytes(b"abc")
                h2.write_bytes(b"abc")
                assert h1 != h2

    def test_write_local_path(self, tmp_path):
        """Inherited ``Holder.write_local_path`` should stream a
        source file's bytes into this holder."""
        src = tmp_path / "src.bin"
        src.write_bytes(b"payload from disk")
        with LocalPath(os.fspath(tmp_path / "dest.bin")) as h:
            n = h.write_local_path(src)
            assert n == len(b"payload from disk")
            assert h.read_bytes() == b"payload from disk"


# ===========================================================================
# temporary flag
# ===========================================================================


class TestTemporary:
    def test_temporary_unlinks_on_close(self, tmp_path):
        p = tmp_path / "tmp.bin"
        h = LocalPath(os.fspath(p), temporary=True)
        h.open()
        h.write_bytes(b"transient")
        assert p.exists()
        h.close()
        # _release → clear() → unlink
        assert not p.exists()

    def test_non_temporary_persists(self, tmp_path):
        p = tmp_path / "keep.bin"
        h = LocalPath(os.fspath(p))
        h.open()
        h.write_bytes(b"persistent")
        h.close()
        assert p.exists()
        assert p.read_bytes() == b"persistent"

    def test_temporary_default_false(self, tmp_path):
        h = LocalPath(os.fspath(tmp_path / "a.bin"))
        assert h.temporary is False


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        with LocalPath(os.fspath(p)) as h:
            assert h.size == 0
            assert h.read_bytes() == b""
            assert bytes(h.memoryview()) == b""

    def test_large_round_trip(self, tmp_path):
        """Stress the pread/pwrite loops over a non-trivial payload —
        catches off-by-one in the short-read/short-write fallback."""
        payload = bytes(range(256)) * 4096  # 1 MiB
        with LocalPath(os.fspath(tmp_path / "big.bin")) as h:
            h.write_bytes(payload)
            assert h.size == len(payload)
            assert h.read_bytes() == payload

    def test_overwrite_in_place(self, tmp_path):
        """Overwriting a slice doesn't grow the file."""
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            h.write_bytes(b"abcdefgh")
            assert h.size == 8
            h.write_bytes(b"XX", pos=3)
            assert h.size == 8
            assert h.read_bytes() == b"abcXXfgh"

    def test_write_extending_grows_file(self, tmp_path):
        """Writing past EOF grows the file via Holder.write_mv → resize
        → truncate. ftruncate zero-pads any gap."""
        p = tmp_path / "a.bin"
        with LocalPath(os.fspath(p)) as h:
            h.write_bytes(b"abc")
            h.write_bytes(b"XYZ", pos=10)
            assert h.size == 13
            assert h.read_bytes() == b"abc\x00\x00\x00\x00\x00\x00\x00XYZ"

    def test_close_release_close_release_idempotent(self, tmp_path):
        """Disposable contract — double close shouldn't raise."""
        h = LocalPath(os.fspath(tmp_path / "a.bin"))
        h.open()
        h.close()
        h.close()  # idempotent