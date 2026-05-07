"""Behavior tests for :class:`yggdrasil.io.holder.Holder` + :class:`Memory`.

`Holder` is the position-addressable byte substrate that BytesIO and
Path stack on top of. Tests pin:

* the abstract primitives (`read_mv` / `write_mv` / `reserve` /
  `truncate` / `clear` / `size`) round-trip cleanly on `Memory`;
* dispatch via `Holder(...)` picks the right subclass for url /
  binary / path / data inputs;
* append-at-end (`pos = -1`) and from-end (`pos = -N`) sentinels
  resolve consistently;
* `temporary=True` honors clear-on-close;
* ``stat`` / ``mtime`` / ``media_type`` accessors stay consistent
  with the underlying mutable :class:`IOStats`.
"""
from __future__ import annotations

import os
import time

import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath


class TestSubclassDispatch:

    def test_no_args_picks_memory(self) -> None:
        assert isinstance(Holder(), Memory)

    def test_binary_picks_memory(self) -> None:
        h = Holder(binary=b"hello")
        assert isinstance(h, Memory)
        assert h.read_bytes() == b"hello"

    def test_path_picks_path_subclass(self, tmp_path) -> None:
        target = str(tmp_path / "out.bin")
        h = Holder(path=target)
        assert isinstance(h, LocalPath)

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scheme"):
            Holder(scheme="not-a-scheme")


class TestMemoryPrimitives:
    """The five primitives + size."""

    def test_empty_construction(self) -> None:
        m = Memory()
        assert m.size == 0
        assert m.capacity == 0
        assert bytes(m) == b""

    def test_capacity_seed(self) -> None:
        m = Memory(8)  # int → reserve, no payload
        assert m.size == 0
        assert m.capacity == 8

    def test_capacity_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 0"):
            Memory(-1)

    def test_write_at_pos_extends(self) -> None:
        m = Memory()
        n = m.write_bytes(b"abc", 0)
        assert n == 3
        assert m.size == 3
        assert m.read_bytes() == b"abc"

    def test_pwrite_in_place_overwrite(self) -> None:
        m = Memory(b"abcdef")
        m.pwrite(b"ZZ", 2)
        assert m.read_bytes() == b"abZZef"

    def test_pread_subrange(self) -> None:
        m = Memory(b"abcdef")
        assert m.pread(3, 1) == b"bcd"
        assert m.pread(2, 4) == b"ef"

    def test_pwrite_at_minus_one_appends(self) -> None:
        m = Memory(b"hi")
        m.pwrite(b"!", -1)
        assert m.read_bytes() == b"hi!"

    def test_pread_at_minus_one_returns_empty(self) -> None:
        m = Memory(b"hi")
        # pos=-1 → end; reads 0 bytes since nothing follows.
        assert m.pread(0, -1) == b""

    def test_pwrite_at_minus_two_indexes_from_end(self) -> None:
        m = Memory(b"abcd")
        m.pwrite(b"X", -2)
        # -2 → size + (-2) = 2 → write 'X' at index 2
        assert m.read_bytes() == b"abXd"

    def test_truncate_shrinks(self) -> None:
        m = Memory(b"abcdef")
        m.truncate(3)
        assert m.size == 3
        assert m.read_bytes() == b"abc"

    def test_truncate_extends_zero_pads(self) -> None:
        m = Memory(b"ab")
        m.truncate(5)
        assert m.size == 5
        assert m.read_bytes() == b"ab\x00\x00\x00"

    def test_truncate_negative_raises(self) -> None:
        m = Memory(b"ab")
        with pytest.raises(ValueError, match="truncate size must be >= 0"):
            m.truncate(-1)

    def test_reserve_grows_capacity_only(self) -> None:
        m = Memory()
        m.write_bytes(b"abc", 0)
        m.reserve(64)
        assert m.capacity >= 64
        assert m.size == 3  # visible size unchanged

    def test_reserve_smaller_is_noop(self) -> None:
        m = Memory(b"abc")
        m.reserve(0)
        assert m.size == 3
        assert m.capacity >= 3

    def test_clear_resets(self) -> None:
        m = Memory(b"abc")
        m.clear()
        assert m.size == 0
        assert m.capacity == 0
        m.write_bytes(b"new", 0)
        assert m.read_bytes() == b"new"


class TestRangeChecks:
    """`read_mv` and `write_mv` enforce bounds with helpful messages."""

    def test_read_past_end_raises(self) -> None:
        m = Memory(b"abc")
        with pytest.raises(ValueError, match="out of bounds"):
            m.read_mv(10, 0)

    def test_read_pos_past_end_raises(self) -> None:
        m = Memory(b"abc")
        with pytest.raises(ValueError, match="out of bounds"):
            m.read_mv(1, 10)

    def test_negative_n_resolves_to_remaining(self) -> None:
        m = Memory(b"abcdef")
        assert bytes(m.read_mv(-1, 2)) == b"cdef"

    def test_negative_pos_clamped_at_negative_one(self) -> None:
        # pos=-1 is the explicit "at end" sentinel.
        m = Memory(b"abc")
        assert bytes(m.read_mv(0, -1)) == b""


class TestStatAndMtime:

    def test_lazy_stat_present_for_memory(self) -> None:
        m = Memory()
        s = m.stat()
        assert s is not None
        assert s.size == 0

    def test_write_bumps_mtime(self) -> None:
        m = Memory()
        before = m.mtime
        time.sleep(0.005)
        m.write_bytes(b"x", 0)
        assert m.mtime >= before

    def test_stat_reflects_live_size(self) -> None:
        m = Memory()
        s = m.stat()
        m.write_bytes(b"abcd", 0)
        # Same instance — caller pinned it.
        assert s.size == 4
        assert m.stat() is s


class TestPredicates:

    def test_memory_predicates(self) -> None:
        m = Memory()
        assert m.is_memory
        assert not m.is_local_path
        assert not m.is_remote_path
        assert m.is_local
        assert not m.is_remote

    def test_local_path_predicates(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "a"))
        assert not lp.is_memory
        assert lp.is_local_path
        assert lp.is_local


class TestTemporaryFlag:

    def test_temporary_clears_on_close(self) -> None:
        m = Memory(b"data", temporary=True)
        assert m.temporary
        m.acquire()
        assert m.size == 4
        m.close()
        # Temporary memory keeps the bytearray reference but clears
        # the visible payload on release.
        assert m.size == 0


class TestEquality:

    def test_equal_to_bytes(self) -> None:
        m = Memory(b"abc")
        assert m == b"abc"
        assert m != b"abz"

    def test_equal_to_other_holder(self) -> None:
        a = Memory(b"abc")
        b = Memory(b"abc")
        assert a == b


class TestWriteLocalPath:

    def test_round_trip_into_memory(self, tmp_path) -> None:
        p = tmp_path / "src.bin"
        p.write_bytes(b"local-bytes")
        m = Memory()
        n = m.write_local_path(os.fspath(p))
        assert n == 11
        assert m.read_bytes() == b"local-bytes"

    def test_partial_n(self, tmp_path) -> None:
        p = tmp_path / "src.bin"
        p.write_bytes(b"123456789")
        m = Memory()
        n = m.write_local_path(os.fspath(p), n=4)
        assert n == 4
        assert m.read_bytes() == b"1234"

    def test_negative_pos_raises(self, tmp_path) -> None:
        p = tmp_path / "src.bin"
        p.write_bytes(b"x")
        m = Memory()
        with pytest.raises(ValueError, match="pos must be >= 0"):
            m.write_local_path(os.fspath(p), pos=-1)


class TestWriteStream:

    def test_drains_bytesio_into_memory(self) -> None:
        import io as _stdio
        m = Memory()
        n = m.write_stream(_stdio.BytesIO(b"stream-bytes"))
        assert n == 12
        assert m.read_bytes() == b"stream-bytes"

    def test_drains_into_local_path(self, tmp_path) -> None:
        import io as _stdio
        target = LocalPath(tmp_path / "out.bin")
        n = target.write_stream(_stdio.BytesIO(b"hello"))
        assert n == 5
        assert (tmp_path / "out.bin").read_bytes() == b"hello"

    def test_atomic_single_write_for_remote_like_backend(self) -> None:
        """Stream-write must hit ``_write_mv`` exactly once.

        Remote backends (VolumePath / S3Path) implement ``_write_mv``
        as an atomic upload + read-modify-rewrite for non-zero pos —
        anything that chunks a stream into per-chunk
        ``write_mv(...,pos=cursor)`` calls would devolve into N
        downloads + N uploads. This test pins the contract.
        """
        import io as _stdio
        m = Memory()
        calls: list[int] = []
        original = type(m)._write_mv

        def _spy(self, data, pos):
            calls.append(len(data))
            return original(self, data, pos)

        type(m)._write_mv = _spy
        try:
            m.write_stream(_stdio.BytesIO(b"x" * (4 * 1024 * 1024)))
        finally:
            type(m)._write_mv = original

        assert calls == [4 * 1024 * 1024]

    def test_empty_stream_is_noop(self) -> None:
        import io as _stdio
        m = Memory(b"keep")
        n = m.write_stream(_stdio.BytesIO(b""))
        assert n == 0
        assert m.read_bytes() == b"keep"

    def test_negative_pos_raises(self) -> None:
        import io as _stdio
        m = Memory()
        with pytest.raises(ValueError, match="pos must be >= 0"):
            m.write_stream(_stdio.BytesIO(b"x"), pos=-1)
