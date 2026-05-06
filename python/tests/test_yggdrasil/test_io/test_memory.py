"""Tests for :class:`yggdrasil.io.memory.Memory` and the :class:`Holder`
convenience surface."""

from __future__ import annotations

import pathlib

import pytest

from yggdrasil.io import Holder, IOStats, Memory


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMemoryConstruction:
    def test_default_is_empty(self):
        m = Memory()
        assert m.size == 0
        assert m.capacity == 0
        assert bytes(m) == b""

    def test_int_reserves_capacity_without_size(self):
        m = Memory(64)
        assert m.size == 0
        assert m.capacity == 64

    def test_bytes_seeds_payload(self):
        m = Memory(b"hello")
        assert m.size == 5
        assert bytes(m) == b"hello"

    def test_memoryview_seeds_payload(self):
        src = memoryview(b"world!")
        m = Memory(src)
        assert m.size == 6
        assert bytes(m) == b"world!"

    def test_copy_from_other_memory(self):
        a = Memory(b"abc")
        b = Memory(a)
        # Distinct buffers — mutation of one doesn't ripple.
        b.write_mv(memoryview(b"X"), 0)
        assert bytes(a) == b"abc"
        assert bytes(b) == b"Xbc"

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError):
            Memory(-1)

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            Memory({"not": "supported"})


# ---------------------------------------------------------------------------
# Holder primitives — read_mv / write_mv / reserve / truncate
# ---------------------------------------------------------------------------


class TestMemoryHolderPrimitives:
    def test_is_a_holder(self):
        assert isinstance(Memory(), Holder)

    def test_read_mv_default_returns_view_to_size(self):
        m = Memory(b"hello")
        view = m.read_mv(-1, 0)
        assert bytes(view) == b"hello"

    def test_read_mv_at_offset(self):
        m = Memory(b"hello")
        view = m.read_mv(3, 1)
        assert bytes(view) == b"ell"

    def test_read_mv_past_end_returns_empty(self):
        m = Memory(b"abc")
        assert bytes(m.read_mv(10, 5)) == b""

    def test_read_mv_negative_pos_raises(self):
        with pytest.raises(ValueError):
            Memory(b"abc").read_mv(1, -1)

    def test_write_mv_appends_grows_size(self):
        m = Memory()
        n = m.write_mv(memoryview(b"hello"), 0)
        assert n == 5
        assert m.size == 5
        assert bytes(m) == b"hello"

    def test_write_mv_at_offset_zero_pads(self):
        m = Memory()
        m.write_mv(memoryview(b"X"), 4)
        assert m.size == 5
        assert bytes(m) == b"\x00\x00\x00\x00X"

    def test_write_mv_overwrites_in_place(self):
        m = Memory(b"hello")
        m.write_mv(memoryview(b"YY"), 1)
        assert bytes(m) == b"hYYlo"
        assert m.size == 5

    def test_write_mv_negative_pos_raises(self):
        with pytest.raises(ValueError):
            Memory().write_mv(memoryview(b"x"), -1)

    def test_reserve_grows_capacity_only(self):
        m = Memory(b"abc")
        m.reserve(100)
        assert m.size == 3
        assert m.capacity >= 100

    def test_reserve_idempotent_when_smaller(self):
        m = Memory(64)
        cap = m.capacity
        m.reserve(10)
        assert m.capacity == cap

    def test_truncate_shrinks(self):
        m = Memory(b"hello")
        m.truncate(2)
        assert m.size == 2
        assert bytes(m) == b"he"

    def test_truncate_extends_zero_pads(self):
        m = Memory(b"hi")
        m.truncate(5)
        assert m.size == 5
        assert bytes(m) == b"hi\x00\x00\x00"

    def test_truncate_zero(self):
        m = Memory(b"abc")
        m.truncate(0)
        assert m.size == 0
        assert bytes(m) == b""

    def test_clear_drops_capacity(self):
        m = Memory(b"abc")
        m.clear()
        assert m.size == 0
        assert m.capacity == 0


# ---------------------------------------------------------------------------
# Holder helpers — bytes / text
# ---------------------------------------------------------------------------


class TestMemoryConvenienceSurface:
    def test_read_bytes_default_is_full(self):
        m = Memory(b"hello world")
        assert m.read_bytes() == b"hello world"

    def test_read_bytes_n_pos(self):
        m = Memory(b"hello world")
        assert m.read_bytes(5, 6) == b"world"

    def test_write_bytes_appends(self):
        m = Memory()
        n = m.write_bytes(b"hi")
        assert n == 2
        assert bytes(m) == b"hi"

    def test_write_bytes_at_pos(self):
        m = Memory(b"hello")
        m.write_bytes(b"YY", 2)
        assert bytes(m) == b"heYYo"

    def test_read_text_default(self):
        m = Memory("héllo".encode("utf-8"))
        assert m.read_text() == "héllo"

    def test_write_text_encodes(self):
        m = Memory()
        m.write_text("héllo")
        assert bytes(m) == "héllo".encode("utf-8")

    def test_write_text_at_pos(self):
        m = Memory(b"prefix:")
        m.write_text("hi", pos=7)
        assert bytes(m) == b"prefix:hi"


# ---------------------------------------------------------------------------
# Local-path bridge — read_local_path / write_local_path
# ---------------------------------------------------------------------------


class TestMemoryLocalPathBridge:
    def test_read_local_path_loads_bytes(self, tmp_path):
        src = tmp_path / "in.bin"
        src.write_bytes(b"file payload")
        m = Memory()
        n = m.read_local_path(src)
        assert n == 12
        assert bytes(m) == b"file payload"

    def test_read_local_path_at_pos(self, tmp_path):
        src = tmp_path / "in.bin"
        src.write_bytes(b"world")
        m = Memory(b"hello ")  # 6 bytes
        n = m.read_local_path(src, pos=6)
        assert n == 5
        assert bytes(m) == b"hello world"

    def test_read_local_path_n_caps_length(self, tmp_path):
        src = tmp_path / "in.bin"
        src.write_bytes(b"abcdefghij")
        m = Memory()
        n = m.read_local_path(src, n=4)
        assert n == 4
        assert bytes(m) == b"abcd"

    def test_read_local_path_n_zero_reads_nothing(self, tmp_path):
        src = tmp_path / "in.bin"
        src.write_bytes(b"abcd")
        m = Memory()
        n = m.read_local_path(src, n=0)
        assert n == 0
        assert bytes(m) == b""

    def test_read_local_path_n_at_pos(self, tmp_path):
        src = tmp_path / "in.bin"
        src.write_bytes(b"abcdef")
        m = Memory(b"_")  # 1 byte
        n = m.read_local_path(src, pos=1, n=3)
        assert n == 3
        assert bytes(m) == b"_abc"

    def test_read_local_path_accepts_str(self, tmp_path):
        src = tmp_path / "in.bin"
        src.write_bytes(b"abc")
        m = Memory()
        n = m.read_local_path(str(src))
        assert n == 3
        assert bytes(m) == b"abc"

    def test_write_local_path_drains_full(self, tmp_path):
        m = Memory(b"persist me")
        dst = tmp_path / "out.bin"
        n = m.write_local_path(dst)
        assert n == 10
        assert dst.read_bytes() == b"persist me"

    def test_write_local_path_creates_parent(self, tmp_path):
        m = Memory(b"x")
        dst = tmp_path / "deep" / "nested" / "out.bin"
        m.write_local_path(dst)
        assert dst.read_bytes() == b"x"

    def test_write_local_path_pos_skips_prefix(self, tmp_path):
        m = Memory(b"PREFIX:payload")
        dst = tmp_path / "out.bin"
        n = m.write_local_path(dst, pos=7)
        assert n == 7
        assert dst.read_bytes() == b"payload"

    def test_write_local_path_n_caps_length(self, tmp_path):
        m = Memory(b"abcdefghij")
        dst = tmp_path / "out.bin"
        n = m.write_local_path(dst, pos=2, n=3)
        assert n == 3
        assert dst.read_bytes() == b"cde"

    def test_round_trip_via_pathlib(self, tmp_path):
        m = Memory(b"round-trip")
        f = tmp_path / "x.bin"
        m.write_local_path(f)

        m2 = Memory()
        m2.read_local_path(pathlib.Path(f))
        assert bytes(m) == bytes(m2)


# ---------------------------------------------------------------------------
# Equality / hashing / dunder
# ---------------------------------------------------------------------------


class TestMemoryStats:
    def test_stats_returns_iostats(self):
        m = Memory(b"hello")
        s = m.stat()
        assert isinstance(s, IOStats)
        assert s.size == 5
        assert s.mtime > 0
        assert s.media_type is None

    def test_stats_after_write_bumps_mtime(self):
        import time as _time
        m = Memory()
        before = m.stat().mtime
        _time.sleep(0.01)
        m.write_bytes(b"x")
        after = m.stat().mtime
        assert after > before

    def test_stats_carries_media_type(self):
        from yggdrasil.data.enums import MediaTypes
        m = Memory(b"abc", media_type=MediaTypes.JSON)
        s = m.stat()
        assert s.media_type is MediaTypes.JSON
        assert s.has_media_type

    def test_iostats_with_default_copies(self):
        s = IOStats(size=10, mtime=1.0, media_type=None)
        s2 = s.with_(size=20)
        assert s.size == 10
        assert s2.size == 20

    def test_iostats_with_inplace(self):
        s = IOStats(size=10, mtime=1.0, media_type=None)
        s.with_(size=20, inplace=True)
        assert s.size == 20

    def test_iostats_clear_media_type(self):
        from yggdrasil.data.enums import MediaTypes
        s = IOStats(size=1, mtime=0.0, media_type=MediaTypes.JSON)
        s.with_(media_type=None, inplace=True)
        assert s.media_type is None
        assert not s.has_media_type

    def test_iostats_iterable(self):
        s = IOStats(size=10, mtime=2.5, media_type=None)
        size, mtime, kind, mode, mt = s
        assert (size, mtime, mode, mt) == (10, 2.5, 0, None)


class TestMemoryDunder:
    def test_len_is_size(self):
        m = Memory(b"abcd")
        assert len(m) == 4

    def test_bytes_protocol(self):
        m = Memory(b"x")
        assert bytes(m) == b"x"

    def test_eq_with_memory(self):
        assert Memory(b"abc") == Memory(b"abc")
        assert Memory(b"abc") != Memory(b"abd")

    def test_eq_with_bytes(self):
        assert Memory(b"abc") == b"abc"

    def test_repr_shows_size_and_capacity(self):
        m = Memory(b"abc")
        m.reserve(64)
        r = repr(m)
        assert "size=3" in r
        assert "capacity=" in r
