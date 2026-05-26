"""Tests for :class:`yggdrasil.io.bytes_io.BytesIO`.

:class:`BytesIO` is the canonical cursor over a :class:`Holder`.
After the Holder ↔ IO merge it inherits :class:`Holder` directly,
so every BytesIO carries a :attr:`parent` (the byte substrate) and
exposes the stdlib :class:`typing.BinaryIO` surface (``read``,
``write``, ``seek``, ``tell``, ``truncate``, ``readline``, …).

These tests pin the cursor surface itself; format-specific behavior
(Parquet / CSV / JSON / …) lives under
``test_yggdrasil/test_io/test_primitive/``.
"""

from __future__ import annotations

import io
import struct

import pytest

from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.path.memory import Memory


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_empty(self) -> None:
        b = BytesIO()
        assert b.size == 0
        assert b.to_bytes() == b""

    def test_seed_bytes(self) -> None:
        b = BytesIO(b"hello")
        assert b.size == 5
        assert b.to_bytes() == b"hello"

    def test_seed_bytearray(self) -> None:
        b = BytesIO(bytearray(b"hello"))
        assert b.to_bytes() == b"hello"

    def test_seed_memoryview(self) -> None:
        b = BytesIO(memoryview(b"hello"))
        assert b.to_bytes() == b"hello"

    def test_parent_kwarg(self) -> None:
        mem = Memory(b"borrowed")
        b = BytesIO(parent=mem, mode="rb")
        assert b.parent is mem
        assert b.read() == b"borrowed"

    def test_holder_kwarg_legacy_alias(self) -> None:
        mem = Memory(b"legacy")
        b = BytesIO(holder=mem, mode="rb")
        assert b.parent is mem

    def test_path_kwarg_creates_local_path_parent(self, tmp_path) -> None:
        from yggdrasil.path.local_path import LocalPath

        b = BytesIO(path=str(tmp_path / "x.bin"))
        assert isinstance(b.parent, LocalPath)

    def test_from_passthrough(self) -> None:
        b = BytesIO(b"abc")
        assert BytesIO.from_(b) is b


# ---------------------------------------------------------------------------
# Sequential read / write — stdlib BinaryIO contract
# ---------------------------------------------------------------------------


class TestRead:

    def test_read_all(self) -> None:
        assert BytesIO(b"abc").read() == b"abc"

    def test_read_advances_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        assert b.read(3) == b"abc"
        assert b.tell() == 3
        assert b.read() == b"def"
        assert b.tell() == 6

    def test_read_zero_returns_empty(self) -> None:
        b = BytesIO(b"abc")
        assert b.read(0) == b""
        assert b.tell() == 0

    def test_read_past_end_returns_remaining(self) -> None:
        b = BytesIO(b"abc")
        assert b.read(100) == b"abc"

    def test_readinto_fills_buffer(self) -> None:
        b = BytesIO(b"abcdef")
        buf = bytearray(4)
        n = b.readinto(buf)
        assert n == 4
        assert buf == b"abcd"
        assert b.tell() == 4

    def test_readline(self) -> None:
        b = BytesIO(b"first\nsecond\nthird")
        assert b.readline() == b"first\n"
        assert b.readline() == b"second\n"
        assert b.readline() == b"third"
        assert b.readline() == b""

    def test_readlines(self) -> None:
        assert BytesIO(b"a\nb\nc\n").readlines() == [b"a\n", b"b\n", b"c\n"]

    def test_iter(self) -> None:
        assert list(BytesIO(b"a\nb\n")) == [b"a\n", b"b\n"]


class TestWrite:

    def test_write_advances_cursor(self) -> None:
        b = BytesIO()
        n = b.write(b"abc")
        assert n == 3
        assert b.tell() == 3

    def test_write_string_encodes_utf8(self) -> None:
        b = BytesIO()
        b.write("héllo")
        assert b.to_bytes() == "héllo".encode("utf-8")

    def test_writelines(self) -> None:
        b = BytesIO()
        b.writelines([b"one\n", b"two\n", b"three"])
        assert b.to_bytes() == b"one\ntwo\nthree"

    def test_truncate_at_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(3)
        b.truncate()
        assert b.size == 3
        assert b.to_bytes() == b"abc"

    def test_truncate_explicit(self) -> None:
        b = BytesIO(b"abcdef")
        b.truncate(2)
        assert b.size == 2


# ---------------------------------------------------------------------------
# Positional (cursor-less) primitives
# ---------------------------------------------------------------------------


class TestPositional:

    def test_pread_does_not_move_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(2)
        assert b.pread(2, 4) == b"ef"
        assert b.tell() == 2

    def test_pwrite_does_not_move_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(2)
        b.pwrite(b"ZZ", 0)
        assert b.to_bytes() == b"ZZcdef"
        assert b.tell() == 2

    def test_pwrite_in_place(self) -> None:
        b = BytesIO(b"abcdef")
        b.pwrite(b"ZZ", 2)
        assert b.to_bytes() == b"abZZef"


# ---------------------------------------------------------------------------
# Seek
# ---------------------------------------------------------------------------


class TestSeek:

    def test_seek_set(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(2)
        assert b.tell() == 2
        assert b.read(2) == b"cd"

    def test_seek_negative_raises(self) -> None:
        b = BytesIO(b"abc")
        with pytest.raises(ValueError, match="Negative SEEK_SET"):
            b.seek(-3)

    def test_seek_minus_one_is_eof_sentinel(self) -> None:
        b = BytesIO(b"abc")
        b.seek(-1)
        assert b.tell() == 3

    def test_seek_end_clamps(self) -> None:
        b = BytesIO(b"abc")
        b.seek(-100, io.SEEK_END)
        assert b.tell() == 0

    def test_seek_invalid_whence(self) -> None:
        b = BytesIO(b"abc")
        with pytest.raises(ValueError, match="Invalid whence"):
            b.seek(0, 99)

    def test_predicates(self) -> None:
        b = BytesIO(b"abc")
        assert b.readable()
        assert b.writable()
        assert b.seekable()
        assert not b.isatty()


# ---------------------------------------------------------------------------
# Mode-driven open semantics
# ---------------------------------------------------------------------------


class TestModes:

    def test_rb_does_not_truncate(self) -> None:
        mem = Memory(b"seed")
        with BytesIO(holder=mem, owns_holder=False, mode="rb") as b:
            assert b.read() == b"seed"
        assert mem.read_bytes() == b"seed"

    def test_wb_truncates_on_open(self) -> None:
        mem = Memory(b"seed")
        with BytesIO(holder=mem, owns_holder=False, mode="wb") as b:
            assert b.size == 0
            b.write(b"new")
        assert mem.read_bytes() == b"new"

    def test_ab_positions_at_end(self) -> None:
        mem = Memory(b"abc")
        with BytesIO(holder=mem, owns_holder=False, mode="ab") as b:
            assert b.tell() == 3
            b.write(b"DEF")
        assert mem.read_bytes() == b"abcDEF"

    def test_xb_raises_on_non_empty(self) -> None:
        mem = Memory(b"existing")
        with pytest.raises(FileExistsError):
            with BytesIO(holder=mem, owns_holder=False, mode="xb"):
                pass

    def test_xb_succeeds_on_empty(self) -> None:
        mem = Memory()
        with BytesIO(holder=mem, owns_holder=False, mode="xb") as b:
            b.write(b"fresh")
        assert mem.read_bytes() == b"fresh"


# ---------------------------------------------------------------------------
# Cursor / parent
# ---------------------------------------------------------------------------


class TestCursorParent:

    def test_writes_land_on_parent_directly(self) -> None:
        mem = Memory(b"original")
        with BytesIO(holder=mem, owns_holder=False, mode="rb+") as b:
            b.seek(0)
            b.write(b"REPLACED")
            assert mem.read_bytes() == b"REPLACED"
        assert mem.read_bytes() == b"REPLACED"

    def test_multiple_cursors_over_one_parent(self) -> None:
        mem = Memory(b"shared")
        a = BytesIO(holder=mem, owns_holder=False, mode="rb")
        b = BytesIO(holder=mem, owns_holder=False, mode="rb")
        a.seek(2)
        assert a.read(2) == b"ar"
        assert b.tell() == 0
        assert b.read(3) == b"sha"


# ---------------------------------------------------------------------------
# Structured primitives
# ---------------------------------------------------------------------------


class TestStructuredPrimitives:

    def test_int32_round_trip(self) -> None:
        b = BytesIO()
        b.write_int32(-42)
        b.seek(0)
        assert b.read_int32() == -42

    def test_uint64_round_trip(self) -> None:
        b = BytesIO()
        b.write_uint64(2**40 + 7)
        b.seek(0)
        assert b.read_uint64() == 2**40 + 7

    def test_f64_round_trip(self) -> None:
        b = BytesIO()
        b.write_f64(3.14159265358979)
        b.seek(0)
        assert b.read_f64() == pytest.approx(3.14159265358979)

    def test_bool_round_trip(self) -> None:
        b = BytesIO()
        b.write_bool(True)
        b.write_bool(False)
        b.seek(0)
        assert b.read_bool() is True
        assert b.read_bool() is False

    def test_length_prefixed_string(self) -> None:
        b = BytesIO()
        b.write_str_u32("hello")
        b.write_str_u32("héllo")
        b.seek(0)
        assert b.read_str_u32() == "hello"
        assert b.read_str_u32() == "héllo"

    def test_short_read_raises_eof(self) -> None:
        b = BytesIO(struct.pack("<I", 8))
        b.seek(0)
        with pytest.raises(EOFError):
            b.read_bytes_u32()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestToBytes:

    def test_to_bytes_returns_copy(self) -> None:
        b = BytesIO(b"abc")
        out = bytearray(b.to_bytes())
        out[0] = ord("Z")
        assert b.to_bytes() == b"abc"

    def test_to_bytes_independent_of_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(3)
        assert b.to_bytes() == b"abcdef"
        assert b.tell() == 3
