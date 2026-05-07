"""Behavior tests for :class:`yggdrasil.io.bytes_io.BytesIO`.

`BytesIO` is the cursor + ``IO[bytes]`` + tabular view that sits on
top of any :class:`Holder`. Tests pin:

* the two operating modes (closed / open with scratch transaction);
* the ``IO[bytes]`` protocol — read / write / readline / readlines /
  writelines / readinto / iter — agrees with stdlib ``io.BytesIO``
  behavior the way external libraries (pandas, polars, json) expect;
* construction routing — bytes / str / file-like / Holder / path /
  another BytesIO all flow through :meth:`from_` correctly;
* the ``with holder.open() as bio: …`` pattern commits exactly the
  scratch buffer onto the durable holder, with both append and
  overwrite modes;
* structured binary primitives (``write_int32`` / ``read_str_u32`` /
  …) round-trip cleanly.
"""
from __future__ import annotations

import io
import json
import struct

import pytest

from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath


class TestConstruction:

    def test_no_args_yields_empty_memory_bio(self) -> None:
        b = BytesIO()
        assert b.size == 0
        assert b.tell() == 0
        assert b.holder.is_memory
        assert b.owns_holder

    def test_bytes_input(self) -> None:
        b = BytesIO(b"hello")
        assert b.size == 5
        assert b.to_bytes() == b"hello"

    def test_borrows_holder_keeps_it_alive_after_close(self) -> None:
        mem = Memory(b"shared")
        c1 = BytesIO(holder=mem)
        c2 = BytesIO(holder=mem)
        assert not c1.owns_holder
        assert not c2.owns_holder
        c1.close()
        # Closing one cursor must not affect the other or the holder.
        assert c2.size == 6
        assert mem.size == 6

    def test_owns_holder_closes_it(self) -> None:
        mem = Memory(b"owned")
        b = BytesIO(holder=mem, owns_holder=True)
        b.close()
        # Memory survives close (its bytearray lives on); ownership
        # just means BytesIO told the holder to release.
        assert mem.size == 5

    def test_data_and_holder_both_raise(self) -> None:
        with pytest.raises(TypeError, match="OR data, not both"):
            BytesIO(b"x", holder=Memory())

    def test_idempotent_from_(self) -> None:
        a = BytesIO(b"abc")
        b = BytesIO.from_(a)
        assert b is a

    def test_from_file_like_drains(self) -> None:
        src = io.BytesIO(b"piped-bytes")
        b = BytesIO.from_(src)
        assert b.to_bytes() == b"piped-bytes"


class TestIOProtocolBasics:

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

    def test_read_more_than_remaining_returns_remaining(self) -> None:
        # Stdlib parity: read(N) where N > remaining returns whatever
        # remains, no exception.
        b = BytesIO(b"abc")
        assert b.read(100) == b"abc"

    def test_readall_alias(self) -> None:
        b = BytesIO(b"abcdef")
        b.read(2)
        assert b.readall() == b"cdef"

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
        b = BytesIO(b"a\nb\nc\n")
        assert b.readlines() == [b"a\n", b"b\n", b"c\n"]

    def test_iter_lines(self) -> None:
        b = BytesIO(b"a\nb\n")
        assert list(b) == [b"a\n", b"b\n"]

    def test_seek_set_clamped(self) -> None:
        b = BytesIO(b"abc")
        with pytest.raises(ValueError, match="Negative SEEK_SET"):
            b.seek(-3)

    def test_seek_set_minus_one_is_eof_sentinel(self) -> None:
        b = BytesIO(b"abc")
        b.seek(-1)
        assert b.tell() == 3

    def test_seek_end_clamps_at_zero(self) -> None:
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


class TestWritePaths:

    def test_write_bytes_advances_cursor(self) -> None:
        b = BytesIO()
        n = b.write(b"abc")
        assert n == 3
        assert b.tell() == 3

    def test_write_string_encodes_utf8(self) -> None:
        b = BytesIO()
        n = b.write("héllo")
        assert n == len("héllo".encode("utf-8"))
        assert b.to_bytes() == "héllo".encode("utf-8")

    def test_write_file_like(self) -> None:
        src = io.BytesIO(b"piped")
        b = BytesIO()
        n = b.write(src)
        assert n == 5
        assert b.to_bytes() == b"piped"

    def test_writelines_concatenates(self) -> None:
        b = BytesIO()
        b.writelines([b"one\n", b"two\n", b"three"])
        assert b.to_bytes() == b"one\ntwo\nthree"

    def test_truncate_at_pos(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(3)
        b.truncate()
        assert b.size == 3
        assert b.to_bytes() == b"abc"

    def test_truncate_explicit_size(self) -> None:
        b = BytesIO(b"abcdef")
        b.truncate(2)
        assert b.size == 2

    def test_pwrite_does_not_touch_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(2)
        b.pwrite(b"ZZ", 0)
        assert b.to_bytes() == b"ZZcdef"
        assert b.tell() == 2

    def test_pread_does_not_touch_cursor(self) -> None:
        b = BytesIO(b"abcdef")
        b.seek(2)
        assert b.pread(2, 4) == b"ef"
        assert b.tell() == 2

    def test_write_update_stat_false_skips_per_write_dirty(self) -> None:
        b = BytesIO(b"seed")
        holder = b._holder
        # Pre-acquired memory holder seeded above is clean to start.
        holder.clear_dirty()
        # In-place overwrites (no resize) with update_stat=False
        # should not flip the holder's dirty bit on each call.
        for offset in range(4):
            b.pwrite(b"X", offset, update_stat=False)
        assert b.to_bytes() == b"XXXX"
        assert holder.is_dirty() is False
        # Default kwarg restores the per-write dirty mark.
        b.pwrite(b"Y", 0)
        assert holder.is_dirty() is True


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
        b = BytesIO(struct.pack("<I", 8))  # claims 8 bytes follow
        b.seek(0)
        with pytest.raises(EOFError):
            b.read_bytes_u32()


class TestTransactionalScratch:
    """`with bio: ...` opens a scratch buffer that commits on close."""

    def test_open_seeds_scratch_from_durable(self) -> None:
        mem = Memory(b"durable-bytes")
        b = BytesIO(holder=mem, owns_holder=False, mode="rb+")
        with b:
            assert b.size == len(b"durable-bytes")
            assert b.read() == b"durable-bytes"

    def test_writes_route_through_scratch_and_commit(self) -> None:
        mem = Memory(b"original")
        b = BytesIO(holder=mem, owns_holder=False, mode="rb+")
        with b:
            b.seek(0)
            b.write(b"REPLACED")
        assert mem.read_bytes() == b"REPLACED"

    def test_wb_mode_truncates_at_open(self) -> None:
        mem = Memory(b"old-bytes")
        b = BytesIO(holder=mem, owns_holder=False, mode="wb")
        with b:
            assert b.size == 0  # scratch starts empty
            b.write(b"new")
        assert mem.read_bytes() == b"new"

    def test_xb_mode_raises_on_non_empty(self) -> None:
        mem = Memory(b"existing")
        b = BytesIO(holder=mem, owns_holder=False, mode="xb")
        with pytest.raises(FileExistsError):
            b.__enter__()

    def test_ab_mode_lands_at_eof(self) -> None:
        mem = Memory(b"head-")
        b = BytesIO(holder=mem, owns_holder=False, mode="ab")
        with b:
            assert b.tell() == 5
            b.write(b"tail")
        assert mem.read_bytes() == b"head-tail"

    def test_flush_commits_without_closing(self) -> None:
        mem = Memory()
        b = BytesIO(holder=mem, owns_holder=False, mode="wb+")
        with b:
            b.write(b"chunk1")
            b.flush()
            assert mem.read_bytes() == b"chunk1"
            b.write(b"chunk2")
        assert mem.read_bytes() == b"chunk1chunk2"

    def test_temporary_holder_discards_scratch(self) -> None:
        mem = Memory(temporary=True)
        b = BytesIO(holder=mem, owns_holder=True, mode="wb+")
        with b:
            b.write(b"throwaway")
        # The temporary clear runs in Holder._release, dropping the
        # bytes the BytesIO scratch had committed.
        assert mem.size == 0


class TestExternalWriters:
    """Standard library writers operate on a raw BytesIO seamlessly.

    These match the integration patterns the user actually hits:

        with holder.open() as b:
            external_lib.dump(b, ...)

    The library expects an object that quacks like ``IO[bytes]``;
    the test asserts the bytes commit to the durable holder once
    the ``with`` block ends.
    """

    def test_json_dump_into_holder_open(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "out.json"))
        with target.open("wb") as bio:
            bio.write(json.dumps({"id": 1, "ok": True}).encode("utf-8"))
        assert json.loads(target.read_text()) == {"id": 1, "ok": True}

    def test_pickle_dump_round_trip(self, tmp_path) -> None:
        import pickle

        target = LocalPath(str(tmp_path / "out.pkl"))
        with target.open("wb") as bio:
            pickle.dump({"x": [1, 2, 3]}, bio)
        with target.open("rb") as bio:
            assert pickle.load(bio) == {"x": [1, 2, 3]}

    def test_pandas_to_csv_into_holder_open(self, tmp_path) -> None:
        pd = pytest.importorskip("pandas")
        target = LocalPath(str(tmp_path / "data.csv"))
        with target.open("wb") as bio:
            pd.DataFrame({"id": [1, 2, 3]}).to_csv(bio, index=False)
        text = target.read_text()
        assert text.splitlines() == ["id", "1", "2", "3"]

    def test_polars_write_csv_into_holder_open(self, tmp_path) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.csv"))
        with target.open("wb") as bio:
            pl.DataFrame({"id": [1, 2, 3]}).write_csv(bio)
        text = target.read_text()
        assert "id" in text and "1" in text and "3" in text

    def test_zip_writer_into_holder_open(self, tmp_path) -> None:
        import zipfile

        target = LocalPath(str(tmp_path / "bundle.zip"))
        with target.open("wb") as bio:
            with zipfile.ZipFile(bio, "w") as zf:
                zf.writestr("hello.txt", "hi")
                zf.writestr("payload.json", '{"ok": true}')
        with zipfile.ZipFile(target.os_path) as zf:
            assert sorted(zf.namelist()) == ["hello.txt", "payload.json"]
            assert zf.read("hello.txt") == b"hi"


class TestConvenienceDrains:

    def test_to_bytes_keeps_cursor(self) -> None:
        b = BytesIO(b"abc")
        b.seek(2)
        assert b.to_bytes() == b"abc"
        assert b.tell() == 2

    def test_decode_text(self) -> None:
        b = BytesIO("héllo".encode("utf-8"))
        assert b.decode() == "héllo"

    def test_to_base64_url_safe(self) -> None:
        b = BytesIO(b"\xff\xfe\xfd")
        assert b.to_base64() == "__79"
