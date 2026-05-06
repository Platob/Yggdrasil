"""Tests for ``yggdrasil.io.buffer.BytesIO``.

Covers the documented public surface — construction shapes,
read/write/seek primitives, the bytes/string/structured surfaces,
hashing, compression, media-type wiring, copy/replace lifecycle,
spill-to-disk behavior, path-bound mode, the view layer
(``BytesIO`` in view mode), and ``pickle`` round-tripping.

Optional-dependency paths (``xxhash``, ``blake3``, ``zstandard``)
are gated behind ``pytest.importorskip`` so the suite passes on a
base install and exercises those branches when the extras are
present.
"""

from __future__ import annotations

import io as _stdio
import os
import pickle
import struct

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.enums.codec import GZIP, ZSTD
from yggdrasil.io.enums.media_type import MediaType
from yggdrasil.io.enums.mime_type import MimeTypes


SMALL = b"Henry Hub prompt settle"
PAYLOAD = b"Brent ICE front-month daily close " * 200


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_is_empty(self):
        buf = BytesIO()
        assert buf.size == 0
        assert buf.tell() == 0
        assert len(buf) == 0

    def test_from_bytes(self):
        buf = BytesIO(SMALL)
        assert buf.size == len(SMALL)
        assert buf.to_bytes() == SMALL

    def test_from_bytearray(self):
        buf = BytesIO(bytearray(SMALL))
        assert buf.to_bytes() == SMALL

    def test_from_memoryview(self):
        buf = BytesIO(memoryview(SMALL))
        assert buf.to_bytes() == SMALL

    def test_from_stdlib_bytesio(self):
        src = _stdio.BytesIO(SMALL)
        buf = BytesIO(src)
        assert buf.to_bytes() == SMALL

    def test_from_stdlib_bytesio_honors_cursor(self):
        src = _stdio.BytesIO(SMALL)
        src.seek(5)
        buf = BytesIO(src)
        assert buf.to_bytes() == SMALL[5:]

    def test_from_filelike_no_seek_drains(self):
        class Stream:
            def __init__(self, data):
                self._chunks = [data[i : i + 4] for i in range(0, len(data), 4)]

            def read(self, n=-1):
                return self._chunks.pop(0) if self._chunks else b""

        buf = BytesIO(Stream(SMALL))
        assert buf.to_bytes() == SMALL

    def test_from_other_bytesio_copy_independent(self):
        original = BytesIO(SMALL)
        twin = BytesIO(original, copy=True)
        twin.seek(0)
        twin.write(b"X")
        assert original.to_bytes() == SMALL

    def test_unsupported_input_type_raises(self):
        with pytest.raises(TypeError):
            BytesIO(123)  # type: ignore[arg-type]

    def test_constructor_accepts_metadata(self):
        buf = BytesIO(SMALL, metadata={"trace": "abc"})
        assert buf.metadata["trace"] == "abc"

    def test_constructor_with_octet_media_type_stays_bytesio(self):
        buf = BytesIO(SMALL, media_type=MediaType(MimeTypes.OCTET_STREAM))
        assert isinstance(buf, BytesIO)
        assert buf.media_type.is_octet


class TestFactories:
    def test_from_idempotent_on_existing_instance(self):
        buf = BytesIO(SMALL)
        assert BytesIO.from_(buf) is buf

    def test_from_wraps_bytes(self):
        wrapped = BytesIO.from_(SMALL)
        assert isinstance(wrapped, BytesIO)
        assert wrapped.to_bytes() == SMALL

    def test_from_returns_default_on_invalid(self):
        sentinel = object()
        result = BytesIO.from_(123, default=sentinel)
        assert result is sentinel

    def test_from_raises_without_default(self):
        with pytest.raises((TypeError, ValueError)):
            BytesIO.from_(123)

    def test_from_path(self, tmp_path):
        p = tmp_path / "factory.bin"
        p.write_bytes(SMALL)
        buf = BytesIO.from_path(str(p), mode="rb")
        try:
            assert buf.to_bytes() == SMALL
        finally:
            buf.close()


# ---------------------------------------------------------------------------
# is_bytish
# ---------------------------------------------------------------------------


class TestIsBytish:
    @pytest.mark.parametrize(
        "obj, expected",
        [
            (b"x", True),
            (bytearray(b"x"), True),
            (memoryview(b"x"), True),
            (BytesIO(), True),
            (_stdio.BytesIO(), True),
            (42, False),
            (None, False),
        ],
    )
    def test_predicate(self, obj, expected):
        assert BytesIO.is_bytish(obj) is expected

    def test_pathish_strings_count_as_bytish(self):
        # Strings are accepted because the constructor binds them as a path.
        assert BytesIO.is_bytish("/tmp/x.bin") is True


# ---------------------------------------------------------------------------
# Cursor read / write / seek
# ---------------------------------------------------------------------------


class TestReadWrite:
    def test_write_then_read(self):
        buf = BytesIO()
        n = buf.write(SMALL)
        assert n == len(SMALL)
        buf.seek(0)
        assert buf.read() == SMALL

    def test_partial_read(self):
        buf = BytesIO(SMALL)
        assert buf.read(5) == SMALL[:5]
        assert buf.tell() == 5

    def test_read_negative_returns_remaining(self):
        buf = BytesIO(SMALL)
        buf.seek(7)
        assert buf.read(-1) == SMALL[7:]

    def test_to_bytes(self):
        assert BytesIO(SMALL).to_bytes() == SMALL

    def test_getvalue_alias(self):
        assert BytesIO(SMALL).getvalue() == SMALL

    def test_decode_default_utf8(self):
        assert BytesIO(b"hello").decode() == "hello"

    def test_decode_empty_returns_empty_str(self):
        assert BytesIO().decode() == ""

    def test_to_base64_urlsafe(self):
        # b"\xff\xfe" -> standard b64 contains '/+', urlsafe replaces.
        buf = BytesIO(b"\xff\xfe")
        assert buf.to_base64(urlsafe=True) == "__4="
        assert buf.to_base64(urlsafe=False) == "//4="

    def test_pread_does_not_move_cursor(self):
        buf = BytesIO(SMALL)
        buf.seek(3)
        assert buf.pread(5, 0) == SMALL[:5]
        assert buf.pread(5, 6) == SMALL[6:11]
        assert buf.tell() == 3

    def test_pread_negative_pos_raises(self):
        buf = BytesIO(SMALL)
        with pytest.raises(ValueError):
            buf.pread(5, -1)

    def test_pwrite_does_not_move_cursor(self):
        buf = BytesIO(b"AAAA")
        buf.seek(2)
        n = buf.pwrite(b"X", 1)
        assert n == 1
        assert buf.to_bytes() == b"AXAA"
        assert buf.tell() == 2

    def test_pwrite_negative_pos_raises(self):
        with pytest.raises(ValueError):
            BytesIO(b"abc").pwrite(b"X", -1)

    def test_readinto(self):
        buf = BytesIO(SMALL)
        target = bytearray(5)
        n = buf.readinto(target)
        assert n == 5
        assert bytes(target) == SMALL[:5]
        assert buf.tell() == 5

    def test_readinto1(self):
        buf = BytesIO(SMALL)
        target = bytearray(4)
        assert buf.readinto1(target) == 4
        assert bytes(target) == SMALL[:4]

    def test_truncate_grow_zero_fills(self):
        buf = BytesIO(b"abc")
        buf.truncate(6)
        assert buf.size == 6
        assert buf.to_bytes() == b"abc\x00\x00\x00"

    def test_truncate_shrink(self):
        buf = BytesIO(b"abcdef")
        buf.truncate(3)
        assert buf.to_bytes() == b"abc"

    def test_truncate_negative_raises(self):
        with pytest.raises((ValueError, OSError)):
            BytesIO(b"abc").truncate(-1)

    def test_truncate_to_cursor_when_size_omitted(self):
        buf = BytesIO(b"abcdef")
        buf.seek(2)
        buf.truncate()
        assert buf.to_bytes() == b"ab"


class TestWriteDispatch:
    def test_write_str_encodes_utf8(self):
        buf = BytesIO()
        n = buf.write("héllo")
        assert n == len("héllo".encode("utf-8"))
        assert buf.to_bytes() == "héllo".encode("utf-8")

    def test_write_bytes(self):
        buf = BytesIO()
        assert buf.write(b"hello") == 5
        assert buf.to_bytes() == b"hello"

    def test_write_filelike_drains(self):
        src = _stdio.BytesIO(b"streamed")
        buf = BytesIO()
        n = buf.write(src)
        assert n == 8
        assert buf.to_bytes() == b"streamed"

    def test_write_none_returns_zero(self):
        buf = BytesIO()
        assert buf.write(None) == 0
        assert buf.size == 0

    def test_write_str_helper(self):
        buf = BytesIO()
        n = buf.write_str("foo")
        assert n == 3
        assert buf.to_bytes() == b"foo"

    def test_writelines(self):
        buf = BytesIO()
        buf.writelines([b"a", b"b", b"c"])
        assert buf.to_bytes() == b"abc"

    def test_write_stream_drains_other_bytesio(self):
        src = BytesIO(SMALL)
        dst = BytesIO()
        n = dst.write_stream(src)
        assert n == len(SMALL)
        assert dst.to_bytes() == SMALL


class TestSeek:
    def test_seek_set(self):
        buf = BytesIO(SMALL)
        assert buf.seek(3) == 3
        assert buf.tell() == 3

    def test_seek_cur(self):
        buf = BytesIO(SMALL)
        buf.seek(2)
        buf.seek(3, _stdio.SEEK_CUR)
        assert buf.tell() == 5

    def test_seek_end(self):
        buf = BytesIO(SMALL)
        assert buf.seek(0, _stdio.SEEK_END) == len(SMALL)

    def test_seek_minus_one_is_end_sentinel(self):
        # Yggdrasil extension: seek(-1) on SEEK_SET maps to end, to
        # mirror the read(-1) "read all" sentinel. Other negative
        # SEEK_SET offsets raise like stdlib io.BytesIO.
        buf = BytesIO(b"helloworld")
        assert buf.seek(-1) == 10
        assert buf.tell() == 10

    def test_seek_other_negative_set_raises(self):
        buf = BytesIO(b"helloworld")
        for bad in (-2, -5, -100):
            with pytest.raises(ValueError):
                buf.seek(bad)
        # cursor unchanged after the failed seeks
        assert buf.tell() == 0

    def test_seek_invalid_whence_raises(self):
        with pytest.raises(ValueError):
            BytesIO(SMALL).seek(0, whence=99)


class TestHeadTail:
    def test_head_returns_prefix(self):
        assert BytesIO(SMALL).head(4) == SMALL[:4]

    def test_tail_returns_suffix(self):
        assert BytesIO(SMALL).tail(4) == SMALL[-4:]

    def test_head_does_not_move_cursor(self):
        buf = BytesIO(SMALL)
        buf.seek(2)
        buf.head(4)
        assert buf.tell() == 2

    def test_head_zero_returns_empty(self):
        assert BytesIO(SMALL).head(0) == b""

    def test_tail_zero_returns_empty(self):
        assert BytesIO(SMALL).tail(0) == b""

    def test_head_on_empty_buffer(self):
        assert BytesIO().head(10) == b""
        assert BytesIO().tail(10) == b""


class TestSyntheticContent:
    def test_short_payload_returns_full(self):
        buf = BytesIO(b"abc")
        assert buf.synthetic_content(n=128) == "abc"

    def test_long_payload_elides_middle(self):
        buf = BytesIO(b"X" * 200)
        out = buf.synthetic_content(n=10)
        assert "..." in out

    def test_empty_buffer_returns_empty(self):
        assert BytesIO().synthetic_content() == ""
        assert BytesIO().synthetic_content(encoding=None) == b""

    def test_encoding_none_returns_bytes(self):
        out = BytesIO(b"abc").synthetic_content(n=128, encoding=None)
        assert out == b"abc"


# ---------------------------------------------------------------------------
# Lines / iteration
# ---------------------------------------------------------------------------


class TestLines:
    def test_readline(self):
        buf = BytesIO(b"line1\nline2\nline3")
        assert buf.readline() == b"line1\n"
        assert buf.readline() == b"line2\n"
        assert buf.readline() == b"line3"
        assert buf.readline() == b""

    def test_readlines(self):
        buf = BytesIO(b"a\nb\nc\n")
        assert buf.readlines() == [b"a\n", b"b\n", b"c\n"]

    def test_iter_protocol(self):
        buf = BytesIO(b"L1\nL2\n")
        assert list(buf) == [b"L1\n", b"L2\n"]


# ---------------------------------------------------------------------------
# Structured I/O
# ---------------------------------------------------------------------------


class TestStructuredInts:
    @pytest.mark.parametrize(
        "writer, reader, value",
        [
            ("write_int8", "read_int8", -7),
            ("write_uint8", "read_uint8", 200),
            ("write_int16", "read_int16", -1234),
            ("write_uint16", "read_uint16", 65000),
            ("write_int32", "read_int32", -10**6),
            ("write_uint32", "read_uint32", 4 * 10**9),
            ("write_int64", "read_int64", -(10**18)),
            ("write_uint64", "read_uint64", 10**18),
        ],
    )
    def test_int_round_trip(self, writer, reader, value):
        buf = BytesIO()
        getattr(buf, writer)(value)
        buf.seek(0)
        assert getattr(buf, reader)() == value

    def test_floats_round_trip(self):
        buf = BytesIO()
        buf.write_f32(3.5)
        buf.write_f64(1.234567890123)
        buf.seek(0)
        assert buf.read_f32() == pytest.approx(3.5)
        assert buf.read_f64() == pytest.approx(1.234567890123)

    def test_bool_round_trip(self):
        buf = BytesIO()
        buf.write_bool(True)
        buf.write_bool(False)
        buf.seek(0)
        assert buf.read_bool() is True
        assert buf.read_bool() is False

    def test_bytes_u32_round_trip(self):
        buf = BytesIO()
        buf.write_bytes_u32(b"prefix-payload")
        buf.seek(0)
        assert buf.read_bytes_u32() == b"prefix-payload"

    def test_str_u32_round_trip(self):
        buf = BytesIO()
        buf.write_str_u32("world")
        buf.seek(0)
        assert buf.read_str_u32() == "world"

    def test_short_read_raises_eof(self):
        buf = BytesIO(b"\x01\x02")
        with pytest.raises(EOFError):
            buf.read_int32()

    def test_endianness_is_little(self):
        buf = BytesIO()
        buf.write_int32(0x01020304)
        assert buf.to_bytes() == struct.pack("<i", 0x01020304)


# ---------------------------------------------------------------------------
# memoryview / arrow_io
# ---------------------------------------------------------------------------


class TestMemoryView:
    def test_memoryview_returns_bytes(self):
        buf = BytesIO(SMALL)
        assert bytes(buf.memoryview()) == SMALL

    def test_memoryview_empty_buffer(self):
        assert bytes(BytesIO().memoryview()) == b""

    def test_memoryview_after_spill(self):
        buf = BytesIO(spill_bytes=8)
        buf.write(b"X" * 200)
        assert buf.spilled
        assert bytes(buf.memoryview()) == b"X" * 200


class TestArrowIO:
    def test_arrow_io_memory_returns_python_file(self):
        import pyarrow as pa

        buf = BytesIO(SMALL)
        f = buf.arrow_io(mode="rb")
        assert isinstance(f, pa.PythonFile)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestHashing:
    def test_xxh3_64_deterministic(self):
        pytest.importorskip("xxhash")
        a = BytesIO(SMALL).xxh3_64().intdigest()
        b = BytesIO(SMALL).xxh3_64().intdigest()
        assert a == b

    def test_xxh3_int64_is_two_complement(self):
        pytest.importorskip("xxhash")
        v = BytesIO(SMALL).xxh3_int64()
        assert -(2**63) <= v < 2**63

    def test_blake3_deterministic(self):
        pytest.importorskip("blake3")
        a = BytesIO(SMALL).blake3().hexdigest()
        b = BytesIO(SMALL).blake3().hexdigest()
        assert a == b

    def test_blake3_after_spill(self):
        pytest.importorskip("blake3")
        buf = BytesIO(b"Y" * 200, spill_bytes=8)
        assert buf.spilled
        assert len(buf.blake3().hexdigest()) == 64


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


class TestCompression:
    def test_gzip_round_trip_with_copy(self):
        original_size = len(PAYLOAD)
        buf = BytesIO(PAYLOAD)
        compressed = buf.compress(codec=GZIP, copy=True)
        # copy=True returns a fresh buffer, leaves the source untouched.
        assert buf.size == original_size
        assert compressed.size < original_size
        decompressed = compressed.decompress(codec=GZIP, copy=True)
        assert decompressed.to_bytes() == PAYLOAD

    def test_gzip_in_place_compress_decompress(self):
        buf = BytesIO(PAYLOAD)
        buf.compress(codec=GZIP)
        assert buf.size < len(PAYLOAD)
        buf.decompress(codec=GZIP)
        assert buf.to_bytes() == PAYLOAD

    def test_compress_by_string_name(self):
        compressed = BytesIO(PAYLOAD).compress(codec="gzip", copy=True)
        assert compressed.size < len(PAYLOAD)

    def test_compress_unknown_codec_raises(self):
        with pytest.raises(ValueError):
            BytesIO(PAYLOAD).compress(codec="not-a-codec")

    def test_decompress_falsy_codec_returns_self(self):
        buf = BytesIO(b"data")
        assert buf.decompress(codec=None) is buf

    def test_zstd_round_trip(self):
        pytest.importorskip("zstandard")
        buf = BytesIO(PAYLOAD)
        compressed = buf.compress(codec=ZSTD, copy=True)
        round_tripped = compressed.decompress(codec=ZSTD, copy=True)
        assert round_tripped.to_bytes() == PAYLOAD

    def test_compress_in_place_records_codec(self):
        buf = BytesIO(b'{"k": 1}' * 200, media_type=MediaType(MimeTypes.JSON))
        buf.compress(codec=GZIP)
        # The media type now carries the gzip codec while keeping the JSON mime.
        assert buf.media_type.codec is GZIP
        assert buf.media_type.mime_type is MimeTypes.JSON


# ---------------------------------------------------------------------------
# Media type
# ---------------------------------------------------------------------------


class TestMediaType:
    def test_default_is_octet_stream(self):
        assert BytesIO().media_type.is_octet

    def test_with_media_type_returns_self_when_not_copied(self):
        buf = BytesIO()
        target = MediaType(MimeTypes.JSON)
        assert buf.with_media_type(target, copy=False) is buf

    def test_with_media_type_copy_yields_new_instance(self):
        buf = BytesIO(b"{}")
        target = MediaType(MimeTypes.JSON)
        copied = buf.with_media_type(target, copy=True)
        assert copied is not buf
        assert copied.media_type.mime_type is MimeTypes.JSON

    def test_change_mime_type_on_nonempty_raises(self):
        # Start with a concrete (non-"any-bytes") mime type so the getter
        # doesn't auto-clear it on the first read.
        buf = BytesIO(b'{"k": 1}', media_type=MediaType(MimeTypes.JSON))
        with pytest.raises(ValueError):
            buf.with_media_type(MediaType(MimeTypes.PARQUET), copy=False)

    def test_media_type_setter(self):
        buf = BytesIO()
        buf.media_type = MediaType(MimeTypes.JSON)
        assert buf.mime_type is MimeTypes.JSON
        assert buf.codec is None


# ---------------------------------------------------------------------------
# IO[bytes] protocol
# ---------------------------------------------------------------------------


class TestIOProtocol:
    def test_default_mode_is_rb_plus(self):
        buf = BytesIO()
        assert buf.mode == "rb+"
        assert buf.readable()
        assert buf.writable()
        assert buf.seekable()
        assert buf.is_writing

    def test_isatty_false(self):
        assert BytesIO().isatty() is False

    def test_fileno_memory_raises(self):
        with pytest.raises(OSError):
            BytesIO(b"abc").fileno()

    def test_fileno_after_spill_returns_int(self):
        buf = BytesIO(b"X" * 200, spill_bytes=8)
        try:
            assert isinstance(buf.fileno(), int)
        finally:
            buf.close()

    def test_writable_false_for_read_only_mode(self, tmp_path):
        p = tmp_path / "ro.bin"
        p.write_bytes(b"abc")
        with BytesIO(path=str(p), mode="rb") as buf:
            assert buf.readable()
            assert not buf.writable()

    def test_url_memory_buffer(self):
        url = BytesIO().url
        assert "mem:" in url.to_string(encode=False)

    def test_name_memory(self):
        assert BytesIO().name == "<memory>"

    def test_is_local_memory(self):
        buf = BytesIO()
        assert buf.is_local
        assert not buf.is_remote


# ---------------------------------------------------------------------------
# Spill behavior / path-bound
# ---------------------------------------------------------------------------


class TestSpill:
    def test_need_spill_threshold(self):
        buf = BytesIO(spill_bytes=100)
        assert buf.need_spill(50) is False
        assert buf.need_spill(200) is True

    def test_auto_spill_on_write(self):
        buf = BytesIO(spill_bytes=8)
        buf.write(b"X" * 64)
        assert buf.spilled
        assert buf.size == 64
        # Round-trip survives the spill.
        assert buf.to_bytes() == b"X" * 64
        buf.close()

    def test_init_from_large_bytes_spills(self):
        buf = BytesIO(b"Y" * 500, spill_bytes=8)
        assert buf.spilled
        assert buf.to_bytes() == b"Y" * 500
        buf.close()

    def test_reserve_below_spill_grows_capacity_only(self):
        buf = BytesIO(spill_bytes=10**9)
        buf.reserve(2048)
        # _size and _pos are not touched by reserve.
        assert buf.size == 0
        assert buf.tell() == 0
        # Capacity is enough that a 2048-byte write doesn't grow further.
        n = buf.write(b"Z" * 2048)
        assert n == 2048

    def test_reserve_above_threshold_spills(self):
        buf = BytesIO(spill_bytes=64)
        buf.reserve(256)
        assert buf.spilled
        buf.close()

    def test_reserve_negative_raises(self):
        with pytest.raises(ValueError):
            BytesIO().reserve(-1)


class TestPathBound:
    def test_round_trip_via_path(self, tmp_path):
        p = tmp_path / "bound.bin"
        with BytesIO(path=str(p), mode="wb+") as buf:
            buf.write(b"durable!")
        assert p.read_bytes() == b"durable!"

        with BytesIO(path=str(p), mode="rb") as buf:
            assert buf.read() == b"durable!"

    def test_path_property_set_when_bound(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"hi")
        with BytesIO(path=str(p), mode="rb") as buf:
            assert buf.path is not None
            assert buf.spilled

    def test_replace_with_payload_external_path_keeps_binding(self, tmp_path):
        p = tmp_path / "bound.bin"
        p.write_bytes(b"original")
        with BytesIO(path=str(p), mode="rb+") as buf:
            buf.replace_with_payload(b"replaced")
        assert p.read_bytes() == b"replaced"

    def test_replace_with_payload_self_raises(self):
        buf = BytesIO(b"abc")
        with pytest.raises(ValueError):
            buf.replace_with_payload(buf)

    def test_replace_with_payload_none_clears(self):
        buf = BytesIO(b"abc")
        buf.replace_with_payload(None)
        assert buf.size == 0


# ---------------------------------------------------------------------------
# Remote-path context-exit auto-flush
#
# When a BytesIO is bound to a non-local path, writes go through an
# in-memory transaction buffer that has to be committed to the backing
# on close. Earlier the dirty bit was only set by a tiny set of
# internal codepaths, so callers that just did the obvious thing —
#   ``with BytesIO(path=remote, mode="wb+") as f: f.write(payload)``
# — saw the file unchanged and had to add ``f.mark_dirty(); f.flush()``
# manually. These tests pin down that writes alone are now enough.
# ---------------------------------------------------------------------------


def _spoof_non_local(monkeypatch, path) -> None:
    """Make a LocalPath behave like a remote backend for dispatch.

    Real non-local backends (S3, Volumes, …) override
    :meth:`write_stream` / :meth:`write_bytes` to push bytes natively.
    The base :class:`Path` implementation, by contrast, falls back to
    opening another path-bound :class:`BytesIO` — which, for a path
    whose ``is_local`` we've spoofed to ``False``, would recurse
    forever through the transaction-buffer commit path.

    To exercise the BytesIO non-local branches without that
    contrivance, we override both:

    * ``is_local`` → ``False`` so dispatch picks the transaction
      buffer.
    * ``write_stream`` / ``write_bytes`` → write straight to the
      underlying local file, the same role a remote backend's
      override would play.
    """
    monkeypatch.setattr(type(path), "is_local", property(lambda self: False))

    def _write_bytes(self, data, *, mode="wb", parents=True):
        import os as _os
        with open(self.full_path(), mode) as fh:
            return fh.write(bytes(data))

    def _write_stream(self, src, *, batch_size=1 << 20, parents=True):
        # Read whatever the source has; mirrors the no-recursion shape
        # a real backend's streaming upload would have.
        if hasattr(src, "seek"):
            src.seek(0)
        data = src.read() if hasattr(src, "read") else bytes(src)
        return _write_bytes(self, data, parents=parents)

    monkeypatch.setattr(type(path), "write_bytes", _write_bytes)
    monkeypatch.setattr(type(path), "write_stream", _write_stream)


class TestRemotePathAutoFlush:
    def test_write_marks_buffer_dirty_for_non_local_path(self, tmp_path, monkeypatch):
        from yggdrasil.io.fs.local_path import LocalPath
        import pathlib

        target = pathlib.Path(str(tmp_path / "remote.dat"))
        target.write_bytes(b"")
        path = LocalPath.from_pathlib(target)
        _spoof_non_local(monkeypatch, path)

        with BytesIO(path=path, mode="wb+") as buf:
            buf.write(b"hello remote")
            assert buf.is_dirty(), (
                "writes against a non-local backing must flag dirty so "
                "context-exit commits without a manual mark_dirty()"
            )

    def test_context_exit_flushes_writes_to_remote_path(self, tmp_path, monkeypatch):
        from yggdrasil.io.fs.local_path import LocalPath
        import pathlib

        target = pathlib.Path(str(tmp_path / "remote.dat"))
        target.write_bytes(b"")
        path = LocalPath.from_pathlib(target)
        _spoof_non_local(monkeypatch, path)

        with BytesIO(path=path, mode="wb+") as buf:
            buf.write(b"durable payload")

        # No manual mark_dirty()/flush() — the context manager handled it.
        assert target.read_bytes() == b"durable payload"

    def test_explicit_flush_still_works(self, tmp_path, monkeypatch):
        from yggdrasil.io.fs.local_path import LocalPath
        import pathlib

        target = pathlib.Path(str(tmp_path / "remote.dat"))
        target.write_bytes(b"")
        path = LocalPath.from_pathlib(target)
        _spoof_non_local(monkeypatch, path)

        with BytesIO(path=path, mode="wb+") as buf:
            buf.write(b"flush mid-context")
            buf.flush()
            assert not buf.is_dirty()
            assert target.read_bytes() == b"flush mid-context"

    def test_truncate_alone_marks_dirty(self, tmp_path, monkeypatch):
        from yggdrasil.io.fs.local_path import LocalPath
        import pathlib

        target = pathlib.Path(str(tmp_path / "remote.dat"))
        target.write_bytes(b"existing-content")
        path = LocalPath.from_pathlib(target)
        _spoof_non_local(monkeypatch, path)

        with BytesIO(path=path, mode="rb+") as buf:
            buf.truncate(4)

        assert target.read_bytes() == b"exis"

    def test_replace_with_payload_none_flushes_zero_bytes(self, tmp_path, monkeypatch):
        from yggdrasil.io.fs.local_path import LocalPath
        import pathlib

        target = pathlib.Path(str(tmp_path / "remote.dat"))
        target.write_bytes(b"existing-content")
        path = LocalPath.from_pathlib(target)
        _spoof_non_local(monkeypatch, path)

        with BytesIO(path=path, mode="rb+") as buf:
            buf.replace_with_payload(None)

        assert target.read_bytes() == b""

    def test_local_path_writes_do_not_set_dirty(self, tmp_path):
        # Local writes hit the kernel via os.pwrite — no commit work
        # is needed, so the dirty bit must stay clear (no spurious
        # double-flushing).
        target = tmp_path / "local.dat"
        with BytesIO(path=str(target), mode="wb+") as buf:
            buf.write(b"local")
            assert not buf.is_dirty()
        assert target.read_bytes() == b"local"

    def test_in_memory_writes_do_not_set_dirty(self):
        # Pure-memory buffers have no durable backing; flagging
        # dirty would just create work for ``commit`` to skip.
        buf = BytesIO()
        with buf:
            buf.write(b"in-memory")
            assert not buf.is_dirty()


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


class TestCopy:
    def test_memory_copy_independent(self):
        original = BytesIO(SMALL)
        original.seek(3)
        cloned = original.copy()
        assert cloned is not original
        assert cloned.to_bytes() == SMALL
        # Cursor is preserved.
        assert cloned.tell() == 3

        cloned.seek(0)
        cloned.write(b"X")
        assert original.to_bytes() == SMALL

    def test_copy_preserves_metadata(self):
        original = BytesIO(SMALL, metadata={"k": "v"})
        cloned = original.copy()
        assert cloned.metadata == {"k": "v"}
        # Mutation independence.
        cloned.metadata["new"] = 1
        assert "new" not in original.metadata

    def test_path_bound_copy_shares_binding(self, tmp_path):
        p = tmp_path / "shared.bin"
        p.write_bytes(b"data")
        with BytesIO(path=str(p), mode="rb+") as a:
            b = a.copy()
            try:
                assert b.path.full_path() == a.path.full_path()
            finally:
                b.close()

    def test_copy_after_spill(self):
        a = BytesIO(b"Q" * 256, spill_bytes=16)
        try:
            b = a.copy()
            try:
                assert b.to_bytes() == b"Q" * 256
                assert b is not a
            finally:
                b.close()
        finally:
            a.close()


# ---------------------------------------------------------------------------
# Stat / size / mtime
# ---------------------------------------------------------------------------


class TestStat:
    def test_stat_in_memory(self):
        buf = BytesIO(SMALL)
        s = buf.stat()
        assert s.size == len(SMALL)
        # Memory mode synthesizes a SOCKET-kind IOStats.
        assert s.kind.name == "SOCKET"

    def test_stat_local_spilled(self):
        buf = BytesIO(b"Z" * 200, spill_bytes=8)
        try:
            s = buf.stat()
            assert s.size == 200
            assert s.kind.name == "FILE"
        finally:
            buf.close()

    def test_size_property_matches_payload(self):
        assert BytesIO(SMALL).size == len(SMALL)

    def test_is_empty_and_remaining_bytes(self):
        buf = BytesIO(b"abc")
        assert buf.is_empty() is False
        buf.seek(1)
        assert buf.remaining_bytes() == 2

    def test_exists_predicate(self):
        assert BytesIO().exists() is False
        assert BytesIO(b"x").exists() is True


# ---------------------------------------------------------------------------
# write_into / write_into_path / to_path
# ---------------------------------------------------------------------------


class TestDrain:
    def test_write_into_filelike(self):
        sink = _stdio.BytesIO()
        n = BytesIO(SMALL).write_into(sink)
        assert n == len(SMALL)
        assert sink.getvalue() == SMALL

    def test_write_into_path(self, tmp_path):
        out = tmp_path / "out.bin"
        n = BytesIO(SMALL).write_into(str(out))
        assert n == len(SMALL)
        assert out.read_bytes() == SMALL

    def test_write_into_path_no_overwrite_raises(self, tmp_path):
        out = tmp_path / "exists.bin"
        out.write_bytes(b"old")
        with pytest.raises(FileExistsError):
            BytesIO(SMALL).write_into(str(out), overwrite=False)

    def test_write_into_unwritable_raises(self):
        with pytest.raises(TypeError):
            BytesIO(SMALL).write_into(123)  # type: ignore[arg-type]

    def test_to_path(self, tmp_path):
        out = tmp_path / "to_path.bin"
        result = BytesIO(SMALL).to_path(str(out))
        assert os.fspath(result) == str(out)
        assert out.read_bytes() == SMALL


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestJsonLoad:
    def test_object(self):
        buf = BytesIO(b'{"a": 1, "b": [2, 3]}')
        loaded = buf.json_load()
        assert loaded == {"a": 1, "b": [2, 3]}

    def test_array(self):
        buf = BytesIO(b"[1, 2, 3]")
        assert buf.json_load() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------------


class TestPickle:
    def test_round_trip_owned_buffer(self):
        original = BytesIO(SMALL)
        clone = pickle.loads(pickle.dumps(original))
        assert clone.to_bytes() == SMALL

    def test_round_trip_preserves_media_type(self):
        original = BytesIO(b'{"x": 1}', media_type=MediaType(MimeTypes.JSON))
        clone = pickle.loads(pickle.dumps(original))
        assert clone.media_type.mime_type == MimeTypes.JSON

    def test_round_trip_path_bound(self, tmp_path):
        p = tmp_path / "p.bin"
        p.write_bytes(SMALL)
        with BytesIO(path=str(p), mode="rb") as buf:
            blob = pickle.dumps(buf)
        clone = pickle.loads(blob)
        try:
            assert clone.to_bytes() == SMALL
        finally:
            clone.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_idempotent(self):
        buf = BytesIO(SMALL)
        buf.close()
        buf.close()  # second close must not raise
        assert buf.closed

    def test_context_manager_closes(self):
        with BytesIO(SMALL) as buf:
            assert buf.read() == SMALL
        assert buf.closed

    def test_clear_releases_state(self):
        buf = BytesIO(SMALL)
        buf.clear()
        assert buf.size == 0
        assert buf.closed

    def test_owned_spill_unlinked_on_close(self):
        # Spill cleanup runs as part of the disposable release path; that
        # only fires if the buffer was opened. Use the context manager so
        # the lifecycle is exercised end-to-end.
        with BytesIO(b"X" * 200, spill_bytes=8) as buf:
            full_path = buf.path.full_path()
            assert os.path.exists(full_path)
        assert not os.path.exists(full_path)

    def test_external_path_not_unlinked_on_close(self, tmp_path):
        p = tmp_path / "external.bin"
        p.write_bytes(b"keep me")
        with BytesIO(path=str(p), mode="rb+"):
            pass
        assert p.exists()


# ---------------------------------------------------------------------------
# Dunder
# ---------------------------------------------------------------------------


class TestDunder:
    def test_len_matches_size(self):
        assert len(BytesIO(SMALL)) == len(SMALL)

    def test_bytes_matches_to_bytes(self):
        assert bytes(BytesIO(SMALL)) == SMALL

    def test_bool_always_true(self):
        assert bool(BytesIO()) is True
        assert bool(BytesIO(SMALL)) is True

    def test_repr_has_basic_structure(self):
        text = repr(BytesIO(SMALL))
        assert "BytesIO" in text
        assert "size=" in text


# ===========================================================================
# BytesIO view-mode (parent + view-relative cursor)
# ===========================================================================


class TestBytesIOView:
    def test_to_bytes_returns_window(self):
        parent = BytesIO(b"0123456789")
        v = parent.view(pos=2, size=4)
        assert v.to_bytes() == b"2345"

    def test_read_advances_view_cursor_only(self):
        parent = BytesIO(b"0123456789")
        v = parent.view(pos=2, size=4)
        assert v.read(2) == b"23"
        assert v.tell() == 2
        assert parent.tell() == 0

    def test_read_remaining_returns_balance(self):
        parent = BytesIO(b"0123456789")
        v = parent.view(pos=2, size=4)
        v.seek(1)
        assert v.read() == b"345"

    def test_pread_does_not_move_cursor(self):
        parent = BytesIO(b"0123456789")
        v = parent.view(pos=2, size=4)
        v.seek(1)
        assert v.pread(2, pos=0) == b"23"
        assert v.tell() == 1

    def test_pwrite_propagates_to_parent(self):
        parent = BytesIO(b"0123456789")
        v = parent.view(pos=0, size=10)
        v.pwrite(b"X", pos=5)
        assert parent.to_bytes() == b"01234X6789"

    def test_pwrite_rejects_non_byteslike(self):
        v = BytesIO(b"abc").view(pos=0, size=3)
        with pytest.raises(TypeError):
            v.pwrite(5, pos=0)  # type: ignore[arg-type]

    def test_pwrite_negative_pos_raises(self):
        v = BytesIO(b"abc").view(pos=0, size=3)
        with pytest.raises(ValueError):
            v.pwrite(b"x", pos=-1)

    def test_write_advances_view_cursor(self):
        parent = BytesIO(b"0000")
        v = parent.view(pos=0, size=4)
        v.write(b"AB")
        assert v.tell() == 2
        assert parent.to_bytes() == b"AB00"

    def test_max_size_caps_growth(self):
        parent = BytesIO(b"\x00" * 10)
        v = parent.view(pos=0, size=2, max_size=4)
        n = v.pwrite(b"123456", pos=0)
        assert n == 4
        assert v.size == 4

    def test_max_size_full_returns_zero(self):
        parent = BytesIO(b"\x00\x00")
        v = parent.view(pos=0, size=2, max_size=2)
        # Already at the cap; further writes return 0.
        n = v.pwrite(b"X", pos=2)
        assert n == 0

    def test_truncate_shrinks_parent(self):
        parent = BytesIO(b"abcdef")
        v = parent.view(pos=0, size=6)
        v.truncate(3)
        assert parent.to_bytes() == b"abc"
        assert v.size == 3

    def test_truncate_to_cursor_when_size_omitted(self):
        parent = BytesIO(b"abcdef")
        v = parent.view(pos=0, size=6)
        v.seek(2)
        v.truncate()
        assert v.size == 2
        assert parent.to_bytes() == b"ab"

    def test_truncate_negative_raises(self):
        v = BytesIO(b"abc").view(pos=0, size=3)
        with pytest.raises(ValueError):
            v.truncate(-1)

    def test_seek_minus_one_is_end_sentinel(self):
        # BytesIO maps seek(-1) to end as a "go to end" sentinel.
        # A view inherits that semantics since it is a BytesIO.
        v = BytesIO(b"abc").view(pos=0, size=3)
        assert v.seek(-1, _stdio.SEEK_SET) == 3
        # Other negative SEEK_SET offsets raise on a view too.
        with pytest.raises(ValueError):
            v.seek(-2, _stdio.SEEK_SET)

    def test_seek_cur_clamps_to_zero(self):
        v = BytesIO(b"abc").view(pos=0, size=3)
        v.seek(1)
        # Going far below 0 clamps.
        assert v.seek(-100, _stdio.SEEK_CUR) == 0

    def test_seek_end_clamps_to_zero(self):
        v = BytesIO(b"abc").view(pos=0, size=3)
        assert v.seek(-100, _stdio.SEEK_END) == 0

    def test_seek_invalid_whence_raises(self):
        v = BytesIO(b"abc").view(pos=0, size=3)
        with pytest.raises(ValueError):
            v.seek(0, whence=42)

    def test_readinto(self):
        parent = BytesIO(b"abcdef")
        v = parent.view(pos=2, size=3)
        target = bytearray(2)
        v.readinto(target)
        assert bytes(target) == b"cd"

    def test_readall(self):
        parent = BytesIO(b"abcdef")
        v = parent.view(pos=2, size=3)
        assert v.readall() == b"cde"
        # readall advances the cursor to size.
        assert v.tell() == 3

    def test_close_marks_closed(self):
        v = BytesIO(b"abc").view(pos=0, size=3)
        v.close()
        assert v.closed

    def test_context_manager_closes(self):
        with BytesIO(b"abc").view(pos=0, size=3) as v:
            assert isinstance(v, BytesIO)
            assert v.is_view
        assert v.closed

    def test_construction_validation(self):
        parent = BytesIO(b"abc")
        with pytest.raises(ValueError):
            BytesIO._make_view(parent, offset=-1)
        with pytest.raises(ValueError):
            BytesIO._make_view(parent, size=-1)
        with pytest.raises(ValueError):
            BytesIO._make_view(parent, pos=-1)
        with pytest.raises(ValueError):
            BytesIO._make_view(parent, max_size=-1)
        with pytest.raises(ValueError):
            BytesIO._make_view(parent, size=10, max_size=5)

    def test_end_and_remaining(self):
        parent = BytesIO(b"abcdef")
        v = parent.view(pos=2, size=3)
        assert v.end == 5
        v.seek(1)
        assert v.remaining == 2
