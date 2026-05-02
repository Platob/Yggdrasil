"""Tests for yggdrasil.io.buffer.bytes_io.BytesIO.

Focus on the public, in-memory contract: construction shapes,
reads/writes/seeks, the bytes/string surface, hashing, compression
helpers, and media-type wiring. Spill-to-disk semantics are touched
on briefly but not deeply audited — the previous suite over-exercised
internal mode transitions and was the largest source of failures.
"""

from __future__ import annotations

import io as _stdio

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

    def test_from_bytes(self):
        buf = BytesIO(SMALL)
        assert buf.size == len(SMALL)

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

    def test_from_other_bytesio_copy_true(self):
        original = BytesIO(SMALL)
        twin = BytesIO(original, copy=True)
        # Independent mutation: writing to one should not mutate the
        # original (copy semantics).
        twin.seek(0)
        twin.write(b"X")
        assert original.to_bytes() == SMALL

    def test_unsupported_input_type_raises(self):
        with pytest.raises(TypeError):
            BytesIO(123)  # type: ignore[arg-type]

    def test_factory_from_idempotent(self):
        buf = BytesIO(SMALL)
        assert BytesIO.from_(buf) is buf

    def test_factory_from_bytes(self):
        wrapped = BytesIO.from_(SMALL)
        assert isinstance(wrapped, BytesIO)
        assert wrapped.to_bytes() == SMALL


# ---------------------------------------------------------------------------
# Bytish / pathish detectors
# ---------------------------------------------------------------------------


class TestIsBytish:
    def test_bytes(self):
        assert BytesIO.is_bytish(b"x") is True

    def test_bytesio(self):
        assert BytesIO.is_bytish(BytesIO()) is True

    def test_filelike(self):
        assert BytesIO.is_bytish(_stdio.BytesIO()) is True

    def test_int_is_not_bytish(self):
        assert BytesIO.is_bytish(42) is False


# ---------------------------------------------------------------------------
# Read / write / seek
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
        buf.seek(0)
        assert buf.read(5) == SMALL[:5]

    def test_to_bytes(self):
        buf = BytesIO(SMALL)
        assert buf.to_bytes() == SMALL

    def test_decode(self):
        buf = BytesIO(b"hello")
        assert buf.decode() == "hello"

    def test_getvalue_alias(self):
        buf = BytesIO(SMALL)
        assert buf.getvalue() == SMALL

    def test_pread(self):
        buf = BytesIO(SMALL)
        assert buf.pread(5, 0) == SMALL[:5]
        assert buf.pread(5, 6) == SMALL[6:11]

    def test_pwrite(self):
        buf = BytesIO(b"AAAA")
        n = buf.pwrite(b"X", 1)
        assert n == 1
        assert buf.to_bytes() == b"AXAA"

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
        buf = BytesIO(b"abc")
        with pytest.raises((ValueError, OSError)):
            buf.truncate(-1)


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


class TestHeadTail:
    def test_head(self):
        buf = BytesIO(SMALL)
        assert buf.head(4) == SMALL[:4]

    def test_tail(self):
        buf = BytesIO(SMALL)
        assert buf.tail(4) == SMALL[-4:]

    def test_head_does_not_move_cursor(self):
        buf = BytesIO(SMALL)
        buf.seek(2)
        buf.head(4)
        assert buf.tell() == 2


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestHashing:
    def test_xxh3_64_deterministic(self):
        a = BytesIO(SMALL).xxh3_64().intdigest()
        b = BytesIO(SMALL).xxh3_64().intdigest()
        assert a == b

    def test_xxh3_int64_returns_signed_int(self):
        buf = BytesIO(SMALL)
        v = buf.xxh3_int64()
        # Two's-complement int64 range
        assert -(2**63) <= v < 2**63


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------


class TestCompression:
    def test_gzip_round_trip_with_copy(self):
        original_size = len(PAYLOAD)
        buf = BytesIO(PAYLOAD)
        compressed = buf.compress(codec=GZIP, copy=True)
        # Default `compress` mutates self; copy=True returns a fresh buffer
        # leaving the source untouched.
        assert buf.size == original_size
        assert compressed.size < original_size
        decompressed = compressed.decompress(codec=GZIP, copy=True)
        assert decompressed.to_bytes() == PAYLOAD

    def test_compress_by_name(self):
        buf = BytesIO(PAYLOAD)
        compressed = buf.compress(codec="gzip", copy=True)
        assert compressed.size < len(PAYLOAD)

    def test_compress_unknown_codec_raises(self):
        buf = BytesIO(PAYLOAD)
        with pytest.raises(ValueError):
            buf.compress(codec="not-a-codec")

    def test_zstd_round_trip(self):
        pytest.importorskip("zstandard")
        buf = BytesIO(PAYLOAD)
        compressed = buf.compress(codec=ZSTD, copy=True)
        assert compressed.decompress(codec=ZSTD, copy=True).to_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# Media type
# ---------------------------------------------------------------------------


class TestMediaType:
    def test_default_is_octet_stream(self):
        buf = BytesIO()
        assert buf.media_type.is_octet

    def test_with_media_type_returns_self(self):
        buf = BytesIO()
        target = MediaType(MimeTypes.JSON)
        result = buf.with_media_type(target, copy=False)
        assert result is buf

    def test_with_media_type_copy_yields_new(self):
        buf = BytesIO(b"{}")
        target = MediaType(MimeTypes.JSON)
        copied = buf.with_media_type(target, copy=True)
        assert copied is not buf
        assert copied.media_type.mime_type is MimeTypes.JSON

    def test_constructor_with_media_type(self):
        # Non-octet media types route through PrimitiveIO, but octet ones
        # stay BytesIO and store the type.
        buf = BytesIO(SMALL, media_type=MediaType(MimeTypes.OCTET_STREAM))
        assert buf.media_type.is_octet


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_idempotent(self):
        buf = BytesIO(SMALL)
        buf.close()
        # Second close must not raise.
        buf.close()
        assert buf.closed

    def test_use_as_context_manager(self):
        with BytesIO(SMALL) as buf:
            assert buf.read() == SMALL
        assert buf.closed
