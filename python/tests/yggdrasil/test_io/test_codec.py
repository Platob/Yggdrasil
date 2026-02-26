# tests/io/test_codec.py
"""Unit tests for yggdrasil.io.enums.codec — Codec enum.

Coverage
--------
- Magic-byte detection: from_io / from_bytes for all six codecs
- _peek / _peek_buf cursor-safe helpers
- compress_bytes / decompress_bytes round-trips (bytes API)
- compress / open stream API with cursor-preservation contract
- compress_with / decompress_with conditional passthrough + str coercion
- roundtrip() diagnostic helper
"""

from __future__ import annotations

import io

import pytest

from .conftest import (
    PAYLOAD,
    SMALL,
    compress_bz2,
    compress_gzip,
    compress_lz4,
    compress_snappy,
    compress_xz,
    compress_zstd,
)

pytest.importorskip("yggdrasil")

from yggdrasil.io.buffer.bytes_io import BytesIO                 # noqa: E402
from yggdrasil.io.enums.codec import Codec, _peek, _peek_buf    # noqa: E402


# ===========================================================================
# Detection — from_io / from_bytes
# ===========================================================================

class TestCodecDetection:
    """Magic-byte detection via from_io and from_bytes."""

    def test_gzip_from_io(self):
        assert Codec.from_io(io.BytesIO(compress_gzip(SMALL))) is Codec.GZIP

    def test_zstd_from_io(self):
        assert Codec.from_io(io.BytesIO(compress_zstd(SMALL))) is Codec.ZSTD

    def test_lz4_from_io(self):
        assert Codec.from_io(io.BytesIO(compress_lz4(SMALL))) is Codec.LZ4

    def test_bzip2_from_io(self):
        assert Codec.from_io(io.BytesIO(compress_bz2(SMALL))) is Codec.BZIP2

    def test_xz_from_io(self):
        assert Codec.from_io(io.BytesIO(compress_xz(SMALL))) is Codec.XZ

    def test_snappy_from_io(self):
        assert Codec.from_io(io.BytesIO(compress_snappy(SMALL))) is Codec.SNAPPY

    def test_uncompressed_returns_none(self):
        assert Codec.from_io(io.BytesIO(b"PAR1\x00" * 10)) is None

    def test_empty_stream_returns_none(self):
        assert Codec.from_io(io.BytesIO(b"")) is None

    def test_from_bytes_gzip(self):
        assert Codec.from_bytes(compress_gzip(SMALL)) is Codec.GZIP

    def test_from_bytes_uncompressed(self):
        assert Codec.from_bytes(b"PAR1hello") is None

    def test_cursor_preserved_after_from_io(self):
        stream = io.BytesIO(compress_gzip(SMALL))
        stream.seek(4)
        Codec.from_io(stream)
        assert stream.tell() == 4

    def test_from_io_accepts_bytesio_wrapper(self):
        """from_io must duck-type .buffer() on the yggdrasil BytesIO wrapper."""
        buf = BytesIO(compress_gzip(SMALL))
        assert Codec.from_io(buf) is Codec.GZIP


# ===========================================================================
# Bytes API — compress_bytes / decompress_bytes
# ===========================================================================

class TestCodecBytesAPI:
    """compress_bytes / decompress_bytes round-trips."""

    @pytest.mark.parametrize("codec,compress_fn", [
        (Codec.GZIP,  compress_gzip),
        (Codec.BZIP2, compress_bz2),
        (Codec.XZ,    compress_xz),
    ])
    def test_stdlib_roundtrip(self, codec, compress_fn):
        compressed = codec.compress_bytes(PAYLOAD)
        assert Codec.from_bytes(compressed) is codec
        assert codec.decompress_bytes(compressed) == PAYLOAD

    def test_zstd_roundtrip(self):
        compressed = Codec.ZSTD.compress_bytes(PAYLOAD)
        assert Codec.from_bytes(compressed) is Codec.ZSTD
        assert Codec.ZSTD.decompress_bytes(compressed) == PAYLOAD

    def test_lz4_roundtrip(self):
        compressed = Codec.LZ4.compress_bytes(PAYLOAD)
        assert Codec.from_bytes(compressed) is Codec.LZ4
        assert Codec.LZ4.decompress_bytes(compressed) == PAYLOAD

    def test_snappy_roundtrip(self):
        compressed = Codec.SNAPPY.compress_bytes(PAYLOAD)
        assert Codec.SNAPPY.decompress_bytes(compressed) == PAYLOAD

    def test_compress_reduces_size_on_repetitive_data(self):
        for codec in (Codec.GZIP, Codec.ZSTD, Codec.LZ4, Codec.BZIP2, Codec.XZ):
            assert len(codec.compress_bytes(PAYLOAD)) < len(PAYLOAD), codec

    def test_empty_bytes_roundtrip(self):
        for codec in (Codec.GZIP, Codec.BZIP2, Codec.XZ):
            assert codec.decompress_bytes(codec.compress_bytes(b"")) == b""


# ===========================================================================
# Stream API — compress / open
# ===========================================================================

class TestCodecStreamAPI:
    """compress() and open() stream-level API."""

    def test_compress_returns_bytesio_at_zero(self):
        out = Codec.GZIP.compress(io.BytesIO(SMALL))
        assert out.tell() == 0
        assert isinstance(out, BytesIO)

    def test_open_returns_bytesio_at_zero(self):
        out = Codec.GZIP.open(io.BytesIO(compress_gzip(SMALL)))
        assert out.tell() == 0
        assert out.read() == SMALL

    def test_compress_preserves_src_cursor(self):
        src = io.BytesIO(SMALL)
        src.seek(5)
        Codec.GZIP.compress(src)
        assert src.tell() == 5

    def test_open_preserves_src_cursor(self):
        # _drain reads from the current position; the stream must sit at 0 for
        # a valid compressed frame.  Cursor is restored to 0 after the call.
        compressed = io.BytesIO(compress_gzip(SMALL))
        compressed.seek(0)
        Codec.GZIP.open(compressed)
        assert compressed.tell() == 0

    def test_compress_then_open_identity(self):
        compressed = Codec.ZSTD.compress(io.BytesIO(PAYLOAD))
        assert Codec.ZSTD.open(compressed).read() == PAYLOAD

    def test_stream_api_accepts_bytesio_wrapper(self):
        compressed = Codec.GZIP.compress(BytesIO(SMALL))
        assert Codec.GZIP.open(compressed).read() == SMALL


# ===========================================================================
# Conditional helpers — compress_with / decompress_with
# ===========================================================================

class TestCodecConditionalHelpers:
    def test_compress_with_none_passthrough(self):
        assert Codec.compress_with(SMALL, None) is SMALL

    def test_decompress_with_none_passthrough(self):
        assert Codec.decompress_with(SMALL, None) is SMALL

    def test_compress_with_string_codec(self):
        result = Codec.compress_with(SMALL, "gzip")
        assert Codec.from_bytes(result) is Codec.GZIP

    def test_decompress_with_string_codec(self):
        compressed = Codec.GZIP.compress_bytes(SMALL)
        assert Codec.decompress_with(compressed, "gzip") == SMALL

    def test_compress_with_codec_instance(self):
        result = Codec.compress_with(SMALL, Codec.ZSTD)
        assert Codec.from_bytes(result) is Codec.ZSTD

    @pytest.mark.parametrize("codec_str", ["gzip", "bzip2", "xz"])
    def test_symmetric_roundtrip(self, codec_str):
        compressed = Codec.compress_with(SMALL, codec_str)
        assert Codec.decompress_with(compressed, codec_str) == SMALL


# ===========================================================================
# Roundtrip diagnostic
# ===========================================================================

class TestCodecRoundtrip:
    @pytest.mark.parametrize("codec", [Codec.GZIP, Codec.BZIP2, Codec.XZ])
    def test_roundtrip_stdlib(self, codec):
        assert codec.roundtrip(PAYLOAD) is True

    def test_roundtrip_zstd(self):
        assert Codec.ZSTD.roundtrip(PAYLOAD) is True

    def test_roundtrip_empty(self):
        assert Codec.GZIP.roundtrip(b"") is True


# ===========================================================================
# _peek / _peek_buf helpers
# ===========================================================================

class TestPeekHelpers:
    def test_peek_returns_bytes_and_restores_cursor(self):
        stream = io.BytesIO(b"ABCDEF")
        stream.seek(2)
        result = _peek(stream, 3)
        assert result == b"CDE"
        assert stream.tell() == 2

    def test_peek_short_stream_does_not_raise(self):
        stream = io.BytesIO(b"AB")
        result = _peek(stream, 10)
        assert result == b"AB"
        assert stream.tell() == 0

    def test_peek_buf_delegates_to_underlying_buffer(self):
        buf = BytesIO(b"HELLO")
        result = _peek_buf(buf, 3)
        assert result == b"HEL"
        assert buf.tell() == 0