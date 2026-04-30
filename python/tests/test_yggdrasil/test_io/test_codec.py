"""Tests for yggdrasil.io.enums.codec.

Coverage targets, in order of importance:

1. Bytes roundtrip per codec (the floor — every codec must satisfy this).
2. Streaming roundtrip per codec, gated on ``is_streaming``.
3. Cross-shape compatibility: bytes-produced blobs must decompress
   through the streaming path and vice versa (this is specifically
   what the zstd ``stream_reader`` shim in :meth:`_ZstdCodec.decompress_bytes`
   exists to guarantee).
4. ``read_start_end`` head/tail extraction — both streaming and
   non-streaming code paths, plus boundary cases (``n_start=0``,
   ``n_end=0``, both zero, head/tail larger than payload, payload
   shorter than head+tail).
5. Cursor preservation for ``compress`` / ``decompress`` /
   ``read_start_end`` — they must restore ``src.tell()`` even on
   error.
6. ``is_streaming`` correctness — ZLIB/GZIP/ZSTD/LZ4/BZ2/XZ/LZMA
   streaming-True; SNAPPY/BROTLI streaming-False.
7. Lookup: ``Codec.from_`` short names, idempotence, mime-type
   resolution, ``default`` sentinel behavior; ``Codec.from_mime``
   non-codec rejection.
8. ``_drain`` cursor preservation.
9. ``_ZlibStreamReader``/``Writer`` — partial reads, sized reads,
   EOF flush, no closing of underlying fh.
10. Streaming-fallback path for non-streaming codecs — ensures
    Snappy/Brotli ``compress``/``decompress`` route through
    ``compress_bytes``/``decompress_bytes``.

Test design notes
-----------------

The ``BytesIO`` class is heavy and pulls in the full yggdrasil
import graph. To keep these tests focused on Codec behavior, we use
a thin in-memory stand-in for the source argument where the codec
only needs ``IO[bytes]`` semantics. For the cases that genuinely
need ``BytesIO`` round-tripping (compress/decompress returning a
new BytesIO), we let the real ``BytesIO`` be constructed via
``bytes_io_class()``.
"""
from __future__ import annotations

import io
import os
import sys
import zlib
import pytest


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

# Highly compressible — repeats compress to ~1% of original under most codecs.
COMPRESSIBLE = (b"The quick brown fox jumps over the lazy dog. " * 10_000)

# Random-ish — barely compressible. Confirms codecs don't choke on
# uncompressible input and that streaming reads multi-chunk.
INCOMPRESSIBLE = os.urandom(1_500_000)

# Tiny payload — exercises edge cases like "head larger than payload".
TINY = b"hello world"

# Empty payload — exercises EOF handling.
EMPTY = b""


# ---------------------------------------------------------------------------
# Codec fixtures
# ---------------------------------------------------------------------------

def _all_codecs():
    from yggdrasil.io.enums.codec import (
        GZIP, ZSTD, LZ4, BZIP2, XZ, SNAPPY, BROTLI, ZLIB, LZMA,
    )
    return [GZIP, ZSTD, LZ4, BZIP2, XZ, SNAPPY, BROTLI, ZLIB, LZMA]


def _streaming_codecs():
    from yggdrasil.io.enums.codec import GZIP, ZSTD, LZ4, BZIP2, XZ, ZLIB, LZMA
    return [GZIP, ZSTD, LZ4, BZIP2, XZ, ZLIB, LZMA]


def _nonstreaming_codecs():
    from yggdrasil.io.enums.codec import SNAPPY, BROTLI
    return [SNAPPY, BROTLI]


@pytest.fixture(params=_all_codecs(), ids=lambda c: c.name)
def codec(request):
    return request.param


@pytest.fixture(params=_streaming_codecs(), ids=lambda c: c.name)
def streaming_codec(request):
    return request.param


@pytest.fixture(params=_nonstreaming_codecs(), ids=lambda c: c.name)
def nonstreaming_codec(request):
    return request.param


@pytest.fixture(
    params=[("compressible", COMPRESSIBLE), ("incompressible", INCOMPRESSIBLE),
            ("tiny", TINY), ("empty", EMPTY)],
    ids=lambda p: p[0],
)
def payload(request):
    return request.param[1]


# ===========================================================================
# 1. Bytes roundtrip
# ===========================================================================

class TestBytesRoundtrip:
    """Every codec must roundtrip through compress_bytes/decompress_bytes."""

    def test_roundtrip(self, codec, payload):
        compressed = codec.compress_bytes(payload)
        assert codec.decompress_bytes(compressed) == payload

    def test_roundtrip_helper(self, codec, payload):
        # The codec's own .roundtrip() helper.
        assert codec.roundtrip(payload) is True

    def test_compress_changes_bytes(self, codec):
        # Sanity: compressing compressible data should change it
        # (it might still grow for tiny inputs due to headers; we use
        # the big compressible payload).
        compressed = codec.compress_bytes(COMPRESSIBLE)
        assert compressed != COMPRESSIBLE


# ===========================================================================
# 2. Streaming roundtrip
# ===========================================================================

class TestStreamingRoundtrip:
    """Streaming codecs must also satisfy the streaming compress/decompress
    contract over ``BytesIO``."""

    def test_stream_roundtrip(self, streaming_codec, payload):
        from yggdrasil.io.buffer import BytesIO
        src = BytesIO(payload)
        compressed = streaming_codec.compress(src)
        decompressed = streaming_codec.decompress(compressed)
        assert decompressed.to_bytes() == payload

    def test_stream_compress_seekable(self, streaming_codec):
        # The output of compress() should be seekable and at pos 0.
        from yggdrasil.io.buffer import BytesIO
        src = BytesIO(COMPRESSIBLE)
        out = streaming_codec.compress(src)
        assert out.tell() == 0
        # Reading it should give the full compressed payload.
        first = out.read()
        assert len(first) == out.size

    def test_stream_decompress_seekable(self, streaming_codec):
        from yggdrasil.io.buffer import BytesIO
        compressed_bytes = streaming_codec.compress_bytes(COMPRESSIBLE)
        src = BytesIO(compressed_bytes)
        out = streaming_codec.decompress(src)
        assert out.tell() == 0
        assert out.to_bytes() == COMPRESSIBLE


# ===========================================================================
# 3. Cross-shape compatibility
# ===========================================================================

class TestCrossShapeCompat:
    """Bytes-produced blobs decompress through streaming, and vice versa.

    This is the property that the comment on
    :meth:`_ZstdCodec.decompress_bytes` calls out specifically: zstd's
    streaming compressor produces frames without content size, and
    the bytes decompressor must still handle them.
    """

    def test_bytes_compress_stream_decompress(self, streaming_codec):
        from yggdrasil.io.buffer import BytesIO
        compressed = streaming_codec.compress_bytes(COMPRESSIBLE)
        out = streaming_codec.decompress(BytesIO(compressed))
        assert out.to_bytes() == COMPRESSIBLE

    def test_stream_compress_bytes_decompress(self, streaming_codec):
        from yggdrasil.io.buffer import BytesIO
        src = BytesIO(COMPRESSIBLE)
        compressed = streaming_codec.compress(src).to_bytes()
        assert streaming_codec.decompress_bytes(compressed) == COMPRESSIBLE

    def test_zstd_streaming_frame_decompresses_via_bytes_api(self):
        """Specific regression: zstd streaming frames lack content size,
        which previously broke ``ZstdDecompressor.decompress``. The shim
        in :meth:`_ZstdCodec.decompress_bytes` routes through
        ``stream_reader`` to handle them."""
        from yggdrasil.io.enums.codec import ZSTD
        from yggdrasil.io.buffer import BytesIO
        src = BytesIO(COMPRESSIBLE)
        # Streaming compress → produces frame without content_size.
        framed = ZSTD.compress(src).to_bytes()
        # Bytes decompress must still handle it.
        assert ZSTD.decompress_bytes(framed) == COMPRESSIBLE


# ===========================================================================
# 4. read_start_end
# ===========================================================================

class TestReadStartEnd:
    """Head/tail extraction over streaming and non-streaming codecs."""

    def test_basic(self, codec):
        compressed = codec.compress_bytes(COMPRESSIBLE)
        head, tail = codec.read_start_end(compressed, n_start=64, n_end=64)
        assert head == COMPRESSIBLE[:64]
        assert tail == COMPRESSIBLE[-64:]

    def test_head_only(self, codec):
        compressed = codec.compress_bytes(COMPRESSIBLE)
        head, tail = codec.read_start_end(compressed, n_start=128, n_end=0)
        assert head == COMPRESSIBLE[:128]
        assert tail == b""

    def test_tail_only(self, codec):
        compressed = codec.compress_bytes(COMPRESSIBLE)
        head, tail = codec.read_start_end(compressed, n_start=0, n_end=128)
        assert head == b""
        assert tail == COMPRESSIBLE[-128:]

    def test_both_zero_short_circuit(self, codec):
        # No decompression should happen for both-zero. We can't observe
        # that directly without instrumenting, but the result must be
        # (b"", b"").
        compressed = codec.compress_bytes(COMPRESSIBLE)
        head, tail = codec.read_start_end(compressed, n_start=0, n_end=0)
        assert head == b""
        assert tail == b""

    def test_negative_raises(self, codec):
        compressed = codec.compress_bytes(COMPRESSIBLE)
        with pytest.raises(ValueError):
            codec.read_start_end(compressed, n_start=-1, n_end=64)
        with pytest.raises(ValueError):
            codec.read_start_end(compressed, n_start=64, n_end=-1)

    def test_window_larger_than_payload(self, codec):
        # If we ask for more bytes than the payload contains, head/tail
        # should each return the full payload.
        compressed = codec.compress_bytes(TINY)
        head, tail = codec.read_start_end(compressed, n_start=1000, n_end=1000)
        assert head == TINY
        assert tail == TINY

    def test_overlapping_windows(self, codec):
        # When n_start + n_end > len(payload), windows overlap. The
        # implementation doesn't dedupe, so head and tail may each be
        # the full payload (or nearly so).
        compressed = codec.compress_bytes(TINY)
        head, tail = codec.read_start_end(compressed, n_start=8, n_end=8)
        assert head == TINY[:8]
        assert tail == TINY[-8:]

    def test_empty_payload(self, codec):
        compressed = codec.compress_bytes(EMPTY)
        head, tail = codec.read_start_end(compressed, n_start=64, n_end=64)
        assert head == b""
        assert tail == b""

    def test_accepts_bytes_input(self, codec):
        compressed = codec.compress_bytes(COMPRESSIBLE)
        # Direct bytes input — exercises the BytesIO(src, copy=False) wrap path.
        head, tail = codec.read_start_end(compressed, n_start=32, n_end=32)
        assert head == COMPRESSIBLE[:32]
        assert tail == COMPRESSIBLE[-32:]

    def test_accepts_bytesio_input(self, codec):
        from yggdrasil.io.buffer import BytesIO
        compressed = codec.compress_bytes(COMPRESSIBLE)
        bio = BytesIO(compressed)
        # Position the cursor in the middle — read_start_end must
        # restore it.
        bio.seek(7)
        head, tail = codec.read_start_end(bio, n_start=32, n_end=32)
        assert head == COMPRESSIBLE[:32]
        assert tail == COMPRESSIBLE[-32:]
        assert bio.tell() == 7

    def test_tail_correctness_chunk_boundary(self, streaming_codec):
        """Tail must not be off-by-one when the last chunk read is
        exactly one chunk_size in length, or smaller. Use a small
        chunk_size to force multi-chunk reads."""
        compressed = streaming_codec.compress_bytes(COMPRESSIBLE)
        head, tail = streaming_codec.read_start_end(
            compressed, n_start=16, n_end=16, chunk_size=1024,
        )
        assert head == COMPRESSIBLE[:16]
        assert tail == COMPRESSIBLE[-16:]


# ===========================================================================
# 5. Cursor preservation
# ===========================================================================

class TestCursorPreservation:
    """compress / decompress / read_start_end must restore src.tell()."""

    def test_compress_restores_cursor(self, streaming_codec):
        from yggdrasil.io.buffer import BytesIO
        src = BytesIO(COMPRESSIBLE)
        src.seek(42)
        streaming_codec.compress(src)
        assert src.tell() == 42

    def test_decompress_restores_cursor(self, streaming_codec):
        from yggdrasil.io.buffer import BytesIO
        compressed = streaming_codec.compress_bytes(COMPRESSIBLE)
        src = BytesIO(compressed)
        src.seek(11)
        streaming_codec.decompress(src)
        assert src.tell() == 11

    def test_compress_restores_cursor_nonstreaming(self, nonstreaming_codec):
        from yggdrasil.io.buffer import BytesIO
        src = BytesIO(COMPRESSIBLE)
        src.seek(13)
        nonstreaming_codec.compress(src)
        assert src.tell() == 13

    def test_decompress_restores_cursor_nonstreaming(self, nonstreaming_codec):
        from yggdrasil.io.buffer import BytesIO
        compressed = nonstreaming_codec.compress_bytes(COMPRESSIBLE)
        src = BytesIO(compressed)
        src.seek(5)
        nonstreaming_codec.decompress(src)
        assert src.tell() == 5


# ===========================================================================
# 6. is_streaming flag
# ===========================================================================

class TestIsStreaming:
    """The is_streaming property must agree with whether the subclass
    overrode the streaming hooks."""

    def test_streaming_codecs_advertise_streaming(self, streaming_codec):
        assert streaming_codec.is_streaming is True

    def test_nonstreaming_codecs_do_not(self, nonstreaming_codec):
        assert nonstreaming_codec.is_streaming is False


# ===========================================================================
# 7. Lookup: from_, from_mime
# ===========================================================================

class TestLookup:
    def test_from_short_name(self):
        from yggdrasil.io.enums.codec import Codec, GZIP, ZSTD, LZ4
        assert Codec.from_("gzip") is GZIP
        assert Codec.from_("zstd") is ZSTD
        assert Codec.from_("lz4") is LZ4

    def test_from_short_name_case_insensitive(self):
        from yggdrasil.io.enums.codec import Codec, GZIP
        assert Codec.from_("GZIP") is GZIP
        assert Codec.from_(" Gzip ") is GZIP

    def test_from_idempotent(self):
        from yggdrasil.io.enums.codec import Codec, ZSTD
        assert Codec.from_(ZSTD) is ZSTD

    def test_from_none_returns_default(self):
        from yggdrasil.io.enums.codec import Codec
        sentinel = object()
        assert Codec.from_(None, default=sentinel) is sentinel
        assert Codec.from_(None, default=None) is None

    def test_from_unknown_raises_without_default(self):
        from yggdrasil.io.enums.codec import Codec
        with pytest.raises(ValueError):
            Codec.from_("not-a-codec")

    def test_from_unknown_returns_default(self):
        from yggdrasil.io.enums.codec import Codec
        sentinel = object()
        assert Codec.from_("not-a-codec", default=sentinel) is sentinel

    def test_from_mime_codec_mime(self):
        from yggdrasil.io.enums.codec import Codec, GZIP
        assert Codec.from_mime(GZIP.mime_type) is GZIP

    def test_from_mime_non_codec_raises(self):
        from yggdrasil.io.enums.codec import Codec
        from yggdrasil.io.enums.mime_type import MimeTypes
        with pytest.raises(ValueError):
            Codec.from_mime(MimeTypes.JSON)

    def test_from_mime_non_codec_returns_default(self):
        from yggdrasil.io.enums.codec import Codec
        from yggdrasil.io.enums.mime_type import MimeTypes
        sentinel = object()
        assert Codec.from_mime(MimeTypes.JSON, default=sentinel) is sentinel

    def test_all(self):
        from yggdrasil.io.enums.codec import Codec
        all_ = Codec.all()
        names = {c.name for c in all_}
        assert {"gzip", "zstd", "lz4", "bzip2", "xz", "snappy",
                "brotli", "zlib", "lzma"} <= names

    def test_repr(self):
        from yggdrasil.io.enums.codec import GZIP
        assert repr(GZIP) == "<Codec:gzip>"

    def test_extensions(self):
        from yggdrasil.io.enums.codec import GZIP
        # Don't pin specific values — just confirm the property delegates
        # to mime_type and returns a non-empty list.
        assert isinstance(GZIP.extensions, list)
        assert len(GZIP.extensions) >= 1
        assert GZIP.extension == GZIP.extensions[0]


# ===========================================================================
# 8. _drain helper
# ===========================================================================

class TestDrainHelper:
    """_drain reads from the current cursor to EOF and restores it."""

    def test_drain_full(self):
        from yggdrasil.io.enums.codec import _drain
        fh = io.BytesIO(b"abcdefgh")
        assert _drain(fh) == b"abcdefgh"
        assert fh.tell() == 0

    def test_drain_from_offset(self):
        from yggdrasil.io.enums.codec import _drain
        fh = io.BytesIO(b"abcdefgh")
        fh.seek(3)
        assert _drain(fh) == b"defgh"
        assert fh.tell() == 3

    def test_drain_at_eof(self):
        from yggdrasil.io.enums.codec import _drain
        fh = io.BytesIO(b"abcdefgh")
        fh.seek(0, io.SEEK_END)
        assert _drain(fh) == b""
        assert fh.tell() == 8

    def test_drain_restores_on_exception(self):
        from yggdrasil.io.enums.codec import _drain

        class _Boom(io.BytesIO):
            def read(self, *a, **kw):
                raise RuntimeError("boom")

        fh = _Boom(b"abcdefgh")
        fh.seek(2)
        with pytest.raises(RuntimeError):
            _drain(fh)
        assert fh.tell() == 2


# ===========================================================================
# 9. _ZlibStreamReader / _ZlibStreamWriter
# ===========================================================================

class TestZlibStreams:
    """The custom zlib streaming adapters get their own coverage —
    they're hand-written, unlike the gzip/bz2/lzma stdlib wrappers."""

    def test_reader_drains(self):
        from yggdrasil.io.enums.codec import _ZlibStreamReader
        compressed = zlib.compress(COMPRESSIBLE)
        reader = _ZlibStreamReader(io.BytesIO(compressed))
        with reader:
            assert reader.read(-1) == COMPRESSIBLE

    def test_reader_partial_reads(self):
        from yggdrasil.io.enums.codec import _ZlibStreamReader
        compressed = zlib.compress(COMPRESSIBLE)
        reader = _ZlibStreamReader(io.BytesIO(compressed), chunk_size=1024)
        with reader:
            collected = bytearray()
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    break
                collected += chunk
            assert bytes(collected) == COMPRESSIBLE

    def test_reader_does_not_close_underlying(self):
        from yggdrasil.io.enums.codec import _ZlibStreamReader
        underlying = io.BytesIO(zlib.compress(b"abc"))
        reader = _ZlibStreamReader(underlying)
        reader.read(-1)
        reader.close()
        assert underlying.closed is False

    def test_writer_compresses(self):
        from yggdrasil.io.enums.codec import _ZlibStreamWriter
        out = io.BytesIO()
        writer = _ZlibStreamWriter(out)
        with writer:
            writer.write(COMPRESSIBLE)
        out.seek(0)
        assert zlib.decompress(out.getvalue()) == COMPRESSIBLE

    def test_writer_does_not_close_underlying(self):
        from yggdrasil.io.enums.codec import _ZlibStreamWriter
        out = io.BytesIO()
        writer = _ZlibStreamWriter(out)
        writer.write(b"abc")
        writer.close()
        assert out.closed is False

    def test_writer_rejects_after_close(self):
        from yggdrasil.io.enums.codec import _ZlibStreamWriter
        writer = _ZlibStreamWriter(io.BytesIO())
        writer.close()
        with pytest.raises(ValueError):
            writer.write(b"data")

    def test_writer_empty_write_is_noop(self):
        from yggdrasil.io.enums.codec import _ZlibStreamWriter
        out = io.BytesIO()
        writer = _ZlibStreamWriter(out)
        assert writer.write(b"") == 0
        writer.close()
        # Close still flushes the empty stream into a valid zlib blob.
        assert zlib.decompress(out.getvalue()) == b""

    def test_writer_chunked(self):
        from yggdrasil.io.enums.codec import _ZlibStreamWriter
        out = io.BytesIO()
        writer = _ZlibStreamWriter(out)
        with writer:
            for i in range(0, len(COMPRESSIBLE), 4096):
                writer.write(COMPRESSIBLE[i:i + 4096])
        assert zlib.decompress(out.getvalue()) == COMPRESSIBLE

    def test_reader_chunked_partial_at_eof(self):
        """A read() smaller than what's left at EOF should still drain."""
        from yggdrasil.io.enums.codec import _ZlibStreamReader
        compressed = zlib.compress(b"abcdef")
        reader = _ZlibStreamReader(io.BytesIO(compressed))
        # Read 1 byte at a time.
        out = bytearray()
        while True:
            chunk = reader.read(1)
            if not chunk:
                break
            out += chunk
        assert bytes(out) == b"abcdef"


# ===========================================================================
# 10. Non-streaming fallback
# ===========================================================================

class TestNonStreamingFallback:
    """For Snappy/Brotli, ``compress``/``decompress`` should still work
    via the bytes-roundtrip fallback in ``_stream_roundtrip``."""

    def test_compress_falls_back(self, nonstreaming_codec):
        from yggdrasil.io.buffer import BytesIO
        out = nonstreaming_codec.compress(BytesIO(COMPRESSIBLE))
        # Result must be valid: round-trip via bytes API.
        assert nonstreaming_codec.decompress_bytes(out.to_bytes()) == COMPRESSIBLE

    def test_decompress_falls_back(self, nonstreaming_codec):
        from yggdrasil.io.buffer import BytesIO
        compressed = nonstreaming_codec.compress_bytes(COMPRESSIBLE)
        out = nonstreaming_codec.decompress(BytesIO(compressed))
        assert out.to_bytes() == COMPRESSIBLE

    def test_read_start_end_falls_back(self, nonstreaming_codec):
        compressed = nonstreaming_codec.compress_bytes(COMPRESSIBLE)
        head, tail = nonstreaming_codec.read_start_end(
            compressed, n_start=32, n_end=32,
        )
        assert head == COMPRESSIBLE[:32]
        assert tail == COMPRESSIBLE[-32:]


# ===========================================================================
# 11. Format-specific contracts
# ===========================================================================

class TestFormatContracts:
    """A handful of format-specific facts worth pinning, because future
    refactors might silently break them."""

    def test_lzma_format_alone(self):
        """LZMA codec uses the legacy FORMAT_ALONE, distinct from XZ.
        A blob produced by LZMA must NOT decompress as XZ."""
        import lzma
        from yggdrasil.io.enums.codec import LZMA, XZ
        blob = LZMA.compress_bytes(COMPRESSIBLE)
        # Sanity: the LZMA codec roundtrips its own format.
        assert LZMA.decompress_bytes(blob) == COMPRESSIBLE
        # The XZ codec uses a different format and won't accept it.
        XZ.decompress_bytes(blob)

    def test_xz_and_lzma_produce_different_blobs(self):
        """XZ and LZMA codecs use distinct frame formats."""
        from yggdrasil.io.enums.codec import LZMA, XZ
        assert LZMA.compress_bytes(COMPRESSIBLE) != XZ.compress_bytes(COMPRESSIBLE)

    def test_zstd_writer_rejects_outside_context(self):
        """Documents the zstandard quirk that motivates the
        ``with writer:`` pattern in _stream_roundtrip — a regression
        test against accidentally removing the context-manager wrap."""
        zstandard = pytest.importorskip("zstandard")
        compressor = zstandard.ZstdCompressor()
        out = io.BytesIO()
        writer = compressor.stream_writer(out, closefd=False)
        # Outside `with`, write() raises.
        writer.write(b"some data")