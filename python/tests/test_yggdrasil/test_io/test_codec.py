"""Unit tests for :mod:`yggdrasil.io.enums.codec`.

Covers:

* Bytes roundtrip for every codec (gzip, zstd, lz4, bz2, xz, snappy,
  brotli, zlib, lzma).
* Streaming roundtrip for codecs that advertise ``is_streaming``.
* ``is_streaming`` property per codec.
* ``Codec.parse`` — codec passthrough, None→default, short names,
  mime strings, invalid inputs.
* ``Codec.from_mime`` / ``Codec.all``.
* ``read_start_end`` — head/tail extraction across streaming and
  bytes-only paths, zero-length cases, payloads smaller than the
  requested window.
* Source-cursor preservation on compress/decompress/read_start_end.
* ``_drain`` helper correctness.
* ``_ZlibStreamReader`` / ``_ZlibStreamWriter`` directly (small reads,
  read(-1), close does not close underlying fh).
* Edge cases: empty input, data at and above the streaming chunk size.

Codecs that depend on runtime-installed packages (zstd, lz4, snappy,
brotli) are auto-skipped when the underlying library isn't importable.
"""
from __future__ import annotations

import io as _io
import zlib
from typing import Callable, List

import pytest

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.enums.codec import (
    BROTLI,
    BZIP2,
    Codec,
    GZIP,
    LZ4,
    LZMA,
    SNAPPY,
    XZ,
    ZLIB,
    ZSTD,
    _CHUNK,
    _ZlibStreamReader,
    _ZlibStreamWriter,
    _drain,
)


# =====================================================================
# Helpers
# =====================================================================

def _is_available(codec: Codec) -> bool:
    """True when the codec's underlying library is importable."""
    try:
        # Trigger the runtime import by calling compress_bytes on a
        # trivial input. If the library is missing, this raises.
        codec.compress_bytes(b"x")
        return True
    except Exception:
        return False


# Discover availability once at collection time.
_AVAILABLE: dict[str, bool] = {c.name: _is_available(c) for c in Codec.all()}


def _require(codec: Codec) -> None:
    if not _AVAILABLE.get(codec.name, False):
        pytest.skip(f"{codec.name} not available in this environment")


# All codecs we want to parametrize over. Skipping happens per-test
# based on runtime availability.
ALL_CODECS: List[Codec] = Codec.all()

# Streaming codecs (the ones that claim is_streaming=True by design).
STREAMING_CODECS: List[Codec] = [GZIP, ZSTD, LZ4, BZIP2, XZ, ZLIB, LZMA]

# Bytes-only codecs (no streaming).
BYTES_ONLY_CODECS: List[Codec] = [SNAPPY, BROTLI]


def _codec_id(c: Codec) -> str:
    return c.name


# =====================================================================
# Bytes roundtrip (all codecs)
# =====================================================================

class TestBytesRoundtrip:
    @pytest.mark.parametrize("codec", ALL_CODECS, ids=_codec_id)
    def test_empty_input(self, codec: Codec):
        _require(codec)
        compressed = codec.compress_bytes(b"")
        recovered = codec.decompress_bytes(compressed)
        assert recovered == b""

    @pytest.mark.parametrize("codec", ALL_CODECS, ids=_codec_id)
    def test_small_payload(self, codec: Codec):
        _require(codec)
        payload = b"hello, commodity trading"
        compressed = codec.compress_bytes(payload)
        assert codec.decompress_bytes(compressed) == payload

    @pytest.mark.parametrize("codec", ALL_CODECS, ids=_codec_id)
    def test_repetitive_payload_compresses(self, codec: Codec):
        """Highly repetitive data should compress to smaller than input."""
        _require(codec)
        payload = b"a" * 10_000
        compressed = codec.compress_bytes(payload)
        # Compression must not expand repetitive input meaningfully.
        # (Some codecs add small framing overhead — 50% is a generous
        # upper bound that all codecs easily beat on this input.)
        assert len(compressed) < len(payload) // 2

    @pytest.mark.parametrize("codec", ALL_CODECS, ids=_codec_id)
    def test_roundtrip_method(self, codec: Codec):
        _require(codec)
        assert codec.roundtrip(b"sample data") is True

    @pytest.mark.parametrize("codec", ALL_CODECS, ids=_codec_id)
    def test_binary_payload(self, codec: Codec):
        """Arbitrary binary bytes must roundtrip losslessly."""
        _require(codec)
        payload = bytes(range(256)) * 100  # all byte values
        compressed = codec.compress_bytes(payload)
        assert codec.decompress_bytes(compressed) == payload


# =====================================================================
# Streaming roundtrip via compress/decompress
# =====================================================================

class TestStreamingRoundtrip:
    @pytest.mark.parametrize("codec", STREAMING_CODECS, ids=_codec_id)
    def test_single_chunk_payload(self, codec: Codec):
        _require(codec)
        payload = b"small payload under one chunk"
        src = BytesIO(payload)

        compressed = codec.compress(src)
        assert isinstance(compressed, BytesIO)

        recovered = codec.decompress(compressed)
        assert recovered.to_bytes() == payload

    @pytest.mark.parametrize("codec", STREAMING_CODECS, ids=_codec_id)
    def test_multi_chunk_payload(self, codec: Codec):
        """Payload large enough to span multiple streaming chunks."""
        _require(codec)
        payload = b"commodity trading " * 200_000  # ~3.4 MB, >> _CHUNK
        assert len(payload) > _CHUNK * 2

        src = BytesIO(payload)
        compressed = codec.compress(src)
        recovered = codec.decompress(compressed)
        assert recovered.to_bytes() == payload

    @pytest.mark.parametrize("codec", STREAMING_CODECS, ids=_codec_id)
    def test_exact_chunk_size_boundary(self, codec: Codec):
        """Payload exactly equal to chunk size — boundary condition."""
        _require(codec)
        payload = b"x" * _CHUNK
        src = BytesIO(payload)
        compressed = codec.compress(src)
        recovered = codec.decompress(compressed)
        assert recovered.to_bytes() == payload

    @pytest.mark.parametrize("codec", STREAMING_CODECS, ids=_codec_id)
    def test_empty_streaming(self, codec: Codec):
        _require(codec)
        src = BytesIO(b"")
        compressed = codec.compress(src)
        recovered = codec.decompress(compressed)
        assert recovered.to_bytes() == b""

    @pytest.mark.parametrize("codec", STREAMING_CODECS, ids=_codec_id)
    def test_streaming_output_matches_bytes(self, codec: Codec):
        """Stream-decompressed output equals bytes-decompressed output.

        The stream path and the bytes path must agree on the decoded
        result — otherwise the streaming wrapper has a bug.
        """
        _require(codec)
        payload = b"commodity trading " * 10_000

        # Via streaming
        src = BytesIO(payload)
        streamed_compressed = codec.compress(src).to_bytes()

        # The bytes-decompress of the streaming-compressed output
        # should equal the original payload.
        assert codec.decompress_bytes(streamed_compressed) == payload

        # Streaming decompress of the bytes-compressed output
        # should also equal the original payload.
        bytes_compressed = codec.compress_bytes(payload)
        streamed_recovered = codec.decompress(BytesIO(bytes_compressed))
        assert streamed_recovered.to_bytes() == payload


# =====================================================================
# Fallback: non-streaming codecs use bytes path
# =====================================================================

class TestBytesOnlyFallback:
    @pytest.mark.parametrize("codec", BYTES_ONLY_CODECS, ids=_codec_id)
    def test_compress_decompress_via_bytes_fallback(self, codec: Codec):
        """Snappy / brotli have no streaming but compress/decompress
        must still work via the bytes fallback."""
        _require(codec)
        payload = b"payload"
        src = BytesIO(payload)
        compressed = codec.compress(src)
        recovered = codec.decompress(compressed)
        assert recovered.to_bytes() == payload


# =====================================================================
# is_streaming property
# =====================================================================

class TestIsStreaming:
    @pytest.mark.parametrize("codec", STREAMING_CODECS, ids=_codec_id)
    def test_streaming_codecs_report_true(self, codec: Codec):
        assert codec.is_streaming is True

    @pytest.mark.parametrize("codec", BYTES_ONLY_CODECS, ids=_codec_id)
    def test_bytes_only_codecs_report_false(self, codec: Codec):
        assert codec.is_streaming is False


# =====================================================================
# Cursor preservation
# =====================================================================

class TestCursorPreservation:
    def test_compress_restores_cursor(self):
        payload = b"commodity trading data"
        src = BytesIO(payload)
        src.seek(5)

        GZIP.compress(src)
        # Cursor should be back at where we left it.
        assert src.tell() == 5

    def test_decompress_restores_cursor(self):
        payload = b"commodity trading data"
        compressed = GZIP.compress_bytes(payload)
        src = BytesIO(compressed)
        src.seek(3)

        GZIP.decompress(src)
        assert src.tell() == 3

    def test_read_start_end_restores_cursor(self):
        payload = b"commodity trading data" * 100
        compressed = GZIP.compress_bytes(payload)
        src = BytesIO(compressed)
        src.seek(7)

        GZIP.read_start_end(src, n_start=8, n_end=8)
        assert src.tell() == 7


# =====================================================================
# _drain helper
# =====================================================================

class TestDrain:
    def test_drains_from_current_cursor(self):
        fh = _io.BytesIO(b"0123456789")
        fh.seek(3)
        assert _drain(fh) == b"3456789"

    def test_preserves_cursor_after_drain(self):
        fh = _io.BytesIO(b"0123456789")
        fh.seek(4)
        _drain(fh)
        assert fh.tell() == 4

    def test_drain_at_eof_returns_empty(self):
        fh = _io.BytesIO(b"0123")
        fh.seek(4)
        assert _drain(fh) == b""
        assert fh.tell() == 4

    def test_drain_from_zero_reads_all(self):
        fh = _io.BytesIO(b"0123456789")
        fh.seek(0)
        assert _drain(fh) == b"0123456789"
        assert fh.tell() == 0


# =====================================================================
# read_start_end
# =====================================================================

class TestReadStartEnd:
    @pytest.mark.parametrize("codec", [GZIP, BZIP2, XZ, ZLIB, LZMA], ids=_codec_id)
    def test_streaming_head_tail(self, codec: Codec):
        payload = b"ABCDEFGHIJ" * 1000  # 10k bytes of known pattern
        compressed = codec.compress_bytes(payload)

        head, tail = codec.read_start_end(compressed, n_start=16, n_end=16)
        assert head == payload[:16]
        assert tail == payload[-16:]

    @pytest.mark.parametrize("codec", BYTES_ONLY_CODECS, ids=_codec_id)
    def test_bytes_only_head_tail(self, codec: Codec):
        """Snappy/brotli read_start_end uses the bytes fallback."""
        _require(codec)
        payload = b"ABCDEFGHIJ" * 100
        compressed = codec.compress_bytes(payload)

        head, tail = codec.read_start_end(compressed, n_start=8, n_end=8)
        assert head == payload[:8]
        assert tail == payload[-8:]

    def test_zero_start_zero_end_short_circuit(self):
        """n_start=0 and n_end=0 must short-circuit without opening a reader."""
        payload = b"x" * 1000
        compressed = GZIP.compress_bytes(payload)
        head, tail = GZIP.read_start_end(compressed, n_start=0, n_end=0)
        assert head == b""
        assert tail == b""

    def test_zero_start_only_tail(self):
        payload = b"0123456789" * 100
        compressed = GZIP.compress_bytes(payload)
        head, tail = GZIP.read_start_end(compressed, n_start=0, n_end=10)
        assert head == b""
        assert tail == payload[-10:]

    def test_zero_end_only_head(self):
        payload = b"0123456789" * 100
        compressed = GZIP.compress_bytes(payload)
        head, tail = GZIP.read_start_end(compressed, n_start=10, n_end=0)
        assert head == payload[:10]
        assert tail == b""

    def test_negative_raises(self):
        compressed = GZIP.compress_bytes(b"data")
        with pytest.raises(ValueError):
            GZIP.read_start_end(compressed, n_start=-1, n_end=0)
        with pytest.raises(ValueError):
            GZIP.read_start_end(compressed, n_start=0, n_end=-1)

    def test_window_larger_than_payload(self):
        """When requested windows exceed the payload, we get what we've got."""
        payload = b"short"
        compressed = GZIP.compress_bytes(payload)
        head, tail = GZIP.read_start_end(compressed, n_start=100, n_end=100)
        assert head == payload
        # tail is payload[-100:] which is the whole payload since
        # len(payload) < 100.
        assert tail == payload

    def test_accepts_bytes_input(self):
        payload = b"0123456789" * 100
        compressed = GZIP.compress_bytes(payload)
        # Pass raw bytes (not a BytesIO).
        head, tail = GZIP.read_start_end(compressed, n_start=5, n_end=5)
        assert head == payload[:5]
        assert tail == payload[-5:]


# =====================================================================
# Codec.parse
# =====================================================================

class TestParse:
    def test_codec_instance_passthrough(self):
        assert Codec.parse(GZIP) is GZIP

    def test_none_returns_default(self):
        assert Codec.parse(None) is None
        assert Codec.parse(None, default=GZIP) is GZIP

    def test_short_name_gzip(self):
        assert Codec.parse("gzip") is GZIP

    def test_short_name_zstd(self):
        assert Codec.parse("zstd") is ZSTD

    def test_short_name_case_insensitive(self):
        assert Codec.parse("GZIP") is GZIP
        assert Codec.parse("  gzip  ") is GZIP
        assert Codec.parse("GZip") is GZIP

    def test_unknown_short_name_falls_through_to_mime_or_default(self):
        # "notacodec" is neither a short name nor a valid mime → default.
        assert Codec.parse("notacodec") is None
        assert Codec.parse("notacodec", default=GZIP) is GZIP

    def test_invalid_input_returns_default(self):
        assert Codec.parse(42, default=GZIP) is GZIP

    def test_from_mime_for_gzip(self):
        assert Codec.from_mime(GZIP.mime_type) is GZIP

    def test_all_returns_every_codec(self):
        all_codecs = Codec.all()
        assert GZIP in all_codecs
        assert ZSTD in all_codecs
        assert ZLIB in all_codecs
        assert len(all_codecs) == 9  # gzip, zstd, lz4, bz2, xz, snappy, brotli, zlib, lzma

    def test_all_returns_fresh_list(self):
        """Codec.all() must return a new list — mutating it mustn't
        affect the internal registry."""
        all_1 = Codec.all()
        all_1.clear()
        all_2 = Codec.all()
        assert len(all_2) == 9


# =====================================================================
# Codec metadata
# =====================================================================

class TestCodecMetadata:
    def test_name_is_non_empty(self):
        for codec in Codec.all():
            assert isinstance(codec.name, str)
            assert codec.name

    def test_extensions_from_mime(self):
        assert GZIP.extensions == list(GZIP.mime_type.extensions)

    def test_extension_from_mime(self):
        assert GZIP.extension == GZIP.mime_type.extension

    def test_repr(self):
        assert repr(GZIP) == "<Codec:gzip>"
        assert repr(ZSTD) == "<Codec:zstd>"


# =====================================================================
# _ZlibStreamReader / _ZlibStreamWriter
# =====================================================================

class TestZlibStreamClasses:
    def test_writer_produces_stdlib_compatible_output(self):
        """Output from _ZlibStreamWriter must be readable by stdlib zlib."""
        payload = b"zlib stream roundtrip payload"
        sink = _io.BytesIO()
        w = _ZlibStreamWriter(sink)
        w.write(payload)
        w.close()

        recovered = zlib.decompress(sink.getvalue())
        assert recovered == payload

    def test_reader_handles_stdlib_compressed(self):
        """_ZlibStreamReader must decode stdlib-compressed data."""
        payload = b"zlib stream roundtrip payload"
        compressed = zlib.compress(payload)

        src = _io.BytesIO(compressed)
        r = _ZlibStreamReader(src)
        recovered = r.read(-1)
        assert recovered == payload

    def test_reader_small_reads(self):
        """Reader must serve small reads correctly from its internal buffer."""
        payload = b"X" * 10_000
        compressed = zlib.compress(payload)

        r = _ZlibStreamReader(_io.BytesIO(compressed))
        pieces = []
        while True:
            piece = r.read(100)
            if not piece:
                break
            pieces.append(piece)

        assert b"".join(pieces) == payload

    def test_reader_read_negative(self):
        """read(-1) drains everything."""
        payload = b"Y" * 5_000
        compressed = zlib.compress(payload)
        r = _ZlibStreamReader(_io.BytesIO(compressed))
        assert r.read(-1) == payload

    def test_reader_read_none(self):
        """read(None) also drains everything (RawIOBase convention)."""
        payload = b"Z" * 5_000
        compressed = zlib.compress(payload)
        r = _ZlibStreamReader(_io.BytesIO(compressed))
        assert r.read(None) == payload

    def test_reader_empty_input(self):
        """Empty compressed input → empty output, no errors."""
        compressed = zlib.compress(b"")
        r = _ZlibStreamReader(_io.BytesIO(compressed))
        assert r.read(-1) == b""

    def test_reader_readable(self):
        r = _ZlibStreamReader(_io.BytesIO(zlib.compress(b"x")))
        assert r.readable() is True

    def test_writer_writable(self):
        w = _ZlibStreamWriter(_io.BytesIO())
        assert w.writable() is True

    def test_writer_write_returns_input_length(self):
        """RawIOBase convention: write returns the number of bytes consumed."""
        w = _ZlibStreamWriter(_io.BytesIO())
        n = w.write(b"hello")
        # zlib buffers internally and may not flush — but write() should
        # report all input bytes as consumed.
        assert n == 5

    def test_writer_empty_write(self):
        w = _ZlibStreamWriter(_io.BytesIO())
        assert w.write(b"") == 0

    def test_writer_close_does_not_close_fh(self):
        """Closing the writer must NOT close the underlying fh."""
        sink = _io.BytesIO()
        w = _ZlibStreamWriter(sink)
        w.write(b"data")
        w.close()
        # sink must still be usable.
        assert sink.closed is False
        sink.write(b"more")

    def test_reader_close_does_not_close_fh(self):
        compressed = zlib.compress(b"data")
        src = _io.BytesIO(compressed)
        r = _ZlibStreamReader(src)
        r.read(-1)
        r.close()
        assert src.closed is False

    def test_writer_write_after_close_raises(self):
        w = _ZlibStreamWriter(_io.BytesIO())
        w.write(b"data")
        w.close()
        with pytest.raises(ValueError):
            w.write(b"more")

    def test_writer_close_idempotent(self):
        """Closing twice must not corrupt output nor raise."""
        sink = _io.BytesIO()
        w = _ZlibStreamWriter(sink)
        w.write(b"data")
        w.close()
        first_output = sink.getvalue()
        w.close()  # second close
        # Output must not have changed.
        assert sink.getvalue() == first_output

    def test_large_roundtrip_via_classes(self):
        """Large payload through writer → reader roundtrips."""
        payload = b"commodity " * 500_000  # 5 MB
        sink = _io.BytesIO()
        w = _ZlibStreamWriter(sink)
        w.write(payload)
        w.close()

        r = _ZlibStreamReader(_io.BytesIO(sink.getvalue()))
        assert r.read(-1) == payload


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_is_streaming_type_consistent(self):
        """is_streaming returns a real bool."""
        for codec in Codec.all():
            assert isinstance(codec.is_streaming, bool)

    def test_parse_accepts_mime_string(self):
        """Codec.parse on a mime string should resolve via mime path."""
        # Use gzip's mime value string.
        result = Codec.parse(GZIP.mime_type.value)
        assert result is GZIP

    def test_compress_large_input_is_bounded_memory(self):
        """Streaming codecs should be able to handle input >> _CHUNK.

        This is a smoke test — we don't measure memory, just verify
        the operation completes without exception.
        """
        payload = b"x" * (_CHUNK * 5)  # 5 chunks worth
        src = BytesIO(payload)
        compressed = GZIP.compress(src)
        recovered = GZIP.decompress(compressed)
        assert recovered.to_bytes() == payload