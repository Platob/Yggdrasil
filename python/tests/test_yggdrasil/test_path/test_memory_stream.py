"""Tests for :class:`yggdrasil.io.memory_stream.MemoryStream`."""
from __future__ import annotations

import gzip
import io
import zlib

import pytest

from yggdrasil.io.memory_stream import MemoryStream


class TestSourceTypes:
    def test_from_bytes(self) -> None:
        s = MemoryStream(b"hello bytes", byte_size=256)
        assert s.read_bytes() == b"hello bytes"
        assert s.eof

    def test_from_callable(self) -> None:
        chunks = iter([b"chunk1", b"chunk2", b""])
        s = MemoryStream(lambda n: next(chunks, b""), byte_size=256)
        assert s.read_bytes() == b"chunk1chunk2"
        assert s.eof

    def test_from_file_like(self) -> None:
        bio = io.BytesIO(b"file-like data")
        s = MemoryStream(bio, byte_size=256)
        assert s.read_bytes() == b"file-like data"
        assert s.eof

    def test_from_iterable(self) -> None:
        def gen():
            yield b"part1"
            yield b"part2"
            yield b"part3"

        s = MemoryStream(gen(), byte_size=256)
        assert s.read_bytes() == b"part1part2part3"
        assert s.eof

    def test_from_none_empty(self) -> None:
        s = MemoryStream(None, byte_size=256)
        assert s.size == 0
        assert s.eof
        assert s.read_bytes() == b""

    def test_from_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="cannot be a str"):
            MemoryStream("not allowed", byte_size=256)


class TestReadWrite:
    def test_read_mv_returns_data(self) -> None:
        s = MemoryStream(b"abcdefgh", byte_size=256)
        mv = s.read_mv(4, 0)
        assert bytes(mv) == b"abcd"

    def test_read_mv_with_offset(self) -> None:
        s = MemoryStream(b"abcdefgh", byte_size=256)
        mv = s.read_mv(3, 5)
        assert bytes(mv) == b"fgh"

    def test_sequential_reads_advance_position(self) -> None:
        s = MemoryStream(b"abcdefghij", byte_size=256)
        first = s.read_mv(3, 0, cursor=True)
        assert bytes(first) == b"abc"
        second = s.read_mv(4, 0, cursor=True)
        assert bytes(second) == b"defg"
        third = s.read_mv(3, 0, cursor=True)
        assert bytes(third) == b"hij"

    def test_write_bytes_appends(self) -> None:
        s = MemoryStream(byte_size=256)
        s.write_bytes(b"hello", offset=0)
        s.write_bytes(b" world", offset=5)
        assert s.read_bytes() == b"hello world"

    def test_size_tracks_total_bytes(self) -> None:
        s = MemoryStream(byte_size=256)
        assert s.size == 0
        s.write_bytes(b"12345", offset=0)
        assert s.size == 5
        s.write_bytes(b"678", offset=5)
        assert s.size == 8

    def test_eof_flag(self) -> None:
        chunks = iter([b"data", b""])
        s = MemoryStream(lambda n: next(chunks, b""), byte_size=256)
        assert not s.eof
        s.read_bytes()
        assert s.eof


class TestContentEncoding:
    def test_gzip_encoding_decompresses_on_read(self) -> None:
        original = b"hello gzip world" * 10
        compressed = gzip.compress(original)
        source = io.BytesIO(compressed)
        s = MemoryStream(source, content_encoding="gzip", byte_size=4096)
        result = s.read_bytes()
        assert result == original

    def test_deflate_encoding_decompresses(self) -> None:
        original = b"hello deflate world" * 10
        compressed = zlib.compress(original)
        source = io.BytesIO(compressed)
        s = MemoryStream(source, content_encoding="deflate", byte_size=4096)
        result = s.read_bytes()
        assert result == original

    def test_unknown_encoding_passes_through(self) -> None:
        raw = b"raw bytes untouched"
        source = io.BytesIO(raw)
        # "bogus-encoding" won't resolve to any known codec, so
        # Codec.from_ returns None (via default=None) and the source
        # read function stays as-is.
        s = MemoryStream(source, content_encoding="bogus-encoding", byte_size=256)
        result = s.read_bytes()
        assert result == raw

    def test_none_encoding_no_op(self) -> None:
        raw = b"no encoding applied"
        source = io.BytesIO(raw)
        s = MemoryStream(source, content_encoding=None, byte_size=256)
        result = s.read_bytes()
        assert result == raw


class TestSpill:
    def test_data_below_threshold_stays_in_memory(self) -> None:
        payload = b"x" * 50
        s = MemoryStream(payload, byte_size=4096, spill_threshold=1024)
        s._pull_to_eof()
        assert not s.has_spill
        assert s.window_start == 0

    def test_data_above_threshold_spills_to_disk(self) -> None:
        # 300 bytes total, in-memory cap 64, total budget 4096.
        # Cold bytes should spill to disk.
        payload = b"y" * 300
        s = MemoryStream(payload, byte_size=4096, spill_threshold=64)
        s._pull_to_eof()
        assert s.has_spill
        assert s.window_start > 0

    def test_spilled_data_still_readable(self) -> None:
        payload = bytes(range(256)) + bytes(range(256))  # 512 bytes
        s = MemoryStream(payload, byte_size=4096, spill_threshold=100)
        s._pull_to_eof()
        assert s.has_spill
        # Spill region is still readable.
        assert s.read_bytes(size=10, offset=0) == payload[:10]
        # Cross-boundary read spanning spill and memory.
        mid = s.window_start - 5
        assert s.read_bytes(size=10, offset=mid) == payload[mid:mid + 10]
        # Pure in-memory read.
        tail = s.window_start + 10
        assert s.read_bytes(size=10, offset=tail) == payload[tail:tail + 10]

    def test_spill_file_cleaned_up_on_close(self) -> None:
        s = MemoryStream(b"z" * 300, byte_size=4096, spill_threshold=64)
        s._pull_to_eof()
        assert s.has_spill
        spill_file = s._spill_file
        assert spill_file is not None
        assert not spill_file.closed
        s.clear()
        assert not s.has_spill
        assert s._spill_file is None


class TestSlidingWindow:
    def test_evicted_bytes_raise_on_reread(self) -> None:
        # 200 bytes through a 32-byte window (byte_size=32, no spill
        # since byte_size <= spill_threshold by default).
        payload = b"a" * 200
        s = MemoryStream(payload, byte_size=32)
        s._pull_to_eof()
        assert s.window_start > 0
        with pytest.raises(ValueError, match="behind the retained region"):
            s.read_bytes(size=10, offset=0)

    def test_window_start_advances_with_eviction(self) -> None:
        payload = b"b" * 500
        s = MemoryStream(payload, byte_size=64)
        s._pull_to_eof()
        assert s.size == 500
        assert s.window_start == 500 - 64
        assert s.window_end == 500
        # Only the last 64 bytes are in memory.
        assert len(s.memoryview()) == 64

    def test_byte_size_caps_total_retention(self) -> None:
        # With spill enabled (byte_size > spill_threshold), the total
        # retained bytes (spill + memory) never exceeds byte_size.
        payload = b"c" * 1000
        s = MemoryStream(payload, byte_size=200, spill_threshold=50)
        s._pull_to_eof()
        retained = s.size - s.spill_start
        assert retained <= 200
        # Oldest bytes are gone.
        assert s.spill_start > 0


class TestEdgeCases:
    def test_empty_source_yields_empty_reads(self) -> None:
        s = MemoryStream(b"", byte_size=256)
        assert s.read_bytes() == b""
        assert s.size == 0
        assert s.eof

    def test_large_read_on_small_source_returns_available(self) -> None:
        s = MemoryStream(b"tiny", byte_size=256)
        result = s.read_bytes(size=10000, offset=0)
        assert result == b"tiny"
        assert s.eof

    def test_multiple_sources_callable_then_manual_write(self) -> None:
        chunks = iter([b"auto", b""])
        s = MemoryStream(lambda n: next(chunks, b""), byte_size=256)
        # Drain the callable source completely (size=-1 pulls to EOF).
        assert s.read_bytes() == b"auto"
        assert s.eof
        # Then manually append more bytes.
        s.write_bytes(b"_manual", offset=4)
        assert s.size == 11
        assert s.read_bytes() == b"auto_manual"
