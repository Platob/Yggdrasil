"""Behavior tests for :class:`yggdrasil.io.memory_stream.MemoryStream`.

Pins:

* **Source shapes** — ``None``, bytes-like, file-like, callable, and
  iterable feeds all produce the same window contents.
* **Window slide** — once buffered bytes exceed ``byte_size`` the
  oldest are evicted and :attr:`window_start` advances; :attr:`size`
  always equals :attr:`window_end`.
* **Seek-within-window** — positions inside ``[window_start,
  window_end]`` are readable and re-readable; reads behind
  :attr:`window_start` raise.
* **EOF semantics** — bounded sources cap reads gracefully; ``n=-1``
  drains to EOF.
* **Manual feed** — :meth:`write_bytes` at end appends and slides;
  in-window writes overwrite; behind-window writes raise.
* **clear / truncate** — clear resets the buffer; truncate shrinks /
  zero-pads while honoring window-start.
"""

from __future__ import annotations

import io

import pytest

from yggdrasil.io.memory_stream import MemoryStream


KB = 1024


class TestSourceShapes:
    def test_none_source_is_empty_eof(self) -> None:
        s = MemoryStream(byte_size=64)
        assert s.size == 0
        assert s.eof
        assert s.read_bytes() == b""

    def test_bytes_source(self) -> None:
        s = MemoryStream(b"hello world", byte_size=64)
        assert s.read_bytes() == b"hello world"
        assert s.eof

    def test_bytearray_source(self) -> None:
        s = MemoryStream(bytearray(b"abc"), byte_size=64)
        assert s.read_bytes() == b"abc"

    def test_memoryview_source(self) -> None:
        s = MemoryStream(memoryview(b"xyz"), byte_size=64)
        assert s.read_bytes() == b"xyz"

    def test_file_like_source(self) -> None:
        s = MemoryStream(io.BytesIO(b"file bytes"), byte_size=64)
        assert s.read_bytes() == b"file bytes"

    def test_callable_source(self) -> None:
        chunks = [b"one", b"two", b""]
        i = iter(chunks)
        s = MemoryStream(lambda n: next(i, b""), byte_size=64)
        assert s.read_bytes() == b"onetwo"
        assert s.eof

    def test_iterable_source(self) -> None:
        def gen():
            yield b"abc"
            yield b"def"
            yield b"ghi"
        s = MemoryStream(gen(), byte_size=64)
        assert s.read_bytes() == b"abcdefghi"

    def test_str_source_rejected(self) -> None:
        with pytest.raises(TypeError, match="cannot be a str"):
            MemoryStream("hello", byte_size=64)

    def test_invalid_source_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be None, bytes-like"):
            MemoryStream(12345, byte_size=64)

    def test_iterable_yielding_str_rejected(self) -> None:
        s = MemoryStream(iter(["not bytes"]), byte_size=64)
        with pytest.raises(TypeError, match="expected bytes-like"):
            s.read_bytes()


class TestByteSizeValidation:
    def test_zero_byte_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="byte_size must be > 0"):
            MemoryStream(byte_size=0)

    def test_negative_byte_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="byte_size must be > 0"):
            MemoryStream(byte_size=-1)

    def test_non_int_byte_size_rejected(self) -> None:
        with pytest.raises(TypeError, match="byte_size must be an int"):
            MemoryStream(byte_size="64")  # type: ignore[arg-type]

    def test_bool_byte_size_rejected(self) -> None:
        # Booleans are ints in Python — reject them explicitly so
        # ``byte_size=True`` doesn't silently make a 1-byte window.
        with pytest.raises(TypeError, match="byte_size must be an int"):
            MemoryStream(byte_size=True)  # type: ignore[arg-type]

    def test_pull_chunk_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="pull_chunk"):
            MemoryStream(b"x", byte_size=64, pull_chunk=0)


class TestWindowSlide:
    def test_window_caps_at_byte_size(self) -> None:
        # 256 bytes fed into a 64-byte window keeps only the last 64.
        s = MemoryStream(b"x" * 256, byte_size=64)
        s._pull_to_eof()
        assert s.size == 256
        assert s.window_start == 256 - 64
        assert s.window_end == 256
        assert len(s.memoryview()) == 64

    def test_window_start_advances_monotonically(self) -> None:
        def gen():
            for _ in range(10):
                yield b"a" * 32
        s = MemoryStream(gen(), byte_size=64)
        s._pull_to_eof()
        assert s.window_start == 10 * 32 - 64
        assert s.size == 10 * 32

    def test_window_below_byte_size_does_not_slide(self) -> None:
        s = MemoryStream(b"hi", byte_size=64)
        s._pull_to_eof()
        assert s.window_start == 0
        assert s.size == 2

    def test_byte_size_property(self) -> None:
        s = MemoryStream(b"", byte_size=128)
        assert s.byte_size == 128


class TestSeekable:
    def test_reread_within_window(self) -> None:
        s = MemoryStream(b"abcdefghij", byte_size=64)
        # Force a pull.
        assert s.read_bytes(size=5, offset=0) == b"abcde"
        # Seek back inside the window — same bytes come back.
        assert s.read_bytes(size=3, offset=2) == b"cde"
        assert s.read_bytes(size=10, offset=0) == b"abcdefghij"

    def test_read_past_window_start_raises(self) -> None:
        s = MemoryStream(b"x" * 200, byte_size=64)
        s._pull_to_eof()  # retained is now [136, 200)
        with pytest.raises(ValueError, match="behind the retained region"):
            s.read_bytes(size=10, offset=0)

    def test_read_within_window_after_slide(self) -> None:
        # Emit 200 bytes through a 64-byte window; bytes [136, 200)
        # remain and stay re-readable.
        payload = bytes(range(200))
        s = MemoryStream(payload, byte_size=64)
        s._pull_to_eof()
        assert s.window_start == 136
        # Position at window_start is the first valid offset.
        assert s.read_bytes(size=10, offset=136) == payload[136:146]
        assert s.read_bytes(size=64, offset=136) == payload[136:200]

    def test_seek_end_sentinel(self) -> None:
        s = MemoryStream(b"hello", byte_size=64)
        # SEEK_END idiom — pos=-1, n=0 returns empty without forcing
        # a pull beyond the current end.
        assert bytes(s.read_mv(0, -1)) == b""

    def test_negative_pos_resolves_against_size(self) -> None:
        s = MemoryStream(b"abcdef", byte_size=64)
        # Drain so size is known.
        s.read_bytes()
        # pos=-2 -> size-2 = 4 -> "ef"
        assert s.read_bytes(size=2, offset=-2) == b"ef"


class TestEOF:
    def test_read_to_eof_with_negative_n(self) -> None:
        s = MemoryStream(b"abcdef", byte_size=64)
        assert s.read_bytes(size=-1, offset=0) == b"abcdef"
        assert s.eof

    def test_read_more_than_available_caps_at_eof(self) -> None:
        s = MemoryStream(b"abc", byte_size=64)
        # Asking for 100 bytes returns only 3.
        assert s.read_bytes(size=100, offset=0) == b"abc"

    def test_partial_pull_then_eof(self) -> None:
        # Source yields one chunk then signals EOF on the next call.
        chunks = iter([b"first", b""])
        s = MemoryStream(lambda n: next(chunks, b""), byte_size=64)
        assert s.read_bytes(size=10, offset=0) == b"first"
        assert s.eof


class TestWrite:
    def test_append_at_end(self) -> None:
        s = MemoryStream(byte_size=64)
        n = s.write_bytes(b"hello", offset=0)
        assert n == 5
        assert s.read_bytes(size=5, offset=0) == b"hello"

    def test_append_slides_window(self) -> None:
        s = MemoryStream(byte_size=8)
        s.write_bytes(b"abcdefghij", offset=0)  # 10 bytes
        assert s.size == 10
        assert s.window_start == 2
        assert bytes(s.memoryview()) == b"cdefghij"

    def test_overwrite_within_window(self) -> None:
        s = MemoryStream(b"abcdef", byte_size=64)
        s.read_bytes()
        s.write_bytes(b"XY", offset=2)
        assert s.read_bytes(size=6, offset=0) == b"abXYef"

    def test_write_behind_window_raises(self) -> None:
        s = MemoryStream(byte_size=4)
        s.write_bytes(b"abcdefgh", offset=0)  # window = [4, 8)
        with pytest.raises(ValueError, match="behind the live window"):
            s.write_bytes(b"!", offset=0)

    def test_write_past_end_raises(self) -> None:
        s = MemoryStream(byte_size=64)
        with pytest.raises(ValueError, match="past current end"):
            s.write_bytes(b"x", offset=10)


class TestTruncateClearReserve:
    def test_truncate_shrinks_within_window(self) -> None:
        s = MemoryStream(b"abcdef", byte_size=64)
        s.read_bytes()
        s.truncate(3)
        assert s.size == 3
        assert s.read_bytes() == b"abc"

    def test_truncate_extends_with_zero_pad(self) -> None:
        s = MemoryStream(b"abc", byte_size=64)
        s.read_bytes()
        s.truncate(6)
        assert s.size == 6
        assert s.read_bytes(size=6, offset=0) == b"abc\x00\x00\x00"

    def test_truncate_below_window_start_raises(self) -> None:
        s = MemoryStream(byte_size=4)
        s.write_bytes(b"abcdefgh", offset=0)  # spill_start = 4
        with pytest.raises(ValueError, match="behind the retained region"):
            s.truncate(2)

    def test_truncate_negative_raises(self) -> None:
        s = MemoryStream(byte_size=64)
        with pytest.raises(ValueError, match="truncate size must be >= 0"):
            s.truncate(-1)

    def test_clear_resets_window(self) -> None:
        s = MemoryStream(byte_size=4)
        s.write_bytes(b"abcdef", offset=0)
        s.clear()
        assert s.size == 0
        assert s.window_start == 0
        assert bytes(s.memoryview()) == b""

    def test_reserve_caps_at_byte_size(self) -> None:
        s = MemoryStream(byte_size=8)
        s.reserve(1024)  # would-be 1024 bytes; capped at 8
        assert len(s.memoryview()) <= 8

    def test_reserve_negative_raises(self) -> None:
        s = MemoryStream(byte_size=64)
        with pytest.raises(ValueError, match="reserve size must be >= 0"):
            s.reserve(-1)


class TestSpill:
    """``spill_threshold`` caps in-memory bytes; cold bytes spill to disk.

    Spill only activates when ``byte_size > spill_threshold``;
    otherwise the holder collapses to the legacy single-window
    eviction shape (covered by :class:`TestWindowSlide`).
    """

    def test_no_spill_below_threshold(self) -> None:
        # 100 bytes total, threshold 1 KiB → wholly in-memory.
        s = MemoryStream(b"x" * 100, byte_size=4096, spill_threshold=1024)
        s._pull_to_eof()
        assert not s.has_spill
        assert s.spill_start == 0
        assert s.window_start == 0

    def test_spill_above_threshold_keeps_bytes_readable(self) -> None:
        # 300 bytes, in-memory cap 100, total budget 1 KiB → cold
        # bytes spill to disk but stay readable.
        payload = bytes(range(256)) + b"abcdefghijklmnopqrstuvwxyz" * 2  # 308 bytes
        payload = payload[:300]
        s = MemoryStream(payload, byte_size=1024, spill_threshold=100)
        s._pull_to_eof()

        assert s.has_spill
        # Total retention fits in 1 KiB → spill_start stayed at 0.
        assert s.spill_start == 0
        # In-memory window is the trailing ≤ 100 bytes; spill holds
        # the head.
        assert s.window_start <= 200
        assert s.size == 300

        # Reads from the spill region still return the original bytes.
        assert s.read_bytes(size=10, offset=0) == payload[:10]
        # Cross-boundary read — spans spill into memory.
        cross_start = s.window_start - 5
        assert s.read_bytes(size=10, offset=cross_start) == payload[
            cross_start:cross_start + 10
        ]
        # In-memory read.
        assert s.read_bytes(size=20, offset=s.window_start) == payload[
            s.window_start:s.window_start + 20
        ]

    def test_retention_cap_evicts_oldest(self) -> None:
        # 500 bytes, in-memory 50, total 200. Once spill + memory
        # would exceed 200, the oldest spilled bytes are evicted —
        # reads behind ``spill_start`` raise.
        payload = bytes((i % 256) for i in range(500))
        s = MemoryStream(payload, byte_size=200, spill_threshold=50)
        s._pull_to_eof()

        # Total retention bounded at 200 bytes.
        assert s.size - s.spill_start <= 200
        assert s.spill_start == 500 - (s.size - s.spill_start)

        # Reads in the retained region work and return the right slice.
        head = s.spill_start
        assert s.read_bytes(size=10, offset=head) == payload[head:head + 10]

        # Reads behind ``spill_start`` raise — the bytes are gone.
        with pytest.raises(ValueError, match="behind the retained region"):
            s.read_bytes(size=10, offset=0)

    def test_clear_drops_spill_file(self) -> None:
        s = MemoryStream(b"x" * 300, byte_size=1024, spill_threshold=100)
        s._pull_to_eof()
        assert s.has_spill
        s.clear()
        assert not s.has_spill
        assert s.size == 0
        assert s.window_start == 0
        assert s.spill_start == 0

    def test_truncate_into_spill_drops_memory_and_shrinks_spill(self) -> None:
        s = MemoryStream(b"x" * 300, byte_size=1024, spill_threshold=100)
        s._pull_to_eof()
        spill_size = s.window_start - s.spill_start
        assert spill_size > 0
        # Truncate to a point inside the spill region.
        target = s.spill_start + spill_size // 2
        s.truncate(target)
        assert s.size == target
        # In-memory window dropped, spill shrunk.
        assert s.window_start == target
        assert len(s.memoryview()) == 0


class TestHolderIntegration:
    def test_is_memory_predicates(self) -> None:
        s = MemoryStream(byte_size=64)
        assert s.is_memory
        assert not s.is_local_path
        assert not s.is_remote_path
        assert s.is_local
        assert not s.is_remote

    def test_size_and_len(self) -> None:
        s = MemoryStream(b"abcd", byte_size=64)
        s.read_bytes()
        assert s.size == 4
        assert len(s) == 4

    def test_bytes_dunder(self) -> None:
        s = MemoryStream(b"abcd", byte_size=64)
        assert bytes(s) == b"abcd"

    def test_context_manager(self) -> None:
        with MemoryStream(b"hello", byte_size=64) as s:
            assert s.read_bytes() == b"hello"

    def test_pull_chunk_default_capped_at_byte_size(self) -> None:
        # byte_size smaller than the default pull chunk — pull chunk
        # should fall back to byte_size, not the 64 KiB default.
        s = MemoryStream(byte_size=128)
        assert s._pull_chunk == 128

    def test_large_streamed_read_yields_full_payload(self) -> None:
        payload = bytes(range(256)) * 16  # 4 KiB
        s = MemoryStream(io.BytesIO(payload), byte_size=8 * KB)
        out = s.read_bytes()
        assert out == payload
        assert s.window_start == 0  # fits within window
