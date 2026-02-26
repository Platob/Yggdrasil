# tests/io/test_bytes_io.py
"""Unit tests for yggdrasil.io.buffer.bytes_io — BytesIO.

Coverage
--------
Construction
    - None, bytes, bytearray, memoryview, io.BytesIO, Path,
      seekable file-like, non-seekable file-like, BytesIO alias
    - Immediate spill when payload exceeds threshold
    - TypeError on unsupported input type

parse_any() factory
    - Returns self for existing BytesIO
    - Wraps stdlib io.BytesIO without copy
    - Opens Path / str as file-backed buffer
    - Wraps raw bytes

I/O interface
    - read / write / seek / tell
    - write accepts str (UTF-8), None (no-op)
    - readable / writable / seekable flags

Spill mechanics
    - Threshold triggers _spill_to_file during write
    - Data integrity preserved across spill
    - Cursor position preserved across spill
    - Spill file deleted on close (default)
    - Spill file retained when keep_spilled_file=True

Introspection
    - size (memory and spilled, via fstat and seek fallback)
    - spilled / path properties
    - __len__ / __bool__

Accessors
    - getvalue (memory and spilled)
    - to_bytes (cursor preserved)
    - memoryview (memory and mmap-backed spilled)
    - open_reader (independent cursor, memory and spilled)
    - decode / decode empty

Structured binary I/O (little-endian)
    - All integer widths: int8/uint8 … int64/uint64
    - f32 / f64
    - bool
    - bytes_u32 / str_u32 (with encoding)
    - _read_exact raises EOFError on short read
    - write_bytes_u32 returns correct total byte count
    - Byte-order verification (int32 = 1 → \x01\x00\x00\x00)

write_any_bytes / _coerce_to_memoryview
    - bytes, bytearray, None, file-like, str
    - TypeError on unsupported type

content_type
    - Delegates to MediaType.from_io without moving cursor

Context manager
    - __enter__ returns self
    - __exit__ closes buffer
    - Spill file deleted on context exit

Lifecycle
    - close() is idempotent
    - write / buffer() raise ValueError on closed buffer
    - cleanup() is an alias for close()
    - flush() is safe on both open and closed buffers
    - __repr__ is safe on closed buffer (does not call buffer())
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from .conftest import PAYLOAD, SMALL

pytest.importorskip("yggdrasil")

from yggdrasil.io.buffer.bytes_io import BytesIO    # noqa: E402
from yggdrasil.io.config import BufferConfig        # noqa: E402
from yggdrasil.io.enums.media_type import MediaType # noqa: E402


# ===========================================================================
# Construction
# ===========================================================================

class TestBytesIOConstruction:
    def test_none_creates_empty_in_memory_buffer(self):
        buf = BytesIO()
        assert buf.size == 0
        assert not buf.spilled

    def test_bytes(self):
        buf = BytesIO(b"hello")
        buf.seek(0)
        assert buf.read() == b"hello"

    def test_bytearray(self):
        buf = BytesIO(bytearray(b"hello"))
        buf.seek(0)
        assert buf.read() == b"hello"

    def test_memoryview(self):
        buf = BytesIO(memoryview(b"hello"))
        buf.seek(0)
        assert buf.read() == b"hello"

    def test_stdlib_bytesio_snapshots_content(self):
        src = io.BytesIO(b"hello")
        src.seek(3)
        buf = BytesIO(src)
        buf.seek(0)
        assert buf.read() == b"hello"

    def test_stdlib_bytesio_preserves_src_cursor(self):
        src = io.BytesIO(b"hello")
        src.seek(3)
        BytesIO(src)
        assert src.tell() == 3

    def test_path_opens_file_backed(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"from file")
        buf = BytesIO(p)
        buf.seek(0)
        assert buf.read() == b"from file"
        assert buf.spilled

    def test_bytesio_alias_shares_backing_store(self):
        original = BytesIO(b"shared")
        alias = BytesIO(original)
        assert alias._mem is original._mem or alias._file is original._file

    def test_seekable_filelike(self):
        buf = BytesIO(io.BytesIO(b"seekable"))
        buf.seek(0)
        assert buf.read() == b"seekable"

    def test_non_seekable_filelike_drained(self):
        class Pipe:
            def read(self):
                return b"piped data"
        buf = BytesIO(Pipe())
        buf.seek(0)
        assert buf.read() == b"piped data"

    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError):
            BytesIO(12345)

    def test_spills_immediately_when_over_threshold(self, spill_config):
        buf = BytesIO(PAYLOAD, config=spill_config)
        assert buf.spilled
        assert buf.path is not None
        assert buf.path.exists()


# ===========================================================================
# parse_any() factory
# ===========================================================================

class TestBytesIOParseAny:
    def test_returns_self_for_existing_bytesio(self):
        buf = BytesIO(SMALL)
        assert BytesIO.parse_any(buf) is buf

    def test_wraps_stdlib_bytesio_without_copy(self):
        src = io.BytesIO(b"data")
        buf = BytesIO.parse_any(src)
        assert buf._mem is src

    def test_opens_path_string_as_file_backed(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"path data")
        buf = BytesIO.parse_any(str(p))
        assert buf.spilled

    def test_opens_path_object_as_file_backed(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"path data")
        buf = BytesIO.parse_any(p)
        assert buf.spilled

    def test_wraps_raw_bytes(self):
        buf = BytesIO.parse_any(b"raw bytes")
        buf.seek(0)
        assert buf.read() == b"raw bytes"


# ===========================================================================
# I/O interface
# ===========================================================================

class TestBytesIOReadWriteSeek:
    def test_write_then_read(self, empty_buf):
        empty_buf.write(b"hello world")
        empty_buf.seek(0)
        assert empty_buf.read() == b"hello world"

    def test_tell_advances_after_write(self, empty_buf):
        empty_buf.write(b"12345")
        assert empty_buf.tell() == 5

    def test_seek_set(self, small_buf):
        small_buf.seek(5)
        assert small_buf.tell() == 5

    def test_seek_end(self, small_buf):
        small_buf.seek(0, io.SEEK_END)
        assert small_buf.tell() == len(SMALL)

    def test_read_partial(self, small_buf):
        assert small_buf.read(5) == SMALL[:5]

    def test_write_string_encoded_as_utf8(self, empty_buf):
        n = empty_buf.write("héllo")
        assert n > 0
        empty_buf.seek(0)
        assert empty_buf.read().decode("utf-8") == "héllo"

    def test_write_none_is_noop(self, empty_buf):
        assert empty_buf.write(None) == 0

    def test_readable(self, empty_buf):
        assert empty_buf.readable()

    def test_writable(self, empty_buf):
        assert empty_buf.writable()

    def test_seekable(self, empty_buf):
        assert empty_buf.seekable()


# ===========================================================================
# Spill mechanics
# ===========================================================================

class TestBytesIOSpill:
    def test_write_past_threshold_triggers_spill(self, spill_config):
        buf = BytesIO(config=spill_config)
        assert not buf.spilled
        buf.write(b"x" * 128)
        assert buf.spilled
        assert buf.path is not None

    def test_data_intact_after_spill(self, spill_config):
        buf = BytesIO(config=spill_config)
        buf.write(b"A" * 128)
        buf.seek(0)
        assert buf.read() == b"A" * 128

    def test_cursor_position_preserved_across_spill(self, spill_config):
        buf = BytesIO(config=spill_config)
        buf.write(b"prefix")
        buf.write(b"x" * 128)
        assert buf.tell() == 6 + 128

    def test_spill_file_deleted_on_close(self, spill_config):
        buf = BytesIO(PAYLOAD, config=spill_config)
        path = buf.path
        assert path.exists()
        buf.close()
        assert not path.exists()

    def test_spill_file_retained_when_keep_flag_set(self, tmp_path):
        cfg = BufferConfig(spill_bytes=64, tmp_dir=tmp_path, keep_spilled_file=True)
        buf = BytesIO(PAYLOAD, config=cfg)
        path = buf.path
        buf.close()
        assert path.exists()


# ===========================================================================
# Introspection
# ===========================================================================

class TestBytesIOIntrospection:
    def test_size_in_memory(self):
        assert BytesIO(b"12345").size == 5

    def test_size_empty(self):
        assert BytesIO().size == 0

    def test_size_spilled(self, spilled_buf):
        assert spilled_buf.size == len(PAYLOAD)

    def test_len_equals_size(self, small_buf):
        assert len(small_buf) == small_buf.size

    def test_bool_true_when_non_empty(self, small_buf):
        assert bool(small_buf)

    def test_bool_false_when_empty(self, empty_buf):
        assert not bool(empty_buf)

    def test_spilled_false_for_memory_buffer(self, small_buf):
        assert not small_buf.spilled

    def test_spilled_true_after_spill(self, spilled_buf):
        assert spilled_buf.spilled

    def test_path_none_for_memory_buffer(self, small_buf):
        assert small_buf.path is None

    def test_path_set_after_spill(self, spilled_buf):
        assert spilled_buf.path is not None


# ===========================================================================
# Accessors
# ===========================================================================

class TestBytesIOAccessors:
    def test_getvalue_memory(self, small_buf):
        assert small_buf.getvalue() == SMALL

    def test_getvalue_spilled(self, spilled_buf):
        assert spilled_buf.getvalue() == PAYLOAD

    def test_to_bytes_returns_full_content(self, small_buf):
        assert small_buf.to_bytes() == SMALL

    def test_to_bytes_preserves_cursor(self, small_buf):
        small_buf.seek(5)
        small_buf.to_bytes()
        assert small_buf.tell() == 5

    def test_memoryview_memory(self, small_buf):
        assert bytes(small_buf.memoryview()) == SMALL

    def test_memoryview_spilled(self, spilled_buf):
        assert bytes(spilled_buf.memoryview()) == PAYLOAD

    def test_open_reader_starts_at_zero(self, small_buf):
        reader = small_buf.open_reader()
        assert reader.read() == SMALL
        reader.close()

    def test_open_reader_does_not_move_buffer_cursor(self, small_buf):
        small_buf.seek(3)
        reader = small_buf.open_reader()
        reader.read()
        assert small_buf.tell() == 3
        reader.close()

    def test_open_reader_spilled(self, spilled_buf):
        reader = spilled_buf.open_reader()
        assert reader.read() == PAYLOAD
        reader.close()

    def test_decode_utf8(self):
        assert BytesIO(b"hello world").decode() == "hello world"

    def test_decode_empty_returns_empty_string(self, empty_buf):
        assert empty_buf.decode() == ""

    def test_repr_memory_contains_state_and_size(self, small_buf):
        r = repr(small_buf)
        assert "memory" in r
        assert "bytes" in r

    def test_repr_spilled_contains_state(self, spilled_buf):
        assert "spilled" in repr(spilled_buf)

    def test_repr_closed_safe_no_exception(self, small_buf):
        small_buf.close()
        r = repr(small_buf)
        assert "closed" in r
        assert "ValueError" not in r


# ===========================================================================
# Structured binary I/O — little-endian
# ===========================================================================

class TestBytesIOStructuredIO:
    """All read_*/write_* pairs encode little-endian two's complement integers
    and IEEE 754 floats, mirroring Rust's byteorder / binrw conventions."""

    def _roundtrip(self, write_method: str, read_method: str, value):
        buf = BytesIO()
        getattr(buf, write_method)(value)
        buf.seek(0)
        return getattr(buf, read_method)()

    @pytest.mark.parametrize("method,value", [
        ("int8",   -42),
        ("uint8",   200),
        ("int16",  -1000),
        ("uint16",  60000),
        ("int32",  -100_000),
        ("uint32",  3_000_000),
        ("int64",  -(2 ** 40)),
        ("uint64",  2 ** 40),
    ])
    def test_integer_roundtrip(self, method, value):
        assert self._roundtrip(f"write_{method}", f"read_{method}", value) == value

    def test_f32_roundtrip(self):
        buf = BytesIO()
        buf.write_f32(3.14)
        buf.seek(0)
        assert abs(buf.read_f32() - 3.14) < 1e-5

    def test_f64_roundtrip(self):
        v = 3.141592653589793
        assert abs(self._roundtrip("write_f64", "read_f64", v) - v) < 1e-15

    def test_bool_true_roundtrip(self):
        assert self._roundtrip("write_bool", "read_bool", True) is True

    def test_bool_false_roundtrip(self):
        assert self._roundtrip("write_bool", "read_bool", False) is False

    def test_bytes_u32_roundtrip(self):
        buf = BytesIO()
        buf.write_bytes_u32(b"settlement price")
        buf.seek(0)
        assert buf.read_bytes_u32() == b"settlement price"

    def test_str_u32_roundtrip(self):
        buf = BytesIO()
        buf.write_str_u32("TTF front-month")
        buf.seek(0)
        assert buf.read_str_u32() == "TTF front-month"

    def test_str_u32_non_utf8_encoding(self):
        buf = BytesIO()
        buf.write_str_u32("héllo", encoding="latin-1")
        buf.seek(0)
        assert buf.read_str_u32(encoding="latin-1") == "héllo"

    def test_read_exact_raises_eof_on_short_read(self):
        buf = BytesIO(b"AB")
        with pytest.raises(EOFError):
            buf._read_exact(10)

    def test_write_bytes_u32_returns_header_plus_payload(self):
        buf = BytesIO()
        n = buf.write_bytes_u32(b"hello")
        assert n == 4 + 5

    def test_int32_is_little_endian_on_wire(self):
        buf = BytesIO()
        buf.write_int32(1)
        buf.seek(0)
        assert buf.read(4) == b"\x01\x00\x00\x00"


# ===========================================================================
# write_any_bytes / _coerce_to_memoryview
# ===========================================================================

class TestBytesIOWriteAnyBytes:
    def test_bytes(self, empty_buf):
        assert empty_buf.write_any_bytes(b"data") == 4

    def test_bytearray(self, empty_buf):
        assert empty_buf.write_any_bytes(bytearray(b"data")) == 4

    def test_none_is_noop(self, empty_buf):
        assert empty_buf.write_any_bytes(None) == 0

    def test_filelike_drained(self, empty_buf):
        assert empty_buf.write_any_bytes(io.BytesIO(b"stream data")) == 11

    def test_unsupported_type_raises(self, empty_buf):
        with pytest.raises(TypeError):
            empty_buf.write_any_bytes(42)


class TestBytesIOCoerce:
    def test_bytes(self):
        assert bytes(BytesIO._coerce_to_memoryview(b"hello")) == b"hello"

    def test_bytearray(self):
        assert bytes(BytesIO._coerce_to_memoryview(bytearray(b"hello"))) == b"hello"

    def test_none_returns_empty_view(self):
        assert bytes(BytesIO._coerce_to_memoryview(None)) == b""

    def test_str_encoded_as_utf8(self):
        assert bytes(BytesIO._coerce_to_memoryview("hello")) == b"hello"

    def test_filelike_drained(self):
        mv = BytesIO._coerce_to_memoryview(io.BytesIO(b"read me"))
        assert bytes(mv) == b"read me"

    def test_unsupported_raises_type_error(self):
        with pytest.raises(TypeError):
            BytesIO._coerce_to_memoryview(42)


# ===========================================================================
# content_type
# ===========================================================================

class TestBytesIOContentType:
    def test_parquet_detected(self):
        buf = BytesIO(b"PAR1" + b"\x00" * 100 + b"PAR1")
        assert buf.content_type.is_parquet

    def test_json_detected(self):
        buf = BytesIO(b'{"symbol": "TTF", "price": 42.5}')
        assert buf.content_type.mime == MediaType.JSON

    def test_cursor_not_moved(self, small_buf):
        small_buf.seek(7)
        _ = small_buf.content_type
        assert small_buf.tell() == 7


# ===========================================================================
# Context manager
# ===========================================================================

class TestBytesIOContextManager:
    def test_enter_returns_self(self):
        buf = BytesIO(SMALL)
        with buf as b:
            assert b is buf

    def test_exit_closes_buffer(self):
        buf = BytesIO(SMALL)
        with buf:
            pass
        assert buf.closed

    def test_spill_file_deleted_on_exit(self, spill_config):
        with BytesIO(PAYLOAD, config=spill_config) as buf:
            path = buf.path
        assert not path.exists()


# ===========================================================================
# Lifecycle
# ===========================================================================

class TestBytesIOLifecycle:
    def test_close_is_idempotent(self, small_buf):
        small_buf.close()
        small_buf.close()   # must not raise

    def test_write_raises_on_closed(self, small_buf):
        small_buf.close()
        with pytest.raises(ValueError):
            small_buf.write(b"x")

    def test_buffer_raises_on_closed(self, small_buf):
        small_buf.close()
        with pytest.raises(ValueError):
            small_buf.buffer()

    def test_cleanup_is_alias_for_close(self, small_buf):
        small_buf.cleanup()
        assert small_buf.closed

    def test_flush_safe_on_open_memory_buffer(self, small_buf):
        small_buf.flush()   # must not raise

    def test_flush_safe_on_closed_buffer(self, small_buf):
        small_buf.close()
        small_buf.flush()   # must not raise (idempotent)