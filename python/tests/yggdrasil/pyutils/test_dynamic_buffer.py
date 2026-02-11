# tests/test_dynamic_buffer.py
from __future__ import annotations

import io
from pathlib import Path

import pytest

from yggdrasil.io.dynamic_buffer import DynamicBuffer, DynamicBufferConfig  # type: ignore


def test_starts_in_memory_and_not_spilled():
    buf = DynamicBuffer(DynamicBufferConfig(spill_bytes=1024))
    assert buf.spilled is False
    assert buf.path is None
    assert buf.size == 0

    buf.write(b"abc")
    assert buf.spilled is False
    assert buf.path is None
    assert buf.size == 3
    assert buf.getvalue() == b"abc"


def test_spills_to_disk_when_threshold_exceeded(tmp_path: Path):
    cfg = DynamicBufferConfig(spill_bytes=16, tmp_dir=tmp_path)
    buf = DynamicBuffer(cfg)

    buf.write(b"a" * 10)
    assert buf.spilled is False
    assert buf.path is None

    buf.write(b"b" * 7)  # 10 + 7 = 17 > 16 => spill
    assert buf.spilled is True
    assert buf.path is not None
    assert buf.path.exists()
    assert buf.size == 17

    buf.seek(0)
    assert buf.read() == b"a" * 10 + b"b" * 7


def test_spill_preserves_cursor_position_and_overwrite_semantics(tmp_path: Path):
    """
    Writing at a non-end position overwrites bytes (does NOT insert).
    """
    cfg = DynamicBufferConfig(spill_bytes=8, tmp_dir=tmp_path)
    buf = DynamicBuffer(cfg)

    buf.write(b"12345")
    buf.seek(2)
    assert buf.tell() == 2

    buf.write(b"abcdefghij")  # triggers spill
    assert buf.spilled is True

    # cursor advanced by len(write)
    assert buf.tell() == 2 + 10

    buf.seek(0)
    # overwrite semantics: '12' + 'abcdefghij' (original tail overwritten)
    assert buf.read() == b"12abcdefghij"


def test_getvalue_raises_after_spill(tmp_path: Path):
    cfg = DynamicBufferConfig(spill_bytes=4, tmp_dir=tmp_path)
    buf = DynamicBuffer(cfg)

    buf.write(b"abcd")  # exactly 4 -> still memory
    assert buf.getvalue() == b"abcd"

    buf.write(b"e")  # spill
    assert buf.spilled is True

    with pytest.raises(RuntimeError, match="spilled"):
        _ = buf.getvalue()


def test_to_bytes_reads_all_data_even_after_spill(tmp_path: Path):
    cfg = DynamicBufferConfig(spill_bytes=8, tmp_dir=tmp_path)
    buf = DynamicBuffer(cfg)

    data = b"x" * 100
    buf.write(data)
    assert buf.spilled is True

    buf.seek(10)
    pos = buf.tell()
    got = buf.to_bytes()
    assert got == data
    assert buf.tell() == pos


def test_open_reader_in_memory_returns_bytesio_copy():
    buf = DynamicBuffer(DynamicBufferConfig(spill_bytes=1024))
    buf.write(b"hello")

    r = buf.open_reader()
    assert isinstance(r, io.BytesIO)
    assert r.read() == b"hello"

    # independent copy
    r2 = buf.open_reader()
    r2.write(b"X")
    assert buf.getvalue() == b"hello"


def test_open_reader_spilled_opens_real_file(tmp_path: Path):
    cfg = DynamicBufferConfig(spill_bytes=4, tmp_dir=tmp_path)
    buf = DynamicBuffer(cfg)
    buf.write(b"01234")  # spill

    assert buf.spilled is True
    p = buf.path
    assert p is not None and p.exists()

    r = buf.open_reader()
    try:
        assert r.read() == b"01234"
    finally:
        r.close()


def test_size_property_does_not_change_cursor_and_accounts_for_overwrite(tmp_path: Path):
    cfg = DynamicBufferConfig(spill_bytes=8, tmp_dir=tmp_path)
    buf = DynamicBuffer(cfg)

    buf.write(b"abcdef")
    buf.seek(2)
    pos = buf.tell()

    assert buf.size == 6
    assert buf.tell() == pos

    # Write 10 bytes at position 2:
    # overwrites 4 bytes and extends to size 2 + 10 = 12
    buf.write(b"0123456789")
    assert buf.spilled is True

    buf.seek(5)
    pos2 = buf.tell()
    assert buf.size == 12
    assert buf.tell() == pos2
