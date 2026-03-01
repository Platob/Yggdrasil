# tests/test_bytes_io.py
from __future__ import annotations

import io
from pathlib import Path

import pytest

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.config import BufferConfig


@pytest.fixture()
def tmp_cfg(tmp_path: Path) -> BufferConfig:
    return BufferConfig(
        spill_bytes=64,
        tmp_dir=tmp_path,
        prefix="test_bytesio_",
        suffix=".bin",
        keep_spilled_file=False,
    )


def _codec_available(name: str) -> bool:
    from yggdrasil.io.enums.codec import Codec as _Codec

    try:
        return _Codec.parse(name) is not None
    except Exception:
        return False


def _detect_name(buf: BytesIO) -> str | None:
    from yggdrasil.io.enums.codec import detect as _detect

    try:
        c = _detect(buf)
        if c is None:
            return None
        name = getattr(c, "name", None) or getattr(c, "value", None) or str(c)
        return str(name).lower()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Core IO + cursor semantics
# -----------------------------------------------------------------------------

def test_empty_buffer_starts_at_zero(tmp_cfg: BufferConfig):
    b = BytesIO(config=tmp_cfg)
    assert b.tell() == 0
    assert b.size == 0
    assert b.read() == b""
    assert b.tell() == 0


def test_write_read_progress_cursor(tmp_cfg: BufferConfig):
    b = BytesIO(config=tmp_cfg)
    assert b.write_bytes(b"hello") == 5
    assert b.tell() == 5
    assert b.size == 5

    b.seek(0)
    assert b.read(2) == b"he"
    assert b.tell() == 2
    assert b.read(10) == b"llo"
    assert b.tell() == 5
    assert b.read(1) == b""


def test_seek_whence_and_bounds(tmp_cfg: BufferConfig):
    b = BytesIO(b"abcdef", config=tmp_cfg)

    b.seek(2)
    assert b.tell() == 2

    b.seek(2, io.SEEK_CUR)
    assert b.tell() == 4

    b.seek(-1, io.SEEK_END)
    assert b.tell() == 5
    assert b.read() == b"f"

    with pytest.raises(ValueError):
        b.seek(-1, io.SEEK_SET)

    with pytest.raises(ValueError):
        b.seek(0, 9999)


def test_head_does_not_move_cursor(tmp_cfg: BufferConfig):
    b = BytesIO(b"0123456789", config=tmp_cfg)
    assert bytes(b.head(4)) == b"0123"
    assert b.tell() == 0

    b.read(3)
    assert b.tell() == 3
    assert bytes(b.head(2)) == b"01"
    assert b.tell() == 3


def test_to_bytes_does_not_move_cursor(tmp_cfg: BufferConfig):
    b = BytesIO(b"abc", config=tmp_cfg)
    b.seek(2)
    assert b.to_bytes() == b"abc"
    assert b.tell() == 2


def test_gap_zero_fill_memory(tmp_cfg: BufferConfig):
    b = BytesIO(config=tmp_cfg)
    b.seek(10)
    b.write_bytes(b"X")
    assert b.size == 11
    b.seek(0)
    assert b.read() == (b"\x00" * 10) + b"X"


def test_spill_threshold_trigger(tmp_cfg: BufferConfig):
    b = BytesIO(config=tmp_cfg)
    assert b.spilled is False

    b.write_bytes(b"A" * 60)
    assert b.spilled is False

    b.write_bytes(b"B" * 10)
    assert b.spilled is True
    assert b.path is not None
    assert b.size == 70


def test_spill_preserves_logical_cursor(tmp_cfg: BufferConfig):
    b = BytesIO(config=tmp_cfg)
    b.write_bytes(b"A" * 32)
    b.seek(10)
    pos = b.tell()

    b.spill_to_file()
    assert b.spilled is True
    assert b.tell() == pos

    b.write_bytes(b"Z")
    b.seek(0)
    out = b.read()
    assert out[:10] == b"A" * 10
    assert out[10:11] == b"Z"


def test_memoryview_memory_and_spilled(tmp_cfg: BufferConfig):
    b = BytesIO(b"hello", config=tmp_cfg)
    mv = b.memoryview()
    assert bytes(mv) == b"hello"
    assert len(mv) == 5

    s = BytesIO(config=tmp_cfg)
    s.write_bytes(b"A" * 100)
    assert s.spilled is True
    mv2 = s.memoryview()
    assert bytes(mv2[:3]) == b"AAA"
    assert len(mv2) == s.size


# -----------------------------------------------------------------------------
# Structured IO
# -----------------------------------------------------------------------------

def test_structured_roundtrip(tmp_cfg: BufferConfig):
    b = BytesIO(config=tmp_cfg)

    b.write_int8(-3)
    b.write_uint8(250)
    b.write_int16(-32000)
    b.write_uint16(65000)
    b.write_int32(-123456)
    b.write_uint32(4000000000)
    b.write_int64(-1234567890123)
    b.write_uint64(12345678901234567890)
    b.write_f32(1.25)
    b.write_f64(2.5)
    b.write_bool(True)
    b.write_bool(False)
    b.write_str_u32("yo")

    b.seek(0)
    assert b.read_int8() == -3
    assert b.read_uint8() == 250
    assert b.read_int16() == -32000
    assert b.read_uint16() == 65000
    assert b.read_int32() == -123456
    assert b.read_uint32() == 4000000000
    assert b.read_int64() == -1234567890123
    assert b.read_uint64() == 12345678901234567890
    assert b.read_f32() == pytest.approx(1.25)
    assert b.read_f64() == pytest.approx(2.5)
    assert b.read_bool() is True
    assert b.read_bool() is False
    assert b.read_str_u32() == "yo"


# -----------------------------------------------------------------------------
# Parse / wrap / close
# -----------------------------------------------------------------------------

def test_parse_from_path(tmp_cfg: BufferConfig, tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")

    b = BytesIO.parse(p, config=tmp_cfg)
    assert b.spilled is True
    assert b.size == 3
    assert b.read(3) == b"abc"


def test_wrap_filelike_readable(tmp_cfg: BufferConfig, tmp_path: Path):
    p = tmp_path / "y.bin"
    with p.open("w+b") as fh:
        fh.write(b"12345")
        fh.flush()

        b = BytesIO.wrap(fh, auto_close=False, config=tmp_cfg)
        b.seek(0)
        assert b.read(2) == b"12"


def test_close_removes_spill_file_when_configured(tmp_cfg: BufferConfig):
    b = BytesIO(config=tmp_cfg)
    b.write_bytes(b"A" * 100)
    assert b.spilled is True
    path = b.path
    assert path is not None and path.exists()

    b.close()
    assert not path.exists()


def test_closed_operations_raise(tmp_cfg: BufferConfig):
    b = BytesIO(b"abc", config=tmp_cfg)
    b.close()
    with pytest.raises(ValueError):
        b.read(1)
    with pytest.raises(ValueError):
        b.write_bytes(b"x")
    with pytest.raises(ValueError):
        b.seek(0)
    with pytest.raises(ValueError):
        b.tell()


# -----------------------------------------------------------------------------
# Compression copy semantics (THE POINT)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("codec", ["gzip"])
def test_compress_copy_true_new_buffer_source_unchanged(tmp_cfg: BufferConfig, codec: str):
    if not _codec_available(codec):
        pytest.skip(f"{codec} codec not available")

    payload = (b"abc123" * 2048) + b"tail"
    src = BytesIO(payload, config=tmp_cfg)

    src.seek(123)
    src_pos = src.tell()
    src_bytes = src.to_bytes()
    src_size = src.size
    src_spilled = src.spilled

    out = src.compress(codec, copy=True)

    assert out is not src
    assert src.tell() == src_pos
    assert src.size == src_size
    assert src.to_bytes() == src_bytes
    assert src.spilled == src_spilled

    assert out.tell() == 0
    if payload:
        assert out.to_bytes() != payload


@pytest.mark.parametrize("codec", ["gzip"])
def test_compress_copy_false_inplace_replaces_self(tmp_cfg: BufferConfig, codec: str):
    if not _codec_available(codec):
        pytest.skip(f"{codec} codec not available")

    payload = (b"abc123" * 2048) + b"tail"
    src = BytesIO(payload, config=tmp_cfg)

    src.seek(10)
    out = src.compress(codec, copy=False)

    assert out is src
    assert src.tell() == 0
    assert src.to_bytes() != payload

    # sanity: explicit restore works
    restored = src.decompress(codec, copy=True)
    assert restored.to_bytes() == payload


@pytest.mark.parametrize("codec", ["gzip"])
def test_decompress_copy_true_new_buffer_source_unchanged(tmp_cfg: BufferConfig, codec: str):
    if not _codec_available(codec):
        pytest.skip(f"{codec} codec not available")

    payload = (b"Z" * 4096) + b"tail"
    raw = BytesIO(payload, config=tmp_cfg)
    comp = raw.compress(codec, copy=True)

    comp.seek(7)
    comp_pos = comp.tell()
    comp_bytes = comp.to_bytes()
    comp_size = comp.size

    out = comp.decompress(codec, copy=True)

    assert out is not comp
    assert comp.tell() == comp_pos
    assert comp.size == comp_size
    assert comp.to_bytes() == comp_bytes

    assert out.tell() == 0
    assert out.to_bytes() == payload


@pytest.mark.parametrize("codec", ["gzip"])
def test_decompress_copy_false_inplace_replaces_self(tmp_cfg: BufferConfig, codec: str):
    if not _codec_available(codec):
        pytest.skip(f"{codec} codec not available")

    payload = (b"Z" * 4096) + b"tail"
    raw = BytesIO(payload, config=tmp_cfg)
    comp = raw.compress(codec, copy=True)

    comp.seek(12)
    out = comp.decompress(codec, copy=False)

    assert out is comp
    assert comp.tell() == 0
    assert comp.to_bytes() == payload


def test_decompress_infer_copy_false_inplace_when_detectable(tmp_cfg: BufferConfig):
    if not _codec_available("gzip"):
        pytest.skip("gzip codec not available")

    payload = b"A" * 2048 + b"tail"
    raw = BytesIO(payload, config=tmp_cfg)
    comp = raw.compress("gzip", copy=True)

    if _detect_name(comp) is None:
        pytest.skip("detect() does not recognize gzip here; infer not testable")

    comp.seek(5)
    out = comp.decompress("infer", copy=False)
    assert out is comp
    assert comp.tell() == 0
    assert comp.to_bytes() == payload


def test_inplace_roundtrip_spilled_source(tmp_cfg: BufferConfig):
    if not _codec_available("gzip"):
        pytest.skip("gzip codec not available")

    payload = b"A" * 256  # force spill
    src = BytesIO(config=tmp_cfg)
    src.write_bytes(payload)
    assert src.spilled is True
    old_path = src.path
    assert old_path is not None and old_path.exists()

    src.seek(11)
    src.compress("gzip", copy=False)

    # should have replaced content, reset cursor
    assert src.tell() == 0
    assert src.to_bytes() != payload

    # old spill file should be gone (copy=False replacement resets backing)
    assert old_path.exists() is False

    # restore
    src.decompress("gzip", copy=False)
    assert src.tell() == 0
    assert src.to_bytes() == payload


@pytest.mark.parametrize("codec", ["zstd", "lz4", "bz2", "xz", "brotli", "snappy"])
def test_optional_codecs_copy_semantics_best_effort(tmp_cfg: BufferConfig, codec: str):
    if not _codec_available(codec):
        pytest.skip(f"{codec} codec not available")

    payload = (b"abc123" * 4096) + b"tail"
    src = BytesIO(payload, config=tmp_cfg)

    # copy=True: unchanged
    src.seek(17)
    pos = src.tell()
    snap = src.to_bytes()
    comp = src.compress(codec, copy=True)
    assert src.tell() == pos
    assert src.to_bytes() == snap
    assert comp.tell() == 0

    # copy=False: in-place
    src.compress(codec, copy=False)
    assert src.tell() == 0
    assert src.to_bytes() != payload

    # explicit restore (infer not guaranteed for brotli/raw-snappy)
    src.decompress(codec, copy=False)
    assert src.tell() == 0
    assert src.to_bytes() == payload