# tests/io/buffer/test_bytes_io.py

from __future__ import annotations

import io
import struct
from pathlib import Path

import pytest

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.config import BufferConfig
from yggdrasil.io.enums import MediaType, MimeType


@pytest.fixture
def cfg(tmp_path: Path) -> BufferConfig:
    return BufferConfig(
        spill_bytes=32,
        tmp_dir=tmp_path,
        prefix="bytes-io-",
        suffix=".bin",
        keep_spilled_file=False,
    )


@pytest.fixture
def keep_cfg(tmp_path: Path) -> BufferConfig:
    return BufferConfig(
        spill_bytes=32,
        tmp_dir=tmp_path,
        prefix="bytes-io-",
        suffix=".bin",
        keep_spilled_file=True,
    )


def test_init_empty(cfg: BufferConfig) -> None:
    b = BytesIO(config=cfg)

    assert b.size == 0
    assert b.tell() == 0
    assert not b.spilled
    assert b.path is None
    assert b.read() == b""
    assert b.exists() is False


def test_init_from_bytes_memory(cfg: BufferConfig) -> None:
    payload = b"hello world"
    b = BytesIO(payload, config=cfg)

    assert b.size == len(payload)
    assert not b.spilled
    assert b.path is None
    assert bytes(b) == payload
    assert b.to_bytes() == payload
    assert b.getvalue() == payload
    assert len(b) == len(payload)
    assert bool(b) is True


def test_init_from_large_bytes_spills(cfg: BufferConfig) -> None:
    payload = b"x" * 128
    b = BytesIO(payload, config=cfg)

    assert b.spilled
    assert b.path is not None
    assert b.path.exists()
    assert b.size == len(payload)
    assert b.to_bytes() == payload


def test_init_from_stdlib_bytesio_memory(cfg: BufferConfig) -> None:
    src = io.BytesIO(b"abcdef")
    src.seek(3)

    b = BytesIO(src, config=cfg)

    assert not b.spilled
    assert b.to_bytes() == b"abcdef"
    assert src.tell() == 3
    assert b.tell() == 0


def test_init_from_stdlib_bytesio_spill(cfg: BufferConfig) -> None:
    payload = b"a" * 100
    src = io.BytesIO(payload)
    src.seek(10)

    b = BytesIO(src, config=cfg)

    assert b.spilled
    assert b.to_bytes() == payload
    assert src.tell() == 10
    assert b.tell() == 0


def test_init_from_path_existing_file(cfg: BufferConfig, tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc123")

    b = BytesIO(path, config=cfg, copy=False)

    assert b.spilled
    assert b.path == path
    assert b.size == 6
    assert b.to_bytes() == b"abc123"


def test_init_from_path_creates_file_if_missing(cfg: BufferConfig, tmp_path: Path) -> None:
    path = tmp_path / "missing.bin"
    assert not path.exists()

    b = BytesIO(path, config=cfg, copy=False)

    assert b.spilled
    assert b.path == path
    assert path.exists()
    assert b.size == 0


def test_init_from_bytesio_instance_copies_memory(cfg: BufferConfig) -> None:
    src = BytesIO(b"abcdef", config=cfg)
    dst = BytesIO(src, config=cfg, copy=True)

    assert dst.to_bytes() == b"abcdef"
    assert dst.tell() == 0

    src.seek(0)
    src.write(b"ZZ")
    assert src.to_bytes() == b"ZZcdef"
    assert dst.to_bytes() == b"abcdef"


def test_init_from_bytesio_instance_copies_spilled(cfg: BufferConfig) -> None:
    src = BytesIO(b"x" * 100, config=cfg)
    assert src.spilled

    dst = BytesIO(src, config=cfg, copy=True)

    assert dst.to_bytes() == b"x" * 100
    assert dst.tell() == 0
    assert src.path != dst.path


def test_init_from_seekable_filelike_uses_remaining_bytes(cfg: BufferConfig) -> None:
    src = io.BytesIO(b"0123456789")
    src.seek(4)

    b = BytesIO(src, config=cfg)

    assert b.to_bytes() == b"0123456789"


def test_init_from_non_seekable_filelike(cfg: BufferConfig) -> None:
    class NonSeekable:
        def __init__(self, payload: bytes) -> None:
            self._bio = io.BytesIO(payload)

        def read(self, n: int = -1) -> bytes:
            return self._bio.read(n)

    b = BytesIO(NonSeekable(b"payload"), config=cfg)
    assert b.to_bytes() == b"payload"
    assert not b.spilled


def test_parse_passthrough_instance(cfg: BufferConfig) -> None:
    b = BytesIO(b"abc", config=cfg)
    out = BytesIO.parse(b, config=cfg)
    assert out is b


def test_parse_other_types(cfg: BufferConfig, tmp_path: Path) -> None:
    assert BytesIO.parse(b"abc", config=cfg).to_bytes() == b"abc"

    path = tmp_path / "f.bin"
    path.write_bytes(b"xyz")
    parsed = BytesIO.parse(path, config=cfg)
    assert parsed.to_bytes() == b"xyz"


def test_repr_memory_and_spilled(cfg: BufferConfig) -> None:
    a = BytesIO(b"abc", config=cfg)
    b = BytesIO(b"x" * 100, config=cfg)

    assert "memory" in repr(a)
    assert "spilled" in repr(b)


def test_seek_and_tell(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdef", config=cfg)

    assert b.seek(2) == 2
    assert b.tell() == 2
    assert b.seek(2, io.SEEK_CUR) == 4
    assert b.tell() == 4
    assert b.seek(-1, io.SEEK_END) == 5
    assert b.tell() == 5

    with pytest.raises(ValueError):
        b.seek(-1)

    with pytest.raises(ValueError):
        b.seek(0, 999)


def test_read_respects_cursor(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdef", config=cfg)

    assert b.read(2) == b"ab"
    assert b.tell() == 2
    assert b.read(3) == b"cde"
    assert b.tell() == 5
    assert b.read(99) == b"f"
    assert b.tell() == 6
    assert b.read() == b""


def test_read_negative_reads_to_end(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdef", config=cfg)
    b.seek(2)
    assert b.read(-1) == b"cdef"


def test_head_does_not_move_cursor(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdef", config=cfg)
    b.seek(3)

    head = b.head(4)

    assert bytes(head) == b"abcd"
    assert b.tell() == 3


def test_head_zero_and_empty(cfg: BufferConfig) -> None:
    b = BytesIO(config=cfg)

    assert bytes(b.head(10)) == b""
    assert bytes(BytesIO(b"abc", config=cfg).head(0)) == b""


def test_write_bytes_memory(cfg: BufferConfig) -> None:
    b = BytesIO(config=cfg)

    n = b.write_bytes(b"abc")
    assert n == 3
    assert b.tell() == 3
    assert b.size == 3
    assert b.to_bytes() == b"abc"


def test_write_string(cfg: BufferConfig) -> None:
    b = BytesIO(config=cfg)

    n = b.write("hé")
    assert n == len("hé".encode("utf-8"))
    assert b.to_bytes() == "hé".encode("utf-8")


def test_write_stream(cfg: BufferConfig) -> None:
    b = BytesIO(config=cfg)
    src = io.BytesIO(b"abcdef")

    n = b.write(src, batch_size=2)

    assert n == 6
    assert b.to_bytes() == b"abcdef"


def test_write_none_is_noop(cfg: BufferConfig) -> None:
    b = BytesIO(b"abc", config=cfg)
    b.seek(1)

    assert b.write(None) == 0
    assert b.tell() == 1
    assert b.to_bytes() == b"abc"


def test_write_overwrite_in_place(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdef", config=cfg)
    b.seek(2)

    b.write(b"ZZ")

    assert b.to_bytes() == b"abZZef"
    assert b.tell() == 4


def test_write_beyond_eof_zero_fills_gap_memory(cfg: BufferConfig) -> None:
    b = BytesIO(b"abc", config=cfg)
    b.seek(5)

    b.write(b"Z")

    assert b.size == 6
    assert b.to_bytes() == b"abc\x00\x00Z"


def test_write_beyond_eof_zero_fills_gap_spilled(cfg: BufferConfig) -> None:
    b = BytesIO(b"x" * 100, config=cfg)
    assert b.spilled

    b.seek(105)
    b.write(b"Q")

    assert b.size == 106
    data = b.to_bytes()
    assert data[:100] == b"x" * 100
    assert data[100:105] == b"\x00" * 5
    assert data[105:] == b"Q"


def test_write_triggers_spill(cfg: BufferConfig) -> None:
    b = BytesIO(b"a" * 16, config=cfg)
    assert not b.spilled

    b.seek(16)
    b.write(b"b" * 20)

    assert b.spilled
    assert b.size == 36
    assert b.to_bytes() == b"a" * 16 + b"b" * 20


def test_spill_to_file_preserves_content_and_cursor(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdefghij", config=cfg)
    b.seek(4)

    b.spill_to_file()

    assert b.spilled
    assert b.tell() == 4
    assert b.to_bytes() == b"abcdefghij"


def test_truncate_smaller_memory(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdef", config=cfg)
    b.seek(5)

    out = b.truncate(3)

    assert out == 3
    assert b.size == 3
    assert b.tell() == 3
    assert b.to_bytes() == b"abc"


def test_truncate_larger_memory_zero_fills(cfg: BufferConfig) -> None:
    b = BytesIO(b"abc", config=cfg)

    out = b.truncate(6)

    assert out == 6
    assert b.size == 6
    assert b.to_bytes() == b"abc\x00\x00\x00"


def test_truncate_none_uses_cursor(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdef", config=cfg)
    b.seek(2)

    out = b.truncate()

    assert out == 2
    assert b.to_bytes() == b"ab"


def test_truncate_spilled(cfg: BufferConfig) -> None:
    b = BytesIO(b"x" * 100, config=cfg)
    assert b.spilled

    out = b.truncate(10)

    assert out == 10
    assert b.size == 10
    assert b.to_bytes() == b"x" * 10


def test_truncate_negative_raises(cfg: BufferConfig) -> None:
    b = BytesIO(b"abc", config=cfg)

    with pytest.raises(ValueError):
        b.truncate(-1)


def test_memoryview_memory_mode_is_zero_copyish(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdef", config=cfg)
    mv = b.memoryview()

    assert bytes(mv) == b"abcdef"
    assert len(mv) == 6


def test_memoryview_spilled_mode(cfg: BufferConfig) -> None:
    b = BytesIO(b"x" * 100, config=cfg)
    mv = b.memoryview()

    assert bytes(mv) == b"x" * 100
    assert len(mv) == 100


def test_to_bytes_empty(cfg: BufferConfig) -> None:
    b = BytesIO(config=cfg)
    assert b.to_bytes() == b""


def test_decode(cfg: BufferConfig) -> None:
    b = BytesIO("hé".encode("utf-8"), config=cfg)
    assert b.decode() == "hé"


def test_open_reader_memory(cfg: BufferConfig) -> None:
    b = BytesIO(b"abcdef", config=cfg)

    with b.open_reader() as fh:
        assert fh.read() == b"abcdef"


def test_open_reader_spilled(cfg: BufferConfig) -> None:
    b = BytesIO(b"x" * 100, config=cfg)

    with b.open_reader() as fh:
        assert fh.read() == b"x" * 100


def test_exists_bool_len(cfg: BufferConfig) -> None:
    empty = BytesIO(config=cfg)
    full = BytesIO(b"a", config=cfg)

    assert empty.exists() is False
    assert bool(empty) is False
    assert len(empty) == 0

    assert full.exists() is True
    assert bool(full) is True
    assert len(full) == 1


def test_iter_yields_ints_like_bytes(cfg: BufferConfig) -> None:
    b = BytesIO(b"abc", config=cfg)
    assert list(b) == list(b"abc")


def test_media_type_property_returns_media_type(cfg: BufferConfig) -> None:
    b = BytesIO(b"{}", config=cfg)
    mt = b.media_type

    assert isinstance(mt, MediaType)


def test_media_io_delegates(cfg: BufferConfig) -> None:
    b = BytesIO(b"{}", config=cfg)
    media = MediaType(MimeType.JSON)

    io_obj = b.media_io(media)

    assert io_obj.buffer is b


def test_structured_roundtrip(cfg: BufferConfig) -> None:
    b = BytesIO(config=cfg)

    b.write_int8(-5)
    b.write_uint8(250)
    b.write_int16(-1234)
    b.write_uint16(54321)
    b.write_int32(-123456)
    b.write_uint32(123456)
    b.write_int64(-123456789)
    b.write_uint64(123456789)
    b.write_f32(1.25)
    b.write_f64(3.5)
    b.write_bool(True)
    b.write_bool(False)
    b.write_bytes_u32(b"hello")
    b.write_str_u32("world")

    b.seek(0)

    assert b.read_int8() == -5
    assert b.read_uint8() == 250
    assert b.read_int16() == -1234
    assert b.read_uint16() == 54321
    assert b.read_int32() == -123456
    assert b.read_uint32() == 123456
    assert b.read_int64() == -123456789
    assert b.read_uint64() == 123456789
    assert b.read_f32() == pytest.approx(1.25)
    assert b.read_f64() == pytest.approx(3.5)
    assert b.read_bool() is True
    assert b.read_bool() is False
    assert b.read_bytes_u32() == b"hello"
    assert b.read_str_u32() == "world"


def test_read_exact_eof(cfg: BufferConfig) -> None:
    b = BytesIO(b"ab", config=cfg)
    with pytest.raises(EOFError):
        b.read_uint32()


def test_write_bytes_u32_matches_struct_layout(cfg: BufferConfig) -> None:
    b = BytesIO(config=cfg)
    b.write_bytes_u32(b"abc")
    assert b.to_bytes() == struct.pack("<I", 3) + b"abc"


def test_xxh3_64_consistent(cfg: BufferConfig) -> None:
    a = BytesIO(b"abcdef", config=cfg)
    b = BytesIO(b"abcdef", config=cfg)

    assert a.xxh3_64().hexdigest() == b.xxh3_64().hexdigest()


def test_blake3_consistent_memory(cfg: BufferConfig) -> None:
    a = BytesIO(b"abcdef", config=cfg)
    b = BytesIO(b"abcdef", config=cfg)

    assert a.blake3().hexdigest() == b.blake3().hexdigest()


def test_blake3_consistent_spilled(cfg: BufferConfig) -> None:
    a = BytesIO(b"x" * 100, config=cfg)
    b = BytesIO(b"x" * 100, config=cfg)

    assert a.blake3().hexdigest() == b.blake3().hexdigest()


@pytest.mark.parametrize("codec", ["gzip", "zstd", "lz4"])
def test_compress_copy_roundtrip(cfg: BufferConfig, codec: str) -> None:
    src = BytesIO(b"hello world" * 20, config=cfg)
    out = src.compress(codec, copy=True)

    assert out is not src
    assert src.to_bytes() == b"hello world" * 20
    assert out.tell() == 0

    dec = out.decompress(codec, copy=True)
    assert dec.to_bytes() == src.to_bytes()


@pytest.mark.parametrize("codec", ["gzip", "zstd", "lz4"])
def test_compress_in_place_roundtrip(cfg: BufferConfig, codec: str) -> None:
    src = BytesIO(b"hello world" * 20, config=cfg)
    original = src.to_bytes()

    src.compress(codec, copy=False)
    assert src.tell() == 0
    assert src.to_bytes() != original

    src.decompress(codec, copy=False)
    assert src.tell() == 0
    assert src.to_bytes() == original


def test_decompress_infer_noop_copy(cfg: BufferConfig) -> None:
    src = BytesIO(b"plain-bytes", config=cfg)
    out = src.decompress("infer", copy=True)

    assert out is not src
    assert out.to_bytes() == b"plain-bytes"
    assert out.tell() == 0
    assert src.to_bytes() == b"plain-bytes"


def test_decompress_infer_noop_in_place(cfg: BufferConfig) -> None:
    src = BytesIO(b"plain-bytes", config=cfg)
    out = src.decompress("infer", copy=False)

    assert out is src
    assert src.to_bytes() == b"plain-bytes"
    assert src.tell() == 0


def test_unknown_codec_raises(cfg: BufferConfig) -> None:
    b = BytesIO(b"abc", config=cfg)

    with pytest.raises(ValueError):
        b.compress("definitely-not-a-codec")

    with pytest.raises(ValueError):
        b.decompress("definitely-not-a-codec")


def test_view_binary_window(cfg: BufferConfig) -> None:
    b = BytesIO(b"0123456789", config=cfg)

    with b.view(start=2, length=4) as fh:
        assert fh.read() == b"2345"

    assert b.tell() == 0


def test_view_text_window(cfg: BufferConfig) -> None:
    b = BytesIO("hello world".encode(), config=cfg)

    with b.view(text=True, start=6, length=5) as fh:
        assert fh.read() == "world"

    assert b.tell() == 0


def test_close_removes_spill_file_by_default(cfg: BufferConfig) -> None:
    b = BytesIO(b"x" * 100, config=cfg)
    path = b.path
    assert path is not None and path.exists()

    b.close()

    assert not path.exists()
    assert b.closed


def test_close_keeps_spill_file_when_configured(keep_cfg: BufferConfig) -> None:
    b = BytesIO(b"x" * 100, config=keep_cfg)
    path = b.path
    assert path is not None and path.exists()

    b.close()

    assert path.exists()
    path.unlink()


def test_cleanup_aliases_close(cfg: BufferConfig) -> None:
    b = BytesIO(b"x" * 100, config=cfg)
    path = b.path
    assert path is not None and path.exists()

    b.cleanup()

    assert not path.exists()
    assert b.closed


def test_context_manager_closes(cfg: BufferConfig) -> None:
    path: Path | None = None
    with BytesIO(b"x" * 100, config=cfg) as b:
        path = b.path
        assert path is not None and path.exists()

    assert path is not None
    assert not path.exists()


def test_operations_on_closed_raise(cfg: BufferConfig) -> None:
    b = BytesIO(b"abc", config=cfg)
    b.close()

    with pytest.raises(ValueError):
        b.read()

    with pytest.raises(ValueError):
        b.write(b"x")

    with pytest.raises(ValueError):
        b.tell()

    with pytest.raises(ValueError):
        b.seek(0)

    with pytest.raises(ValueError):
        b.head()


def test_close_is_idempotent(cfg: BufferConfig) -> None:
    b = BytesIO(b"x" * 100, config=cfg)
    b.close()
    b.close()
    assert b.closed


def test_path_backed_writes_modify_original_file(cfg: BufferConfig, tmp_path: Path) -> None:
    path = tmp_path / "shared.bin"
    path.write_bytes(b"abcdef")

    b = BytesIO(path, config=cfg, copy=False)
    b.seek(2)
    b.write(b"ZZ")

    assert path.read_bytes() == b"abZZef"


def test_path_backed_truncate_modifies_original_file(cfg: BufferConfig, tmp_path: Path) -> None:
    path = tmp_path / "shared.bin"
    path.write_bytes(b"abcdef")

    b = BytesIO(path, config=cfg, copy=False)
    b.truncate(3)

    assert path.read_bytes() == b"abc"


def test_init_from_invalid_type_raises(cfg: BufferConfig) -> None:
    with pytest.raises(TypeError):
        BytesIO(123, config=cfg)  # type: ignore[arg-type]
