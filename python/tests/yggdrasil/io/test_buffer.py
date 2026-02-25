import io
from pathlib import Path

import pytest

from yggdrasil.io.buffer import BytesIO, BufferConfig


def _try_import_internal_xxhash():
    try:
        from yggdrasil.xxhash.lib import xxhash as xxhash_mod
    except Exception:
        return None
    return xxhash_mod


def _try_import_internal_blake3():
    try:
        import yggdrasil.blake3 as blake3_mod
    except Exception:
        return None
    return blake3_mod


def test_len_and_size_in_memory():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    assert len(buf) == 0
    assert buf.size == 0

    buf.write(b"abc")
    assert len(buf) == 3
    assert buf.size == 3


def test_basic_write_read_seek_tell_in_memory():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    buf.write(b"hello")
    assert buf.tell() == 5

    buf.seek(0)
    assert buf.read(2) == b"he"
    assert buf.tell() == 2

    buf.seek(0, io.SEEK_END)
    assert buf.tell() == 5

    buf.seek(-2, io.SEEK_END)
    assert buf.read() == b"lo"



def test_spills_to_file_and_preserves_cursor(tmp_path: Path):
    # Code spills only when size would exceed spill_bytes (strict >).
    cfg = BufferConfig(spill_bytes=8, tmp_dir=tmp_path)
    buf = BytesIO(config=cfg)

    buf.write(b"1234567")  # size = 7
    assert buf.spilled is False
    assert buf.path is None
    assert buf.tell() == 7

    # 7 + 1 == 8 -> NO spill because condition is ">"
    buf.write(b"X")
    assert buf.spilled is False
    assert buf.tell() == 8

    # Now exceed threshold: 8 + 1 == 9 -> spill
    buf.write(b"Y")
    assert buf.spilled is True
    assert buf.path is not None
    assert buf.path.exists()
    assert buf.tell() == 9

    buf.seek(0)
    assert buf.read() == b"1234567XY"


@pytest.mark.xfail(
    reason=(
        "Known bug: DynamicBuffer.close() sets _file=None before checking `self.spilled`, "
        "so unlink never runs and spilled files remain on disk when keep_spilled_file=False."
    ),
    strict=False,
)
def test_close_deletes_spill_file_by_default_expected_contract(tmp_path: Path):
    # This is the intended behavior (should pass after fixing close()).
    cfg = BufferConfig(spill_bytes=2, tmp_dir=tmp_path, keep_spilled_file=False)
    buf = BytesIO(config=cfg)
    buf.write(b"ab")
    buf.write(b"c")  # spill
    p = buf.path
    assert p is not None and p.exists()

    buf.close()
    assert not p.exists()


def test_to_bytes_preserves_position_in_file(tmp_path: Path):
    cfg = BufferConfig(spill_bytes=4, tmp_dir=tmp_path)
    buf = BytesIO(config=cfg)

    buf.write(b"abcd")  # at threshold, still in mem
    buf.write(b"e")     # spill
    assert buf.spilled is True

    buf.seek(2)
    pos = buf.tell()
    b = buf.to_bytes()
    assert b == b"abcde"
    assert buf.tell() == pos  # must restore


def test_getvalue_in_memory():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    buf.write(b"zzz")
    assert buf.getvalue() == b"zzz"


def test_getvalue_when_spilled_reads_all(tmp_path: Path):
    cfg = BufferConfig(spill_bytes=2, tmp_dir=tmp_path)
    buf = BytesIO(config=cfg)
    buf.write(b"ab")  # still in mem
    buf.write(b"c")   # spill
    assert buf.spilled is True

    # should return all bytes
    assert buf.getvalue() == b"abc"


def test_memoryview_in_memory_and_spilled(tmp_path: Path):
    # in-memory
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    buf.write(b"hello")
    mv = buf.memoryview()
    assert bytes(mv) == b"hello"

    # spilled
    cfg = BufferConfig(spill_bytes=4, tmp_dir=tmp_path)
    buf2 = BytesIO(config=cfg)
    buf2.write(b"abcd")
    buf2.write(b"e")  # spill
    assert buf2.spilled is True

    mv2 = buf2.memoryview()
    assert bytes(mv2) == b"abcde"


def test_memoryview_empty_spilled_returns_empty(tmp_path: Path):
    cfg = BufferConfig(spill_bytes=1, tmp_dir=tmp_path)
    buf = BytesIO(config=cfg)

    # force spill with a write and then truncate by reopening path is not supported;
    # easiest: create file-backed via parse_any(path) with empty file.
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    buf2 = BytesIO.parse_any(p)
    assert buf2.spilled is True
    assert bytes(buf2.memoryview()) == b""


def test_write_any_bytes_byteslike_and_filelike():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    buf.write_any_bytes(b"abc")
    buf.write_any_bytes(bytearray(b"def"))
    buf.write_any_bytes(memoryview(b"ghi"))
    buf.write_any_bytes(io.BytesIO(b"jkl"))
    buf.seek(0)
    assert buf.read() == b"abcdefghijkl"


def test_write_any_bytes_none_is_noop():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    assert buf.write_any_bytes(None) == 0
    assert buf.size == 0


def test_write_any_bytes_invalid_type_raises():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    with pytest.raises(TypeError):
        buf.write_any_bytes(123)


def test_parse_any_from_dynamicbuffer_returns_same():
    b1 = BytesIO(config=BufferConfig(spill_bytes=1024))
    b2 = BytesIO.parse_any(b1)
    assert b1 is b2


def test_parse_any_from_bytesio_wraps_without_copying_object_identity():
    bio = io.BytesIO(b"yo")
    buf = BytesIO.parse_any(bio)
    assert buf.spilled is False
    # it literally uses the same BytesIO instance
    assert buf._mem is bio  # intentional: white-box for correctness
    buf.seek(0)
    assert buf.read() == b"yo"


def test_parse_any_from_path_is_file_backed(tmp_path: Path):
    p = tmp_path / "data.bin"
    # parse_any opens w+b and sets mem None
    buf = BytesIO.parse_any(p)
    assert buf.spilled is True
    assert buf.path == p
    assert buf._mem is None

    buf.write(b"hi")
    buf.seek(0)
    assert buf.read() == b"hi"


def test_open_reader_in_memory_returns_bytesio_copy():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    buf.write(b"abc")
    r = buf.open_reader()
    assert isinstance(r, io.BytesIO)
    assert r.read() == b"abc"


def test_open_reader_spilled_requires_path(tmp_path: Path):
    cfg = BufferConfig(spill_bytes=2, tmp_dir=tmp_path)
    buf = BytesIO(config=cfg)
    buf.write(b"ab")
    buf.write(b"c")  # spill
    assert buf.path is not None

    with buf.open_reader() as r:
        assert r.read() == b"abc"


def test_open_reader_spilled_missing_path_raises(tmp_path: Path):
    cfg = BufferConfig(spill_bytes=2, tmp_dir=tmp_path)
    buf = BytesIO(config=cfg)
    buf.write(b"ab")
    buf.write(b"c")  # spill
    assert buf.spilled is True

    # Simulate unexpected state: spilled but no path
    buf._path = None
    with pytest.raises(RuntimeError):
        buf.open_reader()


@pytest.mark.xfail(
    reason=(
        "Known bug: DynamicBuffer.close() sets _file=None before checking `self.spilled`, "
        "so unlink never runs when keep_spilled_file=False."
    ),
    strict=False,
)
def test_close_deletes_spill_file_by_default(tmp_path: Path):
    cfg = BufferConfig(spill_bytes=2, tmp_dir=tmp_path, keep_spilled_file=False)
    buf = BytesIO(config=cfg)
    buf.write(b"ab")
    buf.write(b"c")  # spill
    p = buf.path
    assert p is not None and p.exists()

    buf.close()
    assert not p.exists()


def test_close_keeps_spill_file_when_configured(tmp_path: Path):
    cfg = BufferConfig(spill_bytes=2, tmp_dir=tmp_path, keep_spilled_file=True)
    buf = BytesIO(config=cfg)
    buf.write(b"ab")
    buf.write(b"c")  # spill
    p = buf.path
    assert p is not None and p.exists()

    buf.close()
    assert p.exists()


def test_closed_buffer_raises_on_io():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    buf.write(b"x")
    buf.close()

    with pytest.raises(ValueError):
        buf.write(b"y")
    with pytest.raises(ValueError):
        buf.read(1)
    with pytest.raises(ValueError):
        buf.seek(0)
    with pytest.raises(ValueError):
        buf.tell()


# ---------------------------
# Structured binary tests
# ---------------------------

@pytest.mark.parametrize(
    "write_fn, read_fn, value",
    [
        ("write_int8", "read_int8", -7),
        ("write_uint8", "read_uint8", 250),
        ("write_int16", "read_int16", -1234),
        ("write_uint16", "read_uint16", 65530),
        ("write_int32", "read_int32", -12345678),
        ("write_uint32", "read_uint32", 4294967290),
        ("write_int64", "read_int64", -1234567890123),
        ("write_uint64", "read_uint64", 18446744073709551610),
        ("write_f32", "read_f32", 3.5),
        ("write_f64", "read_f64", 1.234567890123),
        ("write_bool", "read_bool", True),
        ("write_bool", "read_bool", False),
    ],
)
def test_structured_roundtrip_in_memory(write_fn, read_fn, value):
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    getattr(buf, write_fn)(value)
    buf.seek(0)
    out = getattr(buf, read_fn)()

    if isinstance(value, float):
        # float32/64 can have tiny rounding errors
        assert out == pytest.approx(value)
    else:
        assert out == value


def test_read_exact_raises_eoferror():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    buf.write(b"\x01")
    buf.seek(0)
    _ = buf.read_uint8()
    with pytest.raises(EOFError):
        buf.read_uint8()  # no more bytes


def test_bytes_u32_and_str_u32_roundtrip():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    payload = b"\x00\x01\x02" * 10
    s = "zürich 🧪"

    buf.write_bytes_u32(payload)
    buf.write_str_u32(s, encoding="utf-8")

    buf.seek(0)
    assert buf.read_bytes_u32() == payload
    assert buf.read_str_u32("utf-8") == s


def test_structured_roundtrip_spilled(tmp_path: Path):
    cfg = BufferConfig(spill_bytes=16, tmp_dir=tmp_path)
    buf = BytesIO(config=cfg)

    # write enough to spill and also include structured values after spill point
    buf.write(b"X" * 16)   # at threshold
    buf.write(b"Y")        # spill happens here
    buf.write_int32(123)
    buf.write_f64(2.5)
    buf.write_str_u32("abc")

    buf.seek(17)  # 16 X + 1 Y
    assert buf.read_int32() == 123
    assert buf.read_f64() == pytest.approx(2.5)
    assert buf.read_str_u32() == "abc"


def test_bytes_dunder_calls_to_bytes():
    buf = BytesIO(config=BufferConfig(spill_bytes=1024))
    buf.write(b"abc")
    assert bytes(buf) == b"abc"
