# tests/io/buffer/test_zip_io.py
from __future__ import annotations

import io
import zipfile

import pytest

import yggdrasil.pickle.json as json_mod
from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.zip_io import ZipIO, ZipOptions
from yggdrasil.io.enums import MimeType
import pyarrow as pa


def _make_zip_bytes(members: dict[str, bytes], *, compresslevel: int = 8) -> bytes:
    mem = io.BytesIO()
    try:
        with zipfile.ZipFile(
            mem,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=int(compresslevel),
        ) as zf:
            for name, payload in members.items():
                zf.writestr(name, payload)
    except TypeError:
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, payload in members.items():
                zf.writestr(name, payload)
    return mem.getvalue()


def test_zipio_read_empty_returns_empty_table():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.ZIP)

    tb = io_.read_arrow_table()
    assert tb.num_rows == 0


def test_zipio_write_and_read_roundtrip_json_inner():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.ZIP)

    inp = pa.Table.from_pylist([{"a": 1}, {"a": 2}, {"a": 3}])

    # Force inner json to avoid parquet dependency in unit tests
    io_.write_arrow_table(inp, options=ZipOptions(inner_media="json", member="data.json"))

    out = io_.read_arrow_table(options=ZipOptions(member="data.json"))
    assert out.to_pylist() == inp.to_pylist()


def test_zipio_read_concat_all_members_when_member_none():
    # Two json members with same schema => concat should stack rows
    t1 = pa.Table.from_pylist([{"x": 1}, {"x": 2}]).to_pylist()
    t2 = pa.Table.from_pylist([{"x": 3}]).to_pylist()

    m1 = json_mod.dumps(t1)  # bytes
    m2 = json_mod.dumps(t2)  # bytes

    zbytes = _make_zip_bytes({"part1.json": m1, "part2.json": m2})
    buf = BytesIO(zbytes)

    out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(options=ZipOptions(member=None))
    assert out.to_pylist() == (t1 + t2)


def test_zipio_read_single_member_when_specified():
    m1 = json_mod.dumps([{"x": 1}])     # bytes
    m2 = json_mod.dumps([{"x": 999}])   # bytes

    zbytes = _make_zip_bytes({"a.json": m1, "b.json": m2})
    buf = BytesIO(zbytes)

    out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(options=ZipOptions(member="b.json"))
    assert out.to_pylist() == [{"x": 999}]


def test_zipio_member_not_found_raises_keyerror():
    zbytes = _make_zip_bytes({"a.json": json_mod.dumps([{"x": 1}])})
    buf = BytesIO(zbytes)

    with pytest.raises(KeyError):
        MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(options=ZipOptions(member="nope.json"))


def test_zipio_does_not_move_parent_cursor_on_read():
    zbytes = _make_zip_bytes({"a.json": json_mod.dumps([{"x": 1}])})
    buf = BytesIO(zbytes)
    io_ = MediaIO.make(buf, MimeTypes.ZIP)

    buf.seek(7)
    before = buf.tell()

    out = io_.read_arrow_table(options=ZipOptions(member="a.json"))
    after = buf.tell()

    assert out.to_pylist() == [{"x": 1}]
    assert after == before  # view() must not touch parent BytesIO._pos


def test_zipio_read_infers_inner_media_from_extension_json():
    # Tests the "infer children mediatypes" path via member extension.
    payload = json_mod.dumps([{"k": "v"}])  # bytes
    zbytes = _make_zip_bytes({"data.json": payload})
    buf = BytesIO(zbytes)

    out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table()
    assert out.to_pylist() == [{"k": "v"}]


def test_zipio_force_inner_media_overrides_inference():
    # Name looks "wrong" (no extension), payload is json bytes.
    payload = json_mod.dumps([{"a": 1}])  # bytes
    zbytes = _make_zip_bytes({"weirdname": payload})
    buf = BytesIO(zbytes)

    out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
        options=ZipOptions(member="weirdname", inner_media="json", force_inner_media=True)
    )
    assert out.to_pylist() == [{"a": 1}]