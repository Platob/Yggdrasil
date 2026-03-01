# tests/io/buffer/test_json_io.py
from __future__ import annotations

import io

import pytest

import yggdrasil.pickle.json as json_mod
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.json_io import JsonIO


def test_jsonio_read_pylist_empty_returns_empty_list():
    buf = BytesIO()
    io_ = JsonIO(buffer=buf)

    assert buf.size == 0
    assert io_.read_pylist() == []


def test_jsonio_read_pylist_wraps_non_list_payload():
    buf = BytesIO()
    io_ = JsonIO(buffer=buf)

    # Manually write a single JSON object (not a list)
    buf.write_bytes(b'{"a": 1, "b": "x"}')
    buf.seek(0)

    out = io_.read_pylist()
    assert out == [{"a": 1, "b": "x"}]


def test_bytesio_view_does_not_move_parent_cursor_on_read():
    buf = BytesIO(b'[{"x": 1}]')
    io_ = JsonIO(buffer=buf)

    buf.seek(1)
    before = buf.tell()

    out = io_.read_pylist()
    after = buf.tell()

    assert out == [{"x": 1}]
    assert after == before  # view must not touch BytesIO._pos


def test_bytesio_view_does_not_close_parent_on_exit():
    buf = BytesIO(b'[{"x": 1}]')
    assert not buf.closed

    with buf.view() as f:
        # Ensure it's usable as a file-like for json.load
        parsed = json_mod.load(f)

    assert parsed == [{"x": 1}]
    assert not buf.closed  # exiting the view must not close BytesIO


def test_jsonio_write_pylist_and_read_back_roundtrip():
    buf = BytesIO()
    io_ = JsonIO(buffer=buf)

    payload = [{"a": 1}, {"a": 2, "b": "ok"}, {"nested": {"k": 7}}]

    # IMPORTANT CONTRACT:
    # - buf.view() must be writable (TextIOWrapper -> BufferedWriter/RawIOBase.write)
    # - write must start at offset 0 (i.e., view opens at start)
    io_.write_pylist(payload)

    # Parent cursor should still be 0 (or unchanged) because view owns its own cursor.
    # (If your BytesIO constructor sets _pos=0, this should hold.)
    assert buf.tell() == 0

    # And reading should return the same list
    out = io_.read_pylist()
    assert out == payload


def test_bytesio_view_is_seekable_and_reusable_for_json_load():
    buf = BytesIO(b'[{"x": 1}, {"x": 2}]')

    with buf.view() as f:
        assert f.readable()
        assert f.seekable()
        # read once
        p1 = json_mod.load(f)

        # seek back and read again
        f.seek(0, io.SEEK_SET)
        p2 = json_mod.load(f)

    assert p1 == [{"x": 1}, {"x": 2}]
    assert p2 == [{"x": 1}, {"x": 2}]


@pytest.mark.parametrize(
    "start,length,expected",
    [
        (0, None, [{"a": 1}, {"a": 2}]),
        (0, 6, None),  # truncated JSON should fail
    ],
)
def test_bytesio_view_start_length_behavior(start, length, expected):
    buf = BytesIO(b'[{"a": 1}, {"a": 2}]')

    if expected is None:
        with pytest.raises(Exception):
            with buf.view(start=start, length=length) as f:
                json_mod.load(f)
        return

    with buf.view(start=start, length=length) as f:
        parsed = json_mod.load(f)

    assert parsed == expected