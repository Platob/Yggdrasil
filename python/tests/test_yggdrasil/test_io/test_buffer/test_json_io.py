from __future__ import annotations

import io

import pyarrow as pa
import pytest
import yggdrasil.pickle.json as json_mod
from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.enums import GZIP, MediaType


def test_jsonio_read_pylist_empty_returns_empty_list():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.JSON)

    assert buf.size == 0
    assert io_.read_pylist() == []


def test_jsonio_read_pylist_wraps_non_list_payload():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.JSON)

    # Manually write a single JSON object (not a list)
    buf.write_bytes(b'{"a": 1, "b": "x"}')
    buf.seek(0)

    out = io_.read_pylist()
    assert out == [{"a": 1, "b": "x"}]


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
    io_ = MediaIO.make(buf, MimeTypes.JSON)

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


def test_jsonio_gzip_write_arrow_table_and_read_pylist_roundtrip():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MediaType(MimeTypes.JSON, codec=GZIP))

    table = pa.table(
        {
            "a": [1, 2],
            "b": ["x", "y"],
        }
    )

    io_.write_arrow_table(table)

    # Buffer payload should be compressed, not plain JSON text
    raw = buf.to_bytes()
    assert raw
    assert not raw.startswith(b"[{")
    assert not raw.startswith(b'{"')

    out = io_.read_pylist()
    assert out == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_jsonio_gzip_read_pylist_from_write_arrow_table_is_repeatable():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MediaType(MimeTypes.JSON, codec=GZIP))

    table = pa.table(
        {
            "id": [10, 20],
            "name": ["aa", "bb"],
        }
    )

    io_.write_arrow_table(table)

    out1 = io_.read_pylist()
    out2 = io_.read_pylist()

    expected = [
        {"id": 10, "name": "aa"},
        {"id": 20, "name": "bb"},
    ]
    assert out1 == expected
    assert out2 == expected


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
            with buf.view(pos=start, size=length) as f:
                json_mod.load(f)
        return

    with buf.view(pos=start, size=length) as f:
        parsed = json_mod.load(f)

    assert parsed == expected


def test_jsonio_gzip_write_arrow_table_and_read_pylist_roundtrip_nested_types():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MediaType(MimeTypes.JSON, codec=GZIP))

    table = pa.table(
        {
            "id": [1, 2],
            "tags": [
                ["a", "b"],
                ["x"],
            ],
            "attrs": [
                [("k1", 10), ("k2", 20)],
                [("only", 99)],
            ],
            "meta": [
                {"name": "row1", "active": True},
                {"name": "row2", "active": False},
            ],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("attrs", pa.map_(pa.string(), pa.int64())),
                pa.field(
                    "meta",
                    pa.struct(
                        [
                            pa.field("name", pa.string()),
                            pa.field("active", pa.bool_()),
                        ]
                    ),
                ),
            ]
        ),
    )

    io_.write_arrow_table(table)

    raw = buf.to_bytes()
    assert raw
    assert not raw.startswith(b"[{")
    assert not raw.startswith(b'{"')

    out = io_.read_pylist()
    assert out == [
        {
            "id": 1,
            "tags": ["a", "b"],
            "attrs": [["k1", 10], ["k2", 20]],
            "meta": {"name": "row1", "active": True},
        },
        {
            "id": 2,
            "tags": ["x"],
            "attrs": [["only", 99]],
            "meta": {"name": "row2", "active": False},
        },
    ]