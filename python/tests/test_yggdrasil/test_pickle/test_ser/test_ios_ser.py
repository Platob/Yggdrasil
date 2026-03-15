from __future__ import annotations

import io

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.ios import IOSerialized
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def test_io_bytesio_roundtrip() -> None:
    src = io.BytesIO(b"hello io serializer")

    ser = Serialized.from_python_object(src)
    assert ser is not None

    assert isinstance(ser, IOSerialized)
    assert ser.tag == Tags.IO

    metadata = ser.metadata or {}
    assert metadata.get(b"io_kind") == b"binary"

    out = ser.as_python()

    assert isinstance(out, BytesIO)
    assert out.to_bytes() == b"hello io serializer"


def test_io_preserves_name_and_mode_metadata() -> None:
    class Dummy(io.BytesIO):
        name = "dummy.bin"
        mode = "rb"

    src = Dummy(b"abc123")

    ser = Serialized.from_python_object(src)
    assert ser is not None

    assert isinstance(ser, IOSerialized)
    assert ser.tag == Tags.IO

    metadata = ser.metadata or {}
    assert metadata.get(b"io_name") == b"dummy.bin"
    assert metadata.get(b"io_mode") == b"rb"
    assert metadata.get(b"io_kind") == b"binary"


def test_io_read_from_seekable_stream_restores_position() -> None:
    src = io.BytesIO(b"abcdef")
    src.seek(3)

    ser = Serialized.from_python_object(src)
    assert ser is not None

    assert src.tell() == 3

    out = ser.as_python()
    assert isinstance(out, BytesIO)
    assert out.to_bytes() == b"abcdef"
