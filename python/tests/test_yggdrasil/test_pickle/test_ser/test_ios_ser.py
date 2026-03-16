from __future__ import annotations

import io

from yggdrasil.pickle.ser.ios import IOSerialized
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def test_io_bytesio_roundtrip() -> None:
    src = io.BytesIO(b"hello io serializer")

    ser = Serialized.from_python_object(src)
    assert ser is not None

    assert isinstance(ser, IOSerialized)
    assert ser.tag == Tags.IO_BYTES_BUFFER

    metadata = ser.metadata or {}
    assert metadata.get(b"k") == b"bb"

    out = ser.as_python()

    assert isinstance(out, io.BytesIO)
    assert out.getvalue() == b"hello io serializer"


def test_io_read_from_seekable_stream_restores_position() -> None:
    src = io.BytesIO(b"abcdef")
    src.seek(3)

    ser = Serialized.from_python_object(src)
    assert ser is not None

    assert src.tell() == 3

    out = ser.as_python()
    assert isinstance(out, io.BytesIO)
    assert out.getvalue() == b"abcdef"