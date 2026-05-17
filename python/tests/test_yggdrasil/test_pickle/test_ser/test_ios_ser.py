from __future__ import annotations

import io

from yggdrasil.io import BytesIO
from yggdrasil.io.base import IO
from yggdrasil.data.enums.codec import GZIP
from yggdrasil.data.enums.media_type import MediaType
from yggdrasil.data.enums.mime_type import MimeTypes
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


def _media_type_of(buf: BytesIO):
    """Read the buffer's media type via the public stat surface."""
    return buf._parent.stat().media_type


def test_ygg_bytesio_media_type_roundtrip() -> None:
    """Media type on yggdrasil BytesIO is preserved through serialization."""
    mt = MediaType(MimeTypes.JSON)
    src = BytesIO(b'{"a":1}', media_type=mt)

    ser = Serialized.from_python_object(src)
    assert ser is not None

    metadata = ser.metadata or {}
    assert metadata.get(b"mt") == b"JSON"

    out = ser.as_python()
    assert isinstance(out, IO)
    out_mt = _media_type_of(out)
    assert out_mt is not None
    assert out_mt.mime_type is MimeTypes.JSON
    assert out_mt.codec is None
    assert bytes(out) == b'{"a":1}'


def test_ygg_bytesio_media_type_with_codec_roundtrip() -> None:
    """Media type with codec on yggdrasil BytesIO is preserved."""
    mt = MediaType(MimeTypes.PARQUET, codec=GZIP)
    src = BytesIO(b"compressed-payload", media_type=mt)

    ser = Serialized.from_python_object(src)
    assert ser is not None

    metadata = ser.metadata or {}
    assert metadata.get(b"mt") == b"PARQUET+gzip"

    out = ser.as_python()
    assert isinstance(out, IO)
    out_mt = _media_type_of(out)
    assert out_mt is not None
    assert out_mt.mime_type is MimeTypes.PARQUET
    assert out_mt.codec is not None
    assert out_mt.codec.name == "gzip"


def test_ygg_bytesio_no_media_type_no_metadata() -> None:
    """When no media_type is set, no 'mt' key should appear in metadata."""
    src = BytesIO(b"plain bytes")

    ser = Serialized.from_python_object(src)
    assert ser is not None

    metadata = ser.metadata or {}
    assert b"mt" not in metadata

    out = ser.as_python()
    assert isinstance(out, IO)
    assert _media_type_of(out) is None


def test_ygg_bytesio_media_type_promotes_subclass() -> None:
    """Media type on a BytesIO must round-trip to the registered leaf class.

    Pass media_type to the constructor so ``BytesIO.__new__``
    dispatches into the registered :class:`JsonIO` leaf.
    """
    from yggdrasil.io.primitive import JsonIO

    src = BytesIO(b'{"a":1}', media_type=MediaType(MimeTypes.JSON))
    assert isinstance(src, JsonIO)

    ser = Serialized.from_python_object(src)
    assert ser is not None
    out = ser.as_python()

    assert isinstance(out, JsonIO)
    out_mt = _media_type_of(out)
    assert out_mt is not None
    assert out_mt.mime_type is MimeTypes.JSON
