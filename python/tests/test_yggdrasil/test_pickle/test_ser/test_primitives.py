from __future__ import annotations

import pytest

from yggdrasil.pickle.ser.constants import CODEC_GZIP, CODEC_NONE, COMPRESS_THRESHOLD
from yggdrasil.pickle.ser.primitives import UInt32Serialized, UInt8Serialized
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def test_serialized_build_explicit_codec_none() -> None:
    payload = b"a" * (COMPRESS_THRESHOLD + 1024)

    ser = Serialized.build(
        tag=Tags.BYTES,
        data=payload,
        codec=CODEC_NONE,
    )

    assert ser.codec == CODEC_NONE
    assert ser.to_bytes() == payload
    assert ser.decode() == payload


def test_serialized_build_preserves_metadata() -> None:
    metadata = {
        b"foo": b"bar",
        b"answer": b"42",
    }

    ser = Serialized.build(
        tag=Tags.BYTES,
        data=b"xyz",
        metadata=metadata,
    )

    assert ser.metadata == metadata
    assert ser.decode() == b"xyz"


def test_serialized_new_dispatches_primitive_subtype() -> None:
    ser = Serialized.build(
        tag=Tags.UINT32,
        data=(123).to_bytes(4, "big", signed=False),
    )

    assert isinstance(ser, UInt32Serialized)
    assert ser.value == 123
    assert ser.as_python() == 123


def test_serialized_read_from_dispatches_subtype() -> None:
    original = Serialized.build(
        tag=Tags.UINT32,
        data=(77).to_bytes(4, "big", signed=False),
    )

    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, UInt32Serialized)
    assert reread.value == 77


def test_serialized_write_to_roundtrip() -> None:
    original = Serialized.build(
        tag=Tags.BYTES,
        data=b"roundtrip-test",
    )

    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert type(reread) is type(original)
    assert reread.tag == original.tag
    assert reread.codec == original.codec
    assert reread.decode() == b"roundtrip-test"


def test_serialized_write_to_roundtrip_with_metadata() -> None:
    original = Serialized.build(
        tag=Tags.BYTES,
        data=b"roundtrip-metadata",
        metadata={b"k": b"v"},
    )

    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert type(reread) is type(original)
    assert reread.tag == original.tag
    assert reread.codec == original.codec
    assert reread.metadata == {b"k": b"v"}
    assert reread.decode() == b"roundtrip-metadata"


def test_serialized_constructor_dispatches_subtype() -> None:
    ser = Serialized.build(
        tag=Tags.BYTES,
        data=b"abc",
    )
    assert ser.__class__.__name__ == "BytesSerialized"


def test_serialized_codec_label_none() -> None:
    ser = Serialized.build(
        tag=Tags.BYTES,
        data=b"abc",
        codec=CODEC_NONE,
    )

    assert ser.codec_label == "none"


def test_serialized_codec_label_gzip() -> None:
    ser = Serialized.build(
        tag=Tags.BYTES,
        data=b"abc" * 1000,
        codec=CODEC_GZIP,
    )

    assert ser.codec_label == "gzip"


def test_serialized_to_bytes_returns_wire_payload_not_decoded_payload() -> None:
    payload = b"abc" * 1000

    ser = Serialized.build(
        tag=Tags.BYTES,
        data=payload,
        codec=CODEC_GZIP,
    )

    assert ser.decode() == payload
    assert ser.to_bytes() != payload


def test_serialized_from_python_object_int_dispatches_uint32() -> None:
    ser = Serialized.from_python_object(123)

    assert isinstance(ser, UInt8Serialized)
    assert ser.as_python() == 123


def test_serialized_from_python_object_none() -> None:
    ser = Serialized.from_python_object(None)

    assert ser.tag == Tags.NONE
    assert ser.as_python() is None


def test_serialized_unknown_tag_falls_back_to_base_serialized() -> None:
    unknown_tag = 65535

    with pytest.raises(NotImplementedError):
        Serialized.build(
            tag=unknown_tag,
            data=b"some data",
        )