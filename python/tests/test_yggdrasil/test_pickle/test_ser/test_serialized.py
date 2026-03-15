from __future__ import annotations

from yggdrasil.pickle.ser.constants import CODEC_ZSTD, CODEC_GZIP, CODEC_NONE, COMPRESS_THRESHOLD, CODEC_ZLIB
from yggdrasil.pickle.ser.primitives import BytesSerialized, UInt32Serialized
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def test_serialized_build_small_payload_stays_uncompressed() -> None:
    ser = Serialized.build(
        tag=Tags.BYTES,
        data=b"abc",
    )

    assert isinstance(ser, BytesSerialized)
    assert ser.codec == CODEC_NONE
    assert ser.decode() == b"abc"
    assert ser.as_python() == b"abc"


def test_serialized_build_large_payload_auto_compresses() -> None:
    payload = b"a" * (COMPRESS_THRESHOLD + 1024)

    ser = Serialized.build(
        tag=Tags.BYTES,
        data=payload,
    )

    assert isinstance(ser, BytesSerialized)
    assert ser.codec in (CODEC_NONE, CODEC_GZIP, CODEC_ZSTD, CODEC_ZLIB)
    assert ser.decode() == payload


def test_serialized_build_explicit_codec_none() -> None:
    payload = b"a" * (COMPRESS_THRESHOLD + 1024)

    ser = Serialized.build(
        tag=Tags.BYTES,
        data=payload,
        codec=CODEC_NONE,
    )

    assert ser.codec == CODEC_NONE
    assert ser.decode() == payload


def test_serialized_new_dispatches_primitive_subtype() -> None:
    ser = Serialized.build(
        tag=Tags.UINT32,
        data=(123).to_bytes(4, "big", signed=False),
    )

    assert isinstance(ser, UInt32Serialized)
    assert ser.value == 123


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


def test_serialized_constructor_dispatches_subtype() -> None:
    ser = Serialized.build(
        tag=Tags.BYTES,
        data=b"abc",
    )
    assert ser.__class__.__name__ == "BytesSerialized"