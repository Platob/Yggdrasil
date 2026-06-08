from __future__ import annotations

import pickle

import pytest

from yggdrasil.pickle.ser.constants import CODEC_NONE, COMPRESS_THRESHOLD
from yggdrasil.pickle.ser.primitives import UInt32Serialized
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


def test_serialized_instance_can_be_pickled_and_unpickled() -> None:
    ser = Serialized.build(
        tag=Tags.BYTES,
        data=b"pickle-roundtrip",
    )

    restored = pickle.loads(pickle.dumps(ser))

    assert isinstance(restored, type(ser))
    assert restored.decode() == b"pickle-roundtrip"


def test_serialized_pickle_falls_back_to_constructor_when_wire_dump_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ser = Serialized.build(
        tag=Tags.BYTES,
        data=b"fallback-roundtrip",
    )

    monkeypatch.setattr(type(ser), "write_to", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    restored = pickle.loads(pickle.dumps(ser))

    assert isinstance(restored, type(ser))
    assert restored.decode() == b"fallback-roundtrip"
