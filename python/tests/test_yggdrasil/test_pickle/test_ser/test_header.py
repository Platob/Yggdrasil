from __future__ import annotations

from yggdrasil.pickle.ser.constants import CODEC_NONE
from yggdrasil.pickle.ser.header import Header
from yggdrasil.pickle.ser.tags import Tags


def test_header_build_and_read_roundtrip() -> None:
    payload = b"hello world"
    metadata = {b"a": b"1", b"b": b"xyz"}

    head = Header.build(
        tag=Tags.BYTES,
        codec=CODEC_NONE,
        size=len(payload),
        metadata=metadata,
    )
    buf = head.write_to(payload)

    got = Header.read_from(buf, pos=0)

    assert got.tag == Tags.BYTES
    assert got.codec == CODEC_NONE
    assert got.size == len(payload)
    assert got.metadata == metadata
    assert got.payload_view(buf).to_bytes() == payload


def test_header_payload_end_is_absolute() -> None:
    payload = b"abc"
    head = Header.build(
        tag=Tags.BYTES,
        codec=CODEC_NONE,
        size=len(payload),
        metadata=None,
    )
    buf = head.write_to(payload)

    got = Header.read_from(buf, pos=0)
    assert got.payload_end == got.start + len(payload)


def test_header_write_rejects_payload_size_mismatch() -> None:
    head = Header.build(
        tag=Tags.BYTES,
        codec=CODEC_NONE,
        size=10,
        metadata=None,
    )

    try:
        head.write_to(b"abc")
    except ValueError as exc:
        assert "Payload size mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_header_metadata_can_be_none() -> None:
    payload = b"x"
    head = Header.build(
        tag=Tags.BYTES,
        codec=CODEC_NONE,
        size=len(payload),
        metadata=None,
    )
    buf = head.write_to(payload)

    got = Header.read_from(buf, pos=0)
    assert got.metadata is None