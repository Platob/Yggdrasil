from __future__ import annotations

import base64
import binascii

import pytest

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser import dump, dumps, load, loads
from yggdrasil.pickle.ser.constants import FORMAT_VERSION, MAGIC
from yggdrasil.pickle.ser.errors import SerializationError
from yggdrasil.pickle.ser.serialized import Serialized


def test_magic_matches_format_version() -> None:
    assert FORMAT_VERSION == 1
    assert MAGIC == b"YGG1"


def test_dumps_returns_bytes_with_magic_prefix() -> None:
    data = dumps({"a": 1, "b": [1, 2, 3]})

    assert isinstance(data, bytes)
    assert data.startswith(MAGIC)
    assert len(data) > len(MAGIC)


def test_dumps_b64_returns_base64_ascii_string() -> None:
    data = dumps({"a": 1}, b64=True)

    assert isinstance(data, str)

    decoded = base64.urlsafe_b64decode(data.encode("ascii"))
    assert decoded.startswith(MAGIC)
    assert len(decoded) > len(MAGIC)


def test_dump_writes_magic_and_payload_to_file_object() -> None:
    with BytesIO() as buffer:
        result = dump({"x": 42}, buffer)

        # Current implementation does not return anything
        assert result is None

        written = buffer.getvalue()
        assert written.startswith(MAGIC)
        assert len(written) > len(MAGIC)


def test_load_roundtrip_bytes_default_unpickle_true() -> None:
    obj = {
        "int": 123,
        "float": 1.25,
        "str": "abc",
        "list": [1, 2, 3],
        "dict": {"k": "v"},
        "bool": True,
        "none": None,
    }

    payload = dumps(obj)
    loaded = loads(payload)

    assert loaded == obj


def test_load_roundtrip_string_default_unpickle_true() -> None:
    obj = {"hello": "world", "items": [1, 2, 3]}

    payload = dumps(obj, b64=True)
    loaded = loads(payload)

    assert loaded == obj


def test_load_roundtrip_via_file_object() -> None:
    obj = ["alpha", 123, {"nested": True}]

    payload = dumps(obj)

    with BytesIO(payload, copy=False) as buffer:
        loaded = load(buffer)

    assert loaded == obj


def test_loads_bytes_unpickle_false_returns_serialized_instance() -> None:
    obj = {"a": 1}
    payload = dumps(obj)

    loaded = loads(payload, unpickle=False)

    assert isinstance(loaded, Serialized)


def test_loads_string_unpickle_false_returns_serialized_instance() -> None:
    obj = {"a": 1}
    payload = dumps(obj, b64=True)

    loaded = loads(payload, unpickle=False)

    assert isinstance(loaded, Serialized)


def test_load_rejects_invalid_magic_from_bytes() -> None:
    bad_payload = b"not-the-right-magic-header"

    with pytest.raises(SerializationError, match="Invalid magic header"):
        loads(bad_payload)


def test_load_rejects_invalid_magic_from_string() -> None:
    bad_payload = base64.urlsafe_b64encode(b"not-the-right-magic-header").decode("ascii")

    with pytest.raises(SerializationError, match="Invalid magic header"):
        loads(bad_payload)


def test_dump_and_load_with_metadata_bytes_payload() -> None:
    obj = {"hello": "world"}
    metadata = {b"source": b"unit-test", b"env": b"test"}

    payload = dumps(obj, metadata=metadata)
    loaded = loads(payload)

    assert loaded == obj


def test_dump_and_load_with_metadata_string_payload() -> None:
    obj = {"hello": "world"}
    metadata = {b"source": b"unit-test", b"env": b"test"}

    payload = dumps(obj, metadata=metadata, b64=True)
    loaded = loads(payload)

    assert loaded == obj


@pytest.mark.parametrize(
    "obj",
    [
        123,
        1.5,
        "hello",
        b"bytes",
        True,
        None,
        [1, 2, 3],
        {"a": 1, "b": 2},
    ],
)
def test_roundtrip_various_objects_bytes(obj: object) -> None:
    payload = dumps(obj)
    loaded = loads(payload)

    assert loaded == obj


@pytest.mark.parametrize(
    "obj",
    [
        123,
        1.5,
        "hello",
        b"bytes",
        True,
        None,
        [1, 2, 3],
        {"a": 1, "b": 2},
    ],
)
def test_roundtrip_various_objects_string(obj: object) -> None:
    payload = dumps(obj, b64=True)
    loaded = loads(payload)

    assert loaded == obj


def test_dumps_b64_matches_manual_base64_encoding() -> None:
    obj = {"codec": "test"}

    raw = dumps(obj, b64=False)
    encoded = dumps(obj, b64=True)

    expected = base64.urlsafe_b64encode(raw).decode("ascii")

    assert encoded == expected


def test_loads_accepts_base64_string_from_manual_encoding() -> None:
    obj = {"manual": True}

    raw = dumps(obj)
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")

    loaded = loads(encoded)

    assert loaded == obj


def test_invalid_base64_string_raises_binascii_error() -> None:
    with pytest.raises(binascii.Error):
        loads("!!! definitely not base64 !!!")