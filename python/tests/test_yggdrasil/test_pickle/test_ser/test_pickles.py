from __future__ import annotations

import pickle
import socket
import threading
import pytest

from yggdrasil.pickle.ser.constants import CODEC_GZIP
from yggdrasil.pickle.ser.pickles import AnyObjectSerialized, StdlibPickleSerialized
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


class _AnyObjectSample:
    def __init__(self, value: int) -> None:
        self.value = value


class _AnyObjectTextSample:
    def __init__(self, value: str) -> None:
        self.value = value


def test_stdlib_pickle_serialized() -> None:
    obj = {"a": 1, "b": [1, 2, 3]}
    data = pickle.dumps(obj)

    ser = Serialized.build(
        tag=Tags.PICKLE,
        data=data,
    )

    assert isinstance(ser, StdlibPickleSerialized)
    assert ser.as_python() == obj


def test_stdlib_pickle_serialized_gzip() -> None:
    obj = {"x": "y" * 10000}
    data = pickle.dumps(obj)

    ser = Serialized.build(
        tag=Tags.PICKLE,
        data=data,
        codec=CODEC_GZIP,
    )

    assert ser.as_python() == obj


def test_dill_serialized_if_available() -> None:
    try:
        import dill
    except Exception:
        return

    obj = lambda x: x + 1
    data = dill.dumps(obj)

    ser = Serialized.build(
        tag=Tags.DILL,
        data=data,
    )

    fn = ser.as_python()
    assert fn(3) == 4


def test_cloudpickle_serialized_if_available() -> None:
    try:
        import cloudpickle
    except Exception:
        return

    obj = {"f": lambda x: x * 2}
    data = cloudpickle.dumps(obj)

    ser = Serialized.build(
        tag=Tags.CLOUDPICKLE,
        data=data,
    )

    got = ser.as_python()
    assert isinstance(got, dict)
    assert "f" in got
    assert got["f"](5) == 10


def test_any_object_serialized_roundtrip() -> None:
    obj = _AnyObjectSample(7)
    ser = AnyObjectSerialized.from_python_object(obj)

    assert ser is not None
    assert ser.tag == Tags.ANY_OBJECT

    restored = ser.as_python()
    assert isinstance(restored, _AnyObjectSample)
    assert restored.value == 7


def test_serialized_from_python_object_uses_any_object_fallback() -> None:
    obj = _AnyObjectSample(12)

    ser = Serialized.from_python_object(obj)

    assert isinstance(ser, AnyObjectSerialized)
    assert ser.tag == Tags.ANY_OBJECT
    assert ser.as_python().value == 12


def test_serialized_from_python_object_string_keeps_primitive_tag() -> None:
    ser = Serialized.from_python_object("hello-any")
    assert ser.tag == Tags.UTF8_STRING
    assert ser.as_python() == "hello-any"


def test_any_object_serialized_roundtrip_unicode_string_state() -> None:
    obj = _AnyObjectTextSample("héllo-🌲")
    ser = AnyObjectSerialized.from_python_object(obj)
    assert ser is not None
    restored = ser.as_python()
    assert isinstance(restored, _AnyObjectTextSample)
    assert restored.value == "héllo-🌲"


def test_any_object_invalid_payload_short_raises() -> None:
    ser = Serialized.build(tag=Tags.ANY_OBJECT, data=b"\x01")
    with pytest.raises(ValueError, match="too short"):
        ser.as_python()


def test_any_object_invalid_payload_missing_state_raises() -> None:
    module = b"m"
    qualname = b"Q"
    payload = (
        b"\x01"
        + len(module).to_bytes(4, "big") + module
        + len(qualname).to_bytes(4, "big") + qualname
    )
    ser = Serialized.build(tag=Tags.ANY_OBJECT, data=payload)
    with pytest.raises(ValueError, match="missing state payload"):
        ser.as_python()


def test_serialized_from_python_object_socket_uses_sensitive_handler() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        ser = Serialized.from_python_object(s)
        assert ser.tag == Tags.SENSITIVE_OBJECT
        restored = ser.as_python()
        assert restored["kind"] == "sensitive_object"
        assert restored["module"] == "socket"
    finally:
        s.close()


def test_serialized_from_python_object_lock_uses_sensitive_handler() -> None:
    lock = threading.Lock()
    ser = Serialized.from_python_object(lock)
    assert ser.tag == Tags.SENSITIVE_OBJECT
    restored = ser.as_python()
    assert restored["kind"] == "sensitive_object"
    assert "lock" in restored["qualname"].lower()
