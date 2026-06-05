from __future__ import annotations

import pickle
import socket
import threading

import pandas

from yggdrasil.pickle.ser.constants import CODEC_GZIP
from yggdrasil.pickle.ser.pandas import PandasTimestampSerialized
from yggdrasil.pickle.ser.pickles import (
    GenericObjectSerialized,
    RuntimeResourceSerialized,
    StdlibPickleSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


class _Point:
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __getnewargs_ex__(self):
        return (), {}

    def __getstate__(self):
        return {"x": self.x, "y": self.y}

    def __setstate__(self, state):
        self.x = state["x"]
        self.y = state["y"]


def test_pandas() -> None:
    obj = pandas.Timestamp(year=1, month=1, day=1)
    ser = Serialized.from_python_object(obj)

    assert isinstance(ser, PandasTimestampSerialized)
    assert ser.tag == Tags.PANDAS_TIMESTAMP
    out = ser.as_python()
    assert isinstance(out, pandas.Timestamp)
    assert out.year == 1
    assert out.month == 1
    assert out.day == 1


def test_generic_object_serialized_roundtrip() -> None:
    obj = _Point(2, 3)
    ser = Serialized.from_python_object(obj)

    assert isinstance(ser, GenericObjectSerialized)
    out = ser.as_python()
    assert isinstance(out, _Point)
    assert out.x == 2
    assert out.y == 3


def test_generic_object_serialization_skips_local_classes() -> None:
    class _Local:
        def __init__(self, value: int) -> None:
            self.value = value

    try:
        ser = Serialized.from_python_object(_Local(10))
    except ValueError:
        return
    assert ser.tag in (Tags.PICKLE, Tags.DILL, Tags.CLOUDPICKLE)


def test_runtime_thread_lock_serialized_roundtrip() -> None:
    lock = threading.Lock()
    lock.acquire()
    ser = Serialized.from_python_object(lock)

    assert isinstance(ser, RuntimeResourceSerialized)
    restored = ser.as_python()
    assert isinstance(restored, type(lock))
    assert restored.locked() is True


def test_runtime_socket_serialized_roundtrip() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.5)

    try:
        ser = Serialized.from_python_object(sock)
        assert isinstance(ser, RuntimeResourceSerialized)

        restored = ser.as_python()
        assert isinstance(restored, socket.socket)
        assert restored.family == socket.AF_INET
        assert restored.type == socket.SOCK_STREAM
        assert restored.gettimeout() == 2.5
        restored.close()
    finally:
        sock.close()


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

    def obj(x: int) -> int:
        return x + 1

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
