from __future__ import annotations

import pickle

from yggdrasil.pickle.ser.constants import CODEC_GZIP
from yggdrasil.pickle.ser.pickles import StdlibPickleSerialized
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


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