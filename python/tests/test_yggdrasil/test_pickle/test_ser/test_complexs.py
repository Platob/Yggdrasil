from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass
from types import ModuleType

import pytest

from yggdrasil.pickle.ser import Serialized, Tags
from yggdrasil.pickle.ser.complexs import (
    BaseExceptionSerialized,
    ClassSerialized,
    ComplexSerialized,
    ModuleSerialized,
)
from yggdrasil.pickle.ser.dataclasses import (
    DataclassSerialized,
    _dump_dataclass_payload,
    _load_dataclass_payload,
)


class DemoClass:
    value = 123

    def mul(self, x: int) -> int:
        return x * 2


def test_module_serialized_value_roundtrip() -> None:
    ser = ModuleSerialized.build_module(math)

    assert isinstance(ser, ModuleSerialized)
    mod = ser.as_python()

    assert mod is math
    assert mod.sqrt(9) == 3.0


def test_class_serialized_value_roundtrip() -> None:
    ser = ClassSerialized.build_class(DemoClass)

    assert isinstance(ser, ClassSerialized)
    cls = ser.as_python()

    assert cls is DemoClass
    assert cls.value == 123
    assert cls().mul(4) == 8


def test_module_serialized_write_to_roundtrip() -> None:
    original = ModuleSerialized.build_module(math)

    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, ModuleSerialized)
    assert reread.tag == Tags.MODULE
    assert reread.as_python() is math


def test_class_serialized_write_to_roundtrip() -> None:
    original = ClassSerialized.build_class(DemoClass)

    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, ClassSerialized)
    assert reread.tag == Tags.CLASS
    assert reread.as_python() is DemoClass


def test_serialized_from_python_object_dispatches_module() -> None:
    ser = Serialized.from_python_object(math)

    assert isinstance(ser, ModuleSerialized)
    assert ser.as_python() is math


def test_serialized_from_python_object_dispatches_class() -> None:
    ser = Serialized.from_python_object(DemoClass)

    assert isinstance(ser, ClassSerialized)
    assert ser.as_python() is DemoClass


# ===================================================================
# Exception tests
# ===================================================================

class HttpError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(code, msg)
        self.code = code
        self.msg = msg


def test_base_exception_serialized_value_error_roundtrip() -> None:
    exc = ValueError("bad input")

    ser = BaseExceptionSerialized.build_exception(exc)

    assert isinstance(ser, BaseExceptionSerialized)

    got = ser.as_python()
    assert isinstance(got, ValueError)
    assert got.args == ("bad input",)


def test_base_exception_serialized_custom_exception_roundtrip() -> None:
    exc = HttpError(404, "missing")
    ser = BaseExceptionSerialized.build_exception(exc)

    got = ser.as_python()
    assert isinstance(got, HttpError)
    assert got.args == (404, "missing")
    assert got.code == 404
    assert got.msg == "missing"


def test_serialized_from_python_object_dispatches_base_exception() -> None:
    exc = RuntimeError("boom")

    ser = Serialized.from_python_object(exc)

    assert isinstance(ser, BaseExceptionSerialized)
    got = ser.as_python()
    assert isinstance(got, RuntimeError)
    assert got.args == ("boom",)


def test_base_exception_serialized_write_to_roundtrip() -> None:
    exc = KeyError("x")

    original = BaseExceptionSerialized.build_exception(exc)
    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, BaseExceptionSerialized)
    got = reread.as_python()
    assert isinstance(got, KeyError)
    assert got.args == ("x",)


# ===================================================================
# Dataclass tests
# ===================================================================

@dataclass
class Point:
    x: int
    y: int


@dataclass
class Nested:
    name: str
    point: Point


@dataclass(frozen=True)
class FrozenPoint:
    x: int
    y: int


@dataclass
class WithNonInit:
    x: int
    y: int = field(init=False)

    def __post_init__(self) -> None:
        self.y = self.x * 10


@dataclass(slots=True)
class SlotPoint:
    x: int
    y: int


@dataclass
class StatefulPoint:
    x: int
    y: int

    def __getstate__(self):
        return {"x": self.x * 10, "y": self.y * 10}

    def __setstate__(self, state):
        self.x = state["x"] // 10
        self.y = state["y"] // 10


def sample_function(x: int, y: int = 2) -> int:
    return x + y


class BoomError(RuntimeError):
    pass


def test_dump_load_dataclass_payload_simple():
    obj = Point(1, 2)

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, Point)
    assert restored == obj
    assert restored is not obj


def test_dump_load_dataclass_payload_nested():
    obj = Nested(name="nika", point=Point(3, 4))

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, Nested)
    assert restored == obj
    assert isinstance(restored.point, Point)
    assert restored.point == Point(3, 4)


def test_dump_load_dataclass_payload_frozen():
    obj = FrozenPoint(5, 6)

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, FrozenPoint)
    assert restored == obj


def test_dump_load_dataclass_payload_non_init_field():
    obj = WithNonInit(7)
    assert obj.y == 70

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, WithNonInit)
    assert restored.x == 7
    assert restored.y == 70


def test_dump_load_dataclass_payload_extra_dict_state():
    obj = Point(10, 20)
    obj.label = "origin-ish"

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, Point)
    assert restored == Point(10, 20)
    assert restored.label == "origin-ish"


def test_dump_load_dataclass_payload_slots():
    obj = SlotPoint(8, 9)

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, SlotPoint)
    assert restored == obj


def test_dump_load_dataclass_payload_custom_getstate_setstate():
    obj = StatefulPoint(2, 3)

    payload = _dump_dataclass_payload(obj)
    restored = _load_dataclass_payload(payload)

    assert isinstance(restored, StatefulPoint)
    assert restored == StatefulPoint(2, 3)


def test_dataclass_serialized_build_and_restore():
    obj = Point(11, 22)

    ser = DataclassSerialized.build_dataclass(obj)
    restored = ser.as_python()

    assert isinstance(restored, Point)
    assert restored == obj
    assert restored is not obj


def test_dataclass_serialized_value_property():
    obj = Point(4, 5)

    ser = DataclassSerialized.build_dataclass(obj)
    restored = ser.value

    assert isinstance(restored, Point)
    assert restored == obj


def test_complex_serialized_from_python_object_uses_dataclass_serialized():
    obj = Point(100, 200)

    ser = ComplexSerialized.from_python_object(obj)

    assert ser is not None
    assert isinstance(ser, DataclassSerialized)
    restored = ser.as_python()
    assert restored == obj


def test_complex_serialized_dataclass_type_prefers_class_serialized():
    ser = ComplexSerialized.from_python_object(Point)

    assert ser is not None
    assert isinstance(ser, ClassSerialized)
    restored = ser.as_python()
    assert restored is Point


def test_complex_serialized_function_still_uses_function_serialized():
    from yggdrasil.pickle.ser.complexs import FunctionSerialized

    ser = ComplexSerialized.from_python_object(sample_function)

    assert ser is not None
    assert isinstance(ser, FunctionSerialized)
    restored = ser.as_python()
    assert restored(3, 4) == 7
    assert restored(3) == 5


def test_complex_serialized_exception_still_uses_base_exception_serialized():
    exc = BoomError("boom")
    exc.code = 500

    ser = ComplexSerialized.from_python_object(exc)

    assert ser is not None
    assert isinstance(ser, BaseExceptionSerialized)
    restored = ser.as_python()
    assert isinstance(restored, BoomError)
    assert restored.args == ("boom",)
    assert restored.code == 500


def test_complex_serialized_module_still_uses_module_serialized():
    ser = ComplexSerialized.from_python_object(pytest)

    assert ser is not None
    assert isinstance(ser, ModuleSerialized)
    restored = ser.as_python()
    assert isinstance(restored, ModuleType)
    assert restored.__name__ == pytest.__name__


def test_dataclass_serialized_roundtrip_with_nested_and_extra_state():
    obj = Nested(name="alpha", point=Point(1, 9))
    obj.tag = {"kind": "demo", "ok": True}

    ser = DataclassSerialized.build_dataclass(obj)
    restored = ser.as_python()

    assert isinstance(restored, Nested)
    assert restored == Nested(name="alpha", point=Point(1, 9))
    assert restored.tag == {"kind": "demo", "ok": True}


def test_dataclass_serialized_frozen_with_nested_value():
    @dataclass(frozen=True)
    class FrozenNested:
        point: Point
        name: str

    obj = FrozenNested(point=Point(2, 3), name="fp")

    ser = DataclassSerialized.build_dataclass(obj)
    restored = ser.as_python()

    assert is_dataclass(restored)
    assert restored.__class__.__name__ == "FrozenNested"
    assert restored.point == obj.point
    assert restored.name == obj.name


def test_complex_serialized_from_python_object_non_supported_returns_none():
    assert ComplexSerialized.from_python_object(12345) is None
    assert ComplexSerialized.from_python_object("hello") is None
    assert ComplexSerialized.from_python_object([1, 2, 3]) is None


def test_class_serialized_reference_only_for_pandas_class():
    import pandas as pd

    ser = ClassSerialized.build_class(pd.DataFrame)
    cls = ser.as_python()

    assert cls is pd.DataFrame

