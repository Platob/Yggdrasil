from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass
from functools import wraps
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from yggdrasil.mongoengine.decorator import with_mongo_connection
from yggdrasil.pickle.ser import Serialized, Tags
from yggdrasil.pickle.ser.complexs import (
    BaseExceptionSerialized,
    ClassSerialized,
    ComplexSerialized,
    DataclassSerialized,
    FunctionSerialized,
    ModuleSerialized,
    _dump_dataclass_payload,
    _dump_reference_function_payload,
    _load_dataclass_payload,
    _load_reference_function_payload, MethodSerialized,
)

GLOBAL_OFFSET = 5


class DemoClass:
    value = 123

    def mul(self, x: int) -> int:
        return x * 2


def simple_fn(x: int) -> int:
    return x + 1


def global_fn(x: int) -> int:
    return x + GLOBAL_OFFSET


def make_closure(y: int):
    z = 10

    def inner(x: int) -> int:
        return x + y + z

    return inner


def annotated_fn(
    x: int,
    y: int = 2,
    *,
    scale: int = 3,
) -> int:
    return (x + y) * scale


def plain_decorator(fn):
    def wrapper(x: int) -> int:
        return fn(x) + 100

    return wrapper


def multiplier_decorator(factor: int):
    def deco(fn):
        @wraps(fn)
        def wrapper(x: int) -> int:
            return fn(x) * factor

        return wrapper

    return deco


def logging_decorator(fn):
    calls: list[int] = []

    @wraps(fn)
    def wrapper(x: int) -> tuple[int, list[int]]:
        calls.append(x)
        return fn(x), list(calls)

    return wrapper


@plain_decorator
def decorated_plain_fn(x: int) -> int:
    return x * 2


@multiplier_decorator(3)
def decorated_wrapped_fn(x: int) -> int:
    return x + 1


@logging_decorator
def decorated_stateful_fn(x: int) -> int:
    return x + 5


@with_mongo_connection
def decorated_external(x: int) -> int:
    return x + 5


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


def test_function_serialized_simple_roundtrip() -> None:
    ser = FunctionSerialized.build_function(simple_fn)

    assert isinstance(ser, FunctionSerialized)

    fn = ser.as_python()
    assert fn(10) == 11
    assert fn.__name__ == simple_fn.__name__
    assert fn.__qualname__ == simple_fn.__qualname__


def test_function_serialized_with_global_roundtrip() -> None:
    ser = FunctionSerialized.build_function(global_fn)

    fn = ser.as_python()
    assert fn(7) == 12


def test_function_serialized_with_closure_roundtrip() -> None:
    original = make_closure(20)
    ser = FunctionSerialized.build_function(original)

    fn = ser.as_python()
    assert fn(3) == 33


def test_function_serialized_preserves_defaults_kwdefaults_annotations() -> None:
    ser = FunctionSerialized.build_function(annotated_fn)

    fn = ser.as_python()

    assert fn(1) == 9
    assert fn(1, 4, scale=2) == 10
    assert fn.__defaults__ == annotated_fn.__defaults__
    assert fn.__kwdefaults__ == annotated_fn.__kwdefaults__
    assert fn.__annotations__ == annotated_fn.__annotations__


def test_function_serialized_decorated_plain_closure_roundtrip() -> None:
    ser = FunctionSerialized.build_function(decorated_plain_fn)

    fn = ser.as_python()
    assert fn(2) == 104


def test_function_serialized_decorated_with_wraps_roundtrip() -> None:
    ser = FunctionSerialized.build_function(decorated_wrapped_fn)

    fn = ser.as_python()
    assert fn(4) == 15
    assert fn.__name__ == decorated_wrapped_fn.__name__
    assert fn.__qualname__ == decorated_wrapped_fn.__qualname__


def test_function_serialized_decorated_stateful_closure_roundtrip() -> None:
    ser = FunctionSerialized.build_function(decorated_stateful_fn)

    fn = ser.as_python()

    first = fn(1)
    second = fn(2)

    assert first == (6, [1])
    assert second == (7, [1, 2])


def test_function_serialized_decorated_external_roundtrip() -> None:
    ser = FunctionSerialized.build_function(decorated_external)

    fn = ser.as_python()

    first = fn(1)
    second = fn(2)

    assert first == 6
    assert second == 7


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


def test_function_serialized_write_to_roundtrip() -> None:
    original = FunctionSerialized.build_function(global_fn)

    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, FunctionSerialized)
    assert reread.tag == Tags.FUNCTION

    fn = reread.as_python()
    assert fn(10) == 15


def test_serialized_from_python_object_dispatches_module() -> None:
    ser = Serialized.from_python_object(math)

    assert isinstance(ser, ModuleSerialized)
    assert ser.as_python() is math


def test_serialized_from_python_object_dispatches_class() -> None:
    ser = Serialized.from_python_object(DemoClass)

    assert isinstance(ser, ClassSerialized)
    assert ser.as_python() is DemoClass


def test_serialized_from_python_object_dispatches_function() -> None:
    ser = Serialized.from_python_object(simple_fn)

    assert isinstance(ser, FunctionSerialized)
    fn = ser.as_python()
    assert fn(4) == 5


def test_serialized_from_python_object_dispatches_decorated_function() -> None:
    ser = Serialized.from_python_object(decorated_wrapped_fn)

    assert isinstance(ser, FunctionSerialized)
    fn = ser.as_python()
    assert fn(3) == 12


def test_function_serialized_nested_global_objects_use_main_dispatcher() -> None:
    local_state = {
        "nums": [1, 2, 3],
        "flag": True,
    }

    def fn(x: int) -> Any:
        return x + len(local_state["nums"]), local_state["flag"]

    ser = FunctionSerialized.build_function(fn)
    rebuilt = ser.as_python()

    assert rebuilt(5) == (8, True)


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


def test_reference_function_payload_roundtrip_for_pandas_site_packages():
    payload = _dump_reference_function_payload(pd.DataFrame.head)
    fn = _load_reference_function_payload(payload)

    assert callable(fn)
    assert fn is pd.DataFrame.head


def test_function_serialized_uses_reference_only_for_pandas_method():
    ser = FunctionSerialized.build_function(pd.DataFrame.head)
    fn = ser.as_python()

    assert fn is pd.DataFrame.head


def test_serialized_from_python_object_dispatches_reference_only_function():
    ser = Serialized.from_python_object(pd.DataFrame.head)

    assert isinstance(ser, FunctionSerialized)
    fn = ser.as_python()
    assert fn is pd.DataFrame.head


def test_class_serialized_reference_only_for_pandas_class():
    ser = ClassSerialized.build_class(pd.DataFrame)
    cls = ser.as_python()

    assert cls is pd.DataFrame


def test_reference_only_function_roundtrip_through_write_to():
    original = FunctionSerialized.build_function(pd.DataFrame.head)

    buf = original.write_to()
    reread = Serialized.read_from(buf, pos=0)

    assert isinstance(reread, FunctionSerialized)
    fn = reread.as_python()
    assert fn is pd.DataFrame.head


def test_reference_only_bound_method_uses_underlying_function():
    bound = pd.DataFrame({"a": [1, 2]}).head

    ser = Serialized.from_python_object(bound)
    fn = ser.as_python()

    assert fn().equals(bound())
