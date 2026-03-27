from __future__ import annotations

import datetime as dt
import enum
import inspect
import json
import math
import operator
import re
import textwrap
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from functools import partial, wraps
from pathlib import PurePosixPath
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from yggdrasil.mongoengine.decorator import with_mongo_connection
from yggdrasil.pickle.ser import Serialized, Tags
from yggdrasil.pickle.ser.complexs import (
    FunctionSerialized,
    MethodSerialized,
    _FN_FULL_DEFINITION_GLOBALS,
    _FN_FULL_GLOBALS,
    _FN_FULL_MARSHAL,
    _FN_FULL_PY_VERSION,
    _FN_FULL_SOURCE,
    _PYTHON_VERSION,
    _deserialize_nested,
    _dump_function_payload,
    _dump_method_payload,
    _dump_reference_function_payload,
    _load_function_code_payload,
    _load_function_payload,
    _load_method_payload,
    _load_reference_function_payload,
    _serialize_nested,
)

GLOBAL_OFFSET = 5
GLOBAL_DICT = {"key": "value", "count": 42}
GLOBAL_LIST = [10, 20, 30]
GLOBAL_TUPLE = (1, 2, 3)
GLOBAL_SET = frozenset({7, 8, 9})
GLOBAL_NESTED = {"inner": [1, {"deep": True}]}

METHOD_GLOBAL_BONUS = 11
METHOD_GLOBAL_SCALE = 4


def fn_uses_global_int(x: int) -> int:
    return x + GLOBAL_OFFSET


def fn_uses_global_dict(key: str) -> Any:
    return GLOBAL_DICT.get(key)


def fn_uses_global_list() -> int:
    return sum(GLOBAL_LIST)


def fn_uses_global_tuple() -> tuple:
    return GLOBAL_TUPLE + (4,)


def fn_uses_global_set() -> bool:
    return 7 in GLOBAL_SET


def fn_uses_global_nested() -> Any:
    return GLOBAL_NESTED["inner"][1]["deep"]


def fn_uses_multiple_globals(x: int) -> int:
    return x + GLOBAL_OFFSET + len(GLOBAL_LIST)


def fn_uses_builtin_functions(x: list) -> int:
    return len(x) + max(x) + min(x)


def fn_uses_math_module(x: float) -> float:
    return math.sqrt(x) + math.pi


def fn_calls_other_function(x: int) -> int:
    return fn_uses_global_int(x) * 2


def make_adder(y: int):
    def add(x: int) -> int:
        return x + y
    return add


def make_nested_closure(a: int, b: int):
    def middle():
        c = a + b

        def inner(x: int) -> int:
            return x + c

        return inner

    return middle()


def make_closure_over_mutable(items: list):
    def append_and_return(v: Any) -> list:
        items.append(v)
        return list(items)
    return append_and_return


def make_closure_over_dict():
    state = {"count": 0}

    def increment() -> int:
        state["count"] += 1
        return state["count"]

    return increment


def make_counter():
    count = 0
    name = "counter"

    def tick() -> tuple[str, int]:
        nonlocal count
        count += 1
        return name, count

    return tick


def make_closure_with_global_and_local(y: int):
    def compute(x: int) -> int:
        return x + y + GLOBAL_OFFSET
    return compute


def plain_decorator(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) + 100
    return wrapper


def multiplier_decorator(factor: int):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs) * factor
        return wrapper
    return deco


def logging_decorator(fn):
    calls: list[object] = []

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if len(args) >= 2:
            calls.append(args[1])
        elif args:
            calls.append(args[0])
        elif "x" in kwargs:
            calls.append(kwargs["x"])
        else:
            calls.append(None)
        return fn(*args, **kwargs), list(calls)

    return wrapper


@plain_decorator
def decorated_plain(x: int) -> int:
    return x * 2


@multiplier_decorator(3)
def decorated_wraps(x: int) -> int:
    return x + 1


@logging_decorator
def decorated_stateful(x: int) -> int:
    return x + 5


@with_mongo_connection
def decorated_external(x: int) -> int:
    return x + 5


def annotated_fn(
    x: int,
    y: int = 2,
    *,
    scale: int = 3,
) -> int:
    return (x + y) * scale


def fn_no_args() -> str:
    return "hello"


def recursive_factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * recursive_factorial(n - 1)


simple_lambda = lambda x: x * 2  # noqa: E731


def generator_fn(n: int):
    for i in range(n):
        yield i + GLOBAL_OFFSET


@dataclass
class Calculator:
    base: int = 0

    def add(self, x: int) -> int:
        return self.base + x

    def mul(self, x: int) -> int:
        return self.base * x

    @classmethod
    def from_value(cls, v: int) -> "Calculator":
        return cls(base=v)

    @staticmethod
    def static_add(a: int, b: int) -> int:
        return a + b

    @multiplier_decorator(2)
    def decorated_uses_global(self, x: int) -> int:
        return self.base + x + METHOD_GLOBAL_BONUS

    @logging_decorator
    def decorated_stateful_method(self, x: int) -> int:
        return self.base + x + METHOD_GLOBAL_BONUS

    @multiplier_decorator(METHOD_GLOBAL_SCALE)
    def decorated_uses_global_factor(self, x: int) -> int:
        return self.base + x


def _roundtrip(fn):
    ser = FunctionSerialized.build_function(fn)
    return ser.as_python()


def _write_read_roundtrip(fn):
    ser = FunctionSerialized.build_function(fn)
    buf = ser.write_to()
    reread = Serialized.read_from(buf, pos=0)
    assert isinstance(reread, FunctionSerialized)
    return reread.as_python()


def _corrupt_method_function_marshal(method) -> bytes:
    method_bytes = _dump_method_payload(method)
    method_tuple = _deserialize_nested(method_bytes)
    assert isinstance(method_tuple, tuple) and len(method_tuple) == 3

    fn_payload_bytes = method_tuple[1]
    fn_tuple = _deserialize_nested(fn_payload_bytes)
    assert isinstance(fn_tuple, tuple) and len(fn_tuple) == 14

    corrupted_fn = list(fn_tuple)
    corrupted_fn[_FN_FULL_MARSHAL] = b"\x00\xDE\xAD"
    corrupted_fn_bytes = _serialize_nested(tuple(corrupted_fn))

    corrupted_method = (
        method_tuple[0],
        corrupted_fn_bytes,
        method_tuple[2],
    )
    return _serialize_nested(corrupted_method)


class TestGlobalsCapture:
    def test_global_int(self):
        fn = _roundtrip(fn_uses_global_int)
        assert fn(7) == 12

    def test_global_dict(self):
        fn = _roundtrip(fn_uses_global_dict)
        assert fn("key") == "value"
        assert fn("count") == 42
        assert fn("missing") is None

    def test_global_list(self):
        fn = _roundtrip(fn_uses_global_list)
        assert fn() == 60

    def test_global_tuple(self):
        fn = _roundtrip(fn_uses_global_tuple)
        assert fn() == (1, 2, 3, 4)

    def test_global_frozenset(self):
        fn = _roundtrip(fn_uses_global_set)
        assert fn() is True

    def test_global_nested_structure(self):
        fn = _roundtrip(fn_uses_global_nested)
        assert fn() is True

    def test_multiple_globals(self):
        fn = _roundtrip(fn_uses_multiple_globals)
        assert fn(10) == 10 + GLOBAL_OFFSET + len(GLOBAL_LIST)

    def test_builtin_functions_available(self):
        fn = _roundtrip(fn_uses_builtin_functions)
        assert fn([1, 5, 3]) == 3 + 5 + 1

    def test_cross_module_global(self):
        fn = _roundtrip(fn_uses_math_module)
        expected = math.sqrt(4.0) + math.pi
        assert fn(4.0) == pytest.approx(expected)

    def test_function_calling_another_global_function(self):
        fn = _roundtrip(fn_calls_other_function)
        assert fn(7) == (7 + GLOBAL_OFFSET) * 2

    def test_globals_are_snapshot_not_live_reference(self):
        global GLOBAL_OFFSET
        original = GLOBAL_OFFSET
        try:
            ser = FunctionSerialized.build_function(fn_uses_global_int)
            GLOBAL_OFFSET = 999
            fn = ser.as_python()
            assert fn(0) == original
        finally:
            GLOBAL_OFFSET = original


class TestClosuresAndLocals:
    def test_single_level_closure(self):
        original = make_adder(10)
        fn = _roundtrip(original)
        assert fn(5) == 15
        assert fn(0) == 10

    def test_different_closure_values(self):
        fn_a = _roundtrip(make_adder(100))
        fn_b = _roundtrip(make_adder(-3))
        assert fn_a(1) == 101
        assert fn_b(1) == -2

    def test_nested_closure(self):
        original = make_nested_closure(3, 7)
        fn = _roundtrip(original)
        assert fn(10) == 20

    def test_closure_over_mutable_list(self):
        original = make_closure_over_mutable([1, 2])
        fn = _roundtrip(original)
        assert fn(3) == [1, 2, 3]
        assert fn(4) == [1, 2, 3, 4]

    def test_closure_over_dict(self):
        original = make_closure_over_dict()
        fn = _roundtrip(original)
        assert fn() == 1
        assert fn() == 2
        assert fn() == 3

    def test_closure_with_multiple_cells(self):
        original = make_counter()
        fn = _roundtrip(original)
        assert fn() == ("counter", 1)
        assert fn() == ("counter", 2)

    def test_closure_with_global_and_local(self):
        original = make_closure_with_global_and_local(20)
        fn = _roundtrip(original)
        assert fn(5) == 5 + 20 + GLOBAL_OFFSET

    def test_closure_captures_correct_value_not_variable(self):
        fns = [_roundtrip(make_adder(i)) for i in range(5)]
        assert [fn(0) for fn in fns] == [0, 1, 2, 3, 4]


class TestDecorators:
    def test_plain_decorator(self):
        fn = _roundtrip(decorated_plain)
        assert fn(2) == 104

    def test_wraps_decorator_preserves_name(self):
        fn = _roundtrip(decorated_wraps)
        assert fn(4) == 15
        assert fn.__name__ == "decorated_wraps"
        assert fn.__qualname__ == decorated_wraps.__qualname__

    def test_stateful_decorator(self):
        fn = _roundtrip(decorated_stateful)
        assert fn(1) == (6, [1])
        assert fn(2) == (7, [1, 2])

    def test_external_decorator(self):
        fn = _roundtrip(decorated_external)
        assert fn(1) == 6
        assert fn(2) == 7


class TestMetadataPreservation:
    def test_name_and_qualname(self):
        fn = _roundtrip(fn_uses_global_int)
        assert fn.__name__ == "fn_uses_global_int"
        assert fn.__qualname__ == fn_uses_global_int.__qualname__

    def test_defaults(self):
        fn = _roundtrip(annotated_fn)
        assert fn.__defaults__ == (2,)

    def test_kwdefaults(self):
        fn = _roundtrip(annotated_fn)
        assert fn.__kwdefaults__ == {"scale": 3}

    def test_annotations(self):
        fn = _roundtrip(annotated_fn)
        assert fn.__annotations__ == annotated_fn.__annotations__

    def test_defaults_are_used(self):
        fn = _roundtrip(annotated_fn)
        assert fn(1) == 9
        assert fn(1, 4) == 15
        assert fn(1, 4, scale=2) == 10

    def test_no_args_function(self):
        fn = _roundtrip(fn_no_args)
        assert fn() == "hello"

    def test_module_attribute(self):
        fn = _roundtrip(fn_uses_global_int)
        assert fn.__module__ is not None


class TestReferenceOnly:
    def test_reference_payload_roundtrip(self):
        payload = _dump_reference_function_payload(pd.DataFrame.head)
        fn = _load_reference_function_payload(payload)
        assert fn is pd.DataFrame.head

    def test_function_serialized_uses_reference(self):
        ser = FunctionSerialized.build_function(pd.DataFrame.head)
        fn = ser.as_python()
        assert fn is pd.DataFrame.head

    def test_reference_write_read_roundtrip(self):
        fn = _write_read_roundtrip(pd.DataFrame.head)
        assert fn is pd.DataFrame.head

    def test_dispatches_via_from_python_object(self):
        ser = Serialized.from_python_object(pd.DataFrame.head)
        assert isinstance(ser, FunctionSerialized)
        assert ser.as_python() is pd.DataFrame.head

    def test_stdlib_builtin_rejects_non_function_type(self):
        with pytest.raises(TypeError, match="FunctionType"):
            FunctionSerialized.build_function(operator.add)

    def test_math_builtin_rejects_non_function_type(self):
        with pytest.raises(TypeError, match="FunctionType"):
            FunctionSerialized.build_function(math.sqrt)


class TestWriteReadRoundtrip:
    def test_simple_function(self):
        fn = _write_read_roundtrip(fn_uses_global_int)
        assert fn(10) == 15

    def test_closure(self):
        fn = _write_read_roundtrip(make_adder(42))
        assert fn(8) == 50

    def test_decorated(self):
        fn = _write_read_roundtrip(decorated_wraps)
        assert fn(2) == 9

    def test_tag_is_function(self):
        ser = FunctionSerialized.build_function(fn_uses_global_int)
        buf = ser.write_to()
        reread = Serialized.read_from(buf, pos=0)
        assert reread.tag == Tags.FUNCTION

    def test_multiple_globals_binary(self):
        fn = _write_read_roundtrip(fn_uses_multiple_globals)
        assert fn(0) == GLOBAL_OFFSET + len(GLOBAL_LIST)


class TestEdgeCases:
    def test_recursive_function(self):
        fn = _roundtrip(recursive_factorial)
        assert fn(5) == 120
        assert fn(1) == 1
        assert fn(0) == 1

    def test_lambda(self):
        fn = _roundtrip(simple_lambda)
        assert fn(5) == 10

    def test_generator_function(self):
        fn = _roundtrip(generator_fn)
        assert list(fn(3)) == [GLOBAL_OFFSET, GLOBAL_OFFSET + 1, GLOBAL_OFFSET + 2]

    def test_function_with_local_nested_objects(self):
        local_state = {"nums": [1, 2, 3], "flag": True}

        def fn(x: int) -> Any:
            return x + len(local_state["nums"]), local_state["flag"]

        restored = _roundtrip(fn)
        assert restored(5) == (8, True)

    def test_function_referencing_class(self):
        def fn() -> Calculator:
            return Calculator(base=10)

        restored = _roundtrip(fn)
        result = restored()
        assert isinstance(result, Calculator)
        assert result.base == 10

    def test_non_function_raises_type_error(self):
        with pytest.raises(TypeError, match="FunctionType"):
            FunctionSerialized.build_function(42)  # type: ignore[arg-type]

    def test_non_function_class_raises(self):
        with pytest.raises(TypeError, match="FunctionType"):
            FunctionSerialized.build_function(Calculator)  # type: ignore[arg-type]


class TestMethodSerialized:
    def test_instance_method_roundtrip(self):
        calc = Calculator(base=10)
        method = calc.add
        ser = MethodSerialized.build_method(method)
        restored = ser.as_python()
        assert restored(5) == 15

    def test_instance_method_preserves_self_state(self):
        calc = Calculator(base=7)
        ser = MethodSerialized.build_method(calc.mul)
        restored = ser.as_python()
        assert restored(3) == 21

    def test_bound_method_from_python_object(self):
        calc = Calculator(base=3)
        ser = Serialized.from_python_object(calc.add)
        assert isinstance(ser, MethodSerialized)
        fn = ser.as_python()
        assert fn(10) == 13

    def test_reference_only_bound_method(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        bound = df.head
        ser = Serialized.from_python_object(bound)
        fn = ser.as_python()
        assert fn().equals(bound())

    def test_non_method_raises(self):
        with pytest.raises(TypeError, match="MethodType"):
            MethodSerialized.build_method(fn_uses_global_int)  # type: ignore[arg-type]

    def test_method_write_read_roundtrip(self):
        calc = Calculator(base=5)
        ser = MethodSerialized.build_method(calc.add)
        buf = ser.write_to()
        reread = Serialized.read_from(buf, pos=0)
        assert isinstance(reread, MethodSerialized)
        assert reread.as_python()(10) == 15

    def test_decorated_bound_method_roundtrip_preserves_wrapper_behavior(self):
        calc = Calculator(base=10)
        ser = MethodSerialized.build_method(calc.decorated_uses_global)
        restored = ser.as_python()
        assert restored(3) == (10 + 3 + METHOD_GLOBAL_BONUS) * 2

    def test_decorated_stateful_bound_method_roundtrip_preserves_wrapper_behavior(self):
        calc = Calculator(base=10)
        ser = MethodSerialized.build_method(calc.decorated_stateful_method)
        restored = ser.as_python()
        assert restored(1) == (10 + 1 + METHOD_GLOBAL_BONUS, [1])
        assert restored(2) == (10 + 2 + METHOD_GLOBAL_BONUS, [1, 2])

    def test_decorated_bound_method_with_global_decorator_arg(self):
        calc = Calculator(base=3)
        ser = MethodSerialized.build_method(calc.decorated_uses_global_factor)
        restored = ser.as_python()
        assert restored(2) == (3 + 2) * METHOD_GLOBAL_SCALE


class TestFromPythonObjectDispatch:
    def test_dispatches_plain_function(self):
        ser = Serialized.from_python_object(fn_uses_global_int)
        assert isinstance(ser, FunctionSerialized)
        assert ser.as_python()(0) == GLOBAL_OFFSET

    def test_dispatches_decorated_function(self):
        ser = Serialized.from_python_object(decorated_wraps)
        assert isinstance(ser, FunctionSerialized)
        assert ser.as_python()(3) == 12

    def test_dispatches_closure(self):
        closure = make_adder(100)
        ser = Serialized.from_python_object(closure)
        assert isinstance(ser, FunctionSerialized)
        assert ser.as_python()(1) == 101

    def test_dispatches_bound_method(self):
        calc = Calculator(base=2)
        ser = Serialized.from_python_object(calc.add)
        assert isinstance(ser, MethodSerialized)
        assert ser.as_python()(8) == 10


def fn_creates_pandas_dataframe(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data)


def fn_creates_arrow_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.utf8()),
    ])


def fn_creates_arrow_table() -> pa.Table:
    return pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})


def fn_uses_datetime() -> dt.datetime:
    return dt.datetime(2025, 1, 15, 12, 30, 0, tzinfo=dt.timezone.utc)


def fn_datetime_arithmetic(days: int) -> dt.date:
    base = dt.date(2025, 6, 1)
    return base + dt.timedelta(days=days)


def fn_uses_decimal(a: str, b: str) -> Decimal:
    return Decimal(a) + Decimal(b)


def fn_uses_pathlib() -> str:
    return str(PurePosixPath("/usr") / "local" / "bin")


def fn_param_shadows_import(math: int) -> int:
    return math + 1


def fn_local_shadows_from_import() -> str:
    PurePosixPath = lambda value: f"local:{value}"  # noqa: E731
    return PurePosixPath("ok")


def fn_uses_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True)


def fn_uses_regex(text: str) -> list[str]:
    return re.findall(r"\d+", text)


class Colour(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def fn_uses_enum(name: str) -> Colour:
    return Colour[name]


def fn_uses_ordered_dict() -> OrderedDict:
    d = OrderedDict()
    d["a"] = 1
    d["b"] = 2
    return d


def fn_uses_defaultdict() -> dict:
    dd = defaultdict(list)
    dd["x"].append(1)
    dd["x"].append(2)
    return dict(dd)


_base_add = lambda a, b: a + b  # noqa: E731
fn_partial = partial(_base_add, 10)


def fn_multi_library_mix(n: int) -> dict:
    arr = pa.array(range(n))
    df = pd.DataFrame({"v": list(range(n))})
    ts = dt.datetime.now(tz=dt.timezone.utc)
    return {
        "arrow_len": len(arr),
        "pandas_shape": df.shape,
        "has_timestamp": isinstance(ts, dt.datetime),
    }


def make_schema_builder(base_fields: list[tuple[str, pa.DataType]]):
    def build(extra_name: str, extra_type: pa.DataType) -> pa.Schema:
        fields = [pa.field(n, t) for n, t in base_fields] + [
            pa.field(extra_name, extra_type)
        ]
        return pa.schema(fields)
    return build


def fn_with_external_default(
    schema: pa.Schema = pa.schema([pa.field("default_col", pa.int32())]),
) -> list[str]:
    return schema.names


def fn_pandas_chain(data: dict) -> list:
    return (
        pd.DataFrame(data)
        .assign(double=lambda df: df["v"] * 2)
        .query("double > 4")["v"]
        .tolist()
    )


def fn_arrow_record_batch() -> int:
    batch = pa.RecordBatch.from_pydict({"a": [10, 20], "b": [30, 40]})
    return batch.num_rows


def fn_pandas_series_mean(values: list[float]) -> float:
    return pd.Series(values).mean()


class TestExternalImportCapture:
    def test_creates_pandas_dataframe(self):
        fn = _roundtrip(fn_creates_pandas_dataframe)
        result = fn({"a": [1, 2], "b": [3, 4]})
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert result.shape == (2, 2)

    def test_pandas_chain(self):
        fn = _roundtrip(fn_pandas_chain)
        assert fn({"v": [1, 2, 3, 4, 5]}) == [3, 4, 5]

    def test_pandas_series_method(self):
        fn = _roundtrip(fn_pandas_series_mean)
        assert fn([10.0, 20.0, 30.0]) == pytest.approx(20.0)

    def test_creates_arrow_schema(self):
        fn = _roundtrip(fn_creates_arrow_schema)
        schema = fn()
        assert isinstance(schema, pa.Schema)
        assert schema.names == ["id", "name"]
        assert schema.field("id").type == pa.int64()

    def test_creates_arrow_table(self):
        fn = _roundtrip(fn_creates_arrow_table)
        table = fn()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 3
        assert table.column_names == ["x", "y"]

    def test_arrow_record_batch(self):
        fn = _roundtrip(fn_arrow_record_batch)
        assert fn() == 2

    def test_closure_captures_pyarrow_types(self):
        builder = make_schema_builder([("id", pa.int64())])
        fn = _roundtrip(builder)
        schema = fn("name", pa.utf8())
        assert isinstance(schema, pa.Schema)
        assert schema.names == ["id", "name"]

    def test_external_default_pyarrow_schema(self):
        fn = _roundtrip(fn_with_external_default)
        assert fn() == ["default_col"]
        custom = pa.schema([pa.field("x", pa.float64()), pa.field("y", pa.float64())])
        assert fn(custom) == ["x", "y"]

    def test_uses_datetime(self):
        fn = _roundtrip(fn_uses_datetime)
        result = fn()
        assert isinstance(result, dt.datetime)
        assert result.year == 2025
        assert result.tzinfo is not None

    def test_datetime_arithmetic(self):
        fn = _roundtrip(fn_datetime_arithmetic)
        result = fn(10)
        assert result == dt.date(2025, 6, 11)

    def test_uses_decimal(self):
        fn = _roundtrip(fn_uses_decimal)
        result = fn("1.1", "2.2")
        assert isinstance(result, Decimal)
        assert result == Decimal("3.3")

    def test_uses_pathlib(self):
        fn = _roundtrip(fn_uses_pathlib)
        assert fn() == "/usr/local/bin"

    def test_uses_json(self):
        fn = _roundtrip(fn_uses_json)
        assert fn({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'

    def test_uses_regex(self):
        fn = _roundtrip(fn_uses_regex)
        assert fn("abc123def456") == ["123", "456"]

    def test_uses_enum(self):
        fn = _roundtrip(fn_uses_enum)
        result = fn("GREEN")
        assert result == Colour.GREEN
        assert result.value == 2

    def test_uses_ordered_dict(self):
        fn = _roundtrip(fn_uses_ordered_dict)
        result = fn()
        assert isinstance(result, OrderedDict)
        assert list(result.keys()) == ["a", "b"]

    def test_uses_defaultdict(self):
        fn = _roundtrip(fn_uses_defaultdict)
        assert fn() == {"x": [1, 2]}

    def test_partial_rejects_non_function_type(self):
        with pytest.raises(TypeError, match="FunctionType"):
            FunctionSerialized.build_function(fn_partial)

    def test_multi_library_function(self):
        fn = _roundtrip(fn_multi_library_mix)
        result = fn(4)
        assert result["arrow_len"] == 4
        assert result["pandas_shape"] == (4, 1)
        assert result["has_timestamp"] is True

    def test_pandas_function_binary_roundtrip(self):
        fn = _write_read_roundtrip(fn_creates_pandas_dataframe)
        result = fn({"x": [1]})
        assert isinstance(result, pd.DataFrame)

    def test_arrow_function_binary_roundtrip(self):
        fn = _write_read_roundtrip(fn_creates_arrow_schema)
        schema = fn()
        assert schema.names == ["id", "name"]

    def test_datetime_function_binary_roundtrip(self):
        fn = _write_read_roundtrip(fn_uses_datetime)
        assert fn().year == 2025

    def test_multi_library_binary_roundtrip(self):
        fn = _write_read_roundtrip(fn_multi_library_mix)
        result = fn(2)
        assert result["arrow_len"] == 2


def _corrupt_marshal_in_payload(fn) -> bytes:
    payload_bytes = _dump_function_payload(fn)
    payload_tuple = _deserialize_nested(payload_bytes)
    assert isinstance(payload_tuple, tuple) and len(payload_tuple) == 14

    corrupted = list(payload_tuple)
    corrupted[_FN_FULL_MARSHAL] = b"\x00\xDE\xAD"
    return _serialize_nested(tuple(corrupted))


def _fake_version_in_payload(fn) -> bytes:
    payload_bytes = _dump_function_payload(fn)
    payload_tuple = _deserialize_nested(payload_bytes)
    corrupted = list(payload_tuple)
    corrupted[_FN_FULL_PY_VERSION] = (2, 7, 0)
    return _serialize_nested(tuple(corrupted))


def _remove_marshal_in_payload(fn) -> bytes:
    payload_bytes = _dump_function_payload(fn)
    payload_tuple = _deserialize_nested(payload_bytes)
    corrupted = list(payload_tuple)
    corrupted[_FN_FULL_MARSHAL] = None
    return _serialize_nested(tuple(corrupted))


def _remove_source_in_payload(fn) -> bytes:
    payload_bytes = _dump_function_payload(fn)
    payload_tuple = _deserialize_nested(payload_bytes)
    corrupted = list(payload_tuple)
    corrupted[_FN_FULL_SOURCE] = None
    return _serialize_nested(tuple(corrupted))


def _corrupt_both_in_payload(fn) -> bytes:
    payload_bytes = _dump_function_payload(fn)
    payload_tuple = _deserialize_nested(payload_bytes)
    corrupted = list(payload_tuple)
    corrupted[_FN_FULL_MARSHAL] = b"\x00\xDE\xAD"
    corrupted[_FN_FULL_SOURCE] = None
    return _serialize_nested(tuple(corrupted))


class TestSourceFallback:
    def test_simple_function_with_corrupted_marshal(self):
        data = _corrupt_marshal_in_payload(fn_uses_global_int)
        fn = _load_function_payload(data)
        assert fn(10) == 15

    def test_module_import_globals_are_inferred_from_used_source_names(self, monkeypatch):
        import yggdrasil.pickle.ser.complexs as complexs_module

        def fail_getclosurevars(_fn):
            raise RuntimeError("boom")

        monkeypatch.setattr(complexs_module.inspect, "getclosurevars", fail_getclosurevars)

        payload_bytes = _dump_function_payload(fn_uses_math_module)
        payload = _deserialize_nested(payload_bytes)

        assert isinstance(payload, tuple)
        assert len(payload) == 14

        globals_obj = payload[_FN_FULL_GLOBALS]
        assert isinstance(globals_obj, dict)
        assert set(globals_obj) == {"math"}

        fn = _load_function_payload(payload_bytes)
        assert fn(4.0) == pytest.approx(math.sqrt(4.0) + math.pi)

    def test_module_from_import_globals_are_inferred_when_closurevars_fail(self, monkeypatch):
        import yggdrasil.pickle.ser.complexs as complexs_module

        def fail_getclosurevars(_fn):
            raise RuntimeError("boom")

        monkeypatch.setattr(complexs_module.inspect, "getclosurevars", fail_getclosurevars)

        payload_bytes = _dump_function_payload(fn_uses_pathlib)
        payload = _deserialize_nested(payload_bytes)

        assert isinstance(payload, tuple)
        assert len(payload) == 14

        globals_obj = payload[_FN_FULL_GLOBALS]
        assert isinstance(globals_obj, dict)
        assert set(globals_obj) == {"PurePosixPath"}

        fn = _load_function_payload(payload_bytes)
        assert fn() == "/usr/local/bin"

    def test_shadowed_import_parameter_is_not_inferred(self, monkeypatch):
        import yggdrasil.pickle.ser.complexs as complexs_module

        def fail_getclosurevars(_fn):
            raise RuntimeError("boom")

        monkeypatch.setattr(complexs_module.inspect, "getclosurevars", fail_getclosurevars)

        payload_bytes = _dump_function_payload(fn_param_shadows_import)
        payload = _deserialize_nested(payload_bytes)

        assert isinstance(payload, tuple)
        assert len(payload) == 14

        globals_obj = payload[_FN_FULL_GLOBALS]
        assert isinstance(globals_obj, dict)
        assert globals_obj == {}

        fn = _load_function_payload(payload_bytes)
        assert fn(4) == 5

    def test_shadowed_from_import_local_is_not_inferred(self, monkeypatch):
        import yggdrasil.pickle.ser.complexs as complexs_module

        def fail_getclosurevars(_fn):
            raise RuntimeError("boom")

        monkeypatch.setattr(complexs_module.inspect, "getclosurevars", fail_getclosurevars)

        payload_bytes = _dump_function_payload(fn_local_shadows_from_import)
        payload = _deserialize_nested(payload_bytes)

        assert isinstance(payload, tuple)
        assert len(payload) == 14

        globals_obj = payload[_FN_FULL_GLOBALS]
        assert isinstance(globals_obj, dict)
        assert globals_obj == {}

        fn = _load_function_payload(payload_bytes)
        assert fn() == "local:ok"

    def test_simple_function_with_no_marshal(self):
        data = _remove_marshal_in_payload(fn_uses_global_int)
        fn = _load_function_payload(data)
        assert fn(10) == 15

    def test_simple_function_with_fake_version(self):
        data = _fake_version_in_payload(fn_uses_global_int)
        fn = _load_function_payload(data)
        assert fn(10) == 15

    def test_global_dict_via_source(self):
        data = _corrupt_marshal_in_payload(fn_uses_global_dict)
        fn = _load_function_payload(data)
        assert fn("key") == "value"
        assert fn("missing") is None

    def test_global_list_via_source(self):
        data = _corrupt_marshal_in_payload(fn_uses_global_list)
        fn = _load_function_payload(data)
        assert fn() == 60

    def test_multiple_globals_via_source(self):
        data = _corrupt_marshal_in_payload(fn_uses_multiple_globals)
        fn = _load_function_payload(data)
        assert fn(0) == GLOBAL_OFFSET + len(GLOBAL_LIST)

    def test_cross_module_math_via_source(self):
        data = _corrupt_marshal_in_payload(fn_uses_math_module)
        fn = _load_function_payload(data)
        assert fn(4.0) == pytest.approx(math.sqrt(4.0) + math.pi)

    def test_function_calling_another_via_source(self):
        data = _corrupt_marshal_in_payload(fn_calls_other_function)
        fn = _load_function_payload(data)
        assert fn(7) == (7 + GLOBAL_OFFSET) * 2

    def test_closure_with_corrupted_marshal(self):
        original = make_adder(42)
        data = _corrupt_marshal_in_payload(original)
        fn = _load_function_payload(data)
        assert fn(8) == 50

    def test_nested_closure_via_source(self):
        original = make_nested_closure(3, 7)
        data = _corrupt_marshal_in_payload(original)
        fn = _load_function_payload(data)
        assert fn(10) == 20

    def test_closure_with_global_and_local_via_source(self):
        original = make_closure_with_global_and_local(20)
        data = _corrupt_marshal_in_payload(original)
        fn = _load_function_payload(data)
        assert fn(5) == 5 + 20 + GLOBAL_OFFSET

    def test_decorated_wraps_via_source(self):
        data = _corrupt_marshal_in_payload(decorated_wraps)
        fn = _load_function_payload(data)
        assert fn(4) == 5

    def test_decorated_plain_via_source(self):
        data = _corrupt_marshal_in_payload(decorated_plain)
        fn = _load_function_payload(data)
        assert fn(3) == 106

    def test_annotated_defaults_via_source(self):
        data = _corrupt_marshal_in_payload(annotated_fn)
        fn = _load_function_payload(data)
        assert fn.__defaults__ == (2,)
        assert fn.__kwdefaults__ == {"scale": 3}
        assert fn(1) == 9

    def test_name_qualname_via_source(self):
        data = _corrupt_marshal_in_payload(fn_uses_global_int)
        fn = _load_function_payload(data)
        assert fn.__name__ == "fn_uses_global_int"

    def test_recursive_via_source(self):
        data = _corrupt_marshal_in_payload(recursive_factorial)
        fn = _load_function_payload(data)
        assert fn(5) == 120

    def test_generator_via_source(self):
        data = _corrupt_marshal_in_payload(generator_fn)
        fn = _load_function_payload(data)
        assert list(fn(3)) == [GLOBAL_OFFSET, GLOBAL_OFFSET + 1, GLOBAL_OFFSET + 2]

    def test_no_args_function_via_source(self):
        data = _corrupt_marshal_in_payload(fn_no_args)
        fn = _load_function_payload(data)
        assert fn() == "hello"

    def test_pandas_via_source(self):
        data = _corrupt_marshal_in_payload(fn_creates_pandas_dataframe)
        fn = _load_function_payload(data)
        result = fn({"a": [1, 2]})
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 1)

    def test_arrow_via_source(self):
        data = _corrupt_marshal_in_payload(fn_creates_arrow_schema)
        fn = _load_function_payload(data)
        schema = fn()
        assert isinstance(schema, pa.Schema)
        assert schema.names == ["id", "name"]

    def test_datetime_via_source(self):
        data = _corrupt_marshal_in_payload(fn_uses_datetime)
        fn = _load_function_payload(data)
        result = fn()
        assert isinstance(result, dt.datetime)
        assert result.year == 2025

    def test_json_via_source(self):
        data = _corrupt_marshal_in_payload(fn_uses_json)
        fn = _load_function_payload(data)
        assert fn({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'

    def test_regex_via_source(self):
        data = _corrupt_marshal_in_payload(fn_uses_regex)
        fn = _load_function_payload(data)
        assert fn("abc123def456") == ["123", "456"]

    def test_multi_library_via_source(self):
        data = _corrupt_marshal_in_payload(fn_multi_library_mix)
        fn = _load_function_payload(data)
        result = fn(3)
        assert result["arrow_len"] == 3
        assert result["pandas_shape"] == (3, 1)

    def test_marshal_only_still_works(self):
        data = _remove_source_in_payload(fn_uses_global_int)
        fn = _load_function_payload(data)
        assert fn(10) == 15

    def test_both_corrupted_raises_runtime_error(self):
        data = _corrupt_both_in_payload(fn_uses_global_int)
        with pytest.raises(RuntimeError, match="Failed to reconstruct function"):
            _load_function_payload(data)

    def test_load_code_payload_source_only(self):
        source = textwrap.dedent(inspect.getsource(fn_no_args))
        fn = _load_function_code_payload(
            python_version=_PYTHON_VERSION,
            marshal_code=None,
            source_code=source,
            globals_dict={"__builtins__": __builtins__, "__name__": __name__},
            module_name=__name__,
            name="fn_no_args",
            qualname="fn_no_args",
            defaults=None,
            kwdefaults=None,
            annotations={},
            closure=None,
        )
        assert fn() == "hello"

    def test_load_code_payload_marshal_only(self):
        import marshal
        fn = _load_function_code_payload(
            python_version=_PYTHON_VERSION,
            marshal_code=marshal.dumps(fn_no_args.__code__),
            source_code=None,
            globals_dict={"__builtins__": __builtins__, "__name__": __name__},
            module_name=__name__,
            name="fn_no_args",
            qualname="fn_no_args",
            defaults=None,
            kwdefaults=None,
            annotations={},
            closure=None,
        )
        assert fn() == "hello"

    def test_load_code_payload_corrupted_marshal_falls_back_to_source(self):
        source = textwrap.dedent(inspect.getsource(fn_no_args))
        fn = _load_function_code_payload(
            python_version=_PYTHON_VERSION,
            marshal_code=b"\xff\xfe\xfd",
            source_code=source,
            globals_dict={"__builtins__": __builtins__, "__name__": __name__},
            module_name=__name__,
            name="fn_no_args",
            qualname="fn_no_args",
            defaults=None,
            kwdefaults=None,
            annotations={},
            closure=None,
        )
        assert fn() == "hello"

    def test_load_code_payload_version_mismatch_prefers_source(self):
        source = textwrap.dedent(inspect.getsource(fn_no_args))
        import marshal
        fn = _load_function_code_payload(
            python_version=(2, 7, 0),
            marshal_code=marshal.dumps(fn_no_args.__code__),
            source_code=source,
            globals_dict={"__builtins__": __builtins__, "__name__": __name__},
            module_name=__name__,
            name="fn_no_args",
            qualname="fn_no_args",
            defaults=None,
            kwdefaults=None,
            annotations={},
            closure=None,
        )
        assert fn() == "hello"

    def test_load_code_payload_both_none_raises(self):
        with pytest.raises(RuntimeError, match="Failed to reconstruct"):
            _load_function_code_payload(
                python_version=_PYTHON_VERSION,
                marshal_code=None,
                source_code=None,
                globals_dict={"__builtins__": __builtins__},
                module_name=__name__,
                name="nope",
                qualname="nope",
                defaults=None,
                kwdefaults=None,
                annotations={},
                closure=None,
            )

    def test_method_with_corrupted_function_marshal(self):
        calc = Calculator(base=10)
        method = calc.add

        method_bytes = _dump_method_payload(method)
        method_tuple = _deserialize_nested(method_bytes)
        assert isinstance(method_tuple, tuple) and len(method_tuple) == 3

        fn_payload_bytes = method_tuple[1]
        fn_tuple = _deserialize_nested(fn_payload_bytes)

        if isinstance(fn_tuple, tuple) and len(fn_tuple) == 14:
            corrupted_fn = list(fn_tuple)
            corrupted_fn[_FN_FULL_MARSHAL] = b"\x00\xDE\xAD"
            corrupted_fn_bytes = _serialize_nested(tuple(corrupted_fn))

            corrupted_method = (
                method_tuple[0],
                corrupted_fn_bytes,
                method_tuple[2],
            )
            corrupted_method_bytes = _serialize_nested(corrupted_method)

            restored = _load_method_payload(corrupted_method_bytes)
            assert restored(5) == 15

    def test_method_mul_with_corrupted_marshal(self):
        calc = Calculator(base=7)
        method = calc.mul

        method_bytes = _dump_method_payload(method)
        method_tuple = _deserialize_nested(method_bytes)
        fn_payload_bytes = method_tuple[1]
        fn_tuple = _deserialize_nested(fn_payload_bytes)

        if isinstance(fn_tuple, tuple) and len(fn_tuple) == 14:
            corrupted_fn = list(fn_tuple)
            corrupted_fn[_FN_FULL_MARSHAL] = b"\x00\xDE\xAD"
            corrupted_fn_bytes = _serialize_nested(tuple(corrupted_fn))

            corrupted_method = (
                method_tuple[0],
                corrupted_fn_bytes,
                method_tuple[2],
            )
            corrupted_method_bytes = _serialize_nested(corrupted_method)

            restored = _load_method_payload(corrupted_method_bytes)
            assert restored(3) == 21

    def test_decorated_method_payload_captures_inner_global_name(self):
        payload_bytes = _dump_function_payload(Calculator.decorated_uses_global)
        payload = _deserialize_nested(payload_bytes)

        assert isinstance(payload, tuple)
        assert len(payload) == 14

        globals_obj = payload[_FN_FULL_GLOBALS]
        definition_globals_obj = payload[_FN_FULL_DEFINITION_GLOBALS]

        assert isinstance(globals_obj, dict)
        assert isinstance(definition_globals_obj, dict)

        assert (
            "METHOD_GLOBAL_BONUS" in globals_obj
            or "METHOD_GLOBAL_BONUS" in definition_globals_obj
        )

    def test_decorated_method_inner_globals_via_source(self):
        calc = Calculator(base=10)
        restored = _load_method_payload(_corrupt_method_function_marshal(calc.decorated_uses_global))
        assert restored(3) == 10 + 3 + METHOD_GLOBAL_BONUS

    def test_decorated_stateful_method_via_source_uses_inner_function_globals(self):
        calc = Calculator(base=20)
        restored = _load_method_payload(_corrupt_method_function_marshal(calc.decorated_stateful_method))
        assert restored(1) == 20 + 1 + METHOD_GLOBAL_BONUS
        assert restored(2) == 20 + 2 + METHOD_GLOBAL_BONUS

    def test_decorated_method_inner_globals_with_global_decorator_arg_via_source(self):
        calc = Calculator(base=7)
        restored = _load_method_payload(_corrupt_method_function_marshal(calc.decorated_uses_global_factor))
        assert restored(5) == 7 + 5