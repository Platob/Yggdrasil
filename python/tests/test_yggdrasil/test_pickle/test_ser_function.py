"""Unit tests for yggdrasil.pickle.ser.function – FunctionSerialized."""
from __future__ import annotations

import functools
import math
import operator
import unittest

import pytest

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser import (
    FunctionSerialized,
    ModuleSerialized,
    SerdeTags,
    Serialized,
    dumps,
    loads,
)
from yggdrasil.pickle.ser.serialized import ArraySerialized


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

MULTIPLIER = 3


def simple(x: int) -> int:
    return x * 2


def with_global(x: int) -> int:
    return x * MULTIPLIER


def with_default(x: int, factor: int = 10) -> int:
    return x * factor


def with_kwdefault(x: int, *, prefix: str = "hi") -> str:
    return f"{prefix}:{x}"


def uses_math(v: float) -> float:
    return math.sqrt(v)


def uses_operator(a: int, b: int) -> int:
    return operator.add(a, b)


def make_adder(n: int):
    def inner(x: int) -> int:
        return x + n
    return inner


def make_multiplier(n: int):
    def inner(x: int) -> int:
        return x * n
    return inner


def nested_closure(a: int, b: int):
    total = a + b
    def fn() -> int:
        return total
    return fn


# ---------------------------------------------------------------------------
# Decorators for testing
# ---------------------------------------------------------------------------

def my_decorator(fn):
    """Simple pass-through decorator using @wraps."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def logging_decorator(fn):
    """Decorator that adds a custom __dict__ attribute via @wraps."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper.logged = True
    return wrapper


def double_decorator(fn):
    """Multiplies the result by 2."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) * 2
    return wrapper


def prefix_decorator(prefix: str):
    """Parameterised decorator factory."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            return f"{prefix}:{result}"
        return wrapper
    return decorator


@my_decorator
def decorated_simple(x: int) -> int:
    return x * 2


@logging_decorator
def decorated_with_log(x: int) -> int:
    return x + 10


@my_decorator
@double_decorator
def stacked_decorated(x: int) -> int:
    return x + 1


@prefix_decorator("out")
def prefixed_fn(x: int) -> int:
    return x * 3


@my_decorator
def decorated_with_global(x: int) -> int:
    return x * MULTIPLIER


@my_decorator
def decorated_with_default(x: int, factor: int = 5) -> int:
    return x * factor


@my_decorator
def decorated_with_kwdefault(x: int, *, tag: str = "default") -> str:
    return f"{tag}:{x}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFunctionSerializedTag(unittest.TestCase):
    def test_tag(self):
        assert FunctionSerialized.TAG == SerdeTags.FUNCTION


class TestFunctionSerializedBasic(unittest.TestCase):
    """Round-trip behaviour for simple functions."""

    def test_simple_function(self):
        ser = FunctionSerialized.from_value(simple)
        assert ser.value(5) == 10

    def test_lambda(self):
        fn = lambda x: x + 1  # noqa: E731
        ser = FunctionSerialized.from_value(fn)
        assert ser.value(4) == 5

    def test_function_name_preserved(self):
        ser = FunctionSerialized.from_value(simple)
        assert ser.value.__name__ == "simple"

    def test_function_qualname_in_metadata(self):
        ser = FunctionSerialized.from_value(simple)
        assert ser.metadata[b"qualname"] == b"simple"

    def test_function_module_in_metadata(self):
        ser = FunctionSerialized.from_value(simple)
        assert b"module" in ser.metadata

    def test_tag_on_instance(self):
        ser = FunctionSerialized.from_value(simple)
        assert ser.tag == SerdeTags.FUNCTION

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            FunctionSerialized.from_value(42)

    def test_rejects_class(self):
        with pytest.raises(TypeError):
            FunctionSerialized.from_value(int)


class TestFunctionSerializedGlobals(unittest.TestCase):
    """Only referenced globals are captured."""

    def test_global_variable_captured(self):
        ser = FunctionSerialized.from_value(with_global)
        assert ser.value(4) == 4 * MULTIPLIER

    def test_module_global_stays_module_serialized(self):
        ser = FunctionSerialized.from_value(uses_math)
        items = list(ser.iter_())
        globals_item = items[1]
        assert isinstance(globals_item, ArraySerialized)
        pairs = list(globals_item.iter_())
        val_sers = [pairs[i + 1] for i in range(0, len(pairs), 2)]
        assert any(isinstance(v, ModuleSerialized) for v in val_sers)

    def test_module_ref_function_works(self):
        ser = FunctionSerialized.from_value(uses_math)
        assert ser.value(9) == pytest.approx(3.0)

    def test_operator_module_by_ref(self):
        ser = FunctionSerialized.from_value(uses_operator)
        items = list(ser.iter_())
        globals_item = items[1]
        pairs = list(globals_item.iter_())
        val_sers = [pairs[i + 1] for i in range(0, len(pairs), 2)]
        assert any(isinstance(v, ModuleSerialized) for v in val_sers)
        assert ser.value(3, 4) == 7

    def test_unreferenced_globals_excluded(self):
        _UNUSED = object()  # noqa: F841
        ser = FunctionSerialized.from_value(simple)
        items = list(ser.iter_())
        globals_item = items[1]
        pairs = list(globals_item.iter_())
        keys = {pairs[i].value for i in range(0, len(pairs), 2)}
        assert "_UNUSED" not in keys


class TestFunctionSerializedDefaults(unittest.TestCase):
    def test_positional_default(self):
        ser = FunctionSerialized.from_value(with_default)
        fn = ser.value
        assert fn(3) == 30
        assert fn(3, 2) == 6

    def test_kwdefault(self):
        ser = FunctionSerialized.from_value(with_kwdefault)
        fn = ser.value
        assert fn(7) == "hi:7"
        assert fn(7, prefix="yo") == "yo:7"

    def test_no_defaults_ok(self):
        assert FunctionSerialized.from_value(simple).value(5) == 10


class TestFunctionSerializedClosures(unittest.TestCase):
    def test_single_closure(self):
        adder = make_adder(7)
        assert FunctionSerialized.from_value(adder).value(3) == 10

    def test_closure_independence(self):
        ser5 = FunctionSerialized.from_value(make_adder(5))
        ser9 = FunctionSerialized.from_value(make_adder(9))
        assert ser5.value(1) == 6
        assert ser9.value(1) == 10

    def test_nested_closure(self):
        fn = nested_closure(3, 4)
        assert FunctionSerialized.from_value(fn).value() == 7

    def test_no_closure_ok(self):
        ser = FunctionSerialized.from_value(simple)
        assert ser.iter_().__class__  # iter_ works
        items = list(ser.iter_())
        assert items[4].value == []


class TestFunctionSerializedWireFormat(unittest.TestCase):
    def test_six_items_in_payload(self):
        ser = FunctionSerialized.from_value(simple)
        assert len(list(ser.iter_())) == 6  # code, globals, defaults, kwdefaults, closure, wrapper_dict

    def test_bwrite_pread_roundtrip(self):
        ser = FunctionSerialized.from_value(with_global)
        buf = BytesIO()
        ser.bwrite(buf)
        restored, _ = Serialized.pread_from(buf, 0)
        assert isinstance(restored, FunctionSerialized)
        assert restored.value(2) == 2 * MULTIPLIER

    def test_codec_default_no_compression(self):
        assert FunctionSerialized.from_value(simple).codec == 0

    def test_byte_limit_none_no_compression(self):
        assert FunctionSerialized.from_value(simple, byte_limit=None).codec == 0


class TestFunctionSerializedRegistryDispatch(unittest.TestCase):
    def test_from_python_dispatches(self):
        assert isinstance(Serialized.from_python(simple), FunctionSerialized)

    def test_from_python_lambda(self):
        fn = lambda x: x ** 2  # noqa: E731
        ser = Serialized.from_python(fn)
        assert isinstance(ser, FunctionSerialized)
        assert ser.value(4) == 16

    def test_dumps_loads_simple(self):
        assert loads(dumps(simple))(5) == 10

    def test_dumps_loads_with_global(self):
        assert loads(dumps(with_global))(4) == 4 * MULTIPLIER

    def test_dumps_loads_closure(self):
        assert loads(dumps(make_adder(100)))(1) == 101

    def test_dumps_loads_module_ref(self):
        assert loads(dumps(uses_math))(16) == pytest.approx(4.0)

    def test_dumps_loads_kwdefault(self):
        fn = loads(dumps(with_kwdefault))
        assert fn(3) == "hi:3"
        assert fn(3, prefix="hey") == "hey:3"


class TestFunctionSerializedDecorators(unittest.TestCase):
    """Decorated functions – @wraps, stacked, parameterised, __dict__ attrs."""

    # ── @functools.wraps pass-through ───────────────────────────────────────

    def test_wraps_decorated_roundtrip(self):
        ser = FunctionSerialized.from_value(decorated_simple)
        assert ser.value(5) == 10

    def test_wraps_name_preserved(self):
        ser = FunctionSerialized.from_value(decorated_simple)
        assert ser.value.__name__ == "decorated_simple"

    def test_wraps_metadata_name(self):
        ser = FunctionSerialized.from_value(decorated_simple)
        assert ser.metadata[b"name"] == b"decorated_simple"

    def test_wrapped_flag_in_metadata(self):
        """When the input has __wrapped__, metadata must record it."""
        ser = FunctionSerialized.from_value(decorated_simple)
        assert ser.metadata.get(b"wrapped") == b"1"

    def test_plain_function_no_wrapped_flag(self):
        ser = FunctionSerialized.from_value(simple)
        assert b"wrapped" not in ser.metadata

    # ── __dict__ / extra attributes set by decorator ────────────────────────

    def test_wrapper_dict_attribute_preserved(self):
        """`logging_decorator` sets `wrapper.logged = True` after @wraps."""
        ser = FunctionSerialized.from_value(decorated_with_log)
        fn = ser.value
        assert fn(3) == 13           # body: x + 10
        assert fn.__dict__.get("logged") is True

    def test_wrapper_dict_empty_for_plain_function(self):
        items = list(FunctionSerialized.from_value(simple).iter_())
        assert items[5].value == {}  # item 6 = wrapper_dict

    # ── stacked decorators ───────────────────────────────────────────────────

    def test_stacked_decorators_unwraps_to_innermost(self):
        """`@my_decorator(@double_decorator(fn))` – outermost FunctionType serialised."""
        ser = FunctionSerialized.from_value(stacked_decorated)
        fn = ser.value
        # stacked_decorated is a FunctionType (my_decorator's wrapper) so it is
        # serialised directly; the recovered function matches the original.
        assert fn(4) == stacked_decorated(4)

    def test_stacked_decorators_name(self):
        ser = FunctionSerialized.from_value(stacked_decorated)
        assert ser.metadata[b"name"] == b"stacked_decorated"

    # ── parameterised decorator factory ─────────────────────────────────────

    def test_parameterised_decorator_roundtrip(self):
        """The wrapper body (which prepends the prefix) is what's serialised."""
        ser = FunctionSerialized.from_value(prefixed_fn)
        # ser.value is the wrapper function recovered; its body returns f"{prefix}:{x*3}"
        # which matches what prefixed_fn itself produces.
        assert ser.value(2) == prefixed_fn(2)  # "out:6"

    def test_parameterised_decorator_name(self):
        assert FunctionSerialized.from_value(prefixed_fn).metadata[b"name"] == b"prefixed_fn"

    # ── globals / defaults / kwdefaults through wrapper ──────────────────────

    def test_decorated_global_captured(self):
        assert FunctionSerialized.from_value(decorated_with_global).value(3) == 3 * MULTIPLIER

    def test_decorated_positional_default(self):
        fn = FunctionSerialized.from_value(decorated_with_default).value
        assert fn(4) == 20    # default factor=5
        assert fn(4, 2) == 8

    def test_decorated_kwdefault(self):
        fn = FunctionSerialized.from_value(decorated_with_kwdefault).value
        assert fn(7) == "default:7"
        assert fn(7, tag="custom") == "custom:7"

    # ── wire-format ──────────────────────────────────────────────────────────

    def test_decorated_bwrite_pread_roundtrip(self):
        ser = FunctionSerialized.from_value(decorated_with_global)
        buf = BytesIO()
        ser.bwrite(buf)
        restored, _ = Serialized.pread_from(buf, 0)
        assert isinstance(restored, FunctionSerialized)
        assert restored.value(5) == 5 * MULTIPLIER

    def test_decorated_dumps_loads(self):
        assert loads(dumps(decorated_simple))(6) == 12

    def test_decorated_with_log_dumps_loads(self):
        fn = loads(dumps(decorated_with_log))
        assert fn(5) == 15
        assert fn.__dict__.get("logged") is True

    # ── inline closure-based decorator ──────────────────────────────────────

    def test_inline_decorator_with_closure(self):
        """A decorator that captures state via closure, applied inline."""
        call_count = [0]

        def counting_decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                call_count[0] += 1
                return fn(*args, **kwargs)
            return wrapper

        @counting_decorator
        def counted(x):
            return x * 7

        ser = FunctionSerialized.from_value(counted)
        assert ser.value(3) == 21

    # ── reject non-unwrappable callables ────────────────────────────────────

    def test_rejects_callable_without_wrapped(self):
        class MyCallable:
            def __call__(self, x):
                return x

        with pytest.raises(TypeError, match="__wrapped__"):
            FunctionSerialized.from_value(MyCallable())


if __name__ == "__main__":
    unittest.main()

