"""``yggdrasil.data.cast.registry.convert`` — the global cast dispatcher.

The registry is the single dispatch surface every engine plugs into.
The contract under test:

* **Identity fast path** — same source/target produces no cast.
* **Optional unwrap** — ``Optional[T]`` strips the None wrapper before
  dispatch.
* **None handling** — ``None`` to ``T`` returns the registered default
  scalar for ``T``.
* **MRO fallback** — a registration on a base class catches subclass
  values.
* **One-hop composition** — ``X → Y`` plus ``Y → int`` chains
  automatically.
* **Native type support** — Enums (by name + value), dataclasses
  (with future-annotations resolution + private-field skipping),
  list / tuple / set / Mapping with element coercion.
* **Source-type guards** — ``str`` / ``bytes`` aren't valid iterable
  sources for list/set targets.
* **Options pass-through** — ``CastOptions`` is accepted and forwarded.
"""
from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from yggdrasil.arrow.python_defaults import default_scalar
from yggdrasil.data.cast.registry import convert, register_converter
from yggdrasil.data.options import CastOptions


# ---------------------------------------------------------------------------
# Identity / Optional / default fallbacks
# ---------------------------------------------------------------------------


class TestFastPaths:

    def test_identity_int_and_str(self) -> None:
        assert convert(5, int) == 5
        assert convert("x", str) == "x"

    def test_optional_unwraps_none_passthrough(self) -> None:
        assert convert(None, Optional[int]) is None

    def test_none_to_concrete_uses_default_scalar(self) -> None:
        assert convert(None, int) == default_scalar(int)
        assert convert(None, str) == default_scalar(str)


# ---------------------------------------------------------------------------
# Built-in scalar coercions
# ---------------------------------------------------------------------------


class TestStrToInt:

    def test_simple_decimal(self) -> None:
        assert convert("123", int) == 123

    def test_empty_string_falls_to_default(self) -> None:
        assert convert("", int) == 0


class TestStrToFloat:

    def test_simple(self) -> None:
        assert convert("1.5", float) == 1.5

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            convert("", float)


class TestStrToBool:

    @pytest.mark.parametrize("value,expected", [("true", True), ("1", True)])
    def test_truthy_tokens(self, value: str, expected: bool) -> None:
        assert convert(value, bool) is expected

    @pytest.mark.parametrize("value,expected", [("no", False), ("0", False)])
    def test_falsy_tokens(self, value: str, expected: bool) -> None:
        assert convert(value, bool) is expected

    @pytest.mark.parametrize("value", ["", "maybe"])
    def test_unrecognized_raises(self, value: str) -> None:
        with pytest.raises(ValueError):
            convert(value, bool)


class TestIntToStr:

    def test_simple(self) -> None:
        assert convert(7, str) == "7"


# ---------------------------------------------------------------------------
# Dispatch — MRO fallback + one-hop composition
# ---------------------------------------------------------------------------


class TestDispatch:

    def test_mro_fallback_catches_subclass(self) -> None:
        class A:
            def __init__(self, x: int) -> None:
                self.x = x

        class B(A):
            pass

        @register_converter(A, int)
        def a_to_int(v: A, opts: Any) -> int:
            return v.x

        assert convert(B(9), int) == 9

    def test_one_hop_composition(self) -> None:
        class X:
            def __init__(self, n: int) -> None:
                self.n = n

        class Y:
            def __init__(self, s: str) -> None:
                self.s = s

        @register_converter(X, Y)
        def x_to_y(v: X, opts: Any) -> Y:
            return Y(str(v.n))

        @register_converter(Y, int)
        def y_to_int(v: Y, opts: Any) -> int:
            return int(v.s)

        assert convert(X(42), int) == 42


# ---------------------------------------------------------------------------
# Enum dispatch
# ---------------------------------------------------------------------------


class TestEnum:

    def test_resolves_by_member_name(self) -> None:
        class Color(enum.Enum):
            RED = 1
            BLUE = 2

        assert convert("red", Color) is Color.RED
        assert convert("BLUE", Color) is Color.BLUE

    def test_resolves_by_member_value(self) -> None:
        class Color(enum.Enum):
            RED = 1
            BLUE = 2

        assert convert(1, Color) is Color.RED
        assert convert("2", Color) is Color.BLUE

    def test_unknown_raises(self) -> None:
        class Color(enum.Enum):
            RED = 1

        with pytest.raises(TypeError):
            convert("nope", Color)


# ---------------------------------------------------------------------------
# Dataclass dispatch — incl. the future-annotations regression
# ---------------------------------------------------------------------------


class TestDataclass:

    def test_defaults_and_default_factories_with_future_annotations(self) -> None:
        # Regression: with ``from __future__ import annotations``,
        # dataclasses.Field.type is a string ("int"), so defaulting must
        # use ``get_type_hints`` to resolve to real types.
        @dataclass
        class Cfg:
            a: int
            b: str = "x"
            c: int = field(default_factory=lambda: 7)

        out = convert({}, Cfg)
        assert isinstance(out, Cfg)
        assert out.a == default_scalar(int)
        assert out.b == "x"
        assert out.c == 7

        out2 = convert({"a": "3", "b": 5}, Cfg)
        assert out2 == Cfg(a=3, b="5", c=7)

    def test_init_false_and_private_fields_ignored(self) -> None:
        @dataclass
        class PrivateFields:
            a: int
            _skip: int = 10
            b: int = dataclasses.field(init=False, default=11)

        out = convert({"a": "9", "_skip": "999", "b": "999"}, PrivateFields)

        assert out.a == 9
        assert out._skip == 10
        assert out.b == 11


# ---------------------------------------------------------------------------
# Containers — list / tuple / set / dict / Mapping
# ---------------------------------------------------------------------------


class TestContainers:

    def test_list_int(self) -> None:
        assert convert(["1", "2"], list[int]) == [1, 2]

    def test_set_int(self) -> None:
        assert convert({"1", "2"}, set[int]) == {1, 2}

    def test_variadic_tuple_int(self) -> None:
        assert convert(["1", "2"], tuple[int, ...]) == (1, 2)

    def test_fixed_tuple_int_str(self) -> None:
        assert convert(["1", 2], tuple[int, str]) == (1, "2")

    def test_fixed_tuple_length_mismatch_raises(self) -> None:
        with pytest.raises(TypeError):
            convert([1, 2, 3], tuple[int, str])

    def test_dict_int_int(self) -> None:
        assert convert({"1": "2"}, dict[int, int]) == {1: 2}

    def test_mapping_int_int(self) -> None:
        out = convert({"1": "2"}, Mapping[int, int])

        assert dict(out) == {1: 2}


# ---------------------------------------------------------------------------
# Source-type guards — str / bytes aren't iterable sources for collections.
# ---------------------------------------------------------------------------


class TestIterableGuards:

    def test_str_to_list_int_raises(self) -> None:
        with pytest.raises(TypeError):
            convert("123", list[int])

    def test_bytes_to_set_int_raises(self) -> None:
        with pytest.raises(TypeError):
            convert(b"123", set[int])


# ---------------------------------------------------------------------------
# Options pass-through
# ---------------------------------------------------------------------------


class TestOptions:

    def test_cast_options_argument_is_threaded_through(self) -> None:
        # Don't assume specific fields exist; just ensure the path doesn't
        # reject the options object.
        opts = CastOptions.check()
        assert convert("2", int, options=opts) == 2
