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
from types import SimpleNamespace
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

    def test_str_to_float_empty_string_raises_without_options(self) -> None:
        with pytest.raises(ValueError):
            convert("", float)

    def test_str_to_bool_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            convert("", bool)


# ---------------------------------------------------------------------------
# Dispatch cache — populated on first call, reused on subsequent calls
# ---------------------------------------------------------------------------


class TestDispatchCache:
    """_find_cache memoizes resolved (from_type, to_hint) pairs so repeated
    find_converter calls pay a single dict lookup instead of MRO scanning."""

    def test_same_converter_returned_on_repeated_calls(self) -> None:
        from yggdrasil.data.cast.registry import find_converter

        c1 = find_converter(str, int)
        c2 = find_converter(str, int)
        assert c1 is c2

    def test_mro_miss_is_cached_as_none(self) -> None:
        from yggdrasil.data.cast.registry import find_converter

        # A pair with no registered converter should resolve to None and
        # the None result should itself be cached so the MRO scan doesn't
        # repeat on the next call.
        class _Src:
            pass

        class _Tgt:
            pass

        c1 = find_converter(_Src, _Tgt)
        c2 = find_converter(_Src, _Tgt)
        assert c1 is None
        assert c2 is None

    def test_cache_invalidated_on_new_registration(self) -> None:
        from yggdrasil.data.cast.registry import _find_cache, find_converter

        class _A:
            pass

        class _B:
            pass

        # Populate cache with None.
        c_before = find_converter(_A, _B)
        assert c_before is None
        assert (_A, _B) in _find_cache

        # New registration invalidates the cache.
        @register_converter(_A, _B)
        def a_to_b(v: _A, opts: Any) -> _B:
            return _B()

        assert (_A, _B) not in _find_cache

        c_after = find_converter(_A, _B)
        assert c_after is a_to_b

    def test_identity_is_cached_for_any_target(self) -> None:
        from yggdrasil.data.cast.registry import _find_cache, find_converter, identity
        from typing import Any as TypingAny

        find_converter(str, TypingAny)
        assert _find_cache.get((str, TypingAny)) is identity


# ---------------------------------------------------------------------------
# Extended container / dataclass / enum edge cases
# ---------------------------------------------------------------------------


class TestContainersEdgeCases:

    def test_list_no_element_type_annotation_passes_through(self) -> None:
        # ``list`` without a subscript → element hint defaults to Any → fast path.
        result = convert([1, "2", True], list)
        assert result == [1, "2", True]

    def test_tuple_empty_passes_through(self) -> None:
        assert convert([], tuple) == ()

    def test_dict_value_only_cast(self) -> None:
        # key hint is Any → only values are cast.
        out = convert({"a": "1", "b": "2"}, dict[str, int])
        assert out == {"a": 1, "b": 2}

    def test_dict_key_only_cast(self) -> None:
        out = convert({"1": "x", "2": "y"}, dict[int, str])
        assert out == {1: "x", 2: "y"}

    def test_nested_list_int(self) -> None:
        assert convert([["1", "2"], ["3"]], list[list[int]]) == [[1, 2], [3]]

    def test_str_to_list_raises(self) -> None:
        with pytest.raises(TypeError):
            convert("abc", list[str])

    def test_bytes_to_tuple_raises(self) -> None:
        with pytest.raises(TypeError):
            convert(b"abc", tuple[int, ...])


class TestDataclassEdgeCases:

    def test_non_mapping_input_raises(self) -> None:
        @dataclass
        class Cfg:
            x: int = 0

        with pytest.raises(TypeError):
            convert([1, 2], Cfg)

    def test_already_instance_returns_same_object(self) -> None:
        @dataclass
        class Cfg:
            x: int = 0

        obj = Cfg(x=5)
        assert convert(obj, Cfg) is obj

    def test_extra_mapping_keys_are_ignored(self) -> None:
        @dataclass
        class Cfg:
            a: int = 0

        out = convert({"a": 3, "extra": "ignored"}, Cfg)
        assert out.a == 3


class TestEnumEdgeCases:

    def test_already_member_returns_same_object(self) -> None:
        class Color(enum.Enum):
            RED = 1

        assert convert(Color.RED, Color) is Color.RED

    def test_str_value_lookup_works(self) -> None:
        class Status(enum.Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        assert convert("active", Status) is Status.ACTIVE

    def test_int_value_lookup_works(self) -> None:
        class Priority(enum.Enum):
            LOW = 1
            HIGH = 2

        assert convert(1, Priority) is Priority.LOW

    def test_case_insensitive_name_lookup(self) -> None:
        class Color(enum.Enum):
            RED = 1

        assert convert("RED", Color) is Color.RED
        assert convert("red", Color) is Color.RED

    def test_unhashable_member_value_falls_back_to_name(self) -> None:
        # Enum members whose values are unhashable (list, dict) cannot be
        # put in the value_lookup dict; the fallback is the name-based lookup.
        class ListEnum(enum.Enum):
            A = [1, 2]
            B = [3, 4]

        assert convert("a", ListEnum) is ListEnum.A
        assert convert("B", ListEnum) is ListEnum.B

    def test_empty_enum_raises_type_error(self) -> None:
        class Empty(enum.Enum):
            pass

        with pytest.raises(TypeError, match="empty Enum"):
            convert("x", Empty)


# ---------------------------------------------------------------------------
# Options pass-through — default_value on str→float and str→bool converters
#
# The converters use ``getattr(opts, "default_value", None)`` so any object
# with that attribute works — CastOptions doesn't carry that field, so we
# drive the low-level converters directly here with a SimpleNamespace.
# ---------------------------------------------------------------------------


class TestOptionsDefaultValue:

    def test_str_to_float_empty_string_uses_default_value(self) -> None:
        from yggdrasil.data.cast.registry import str_to_float

        opts = SimpleNamespace(default_value=0.0)
        assert str_to_float("", opts) == 0.0

    def test_str_to_float_non_empty_ignores_default_value(self) -> None:
        from yggdrasil.data.cast.registry import str_to_float

        opts = SimpleNamespace(default_value=99.9)
        assert str_to_float("1.5", opts) == pytest.approx(1.5)

    def test_str_to_bool_empty_string_uses_default_value(self) -> None:
        from yggdrasil.data.cast.registry import str_to_bool

        opts = SimpleNamespace(default_value=False)
        assert str_to_bool("", opts) is False

    def test_str_to_bool_non_empty_ignores_default_value(self) -> None:
        from yggdrasil.data.cast.registry import str_to_bool

        opts = SimpleNamespace(default_value=False)
        assert str_to_bool("true", opts) is True


# ---------------------------------------------------------------------------
# Composition cache — composed result lands in _find_cache
# ---------------------------------------------------------------------------


class TestCompositionCache:

    def test_composed_result_cached_in_find_cache(self) -> None:
        from yggdrasil.data.cast.registry import _find_cache, find_converter

        class _P:
            def __init__(self, n: int) -> None:
                self.n = n

        class _Q:
            def __init__(self, s: str) -> None:
                self.s = s

        @register_converter(_P, _Q)
        def p_to_q(v: _P, opts: Any) -> _Q:
            return _Q(str(v.n))

        @register_converter(_Q, float)
        def q_to_float(v: _Q, opts: Any) -> float:
            return float(v.s)

        # Clear so this pair starts cold.
        _find_cache.pop((_P, float), None)

        result = find_converter(_P, float)
        assert result is not None

        # Second call must return the exact same object (cache hit).
        result2 = find_converter(_P, float)
        assert result is result2
        assert (_P, float) in _find_cache

    def test_composition_produces_correct_value(self) -> None:
        class _R:
            def __init__(self, n: int) -> None:
                self.n = n

        class _S:
            def __init__(self, s: str) -> None:
                self.s = s

        @register_converter(_R, _S)
        def r_to_s(v: _R, opts: Any) -> _S:
            return _S(str(v.n * 10))

        @register_converter(_S, int)
        def s_to_int(v: _S, opts: Any) -> int:
            return int(v.s)

        assert convert(_R(7), int) == 70


# ---------------------------------------------------------------------------
# _registry_by_from index — stays consistent across registrations
# ---------------------------------------------------------------------------


class TestRegistryByFromIndex:

    def test_index_is_rebuilt_after_stale_marking(self) -> None:
        from yggdrasil.data.cast.registry import (
            _registry_by_from,
            _registry_index_stale,
            find_converter,
        )

        class _T1:
            pass

        class _T2:
            pass

        # Ensure a composition call triggers a rebuild.
        find_converter(_T1, _T2)  # populates cache, rebuilds index if stale
        assert not _registry_index_stale  # index is fresh after call

    def test_index_marked_stale_on_registration(self) -> None:
        import yggdrasil.data.cast.registry as _reg

        # Trigger a rebuild first.
        from yggdrasil.data.cast.registry import find_converter

        class _U1:
            pass

        class _U2:
            pass

        find_converter(_U1, _U2)
        assert not _reg._registry_index_stale

        # A new registration must mark it stale again.
        @register_converter(_U1, _U2)
        def u1_to_u2(v: Any, opts: Any) -> _U2:
            return _U2()

        assert _reg._registry_index_stale
