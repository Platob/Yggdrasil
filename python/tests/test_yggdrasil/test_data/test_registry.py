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


# ---------------------------------------------------------------------------
# Enum lookup cache — populated on first call, reused per Enum class
# ---------------------------------------------------------------------------


class TestEnumLookupCache:
    """_enum_lookup_cache builds name/value maps once per Enum class and
    reuses them on every subsequent call, making repeated conversions O(1)."""

    def test_cache_populated_after_first_call(self) -> None:
        from yggdrasil.data.cast.registry import _enum_lookup_cache

        class _Fruit(enum.Enum):
            APPLE = 1
            BANANA = 2

        _enum_lookup_cache.pop(_Fruit, None)
        convert("apple", _Fruit)
        assert _Fruit in _enum_lookup_cache

    def test_cache_returns_same_entry_on_repeated_calls(self) -> None:
        from yggdrasil.data.cast.registry import _enum_lookup_cache

        class _Color(enum.Enum):
            RED = 1

        _enum_lookup_cache.pop(_Color, None)
        convert("red", _Color)
        entry_first = _enum_lookup_cache[_Color]
        convert("red", _Color)
        assert _enum_lookup_cache[_Color] is entry_first

    def test_different_enum_classes_get_separate_cache_entries(self) -> None:
        from yggdrasil.data.cast.registry import _enum_lookup_cache

        class _A(enum.Enum):
            X = 1

        class _B(enum.Enum):
            X = 2

        _enum_lookup_cache.pop(_A, None)
        _enum_lookup_cache.pop(_B, None)
        convert("x", _A)
        convert("x", _B)
        assert _enum_lookup_cache[_A] is not _enum_lookup_cache[_B]


# ---------------------------------------------------------------------------
# Wildcard Any → T converters
# ---------------------------------------------------------------------------


class TestWildcardAny:
    """Converters registered with from_hint=Any (or object) match any source
    type and are stored in _any_registry, not _registry."""

    def test_any_source_converter_is_dispatched(self) -> None:
        from typing import Any as TypingAny

        class _Target:
            def __init__(self, v: str) -> None:
                self.v = v

        @register_converter(TypingAny, _Target)
        def anything_to_target(value: object, opts: Any) -> _Target:
            return _Target(str(value))

        result = convert(42, _Target)
        assert isinstance(result, _Target)
        assert result.v == "42"

    def test_exact_match_beats_wildcard(self) -> None:
        from typing import Any as TypingAny

        class _Tgt:
            pass

        calls: list[str] = []

        @register_converter(TypingAny, _Tgt)
        def wild(v: object, opts: Any) -> _Tgt:
            calls.append("wild")
            return _Tgt()

        @register_converter(str, _Tgt)
        def exact(v: str, opts: Any) -> _Tgt:
            calls.append("exact")
            return _Tgt()

        convert("hello", _Tgt)
        assert calls == ["exact"]


# ---------------------------------------------------------------------------
# str_to_float / str_to_bool — default_value via duck-typed opts
# ---------------------------------------------------------------------------


class TestDefaultValueOption:
    """The str_to_float and str_to_bool converters honor a ``default_value``
    attribute on the options object for empty-string inputs.  This is accessed
    via ``getattr(opts, 'default_value', None)`` so any duck-typed opts works
    — the converter functions are called directly here to bypass the
    ``CastOptions.check`` normalization that ``convert()`` applies."""

    def test_str_to_float_empty_uses_default_via_duck_opts(self) -> None:
        from yggdrasil.data.cast.registry import str_to_float

        class _Opts:
            default_value = 0.0

        assert str_to_float("", _Opts()) == 0.0

    def test_str_to_bool_empty_uses_default_via_duck_opts(self) -> None:
        from yggdrasil.data.cast.registry import str_to_bool

        class _Opts:
            default_value = False

        assert str_to_bool("", _Opts()) is False

    def test_str_to_float_non_empty_ignores_default(self) -> None:
        from yggdrasil.data.cast.registry import str_to_float

        class _Opts:
            default_value = 99.0

        assert str_to_float("3.14", _Opts()) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# unwrap_optional / is_runtime_value — documented utility functions
# ---------------------------------------------------------------------------


class TestHintUtilities:

    def test_unwrap_optional_pipe_syntax(self) -> None:
        from yggdrasil.data.cast.registry import unwrap_optional

        is_opt, base = unwrap_optional(int | None)
        assert is_opt is True
        assert base is int

    def test_unwrap_optional_typing_Optional(self) -> None:
        from typing import Optional
        from yggdrasil.data.cast.registry import unwrap_optional

        is_opt, base = unwrap_optional(Optional[str])
        assert is_opt is True
        assert base is str

    def test_unwrap_optional_non_optional_passthrough(self) -> None:
        from yggdrasil.data.cast.registry import unwrap_optional

        is_opt, base = unwrap_optional(int)
        assert is_opt is False
        assert base is int

    def test_is_runtime_value_class_is_false(self) -> None:
        from yggdrasil.data.cast.registry import is_runtime_value

        assert is_runtime_value(int) is False
        assert is_runtime_value(str) is False

    def test_is_runtime_value_instance_is_true(self) -> None:
        from yggdrasil.data.cast.registry import is_runtime_value

        assert is_runtime_value(42) is True
        assert is_runtime_value([]) is True
        assert is_runtime_value("hello") is True

    def test_is_runtime_value_generic_alias_is_false(self) -> None:
        from yggdrasil.data.cast.registry import is_runtime_value

        assert is_runtime_value(list[int]) is False


# ---------------------------------------------------------------------------
# _find_cache — composition path hits cache on second call
# ---------------------------------------------------------------------------


class TestCompositionCache:
    """One-hop composed converters are also stored in _find_cache so the
    composition scan only runs once per (from_type, to_type) pair."""

    def test_composed_result_is_cached(self) -> None:
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

        _find_cache.pop((_P, float), None)
        c1 = find_converter(_P, float)
        assert c1 is not None
        assert (_P, float) in _find_cache

        c2 = find_converter(_P, float)
        assert c2 is c1
