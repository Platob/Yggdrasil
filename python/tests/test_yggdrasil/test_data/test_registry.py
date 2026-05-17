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
# Wildcard Any-source dispatch via _any_registry
# ---------------------------------------------------------------------------


class TestWildcardAnyDispatch:
    """register_converter(Any, T) stores in _any_registry[T] and is reached
    as step 3 of find_converter (after exact match and identity checks)."""

    def test_any_source_converter_fires_for_any_type(self) -> None:
        from typing import Any as TypingAny

        class Tag:
            def __init__(self, s: str) -> None:
                self.s = s

        @register_converter(TypingAny, Tag)
        def anything_to_tag(v: TypingAny, opts: Any) -> Tag:
            return Tag(str(v))

        assert isinstance(convert(42, Tag), Tag)
        assert isinstance(convert("hello", Tag), Tag)
        assert isinstance(convert(3.14, Tag), Tag)

    def test_exact_registration_beats_wildcard(self) -> None:
        """A concrete (from_type, to_hint) registration takes priority over
        the Any wildcard handler for the same target."""
        from typing import Any as TypingAny

        class Box:
            def __init__(self, val: Any) -> None:
                self.val = val

        @register_converter(TypingAny, Box)
        def any_to_box(v: TypingAny, opts: Any) -> Box:
            return Box(("any", v))

        @register_converter(int, Box)
        def int_to_box(v: int, opts: Any) -> Box:
            return Box(("int", v))

        result = convert(99, Box)
        assert result.val == ("int", 99)

        result_str = convert("x", Box)
        assert result_str.val == ("any", "x")


# ---------------------------------------------------------------------------
# Cache preservation — valid non-None hits survive new registrations
# ---------------------------------------------------------------------------


class TestCachePreservation:
    """After the targeted-invalidation optimization, cached hits for unrelated
    type pairs must not be evicted when a new converter is registered."""

    def test_unrelated_hit_survives_new_registration(self) -> None:
        from yggdrasil.data.cast.registry import _find_cache, find_converter

        class _Src:
            pass

        class _Dst:
            pass

        class _Unrelated:
            pass

        # Populate cache for (_Src, _Dst) → None (no path).
        find_converter(_Src, _Dst)
        # Also populate (str, int) which has a real converter.
        c_str_int = find_converter(str, int)
        assert c_str_int is not None
        assert (str, int) in _find_cache
        cached_before = _find_cache[str, int]

        # Register something completely unrelated.
        @register_converter(_Unrelated, _Unrelated)
        def noop(v: Any, opts: Any) -> Any:
            return v

        # The non-None hit for (str, int) must still be there.
        assert (str, int) in _find_cache
        assert _find_cache[str, int] is cached_before

    def test_none_entries_are_evicted_on_new_registration(self) -> None:
        from yggdrasil.data.cast.registry import _find_cache, find_converter

        class _A2:
            pass

        class _B2:
            pass

        class _C2:
            pass

        # Cache a None entry for (_A2, _B2).
        find_converter(_A2, _B2)
        assert _find_cache.get((_A2, _B2)) is None

        # Register something that doesn't directly connect _A2 → _B2.
        @register_converter(_C2, _C2)
        def noop2(v: Any, opts: Any) -> Any:
            return v

        # The None entry must be gone — it will be re-evaluated on demand.
        assert (_A2, _B2) not in _find_cache

    def test_exact_key_evicted_on_direct_registration(self) -> None:
        """When a converter is registered for a pair that was cached
        (e.g. as None), the cache entry for that exact pair is dropped
        so the new converter is found on the next dispatch call."""
        from yggdrasil.data.cast.registry import _find_cache, find_converter

        class _P:
            pass

        class _Q:
            pass

        # Ensure no path exists yet; cache the None result.
        assert find_converter(_P, _Q) is None
        assert (_P, _Q) in _find_cache

        @register_converter(_P, _Q)
        def p_to_q(v: _P, opts: Any) -> _Q:
            return _Q()

        # Cache entry must be gone so the new converter is found.
        assert (_P, _Q) not in _find_cache
        assert find_converter(_P, _Q) is p_to_q


# ---------------------------------------------------------------------------
# identity function contract
# ---------------------------------------------------------------------------


class TestIdentityFunction:

    def test_identity_returns_value_unchanged(self) -> None:
        from yggdrasil.data.cast.registry import identity

        obj = object()
        assert identity(obj) is obj

    def test_identity_accepts_extra_args(self) -> None:
        from yggdrasil.data.cast.registry import identity

        # Converter signature is func(value, options); identity absorbs both.
        assert identity(42, None) == 42
        assert identity("x", object()) == "x"
