from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import pyarrow as pa
import pytest

from yggdrasil.arrow.python_defaults import default_scalar
from yggdrasil.data.cast import CastOptions
from yggdrasil.data.cast.registry import convert, register_converter


def test_fast_path_identity_no_options_no_kwargs() -> None:
    assert convert(5, int) == 5
    assert convert("x", str) == "x"


def test_optional_unwrap_and_none_handling() -> None:
    assert convert(None, Optional[int]) is None
    assert convert(None, int) == default_scalar(int)
    assert convert(None, str) == default_scalar(str)


def test_default_str_to_int() -> None:
    assert convert("123", int) == 123
    assert convert("", int) == 0


def test_default_str_to_float() -> None:
    assert convert("1.5", float) == 1.5
    with pytest.raises(ValueError):
        convert("", float)


def test_default_str_to_bool() -> None:
    assert convert("true", bool) is True
    assert convert("1", bool) is True
    assert convert("no", bool) is False
    assert convert("0", bool) is False
    with pytest.raises(ValueError):
        convert("", bool)
    with pytest.raises(ValueError):
        convert("maybe", bool)


def test_default_int_to_str() -> None:
    assert convert(7, str) == "7"


def test_mro_fallback_uses_base_registration() -> None:
    class A:
        def __init__(self, x: int):
            self.x = x

    class B(A):
        pass

    @register_converter(A, int)
    def a_to_int(v: A, opts: Any) -> int:
        return v.x

    assert convert(B(9), int) == 9


def test_one_hop_composition_from_to_mid_to_target() -> None:
    class X:
        def __init__(self, n: int):
            self.n = n

    class Y:
        def __init__(self, s: str):
            self.s = s

    @register_converter(X, Y)
    def x_to_y(v: X, opts: Any) -> Y:
        return Y(str(v.n))

    @register_converter(Y, int)
    def y_to_int(v: Y, opts: Any) -> int:
        return int(v.s)

    assert convert(X(42), int) == 42


def test_enum_conversion_by_name_and_value() -> None:
    class Color(enum.Enum):
        RED = 1
        BLUE = 2

    assert convert("red", Color) is Color.RED
    assert convert("BLUE", Color) is Color.BLUE
    assert convert(1, Color) is Color.RED
    assert convert("2", Color) is Color.BLUE

    with pytest.raises(TypeError):
        convert("nope", Color)


def test_dataclass_conversion_defaults_and_factories_future_annotations_regression() -> None:
    # This test is the regression for the failure you hit:
    # when __future__.annotations is enabled, dataclasses.Field.type can be a string ("int"),
    # so defaulting must use get_type_hints() resolved types.
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

    # init=False and private fields ignored
    @dataclass
    class PrivateFields:
        a: int
        _skip: int = 10
        b: int = dataclasses.field(init=False, default=11)

    out3 = convert({"a": "9", "_skip": "999", "b": "999"}, PrivateFields)
    assert out3.a == 9
    assert out3._skip == 10
    assert out3.b == 11


def test_list_set_tuple_and_mapping_recursive_casts() -> None:
    assert convert(["1", "2"], list[int]) == [1, 2]
    assert convert({"1", "2"}, set[int]) == {1, 2}

    assert convert(["1", "2"], tuple[int, ...]) == (1, 2)
    assert convert(["1", 2], tuple[int, str]) == (1, "2")

    with pytest.raises(TypeError):
        convert([1, 2, 3], tuple[int, str])  # length mismatch

    m = convert({"1": "2"}, dict[int, int])
    assert m == {1: 2}

    m2 = convert({"1": "2"}, Mapping[int, int])
    assert dict(m2) == {1: 2}


def test_iterable_conversion_rejects_str_and_bytes_sources() -> None:
    with pytest.raises(TypeError):
        convert("123", list[int])
    with pytest.raises(TypeError):
        convert(b"123", set[int])


def test_arrow_array_to_list_recursively_casts_elements() -> None:
    arr = pa.array(["1", "2", "3"])
    out = convert(arr, list[int])
    assert out == [1, 2, 3]


def test_options_object_is_accepted() -> None:
    # Don’t assume specific fields exist; just ensure the arg path works.
    opts = CastOptions.check_arg()
    assert convert("2", int, options=opts) == 2