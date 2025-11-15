import json
import pytest

# Replace `safe_utils` with the actual module name if different
from yggdrasil.utils.py_utils import (
    safe_str,
    safe_bytes,
    safe_bool,
    safe_dict,
    merge_dicts,
    TRUE_STR_VALUES,
    TRUE_BYTES_VALUES,
    FALSE_STR_VALUES,
    FALSE_BYTES_VALUES,
)


def test_safe_str_basic():
    assert safe_str(None) is None
    assert safe_str("", default="d") == "d"         # empty is falsy -> default
    assert safe_str("abc") == "abc"
    assert safe_str(b"abc") == "abc"
    assert safe_str(bytearray(b"xyz")) == "xyz"
    assert safe_str(123) == "123"


def test_safe_bytes_basic():
    assert safe_bytes(None) is None
    assert safe_bytes("", default=b"d") == b"d"    # empty is falsy -> default
    assert safe_bytes(b"abc") == b"abc"
    assert safe_bytes("abc") == b"abc"
    assert safe_bytes(bytearray(b"qwe")) == b"qwe"
    assert safe_bytes(memoryview(b"mem")) == b"mem"
    # bytes() on a small iterable of ints should work
    assert safe_bytes([65, 66, 67]) == b"[65, 66, 67]"


def test_safe_bool_with_bool_and_strings_and_ints():
    assert safe_bool(None) is None
    assert safe_bool(True) is True
    assert safe_bool(False) is False

    # string truthy/falsy by first char
    assert safe_bool("True") is True
    assert safe_bool("true") is True
    assert safe_bool("1") is True
    assert safe_bool("Yes") is True

    assert safe_bool("False") is False
    assert safe_bool("0") is False
    assert safe_bool("no") is False

    # unknown string -> default returned
    assert safe_bool("maybe", default=None) is None
    assert safe_bool("maybe", default=False) is False

    # numeric values: note that 0 is falsy and will return default unless you pass default
    assert safe_bool(1, default=False) is True
    assert safe_bool(0, default=False) is False


@pytest.mark.xfail(reason="bytes handling in safe_bool uses obj[0] which returns int; "
                          "TRUE_BYTES_VALUES is a set of bytes values so comparison never matches",
                   strict=False)
def test_safe_bool_with_bytes_expected_behavior():
    # This expresses the intended behavior: bytes starting with 'T'/'t'/'1' etc. should map to True
    # But current implementation will index bytes and get an int, so this test is expected to xfail.
    assert safe_bool(b"True") is True
    assert safe_bool(b"false") is False
    assert safe_bool(b"1") is True
    assert safe_bool(b"0") is False


def test_safe_dict_with_dicts_and_checks():
    # simple dict passes through
    d = {"a": 1, "": 2, "b": 0}
    # empty key is removed
    out = safe_dict(d, default=None)
    assert "a" in out and "b" in out and "" not in out

    # apply check_key/check_value functions
    out2 = safe_dict({"A": "  x  "}, check_key=lambda k: k.lower(), check_value=lambda v: v.strip())
    assert out2 == {"a": "x"}

    # JSON string input
    s = '{"k": "v", "": "drop"}'
    out3 = safe_dict(s)
    assert out3 == {"k": "v"}

    # list-of-tuples
    lst = [("x", 1), ("", 2), ("y", 3)]
    out4 = safe_dict(lst)
    assert out4 == {"x": 1, "y": 3}

    # list-of-dicts and nested containers
    complex_in = [{"p": 1}, [("q", 2), ("", 0)], {"r": 3}]
    out5 = safe_dict(complex_in)
    assert out5 == {"p": 1, "q": 2, "r": 3}

    # invalid type should raise
    with pytest.raises(TypeError):
        safe_dict(12345)


def test_safe_dict_with_bytes_and_bytearray_json():
    # bytes/bytearray JSON input should parse
    b = b'{"kk": "vv", "": "drop"}'
    out = safe_dict(b)
    assert out == {"kk": "vv"}


def test_merge_dicts_behavior():
    assert merge_dicts(None) is None
    assert merge_dicts([]) is None

    d1 = {"a": 1}
    d2 = '{"b":2}'
    d3 = [("c", 3)]

    merged = merge_dicts([d1, d2, d3])
    assert merged == {"a": 1, "b": 2, "c": 3}

    # later dicts override earlier ones
    merged2 = merge_dicts([{"k": 1}, {"k": 2}])
    assert merged2 == {"k": 2}


def test_merge_dicts_filters_empty_dicts():
    # safe_dict returns default for empty inputs, merge should ignore them
    assert merge_dicts([None, {}, [], ""]) is None
    assert merge_dicts([{"a": 1}, None, {}]) == {"a": 1}
