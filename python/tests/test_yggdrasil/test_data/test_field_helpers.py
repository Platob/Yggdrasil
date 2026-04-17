from __future__ import annotations

import pyarrow as pa
from yggdrasil.data.data_field import (
    _TYPE_JSON_METADATA_KEY,
    _attach_type_json_metadata,
    _parse_field_name_token,
    _safe_issubclass,
    _split_field_shorthand,
    _strip_internal_metadata,
    _strip_matching_quotes,
)


def test_strip_matching_quotes_removes_matching_outer_quotes_only():
    assert _strip_matching_quotes('"abc"') == "abc"
    assert _strip_matching_quotes("'abc'") == "abc"
    assert _strip_matching_quotes("`abc`") == "abc"
    assert _strip_matching_quotes(" abc ") == "abc"


def test_parse_field_name_token_supports_nullable_suffix():
    assert _parse_field_name_token("value?") == ("value", True)
    assert _parse_field_name_token("value!") == ("value", False)
    assert _parse_field_name_token('"value"') == ("value", None)


def test_split_field_shorthand_simple():
    left, right = _split_field_shorthand("price:int64")

    assert left == "price"
    assert right == "int64"


def test_split_field_shorthand_ignores_nested_colons():
    left, right = _split_field_shorthand("payload:struct<a:int64,b:string>")

    assert left == "payload"
    assert right == "struct<a:int64,b:string>"


def test_safe_issubclass_handles_non_types():
    assert _safe_issubclass(int, int) is True
    assert _safe_issubclass("int", int) is False
    assert _safe_issubclass(123, int) is False


def test_attach_type_json_metadata_adds_internal_key():
    metadata = {b"comment": b"hello"}

    out = _attach_type_json_metadata(pa.int64(), metadata)

    assert out[b"comment"] == b"hello"
    assert _TYPE_JSON_METADATA_KEY in out
    assert isinstance(out[_TYPE_JSON_METADATA_KEY], (bytes, bytearray))


def test_strip_internal_metadata_removes_internal_key_only():
    metadata = {
        b"comment": b"hello",
        _TYPE_JSON_METADATA_KEY: b'{"id":3}',
    }

    out = _strip_internal_metadata(metadata)

    assert out == {b"comment": b"hello"}


def test_strip_internal_metadata_returns_none_when_only_internal_key_present():
    metadata = {_TYPE_JSON_METADATA_KEY: b'{"id":3}'}

    out = _strip_internal_metadata(metadata)

    assert out is None
