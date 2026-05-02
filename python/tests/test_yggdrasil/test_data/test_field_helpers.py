"""Private helpers in :mod:`yggdrasil.data.data_field`.

These are internal — but they're load-bearing for ``Field.from_str``
parsing and for the type-json-metadata round-trip that lets a
yggdrasil :class:`Field` survive an Arrow IPC round-trip without
losing its dtype intent. Worth pinning so a refactor on either side
fails loud here, not silently in the user-visible API.
"""
from __future__ import annotations

from yggdrasil.data.data_field import (
    _TYPE_JSON_METADATA_KEY,
    _parse_field_name_token,
    _safe_issubclass,
    _split_field_shorthand,
    _strip_internal_metadata,
    _strip_matching_quotes,
)


class TestStripMatchingQuotes:

    def test_double_quotes(self) -> None:
        assert _strip_matching_quotes('"abc"') == "abc"

    def test_single_quotes(self) -> None:
        assert _strip_matching_quotes("'abc'") == "abc"

    def test_backticks(self) -> None:
        assert _strip_matching_quotes("`abc`") == "abc"

    def test_only_strips_outer_whitespace_when_no_quotes(self) -> None:
        assert _strip_matching_quotes(" abc ") == "abc"


class TestParseFieldNameToken:

    def test_question_mark_marks_nullable_true(self) -> None:
        assert _parse_field_name_token("value?") == ("value", True)

    def test_bang_marks_nullable_false(self) -> None:
        assert _parse_field_name_token("value!") == ("value", False)

    def test_quoted_name_keeps_nullability_undecided(self) -> None:
        assert _parse_field_name_token('"value"') == ("value", None)


class TestSplitFieldShorthand:

    def test_simple_pair(self) -> None:
        assert _split_field_shorthand("price:int64") == ("price", "int64")

    def test_ignores_colons_inside_nested_brackets(self) -> None:
        left, right = _split_field_shorthand(
            "payload:struct<a:int64,b:string>"
        )

        assert left == "payload"
        assert right == "struct<a:int64,b:string>"


class TestSafeIssubclass:

    def test_real_subclass_pair(self) -> None:
        assert _safe_issubclass(int, int) is True

    def test_non_type_object_returns_false_without_raising(self) -> None:
        assert _safe_issubclass("int", int) is False
        assert _safe_issubclass(123, int) is False


class TestStripInternalMetadata:

    def test_drops_internal_key_only(self) -> None:
        metadata = {
            b"comment": b"hello",
            _TYPE_JSON_METADATA_KEY: b'{"id":3}',
        }

        assert _strip_internal_metadata(metadata) == {b"comment": b"hello"}

    def test_returns_none_when_only_internal_key_remains(self) -> None:
        metadata = {_TYPE_JSON_METADATA_KEY: b'{"id":3}'}

        assert _strip_internal_metadata(metadata) is None
