"""Tests for Hive partition helpers in :mod:`yggdrasil.url.hive`."""
from __future__ import annotations

import pytest
import pyarrow as pa

from yggdrasil.url.hive import (
    HIVE_DEFAULT_PARTITION,
    hive_cast_value,
    hive_decode,
    hive_encode,
    hive_split,
)


class TestHiveEncode:

    def test_none_returns_default_partition(self):
        assert hive_encode(None) == HIVE_DEFAULT_PARTITION

    def test_string_value_is_url_quoted(self):
        assert hive_encode("hello world") == "hello%20world"

    def test_integer_value(self):
        assert hive_encode(42) == "42"

    def test_equals_sign_is_encoded(self):
        assert hive_encode("a=b") == "a%3Db"

    def test_slash_is_encoded(self):
        assert hive_encode("a/b") == "a%2Fb"

    def test_space_is_encoded(self):
        assert hive_encode("a b") == "a%20b"

    def test_special_characters_combined(self):
        encoded = hive_encode("col=val/foo bar")
        assert "=" not in encoded
        assert "/" not in encoded
        assert " " not in encoded


class TestHiveDecode:

    def test_default_partition_returns_none(self):
        assert hive_decode(HIVE_DEFAULT_PARTITION) is None

    def test_url_encoded_string_decoded(self):
        assert hive_decode("hello%20world") == "hello world"

    def test_plain_string_passes_through(self):
        assert hive_decode("plain") == "plain"


class TestHiveSplit:

    def test_col_eq_val(self):
        result = hive_split("year=2024")
        assert result == ("year", "2024")

    def test_no_equals_returns_none(self):
        assert hive_split("plain_dir") is None

    def test_empty_col_returns_none(self):
        assert hive_split("=val") is None

    def test_multiple_equals_splits_on_first(self):
        result = hive_split("col=a%3Db")
        assert result is not None
        col, val = result
        assert col == "col"
        assert val == "a=b"

    def test_decoded_value(self):
        result = hive_split("city=New%20York")
        assert result == ("city", "New York")


class TestHiveCastValue:

    def test_none_value_returns_none(self):
        assert hive_cast_value(None, pa.int64()) is None

    def test_none_dtype_returns_value_as_is(self):
        assert hive_cast_value("hello", None) == "hello"

    def test_both_none(self):
        assert hive_cast_value(None, None) is None

    def test_int64_cast(self):
        result = hive_cast_value("42", pa.int64())
        assert result == 42
        assert isinstance(result, int)

    def test_float64_cast(self):
        result = hive_cast_value("3.14", pa.float64())
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_bool_true(self):
        assert hive_cast_value("true", pa.bool_()) is True

    def test_bool_false(self):
        assert hive_cast_value("false", pa.bool_()) is False

    def test_string_dtype_keeps_string(self):
        result = hive_cast_value("hello", pa.string())
        assert result == "hello"
        assert isinstance(result, str)

    def test_invalid_cast_returns_original(self):
        result = hive_cast_value("not_a_number", pa.int64())
        assert result == "not_a_number"

    @pytest.mark.parametrize("raw,expected", [
        ("1", True),
        ("yes", True),
        ("t", True),
        ("TRUE", True),
        ("0", False),
        ("no", False),
        ("f", False),
        ("FALSE", False),
    ])
    def test_bool_edge_cases(self, raw, expected):
        assert hive_cast_value(raw, pa.bool_()) is expected

    def test_int32_cast(self):
        result = hive_cast_value("7", pa.int32())
        assert result == 7
        assert isinstance(result, int)

    def test_large_string_dtype(self):
        result = hive_cast_value("abc", pa.large_string())
        assert result == "abc"


class TestHiveRoundTrip:

    def test_string_round_trip(self):
        original = "hello world"
        encoded = hive_encode(original)
        decoded = hive_decode(encoded)
        assert decoded == original

    def test_none_round_trip(self):
        encoded = hive_encode(None)
        decoded = hive_decode(encoded)
        assert decoded is None

    def test_encode_split_cast_int(self):
        col = "year"
        value = 2024
        segment = f"{col}={hive_encode(value)}"
        result = hive_split(segment)
        assert result is not None
        split_col, split_val = result
        assert split_col == col
        casted = hive_cast_value(split_val, pa.int64())
        assert casted == value
        assert isinstance(casted, int)

    def test_encode_split_cast_float(self):
        col = "score"
        value = 9.5
        segment = f"{col}={hive_encode(value)}"
        result = hive_split(segment)
        assert result is not None
        split_col, split_val = result
        assert split_col == col
        casted = hive_cast_value(split_val, pa.float64())
        assert casted == pytest.approx(value)

    def test_special_chars_round_trip(self):
        original = "path/to=some thing"
        encoded = hive_encode(original)
        decoded = hive_decode(encoded)
        assert decoded == original
