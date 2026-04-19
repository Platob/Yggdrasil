"""Tests for ``DataType._convert_pyobj`` — the single-value coercion hook.

Every concrete DataType should be able to consume at least ``str`` and
``bytes`` inputs (the priority paths) plus the obvious Python native form
for the type. ``safe=True`` turns parse failures into ``ValueError``;
``safe=False`` (default) returns ``None`` on unparseable input.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from yggdrasil.data.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DurationType,
    FloatingPointType,
    IntegerType,
    MapType,
    NullType,
    StringType,
    StructType,
    TimeType,
    TimestampType,
)


# ---------------------------------------------------------------------------
# NullType
# ---------------------------------------------------------------------------

class TestNullType:
    def test_always_returns_none(self) -> None:
        assert NullType().convert_pyobj(None, nullable=True) is None
        assert NullType().convert_pyobj("anything", nullable=True) is None
        assert NullType().convert_pyobj(42, nullable=True) is None


# ---------------------------------------------------------------------------
# BooleanType
# ---------------------------------------------------------------------------

class TestBooleanType:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("y", True),
            ("on", True),
            ("1", True),
            ("false", False),
            ("no", False),
            ("off", False),
            ("0", False),
            ("", False),
            (b"true", True),
            (b"false", False),
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            (1.5, True),
            (0.0, False),
            (Decimal("1"), True),
            (Decimal("0"), False),
        ],
    )
    def test_recognized(self, value, expected) -> None:
        assert BooleanType().convert_pyobj(value, nullable=True) is expected

    def test_whitespace_string(self) -> None:
        assert BooleanType().convert_pyobj("  true  ", nullable=True) is True

    def test_unparseable_unsafe_returns_none(self) -> None:
        assert BooleanType().convert_pyobj("garbage", nullable=True) is None

    def test_unparseable_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse bool"):
            BooleanType().convert_pyobj("garbage", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# IntegerType
# ---------------------------------------------------------------------------

class TestIntegerType:
    def test_str_decimal(self) -> None:
        assert IntegerType().convert_pyobj("42", nullable=True) == 42

    def test_str_hex(self) -> None:
        assert IntegerType().convert_pyobj("0x1a", nullable=True) == 26

    def test_str_bin(self) -> None:
        assert IntegerType().convert_pyobj("0b1010", nullable=True) == 10

    def test_str_float(self) -> None:
        assert IntegerType().convert_pyobj("3.9", nullable=True) == 3

    def test_bytes(self) -> None:
        assert IntegerType().convert_pyobj(b"42", nullable=True) == 42

    def test_bool(self) -> None:
        assert IntegerType().convert_pyobj(True, nullable=True) == 1
        assert IntegerType().convert_pyobj(False, nullable=True) == 0

    def test_float_truncates(self) -> None:
        assert IntegerType().convert_pyobj(3.9, nullable=True) == 3
        assert IntegerType().convert_pyobj(-3.9, nullable=True) == -3

    def test_decimal(self) -> None:
        assert IntegerType().convert_pyobj(Decimal("5.5"), nullable=True) == 5

    def test_nan_unsafe(self) -> None:
        assert IntegerType().convert_pyobj(float("nan"), nullable=True) is None

    def test_nan_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            IntegerType().convert_pyobj(float("nan"), nullable=True, safe=True)

    def test_empty_string_unsafe(self) -> None:
        assert IntegerType().convert_pyobj("", nullable=True) is None

    def test_empty_string_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="empty string"):
            IntegerType().convert_pyobj("", nullable=True, safe=True)

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse int"):
            IntegerType().convert_pyobj("garbage", nullable=True, safe=True)

    def test_garbage_unsafe_returns_none(self) -> None:
        assert IntegerType().convert_pyobj("garbage", nullable=True) is None


# ---------------------------------------------------------------------------
# FloatingPointType
# ---------------------------------------------------------------------------

class TestFloatingPointType:
    def test_str(self) -> None:
        assert FloatingPointType().convert_pyobj("3.14", nullable=True) == 3.14

    def test_scientific(self) -> None:
        assert FloatingPointType().convert_pyobj("1e3", nullable=True) == 1000.0

    def test_bytes(self) -> None:
        assert FloatingPointType().convert_pyobj(b"2.5", nullable=True) == 2.5

    def test_int(self) -> None:
        assert FloatingPointType().convert_pyobj(5, nullable=True) == 5.0

    def test_decimal(self) -> None:
        assert FloatingPointType().convert_pyobj(Decimal("1.5"), nullable=True) == 1.5

    def test_inf(self) -> None:
        assert FloatingPointType().convert_pyobj("inf", nullable=True) == float("inf")

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse float"):
            FloatingPointType().convert_pyobj("nope", nullable=True, safe=True)

    def test_garbage_unsafe(self) -> None:
        assert FloatingPointType().convert_pyobj("nope", nullable=True) is None


# ---------------------------------------------------------------------------
# DecimalType
# ---------------------------------------------------------------------------

class TestDecimalType:
    def test_str(self) -> None:
        assert DecimalType().convert_pyobj("3.14", nullable=True) == Decimal("3.14")

    def test_bytes(self) -> None:
        assert DecimalType().convert_pyobj(b"1.5", nullable=True) == Decimal("1.5")

    def test_int(self) -> None:
        assert DecimalType().convert_pyobj(5, nullable=True) == Decimal(5)

    def test_float(self) -> None:
        # float → str(float) → Decimal preserves visible repr.
        assert DecimalType().convert_pyobj(0.1, nullable=True) == Decimal("0.1")

    def test_nan_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            DecimalType().convert_pyobj(float("nan"), nullable=True, safe=True)

    def test_garbage_unsafe(self) -> None:
        assert DecimalType().convert_pyobj("nope", nullable=True) is None


# ---------------------------------------------------------------------------
# StringType
# ---------------------------------------------------------------------------

class TestStringType:
    def test_str_passthrough(self) -> None:
        assert StringType().convert_pyobj("hello", nullable=True) == "hello"

    def test_bytes_decode(self) -> None:
        assert StringType().convert_pyobj(b"hello", nullable=True) == "hello"

    def test_int(self) -> None:
        assert StringType().convert_pyobj(42, nullable=True) == "42"

    def test_bool(self) -> None:
        assert StringType().convert_pyobj(True, nullable=True) == "true"
        assert StringType().convert_pyobj(False, nullable=True) == "false"

    def test_float(self) -> None:
        assert StringType().convert_pyobj(3.14, nullable=True) == "3.14"

    def test_date(self) -> None:
        assert StringType().convert_pyobj(dt.date(2024, 1, 1), nullable=True) == "2024-01-01"

    def test_bad_utf8_unsafe(self) -> None:
        assert StringType().convert_pyobj(b"\xff\xfe", nullable=True) is None

    def test_bad_utf8_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="decode bytes"):
            StringType().convert_pyobj(b"\xff\xfe", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# BinaryType
# ---------------------------------------------------------------------------

class TestBinaryType:
    def test_bytes_passthrough(self) -> None:
        assert BinaryType().convert_pyobj(b"abc", nullable=True) == b"abc"

    def test_str_encode(self) -> None:
        assert BinaryType().convert_pyobj("abc", nullable=True) == b"abc"

    def test_bytearray(self) -> None:
        assert BinaryType().convert_pyobj(bytearray(b"xy"), nullable=True) == b"xy"

    def test_memoryview(self) -> None:
        assert BinaryType().convert_pyobj(memoryview(b"mv"), nullable=True) == b"mv"

    def test_fixed_size_pads(self) -> None:
        assert BinaryType(byte_size=5).convert_pyobj(b"ab", nullable=True) == b"ab\x00\x00\x00"

    def test_fixed_size_truncates_unsafe(self) -> None:
        assert BinaryType(byte_size=2).convert_pyobj(b"abcd", nullable=True) == b"ab"

    def test_fixed_size_truncates_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds fixed"):
            BinaryType(byte_size=2).convert_pyobj(b"abcd", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# DateType
# ---------------------------------------------------------------------------

class TestDateType:
    def test_iso_str(self) -> None:
        assert DateType().convert_pyobj("2024-01-15", nullable=True) == dt.date(2024, 1, 15)

    def test_iso_bytes(self) -> None:
        assert DateType().convert_pyobj(b"2024-01-15", nullable=True) == dt.date(2024, 1, 15)

    def test_iso_datetime_str(self) -> None:
        assert DateType().convert_pyobj(
            "2024-01-15T10:30:45", nullable=True
        ) == dt.date(2024, 1, 15)

    def test_datetime_passthrough(self) -> None:
        assert DateType().convert_pyobj(
            dt.datetime(2024, 1, 15, 10, 30), nullable=True
        ) == dt.date(2024, 1, 15)

    def test_date_passthrough(self) -> None:
        d = dt.date(2024, 1, 15)
        assert DateType().convert_pyobj(d, nullable=True) == d

    def test_int_days_since_epoch(self) -> None:
        assert DateType().convert_pyobj(0, nullable=True) == dt.date(1970, 1, 1)
        assert DateType().convert_pyobj(1, nullable=True) == dt.date(1970, 1, 2)

    def test_garbage_unsafe(self) -> None:
        assert DateType().convert_pyobj("not a date", nullable=True) is None

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse date"):
            DateType().convert_pyobj("not a date", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# TimeType
# ---------------------------------------------------------------------------

class TestTimeType:
    def test_iso_str(self) -> None:
        assert TimeType().convert_pyobj("10:30:45", nullable=True) == dt.time(10, 30, 45)

    def test_iso_bytes(self) -> None:
        assert TimeType().convert_pyobj(b"10:30:45", nullable=True) == dt.time(10, 30, 45)

    def test_datetime_extracts_time(self) -> None:
        assert TimeType().convert_pyobj(
            dt.datetime(2024, 1, 1, 10, 30, 45), nullable=True
        ) == dt.time(10, 30, 45)

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse time"):
            TimeType().convert_pyobj("not a time", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# TimestampType
# ---------------------------------------------------------------------------

class TestTimestampType:
    def test_iso_str_naive(self) -> None:
        assert TimestampType().convert_pyobj(
            "2024-01-15T10:30:45", nullable=True
        ) == dt.datetime(2024, 1, 15, 10, 30, 45)

    def test_iso_bytes_z(self) -> None:
        out = TimestampType(tz="UTC").convert_pyobj(
            b"2024-01-15T10:30:45Z", nullable=True
        )
        assert out == dt.datetime(2024, 1, 15, 10, 30, 45, tzinfo=dt.timezone.utc)

    def test_space_separator(self) -> None:
        assert TimestampType().convert_pyobj(
            "2024-01-15 10:30:45", nullable=True
        ) == dt.datetime(2024, 1, 15, 10, 30, 45)

    def test_date_only_str(self) -> None:
        assert TimestampType().convert_pyobj(
            "2024-01-15", nullable=True
        ) == dt.datetime(2024, 1, 15)

    def test_naive_target_strips_tz(self) -> None:
        out = TimestampType(tz=None).convert_pyobj(
            "2024-01-15T10:30:45+05:00", nullable=True
        )
        # Input is 10:30:45 at +05:00 → 05:30:45 UTC (naive).
        assert out == dt.datetime(2024, 1, 15, 5, 30, 45)

    def test_epoch_seconds(self) -> None:
        out = TimestampType(unit="s", tz="UTC").convert_pyobj(0, nullable=True)
        assert out == dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

    def test_date_input(self) -> None:
        assert TimestampType().convert_pyobj(
            dt.date(2024, 1, 15), nullable=True
        ) == dt.datetime(2024, 1, 15)

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            TimestampType().convert_pyobj("not a timestamp", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# DurationType
# ---------------------------------------------------------------------------

class TestDurationType:
    def test_iso_str(self) -> None:
        assert DurationType().convert_pyobj("PT1H30M", nullable=True) == dt.timedelta(
            hours=1, minutes=30
        )

    def test_iso_str_seconds(self) -> None:
        assert DurationType().convert_pyobj(
            "PT45.5S", nullable=True
        ) == dt.timedelta(seconds=45, microseconds=500_000)

    def test_iso_str_negative(self) -> None:
        assert DurationType().convert_pyobj("-PT1H", nullable=True) == dt.timedelta(
            hours=-1
        )

    def test_clock_str(self) -> None:
        assert DurationType().convert_pyobj("01:30:00", nullable=True) == dt.timedelta(
            hours=1, minutes=30
        )

    def test_clock_bytes(self) -> None:
        assert DurationType().convert_pyobj(
            b"01:30:00", nullable=True
        ) == dt.timedelta(hours=1, minutes=30)

    def test_clock_negative(self) -> None:
        assert DurationType().convert_pyobj(
            "-01:30:00", nullable=True
        ) == dt.timedelta(hours=-1, minutes=-30)

    def test_numeric_str_in_unit(self) -> None:
        # Default unit is 'us' (microseconds).
        assert DurationType(unit="us").convert_pyobj(
            "1000000", nullable=True
        ) == dt.timedelta(seconds=1)

    def test_int_in_unit(self) -> None:
        assert DurationType(unit="s").convert_pyobj(60, nullable=True) == dt.timedelta(
            seconds=60
        )

    def test_timedelta_passthrough(self) -> None:
        td = dt.timedelta(seconds=42)
        assert DurationType().convert_pyobj(td, nullable=True) is td

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse duration"):
            DurationType().convert_pyobj("not a duration", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# ArrayType
# ---------------------------------------------------------------------------

class TestArrayType:
    def _int_array(self) -> ArrayType:
        return ArrayType.from_item_field(IntegerType().to_field(name="item"))

    def test_json_str(self) -> None:
        assert self._int_array().convert_pyobj("[1, 2, 3]", nullable=True) == [1, 2, 3]

    def test_json_bytes(self) -> None:
        assert self._int_array().convert_pyobj(b"[1, 2, 3]", nullable=True) == [1, 2, 3]

    def test_list_coerces_items(self) -> None:
        assert self._int_array().convert_pyobj([1, "2", 3.5], nullable=True) == [1, 2, 3]

    def test_tuple_input(self) -> None:
        assert self._int_array().convert_pyobj((1, 2, 3), nullable=True) == [1, 2, 3]

    def test_set_input(self) -> None:
        result = self._int_array().convert_pyobj({1, 2, 3}, nullable=True)
        assert sorted(result) == [1, 2, 3]

    def test_bad_json_unsafe(self) -> None:
        assert self._int_array().convert_pyobj("not json", nullable=True) is None

    def test_bad_json_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse array"):
            self._int_array().convert_pyobj("not json", nullable=True, safe=True)

    def test_scalar_input_unsafe(self) -> None:
        assert self._int_array().convert_pyobj(42, nullable=True) is None


# ---------------------------------------------------------------------------
# MapType
# ---------------------------------------------------------------------------

class TestMapType:
    def _str_int_map(self) -> MapType:
        return MapType.from_key_value(
            key_field=StringType().to_field(),
            value_field=IntegerType().to_field(),
        )

    def test_json_str(self) -> None:
        assert self._str_int_map().convert_pyobj(
            '{"a": 1, "b": 2}', nullable=True
        ) == {"a": 1, "b": 2}

    def test_json_bytes(self) -> None:
        assert self._str_int_map().convert_pyobj(
            b'{"a": 1}', nullable=True
        ) == {"a": 1}

    def test_dict_coerces(self) -> None:
        assert self._str_int_map().convert_pyobj({"x": "10"}, nullable=True) == {"x": 10}

    def test_list_of_pairs(self) -> None:
        assert self._str_int_map().convert_pyobj(
            [("a", 1), ("b", 2)], nullable=True
        ) == {"a": 1, "b": 2}

    def test_list_of_entry_dicts(self) -> None:
        assert self._str_int_map().convert_pyobj(
            [{"key": "a", "value": 1}], nullable=True
        ) == {"a": 1}

    def test_bad_json_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse map"):
            self._str_int_map().convert_pyobj("garbage", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# StructType
# ---------------------------------------------------------------------------

class TestStructType:
    def _struct(self) -> StructType:
        return StructType(
            fields=[
                IntegerType().to_field(name="a"),
                StringType().to_field(name="b"),
            ]
        )

    def test_json_str(self) -> None:
        assert self._struct().convert_pyobj(
            '{"a": 1, "b": "hi"}', nullable=True
        ) == {"a": 1, "b": "hi"}

    def test_json_bytes(self) -> None:
        assert self._struct().convert_pyobj(
            b'{"a": 1, "b": "hi"}', nullable=True
        ) == {"a": 1, "b": "hi"}

    def test_dict_coerces_children(self) -> None:
        assert self._struct().convert_pyobj(
            {"a": "42", "b": 99}, nullable=True
        ) == {"a": 42, "b": "99"}

    def test_list_positional(self) -> None:
        assert self._struct().convert_pyobj([1, "x"], nullable=True) == {
            "a": 1,
            "b": "x",
        }

    def test_missing_child_uses_default(self) -> None:
        # Missing "b" → None since the field is nullable by default.
        assert self._struct().convert_pyobj({"a": 1}, nullable=True) == {
            "a": 1,
            "b": None,
        }

    def test_object_with_dict(self) -> None:
        class Payload:
            def __init__(self) -> None:
                self.a = 1
                self.b = "x"

        assert self._struct().convert_pyobj(Payload(), nullable=True) == {
            "a": 1,
            "b": "x",
        }

    def test_bad_json_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse struct"):
            self._struct().convert_pyobj("garbage", nullable=True, safe=True)
