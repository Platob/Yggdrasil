"""``DataType.convert_pyobj`` — single-value coercion per concrete type.

Two surfaces under test for every type:

* The ``str`` / ``bytes`` priority paths (CSV / JSON / Excel ingest
  hits these constantly).
* The native Python form (``int`` for IntegerType, ``dt.date`` for
  DateType, etc.).

``safe=True`` turns parse failures into ``ValueError``; ``safe=False``
(the default) returns ``None`` on unparseable input.
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
# NullType — always None
# ---------------------------------------------------------------------------


class TestNullType:

    @pytest.mark.parametrize("value", [None, "anything", 42])
    def test_returns_none_for_anything(self, value) -> None:
        assert NullType().convert_pyobj(value, nullable=True) is None


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
    def test_known_inputs(self, value, expected) -> None:
        assert BooleanType().convert_pyobj(value, nullable=True) is expected

    def test_whitespace_string_is_stripped(self) -> None:
        assert BooleanType().convert_pyobj("  true  ", nullable=True) is True

    def test_garbage_unsafe_returns_none(self) -> None:
        assert BooleanType().convert_pyobj("garbage", nullable=True) is None

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse bool"):
            BooleanType().convert_pyobj("garbage", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# IntegerType
# ---------------------------------------------------------------------------


class TestIntegerType:

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("42", 42),
            ("0x1a", 26),
            ("0b1010", 10),
            ("3.9", 3),
            (b"42", 42),
            (True, 1),
            (False, 0),
            (3.9, 3),
            (-3.9, -3),
            (Decimal("5.5"), 5),
        ],
    )
    def test_recognized_inputs(self, value, expected: int) -> None:
        assert IntegerType().convert_pyobj(value, nullable=True) == expected

    def test_nan_unsafe_returns_none(self) -> None:
        assert IntegerType().convert_pyobj(float("nan"), nullable=True) is None

    def test_nan_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            IntegerType().convert_pyobj(
                float("nan"), nullable=True, safe=True
            )

    def test_empty_string_unsafe_returns_none(self) -> None:
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

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("3.14", 3.14),
            ("1e3", 1000.0),
            (b"2.5", 2.5),
            (5, 5.0),
            (Decimal("1.5"), 1.5),
            ("inf", float("inf")),
        ],
    )
    def test_recognized_inputs(self, value, expected: float) -> None:
        assert FloatingPointType().convert_pyobj(value, nullable=True) == expected

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse float"):
            FloatingPointType().convert_pyobj(
                "nope", nullable=True, safe=True
            )

    def test_garbage_unsafe_returns_none(self) -> None:
        assert FloatingPointType().convert_pyobj("nope", nullable=True) is None


# ---------------------------------------------------------------------------
# DecimalType
# ---------------------------------------------------------------------------


class TestDecimalType:

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("3.14", Decimal("3.14")),
            (b"1.5", Decimal("1.5")),
            (5, Decimal(5)),
            # float → Decimal goes through ``str(float)`` so the visible repr
            # is preserved, not the binary float approximation.
            (0.1, Decimal("0.1")),
        ],
    )
    def test_recognized_inputs(self, value, expected: Decimal) -> None:
        assert DecimalType().convert_pyobj(value, nullable=True) == expected

    def test_nan_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            DecimalType().convert_pyobj(
                float("nan"), nullable=True, safe=True
            )

    def test_garbage_unsafe_returns_none(self) -> None:
        assert DecimalType().convert_pyobj("nope", nullable=True) is None


# ---------------------------------------------------------------------------
# StringType
# ---------------------------------------------------------------------------


class TestStringType:

    def test_str_passthrough(self) -> None:
        assert StringType().convert_pyobj("hello", nullable=True) == "hello"

    def test_bytes_decoded_as_utf8(self) -> None:
        assert StringType().convert_pyobj(b"hello", nullable=True) == "hello"

    def test_int_stringified(self) -> None:
        assert StringType().convert_pyobj(42, nullable=True) == "42"

    @pytest.mark.parametrize(
        "value,expected", [(True, "true"), (False, "false")]
    )
    def test_bool_lowercased(self, value: bool, expected: str) -> None:
        assert StringType().convert_pyobj(value, nullable=True) == expected

    def test_float_stringified(self) -> None:
        assert StringType().convert_pyobj(3.14, nullable=True) == "3.14"

    def test_date_isoformatted(self) -> None:
        assert StringType().convert_pyobj(
            dt.date(2024, 1, 1), nullable=True
        ) == "2024-01-01"

    def test_bad_utf8_unsafe_returns_none(self) -> None:
        assert StringType().convert_pyobj(b"\xff\xfe", nullable=True) is None

    def test_bad_utf8_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="decode bytes"):
            StringType().convert_pyobj(
                b"\xff\xfe", nullable=True, safe=True
            )


# ---------------------------------------------------------------------------
# BinaryType
# ---------------------------------------------------------------------------


class TestBinaryType:

    def test_bytes_passthrough(self) -> None:
        assert BinaryType().convert_pyobj(b"abc", nullable=True) == b"abc"

    def test_str_encoded_utf8(self) -> None:
        assert BinaryType().convert_pyobj("abc", nullable=True) == b"abc"

    def test_bytearray_normalised_to_bytes(self) -> None:
        assert (
            BinaryType().convert_pyobj(bytearray(b"xy"), nullable=True) == b"xy"
        )

    def test_memoryview_normalised_to_bytes(self) -> None:
        assert (
            BinaryType().convert_pyobj(memoryview(b"mv"), nullable=True) == b"mv"
        )

    def test_fixed_size_pads_with_null_bytes(self) -> None:
        assert BinaryType(byte_size=5).convert_pyobj(
            b"ab", nullable=True
        ) == b"ab\x00\x00\x00"

    def test_fixed_size_truncates_unsafe(self) -> None:
        assert (
            BinaryType(byte_size=2).convert_pyobj(b"abcd", nullable=True) == b"ab"
        )

    def test_fixed_size_truncates_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds fixed"):
            BinaryType(byte_size=2).convert_pyobj(
                b"abcd", nullable=True, safe=True
            )


# ---------------------------------------------------------------------------
# DateType
# ---------------------------------------------------------------------------


class TestDateType:

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("2024-01-15", dt.date(2024, 1, 15)),
            (b"2024-01-15", dt.date(2024, 1, 15)),
            ("2024-01-15T10:30:45", dt.date(2024, 1, 15)),
            (dt.datetime(2024, 1, 15, 10, 30), dt.date(2024, 1, 15)),
            (dt.date(2024, 1, 15), dt.date(2024, 1, 15)),
            (0, dt.date(1970, 1, 1)),
            (1, dt.date(1970, 1, 2)),
        ],
    )
    def test_recognized(self, value, expected: dt.date) -> None:
        assert DateType().convert_pyobj(value, nullable=True) == expected

    def test_garbage_unsafe(self) -> None:
        assert DateType().convert_pyobj("not a date", nullable=True) is None

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse date"):
            DateType().convert_pyobj("not a date", nullable=True, safe=True)


# ---------------------------------------------------------------------------
# TimeType
# ---------------------------------------------------------------------------


class TestTimeType:

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("10:30:45", dt.time(10, 30, 45)),
            (b"10:30:45", dt.time(10, 30, 45)),
            (dt.datetime(2024, 1, 1, 10, 30, 45), dt.time(10, 30, 45)),
        ],
    )
    def test_recognized(self, value, expected: dt.time) -> None:
        assert TimeType().convert_pyobj(value, nullable=True) == expected

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

    def test_iso_bytes_with_z_attaches_utc(self) -> None:
        out = TimestampType(tz="UTC").convert_pyobj(
            b"2024-01-15T10:30:45Z", nullable=True
        )
        assert out == dt.datetime(2024, 1, 15, 10, 30, 45, tzinfo=dt.timezone.utc)

    def test_space_separator(self) -> None:
        assert TimestampType().convert_pyobj(
            "2024-01-15 10:30:45", nullable=True
        ) == dt.datetime(2024, 1, 15, 10, 30, 45)

    def test_date_only_string_widens_to_midnight(self) -> None:
        assert TimestampType().convert_pyobj(
            "2024-01-15", nullable=True
        ) == dt.datetime(2024, 1, 15)

    def test_naive_target_strips_offset_via_utc(self) -> None:
        out = TimestampType(tz=None).convert_pyobj(
            "2024-01-15T10:30:45+05:00", nullable=True
        )

        # 10:30:45 at +05:00 → 05:30:45 UTC, naive.
        assert out == dt.datetime(2024, 1, 15, 5, 30, 45)

    def test_epoch_seconds(self) -> None:
        out = TimestampType(unit="s", tz="UTC").convert_pyobj(0, nullable=True)
        assert out == dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

    def test_date_input_widens_to_midnight(self) -> None:
        assert TimestampType().convert_pyobj(
            dt.date(2024, 1, 15), nullable=True
        ) == dt.datetime(2024, 1, 15)

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            TimestampType().convert_pyobj(
                "not a timestamp", nullable=True, safe=True
            )


# ---------------------------------------------------------------------------
# DurationType
# ---------------------------------------------------------------------------


class TestDurationType:

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("PT1H30M", dt.timedelta(hours=1, minutes=30)),
            ("PT45.5S", dt.timedelta(seconds=45, microseconds=500_000)),
            ("-PT1H", dt.timedelta(hours=-1)),
            ("01:30:00", dt.timedelta(hours=1, minutes=30)),
            (b"01:30:00", dt.timedelta(hours=1, minutes=30)),
            ("-01:30:00", dt.timedelta(hours=-1, minutes=-30)),
        ],
    )
    def test_string_inputs(self, value, expected: dt.timedelta) -> None:
        assert DurationType().convert_pyobj(value, nullable=True) == expected

    def test_numeric_str_in_default_unit(self) -> None:
        # Default unit is microseconds.
        assert DurationType(unit="us").convert_pyobj(
            "1000000", nullable=True
        ) == dt.timedelta(seconds=1)

    def test_int_in_explicit_unit(self) -> None:
        assert DurationType(unit="s").convert_pyobj(
            60, nullable=True
        ) == dt.timedelta(seconds=60)

    def test_timedelta_passthrough_identity(self) -> None:
        td = dt.timedelta(seconds=42)
        assert DurationType().convert_pyobj(td, nullable=True) is td

    def test_garbage_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse duration"):
            DurationType().convert_pyobj(
                "not a duration", nullable=True, safe=True
            )


# ---------------------------------------------------------------------------
# ArrayType
# ---------------------------------------------------------------------------


class TestArrayType:

    @staticmethod
    def _int_array() -> ArrayType:
        return ArrayType.from_item(IntegerType().to_field(name="item"))

    def test_json_string_input(self) -> None:
        assert self._int_array().convert_pyobj(
            "[1, 2, 3]", nullable=True
        ) == [1, 2, 3]

    def test_json_bytes_input(self) -> None:
        assert self._int_array().convert_pyobj(
            b"[1, 2, 3]", nullable=True
        ) == [1, 2, 3]

    def test_list_input_coerces_each_item(self) -> None:
        assert self._int_array().convert_pyobj(
            [1, "2", 3.5], nullable=True
        ) == [1, 2, 3]

    def test_tuple_input(self) -> None:
        assert self._int_array().convert_pyobj((1, 2, 3), nullable=True) == [
            1, 2, 3,
        ]

    def test_set_input(self) -> None:
        result = self._int_array().convert_pyobj({1, 2, 3}, nullable=True)

        assert sorted(result) == [1, 2, 3]

    def test_bad_json_unsafe_returns_none(self) -> None:
        assert self._int_array().convert_pyobj("not json", nullable=True) is None

    def test_bad_json_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse array"):
            self._int_array().convert_pyobj(
                "not json", nullable=True, safe=True
            )

    def test_scalar_input_unsafe_returns_none(self) -> None:
        assert self._int_array().convert_pyobj(42, nullable=True) is None


# ---------------------------------------------------------------------------
# MapType
# ---------------------------------------------------------------------------


class TestMapType:

    @staticmethod
    def _str_int_map() -> MapType:
        return MapType.from_key_value(
            key_field=StringType().to_field(),
            value_field=IntegerType().to_field(),
        )

    def test_json_string_input(self) -> None:
        assert self._str_int_map().convert_pyobj(
            '{"a": 1, "b": 2}', nullable=True
        ) == {"a": 1, "b": 2}

    def test_json_bytes_input(self) -> None:
        assert self._str_int_map().convert_pyobj(
            b'{"a": 1}', nullable=True
        ) == {"a": 1}

    def test_dict_input_coerces_values(self) -> None:
        assert self._str_int_map().convert_pyobj(
            {"x": "10"}, nullable=True
        ) == {"x": 10}

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
            self._str_int_map().convert_pyobj(
                "garbage", nullable=True, safe=True
            )


# ---------------------------------------------------------------------------
# StructType
# ---------------------------------------------------------------------------


class TestStructType:

    @staticmethod
    def _ab_struct() -> StructType:
        return StructType(
            fields=[
                IntegerType().to_field(name="a"),
                StringType().to_field(name="b"),
            ]
        )

    def test_json_string_input(self) -> None:
        assert self._ab_struct().convert_pyobj(
            '{"a": 1, "b": "hi"}', nullable=True
        ) == {"a": 1, "b": "hi"}

    def test_json_bytes_input(self) -> None:
        assert self._ab_struct().convert_pyobj(
            b'{"a": 1, "b": "hi"}', nullable=True
        ) == {"a": 1, "b": "hi"}

    def test_dict_coerces_each_child(self) -> None:
        assert self._ab_struct().convert_pyobj(
            {"a": "42", "b": 99}, nullable=True
        ) == {"a": 42, "b": "99"}

    def test_list_input_treated_as_positional(self) -> None:
        assert self._ab_struct().convert_pyobj([1, "x"], nullable=True) == {
            "a": 1,
            "b": "x",
        }

    def test_missing_child_yields_none_when_nullable(self) -> None:
        assert self._ab_struct().convert_pyobj({"a": 1}, nullable=True) == {
            "a": 1,
            "b": None,
        }

    def test_object_with_dict_attribute(self) -> None:
        class _Payload:
            def __init__(self) -> None:
                self.a = 1
                self.b = "x"

        assert self._ab_struct().convert_pyobj(_Payload(), nullable=True) == {
            "a": 1,
            "b": "x",
        }

    def test_bad_json_safe_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse struct"):
            self._ab_struct().convert_pyobj(
                "garbage", nullable=True, safe=True
            )
