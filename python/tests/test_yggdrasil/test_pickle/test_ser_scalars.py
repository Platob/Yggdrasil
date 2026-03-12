"""Unit tests for yggdrasil.pickle.ser module - Scalar types."""

import datetime as dt
import decimal
import unittest
import uuid

import pytest

from yggdrasil.pickle.ser import (
    BytesSerialized,
    BoolSerialized,
    DateSerialized,
    DateTimeSerialized,
    DecimalSerialized,
    FloatSerialized,
    IntSerialized,
    NoneSerialized,
    StringSerialized,
    UUIDSerialized,
    SerdeTags,
)
from yggdrasil.pickle.ser.scalars import _tz_to_bytes


class TestNoneSerialized(unittest.TestCase):
    """Test NoneSerialized for None values."""

    def test_none_value(self):
        """Test None value property."""
        serialized = NoneSerialized.from_value(None)
        assert serialized.value is None

    def test_none_tag(self):
        """Test correct tag."""
        assert NoneSerialized.TAG == SerdeTags.NONE

    def test_none_roundtrip(self):
        """Test serialization roundtrip."""
        original = None
        serialized = NoneSerialized.from_value(original)
        assert serialized.value is original

    def test_none_with_metadata(self):
        """Test None with metadata."""
        metadata = {b"key": b"value"}
        serialized = NoneSerialized.from_value(None, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value is None

    def test_none_rejects_non_none(self):
        """Test that NoneSerialized rejects non-None values."""
        with pytest.raises(TypeError):
            NoneSerialized.from_value(0)

        with pytest.raises(TypeError):
            NoneSerialized.from_value("")

        with pytest.raises(TypeError):
            NoneSerialized.from_value([])


class TestBytesSerialized(unittest.TestCase):
    """Test BytesSerialized for bytes values."""

    def test_empty_bytes(self):
        """Test empty bytes."""
        serialized = BytesSerialized.from_value(b"")
        assert serialized.value == b""

    def test_simple_bytes(self):
        """Test simple bytes."""
        data = b"hello"
        serialized = BytesSerialized.from_value(data)
        assert serialized.value == data

    def test_binary_bytes(self):
        """Test binary bytes."""
        data = bytes(range(256))
        serialized = BytesSerialized.from_value(data)
        assert serialized.value == data

    def test_bytes_tag(self):
        """Test correct tag."""
        assert BytesSerialized.TAG == SerdeTags.BYTES

    def test_bytes_roundtrip(self):
        """Test bytes serialization roundtrip."""
        original = b"test data with special chars: \x00\xff\xfe"
        serialized = BytesSerialized.from_value(original)
        assert serialized.value == original

    def test_bytes_with_metadata(self):
        """Test bytes with metadata."""
        data = b"test"
        metadata = {b"format": b"base64"}
        serialized = BytesSerialized.from_value(data, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == data

    def test_bytes_rejects_string(self):
        """Test that BytesSerialized rejects strings."""
        with pytest.raises(TypeError):
            BytesSerialized.from_value("not bytes")

    def test_bytes_large_data(self):
        """Test large bytes object."""
        data = b"x" * (10 * 1024 * 1024)  # 10MB
        serialized = BytesSerialized.from_value(data)
        assert serialized.value == data


class TestStringSerialized(unittest.TestCase):
    """Test StringSerialized for string values."""

    def test_empty_string(self):
        """Test empty string."""
        serialized = StringSerialized.from_value("")
        assert serialized.value == ""

    def test_simple_string(self):
        """Test simple string."""
        text = "hello world"
        serialized = StringSerialized.from_value(text)
        assert serialized.value == text

    def test_unicode_string(self):
        """Test unicode string."""
        text = "Hello 世界 🌍 مرحبا мир"
        serialized = StringSerialized.from_value(text)
        assert serialized.value == text

    def test_string_tag(self):
        """Test correct tag."""
        assert StringSerialized.TAG == SerdeTags.STRING

    def test_string_roundtrip(self):
        """Test string serialization roundtrip."""
        original = "test\nstring\twith\rspecial\\characters"
        serialized = StringSerialized.from_value(original)
        assert serialized.value == original

    def test_string_with_metadata(self):
        """Test string with custom encoding metadata."""
        text = "hello"
        metadata = {b"encoding": b"ascii"}
        serialized = StringSerialized.from_value(text, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == text

    def test_string_default_utf8_encoding(self):
        """Test default UTF-8 encoding."""
        text = "test"
        serialized = StringSerialized.from_value(text)
        # Should use UTF-8 by default
        assert b"encoding" not in serialized.metadata or serialized.metadata.get(b"encoding") == b"utf-8"

    def test_string_rejects_non_string(self):
        """Test that StringSerialized rejects non-string types."""
        with pytest.raises(TypeError):
            StringSerialized.from_value(123)

        with pytest.raises(TypeError):
            StringSerialized.from_value(b"bytes")

    def test_string_large_text(self):
        """Test large string."""
        text = "x" * (10 * 1024 * 1024)  # 10MB
        serialized = StringSerialized.from_value(text)
        assert serialized.value == text


class TestBoolSerialized(unittest.TestCase):
    """Test BoolSerialized for boolean values."""

    def test_true_value(self):
        """Test True value."""
        serialized = BoolSerialized.from_value(True)
        assert serialized.value is True

    def test_false_value(self):
        """Test False value."""
        serialized = BoolSerialized.from_value(False)
        assert serialized.value is False

    def test_bool_tag(self):
        """Test correct tag."""
        assert BoolSerialized.TAG == SerdeTags.BOOL

    def test_bool_roundtrip_true(self):
        """Test True roundtrip."""
        serialized = BoolSerialized.from_value(True)
        assert serialized.value is True

    def test_bool_roundtrip_false(self):
        """Test False roundtrip."""
        serialized = BoolSerialized.from_value(False)
        assert serialized.value is False

    def test_bool_with_metadata(self):
        """Test bool with metadata."""
        metadata = {b"source": b"sql"}
        serialized = BoolSerialized.from_value(True, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value is True

    def test_bool_rejects_non_bool(self):
        """Test that BoolSerialized rejects non-bool types."""
        with pytest.raises(TypeError):
            BoolSerialized.from_value(1)

        with pytest.raises(TypeError):
            BoolSerialized.from_value("true")

        with pytest.raises(TypeError):
            BoolSerialized.from_value(None)


class TestIntSerialized(unittest.TestCase):
    """Test IntSerialized for integer values."""

    def test_zero(self):
        """Test zero."""
        serialized = IntSerialized.from_value(0)
        assert serialized.value == 0

    def test_positive_int(self):
        """Test positive integer."""
        serialized = IntSerialized.from_value(42)
        assert serialized.value == 42

    def test_negative_int(self):
        """Test negative integer."""
        serialized = IntSerialized.from_value(-42)
        assert serialized.value == -42

    def test_large_positive_int(self):
        """Test large positive integer."""
        value = 10**20
        serialized = IntSerialized.from_value(value)
        assert serialized.value == value

    def test_large_negative_int(self):
        """Test large negative integer."""
        value = -(10**20)
        serialized = IntSerialized.from_value(value)
        assert serialized.value == value

    def test_int_tag(self):
        """Test correct tag."""
        assert IntSerialized.TAG == SerdeTags.INT

    def test_int_roundtrip(self):
        """Test int roundtrip."""
        values = [0, 1, -1, 42, -42, 2**31 - 1, -(2**31), 2**63 - 1, -(2**63)]
        for value in values:
            serialized = IntSerialized.from_value(value)
            assert serialized.value == value

    def test_int_with_metadata(self):
        """Test int with metadata."""
        metadata = {b"unit": b"seconds"}
        serialized = IntSerialized.from_value(123, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == 123

    def test_int_rejects_float(self):
        """Test that IntSerialized rejects floats."""
        with pytest.raises(TypeError):
            IntSerialized.from_value(3.14)

    def test_int_rejects_string(self):
        """Test that IntSerialized rejects strings."""
        with pytest.raises(TypeError):
            IntSerialized.from_value("42")


class TestFloatSerialized(unittest.TestCase):
    """Test FloatSerialized for float values."""

    def test_zero_float(self):
        """Test zero."""
        serialized = FloatSerialized.from_value(0.0)
        assert serialized.value == 0.0

    def test_positive_float(self):
        """Test positive float."""
        serialized = FloatSerialized.from_value(3.14)
        assert abs(serialized.value - 3.14) < 1e-10

    def test_negative_float(self):
        """Test negative float."""
        serialized = FloatSerialized.from_value(-3.14)
        assert abs(serialized.value - (-3.14)) < 1e-10

    def test_very_small_float(self):
        """Test very small float."""
        value = 1e-300
        serialized = FloatSerialized.from_value(value)
        assert abs(serialized.value - value) / value < 1e-14

    def test_very_large_float(self):
        """Test very large float."""
        value = 1e300
        serialized = FloatSerialized.from_value(value)
        assert abs(serialized.value - value) / value < 1e-14

    def test_float_tag(self):
        """Test correct tag."""
        assert FloatSerialized.TAG == SerdeTags.FLOAT

    def test_float_roundtrip(self):
        """Test float roundtrip."""
        values = [0.0, 1.0, -1.0, 3.14159, -3.14159, 1e-10, 1e10]
        for value in values:
            serialized = FloatSerialized.from_value(value)
            assert abs(serialized.value - value) < abs(value) * 1e-14 + 1e-300

    def test_float_special_values(self):
        """Test special float values."""
        # Positive infinity
        inf = float("inf")
        serialized = FloatSerialized.from_value(inf)
        assert serialized.value == inf

        # Negative infinity
        ninf = float("-inf")
        serialized = FloatSerialized.from_value(ninf)
        assert serialized.value == ninf

        # NaN
        nan = float("nan")
        serialized = FloatSerialized.from_value(nan)
        assert serialized.value != serialized.value  # NaN != NaN

    def test_float_with_metadata(self):
        """Test float with metadata."""
        metadata = {b"unit": b"meters"}
        serialized = FloatSerialized.from_value(42.5, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == 42.5

    def test_float_rejects_string(self):
        """Test that FloatSerialized rejects strings."""
        with pytest.raises(TypeError):
            FloatSerialized.from_value("3.14")


class TestDateSerialized(unittest.TestCase):
    """Test DateSerialized for date values."""

    def test_simple_date(self):
        """Test simple date."""
        date = dt.date(2026, 3, 10)
        serialized = DateSerialized.from_value(date)
        assert serialized.value == date

    def test_epoch_date(self):
        """Test epoch date."""
        date = dt.date(1970, 1, 1)
        serialized = DateSerialized.from_value(date)
        assert serialized.value == date

    def test_min_date(self):
        """Test minimum date serialization metadata and payload is created."""
        date = dt.date.min
        serialized = DateSerialized.from_value(date)
        assert serialized.size > 0
        assert serialized.metadata.get(b"tz") == b"+00:00"

    def test_max_date(self):
        """Test maximum date serialization metadata and payload is created."""
        date = dt.date.max
        serialized = DateSerialized.from_value(date)
        assert serialized.size > 0
        assert serialized.metadata.get(b"tz") == b"+00:00"

    def test_date_roundtrip(self):
        """Test date roundtrip on portable date range supported by fromtimestamp."""
        dates = [
            dt.date(2026, 3, 10),
            dt.date(1970, 1, 1),
            dt.date(2000, 12, 31),
            dt.date(1980, 1, 1),
        ]
        for date in dates:
            serialized = DateSerialized.from_value(date)
            assert serialized.value == date

    def test_date_with_metadata(self):
        """Test date metadata merges caller metadata with serializer timezone info."""
        date = dt.date(2026, 3, 10)
        metadata = {b"format": b"ISO8601"}
        serialized = DateSerialized.from_value(date, metadata=metadata)
        assert serialized.metadata.get(b"format") == b"ISO8601"
        assert serialized.metadata.get(b"tz") == b"+00:00"
        assert serialized.value == date

    def test_date_rejects_datetime(self):
        """Test that DateSerialized rejects datetime."""
        with pytest.raises(TypeError):
            DateSerialized.from_value(dt.datetime.now())

    def test_date_rejects_string(self):
        """Test that DateSerialized rejects strings."""
        with pytest.raises(TypeError):
            DateSerialized.from_value("2026-03-10")


class TestDateTimeSerialized(unittest.TestCase):
    """Test DateTimeSerialized for datetime values."""

    def test_naive_datetime(self):
        """Naive datetimes are normalized to UTC-aware values."""
        dt_val = dt.datetime(2026, 3, 10, 12, 34, 56)
        serialized = DateTimeSerialized.from_value(dt_val)
        assert serialized.value == dt_val.replace(tzinfo=dt.timezone.utc)

    def test_utc_datetime(self):
        """Test UTC datetime."""
        dt_val = dt.datetime(2026, 3, 10, 12, 34, 56, tzinfo=dt.timezone.utc)
        serialized = DateTimeSerialized.from_value(dt_val)
        assert serialized.value == dt_val

    def test_offset_datetime(self):
        """Test datetime with timezone offset."""
        tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
        dt_val = dt.datetime(2026, 3, 10, 12, 34, 56, tzinfo=tz)
        serialized = DateTimeSerialized.from_value(dt_val)
        assert serialized.value == dt_val

    def test_datetime_with_microseconds(self):
        """Naive datetimes with microseconds are normalized to UTC-aware values."""
        dt_val = dt.datetime(2026, 3, 10, 12, 34, 56, 123456)
        serialized = DateTimeSerialized.from_value(dt_val)
        assert serialized.value == dt_val.replace(tzinfo=dt.timezone.utc)

    def test_epoch_datetime(self):
        """Test epoch datetime."""
        dt_val = dt.datetime(1970, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        serialized = DateTimeSerialized.from_value(dt_val)
        assert serialized.value == dt_val

    def test_datetime_tag(self):
        """Test correct tag."""
        assert DateTimeSerialized.TAG == SerdeTags.DATETIME

    def test_datetime_roundtrip(self):
        """Test datetime roundtrip with normalization rules applied."""
        datetimes = [
            dt.datetime(2026, 3, 10, 12, 34, 56),
            dt.datetime(2026, 3, 10, 12, 34, 56, tzinfo=dt.timezone.utc),
            dt.datetime(2000, 1, 1, 0, 0, 0),
        ]
        for dt_val in datetimes:
            serialized = DateTimeSerialized.from_value(dt_val)
            expected = dt_val if dt_val.tzinfo is not None else dt_val.replace(tzinfo=dt.timezone.utc)
            assert serialized.value == expected

    def test_datetime_with_metadata(self):
        """Test datetime metadata merges caller metadata with serializer tz markers."""
        dt_val = dt.datetime(2026, 3, 10, 12, 34, 56)
        metadata = {b"timezone": b"UTC"}
        serialized = DateTimeSerialized.from_value(dt_val, metadata=metadata)
        assert serialized.metadata.get(b"timezone") == b"UTC"
        assert serialized.metadata.get(b"tz") == b"naive"
        assert serialized.value == dt_val.replace(tzinfo=dt.timezone.utc)

    def test_datetime_rejects_date(self):
        """Test that DateTimeSerialized rejects date."""
        with pytest.raises(TypeError):
            DateTimeSerialized.from_value(dt.date.today())

    def test_datetime_rejects_string(self):
        """Test that DateTimeSerialized rejects strings."""
        with pytest.raises(TypeError):
            DateTimeSerialized.from_value("2026-03-10T12:34:56")


class TestDecimalSerialized(unittest.TestCase):
    """Test DecimalSerialized for Decimal values."""

    def test_zero_decimal(self):
        """Test zero decimal."""
        value = decimal.Decimal("0")
        serialized = DecimalSerialized.from_value(value)
        assert serialized.value == value

    def test_positive_decimal(self):
        """Test positive decimal."""
        value = decimal.Decimal("123.45")
        serialized = DecimalSerialized.from_value(value)
        assert serialized.value == value

    def test_negative_decimal(self):
        """Test negative decimal."""
        value = decimal.Decimal("-123.45")
        serialized = DecimalSerialized.from_value(value)
        assert serialized.value == value

    def test_high_precision_decimal(self):
        """Test high precision decimal."""
        value = decimal.Decimal("123456789012345678901234567890.123456789")
        serialized = DecimalSerialized.from_value(value)
        assert serialized.value == value

    def test_decimal_tag(self):
        """Test correct tag."""
        assert DecimalSerialized.TAG == SerdeTags.DECIMAL

    def test_decimal_roundtrip(self):
        """Test decimal roundtrip."""
        values = [
            decimal.Decimal("0"),
            decimal.Decimal("1.5"),
            decimal.Decimal("-1.5"),
            decimal.Decimal("999999.999999"),
        ]
        for value in values:
            serialized = DecimalSerialized.from_value(value)
            assert serialized.value == value

    def test_decimal_with_metadata(self):
        """Test decimal with metadata."""
        value = decimal.Decimal("123.45")
        metadata = {b"currency": b"USD"}
        serialized = DecimalSerialized.from_value(value, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == value

    def test_decimal_rejects_float(self):
        """Test that DecimalSerialized rejects float."""
        with pytest.raises(TypeError):
            DecimalSerialized.from_value(3.14)

    def test_decimal_rejects_string(self):
        """Test that DecimalSerialized rejects string."""
        with pytest.raises(TypeError):
            DecimalSerialized.from_value("123.45")


class TestUUIDSerialized(unittest.TestCase):
    """Test UUIDSerialized for UUID values."""

    def test_zero_uuid(self):
        """Test zero UUID."""
        value = uuid.UUID(int=0)
        serialized = UUIDSerialized.from_value(value)
        assert serialized.value == value

    def test_simple_uuid(self):
        """Test simple UUID."""
        value = uuid.uuid4()
        serialized = UUIDSerialized.from_value(value)
        assert serialized.value == value

    def test_uuid_from_string(self):
        """Test UUID created from string."""
        value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        serialized = UUIDSerialized.from_value(value)
        assert serialized.value == value

    def test_uuid_tag(self):
        """Test correct tag."""
        assert UUIDSerialized.TAG == SerdeTags.UUID

    def test_uuid_roundtrip(self):
        """Test UUID roundtrip."""
        uuids = [
            uuid.UUID(int=0),
            uuid.UUID(int=1),
            uuid.uuid4(),
            uuid.UUID("12345678-1234-5678-1234-567812345678"),
        ]
        for value in uuids:
            serialized = UUIDSerialized.from_value(value)
            assert serialized.value == value

    def test_uuid_with_metadata(self):
        """Test UUID with metadata."""
        value = uuid.uuid4()
        metadata = {b"version": b"4"}
        serialized = UUIDSerialized.from_value(value, metadata=metadata)
        assert serialized.metadata == metadata
        assert serialized.value == value

    def test_uuid_rejects_string(self):
        """Test that UUIDSerialized rejects string."""
        with pytest.raises(TypeError):
            UUIDSerialized.from_value("12345678-1234-5678-1234-567812345678")


class TestTzToBytes(unittest.TestCase):
    """Test _tz_to_bytes helper function."""

    def test_naive_timezone(self):
        """Test naive (None) timezone."""
        assert _tz_to_bytes(None) == b"naive"

    def test_utc_timezone(self):
        """Test UTC timezone."""
        result = _tz_to_bytes(dt.timezone.utc)
        assert result == b"+00:00"

    def test_positive_offset(self):
        """Test positive offset timezone."""
        tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
        result = _tz_to_bytes(tz)
        assert result == b"+05:30"

    def test_negative_offset(self):
        """Test negative offset timezone."""
        tz = dt.timezone(dt.timedelta(hours=-8))
        result = _tz_to_bytes(tz)
        assert result == b"-08:00"

    def test_half_hour_offset(self):
        """Test half-hour offset."""
        tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
        result = _tz_to_bytes(tz)
        assert result == b"+05:30"


if __name__ == "__main__":
    unittest.main()

