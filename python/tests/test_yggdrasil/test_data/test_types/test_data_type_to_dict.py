"""``DataType.to_dict`` ↔ :meth:`DataType.from_dict` round-trips.

Each primitive type renders to a dict that ``DataType.from_dict``
should resolve back to the same concrete subclass with the same
parameters. Carrying parameters through the dict is what lets schemas
travel through JSON / Databricks tables / cache files without the
yggdrasil objects being importable on the consumer side.
"""
from __future__ import annotations

import unittest

from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.primitive import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DurationType,
    FloatingPointType,
    IntegerType,
    NullType,
    StringType,
    TimeType,
    TimestampType,
)


class TestPrimitiveDictRoundTrip(unittest.TestCase):

    def test_integer_carries_byte_size_and_signed(self) -> None:
        original = IntegerType(byte_size=8, signed=True)
        d = original.to_dict()
        restored = DataType.from_dict(d)

        # ``IntegerType(byte_size=8, signed=True)`` redirects to the
        # specialized ``Int64Type`` via ``__new__``; the dict id reflects
        # the concrete subclass rather than the abstract ``INTEGER``.
        self.assertEqual(d["id"], int(DataTypeId.INT64))
        self.assertIsInstance(restored, IntegerType)
        self.assertEqual(restored.byte_size, 8)
        self.assertTrue(restored.signed)

    def test_string_round_trip(self) -> None:
        d = StringType().to_dict()
        self.assertEqual(d["id"], int(DataTypeId.STRING))
        self.assertIsInstance(DataType.from_dict(d), StringType)

    def test_boolean_round_trip(self) -> None:
        self.assertIsInstance(DataType.from_dict(BooleanType().to_dict()), BooleanType)

    def test_null_round_trip(self) -> None:
        self.assertIsInstance(DataType.from_dict(NullType().to_dict()), NullType)

    def test_binary_round_trip(self) -> None:
        self.assertIsInstance(DataType.from_dict(BinaryType().to_dict()), BinaryType)

    def test_decimal_carries_precision_and_scale(self) -> None:
        original = DecimalType(precision=10, scale=2)
        restored = DataType.from_dict(original.to_dict())

        self.assertIsInstance(restored, DecimalType)
        self.assertEqual(restored.precision, 10)
        self.assertEqual(restored.scale, 2)

    def test_float_carries_byte_size(self) -> None:
        restored = DataType.from_dict(FloatingPointType(byte_size=8).to_dict())

        self.assertIsInstance(restored, FloatingPointType)
        self.assertEqual(restored.byte_size, 8)

    def test_date_round_trip(self) -> None:
        self.assertIsInstance(DataType.from_dict(DateType().to_dict()), DateType)

    def test_time_round_trip(self) -> None:
        self.assertIsInstance(DataType.from_dict(TimeType().to_dict()), TimeType)

    def test_timestamp_carries_unit_and_tz(self) -> None:
        original = TimestampType(tz="UTC", unit="us")
        restored = DataType.from_dict(original.to_dict())

        self.assertIsInstance(restored, TimestampType)
        self.assertEqual(restored.tz, "UTC")
        self.assertEqual(restored.unit, "us")

    def test_timestamp_naive_preserves_none_tz(self) -> None:
        restored = DataType.from_dict(TimestampType(tz=None, unit="us").to_dict())

        self.assertIsInstance(restored, TimestampType)
        self.assertIsNone(restored.tz)

    def test_duration_carries_unit(self) -> None:
        restored = DataType.from_dict(DurationType(unit="us").to_dict())

        self.assertIsInstance(restored, DurationType)
        self.assertEqual(restored.unit, "us")


class TestRoundTripMatrix(unittest.TestCase):
    """One subTest per primitive — exact subclass survives the round-trip."""

    def test_all_primitive_round_trips(self) -> None:
        primitives = [
            NullType(),
            BinaryType(),
            StringType(),
            BooleanType(),
            IntegerType(byte_size=1, signed=True),
            IntegerType(byte_size=2, signed=True),
            IntegerType(byte_size=4, signed=True),
            IntegerType(byte_size=8, signed=True),
            IntegerType(byte_size=1, signed=False),
            IntegerType(byte_size=2, signed=False),
            IntegerType(byte_size=4, signed=False),
            IntegerType(byte_size=8, signed=False),
            FloatingPointType(byte_size=4),
            FloatingPointType(byte_size=8),
            DecimalType(precision=38, scale=18),
            DateType(),
            TimeType(),
            TimestampType(tz="UTC"),
            TimestampType(tz=None),
            DurationType(),
        ]
        for original in primitives:
            with self.subTest(dtype=str(original)):
                restored = DataType.from_dict(original.to_dict())
                self.assertEqual(type(restored), type(original))
