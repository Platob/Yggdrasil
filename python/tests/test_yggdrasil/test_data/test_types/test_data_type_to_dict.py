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


class TestDataTypeDictRoundTrip(unittest.TestCase):

    def test_integer_round_trip(self):
        original = IntegerType(byte_size=8, signed=True)
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, IntegerType)
        self.assertEqual(restored.byte_size, 8)
        self.assertTrue(restored.signed)
        self.assertEqual(d["id"], int(DataTypeId.INTEGER))

    def test_string_round_trip(self):
        original = StringType()
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, StringType)
        self.assertEqual(d["id"], int(DataTypeId.STRING))

    def test_boolean_round_trip(self):
        original = BooleanType()
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, BooleanType)

    def test_null_round_trip(self):
        original = NullType()
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, NullType)

    def test_binary_round_trip(self):
        original = BinaryType()
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, BinaryType)

    def test_decimal_round_trip(self):
        original = DecimalType(precision=10, scale=2)
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, DecimalType)
        self.assertEqual(restored.precision, 10)
        self.assertEqual(restored.scale, 2)

    def test_float_round_trip(self):
        original = FloatingPointType(byte_size=8)
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, FloatingPointType)
        self.assertEqual(restored.byte_size, 8)

    def test_date_round_trip(self):
        original = DateType()
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, DateType)

    def test_time_round_trip(self):
        original = TimeType()
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, TimeType)

    def test_timestamp_round_trip(self):
        original = TimestampType(tz="UTC", unit="us")
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, TimestampType)
        self.assertEqual(restored.tz, "UTC")
        self.assertEqual(restored.unit, "us")

    def test_timestamp_ntz_round_trip(self):
        original = TimestampType(tz=None, unit="us")
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, TimestampType)
        self.assertIsNone(restored.tz)

    def test_duration_round_trip(self):
        original = DurationType(unit="us")
        d = original.to_dict()
        restored = DataType.from_dict(d)

        self.assertIsInstance(restored, DurationType)
        self.assertEqual(restored.unit, "us")

    def test_all_primitive_round_trips(self):
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
                d = original.to_dict()
                restored = DataType.from_dict(d)
                self.assertEqual(type(restored), type(original))
