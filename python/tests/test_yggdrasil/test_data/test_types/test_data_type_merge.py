from __future__ import annotations

import unittest

import pyarrow as pa

from yggdrasil.data.types.primitive import (
    DecimalType,
    FloatingPointType,
    IntegerType,
    NullType,
    StringType,
)
from yggdrasil.io import SaveMode


class TestPrimitiveTypeMerge(unittest.TestCase):

    def test_int_same_size_no_cast_returns_new_with_same_size(self):
        a = IntegerType(byte_size=8, signed=True)
        b = IntegerType(byte_size=8, signed=True)

        result = a.merge_with(b)

        self.assertIsInstance(result, IntegerType)
        self.assertEqual(result.byte_size, 8)
        self.assertTrue(result.signed)

    def test_int_merge_upcast(self):
        small = IntegerType(byte_size=4, signed=True)
        large = IntegerType(byte_size=8, signed=True)

        result = small.merge_with(large, upcast=True)

        self.assertIsInstance(result, IntegerType)
        self.assertEqual(result.byte_size, 8)

    def test_int_merge_downcast(self):
        small = IntegerType(byte_size=4, signed=True)
        large = IntegerType(byte_size=8, signed=True)

        result = large.merge_with(small, downcast=True)

        self.assertIsInstance(result, IntegerType)
        self.assertEqual(result.byte_size, 4)

    def test_int_merge_no_cast_uses_max(self):
        a = IntegerType(byte_size=4, signed=True)
        b = IntegerType(byte_size=8, signed=True)

        result = a.merge_with(b)

        self.assertIsInstance(result, IntegerType)
        self.assertEqual(result.byte_size, 8)

    def test_int_merge_upcast_and_downcast_raises(self):
        a = IntegerType(byte_size=4, signed=True)
        b = IntegerType(byte_size=8, signed=True)

        with self.assertRaises(pa.ArrowInvalid):
            a.merge_with(b, upcast=True, downcast=True)

    def test_float_merge_upcast(self):
        small = FloatingPointType(byte_size=4)
        large = FloatingPointType(byte_size=8)

        result = small.merge_with(large, upcast=True)

        self.assertIsInstance(result, FloatingPointType)
        self.assertEqual(result.byte_size, 8)

    def test_float_merge_downcast(self):
        small = FloatingPointType(byte_size=4)
        large = FloatingPointType(byte_size=8)

        result = large.merge_with(small, downcast=True)

        self.assertIsInstance(result, FloatingPointType)
        self.assertEqual(result.byte_size, 4)

    def test_float_merge_same_size_returns_self(self):
        a = FloatingPointType(byte_size=8)
        b = FloatingPointType(byte_size=8)

        result = a.merge_with(b)

        self.assertIs(result, a)

    def test_null_merge_with_int(self):
        null_type = NullType()
        int_type = IntegerType(byte_size=8, signed=True)

        result = null_type.merge_with(int_type)

        self.assertIsInstance(result, IntegerType)

    def test_int_merge_with_null_returns_self(self):
        int_type = IntegerType(byte_size=8, signed=True)
        null_type = NullType()

        result = int_type.merge_with(null_type)

        self.assertIs(result, int_type)

    def test_string_merge_with_string_returns_self(self):
        a = StringType()
        b = StringType()

        result = a.merge_with(b)

        self.assertIs(result, a)

    def test_decimal_merge_preserves_precision_scale(self):
        a = DecimalType(precision=10, scale=2)
        b = DecimalType(precision=18, scale=4)

        result = a.merge_with(b)

        self.assertIsInstance(result, DecimalType)

    def test_overwrite_mode_returns_other(self):
        a = IntegerType(byte_size=4, signed=True)
        b = IntegerType(byte_size=8, signed=True)

        result = a.merge_with(b, mode=SaveMode.OVERWRITE)

        self.assertEqual(result.byte_size, 8)

    def test_merge_different_primitive_types_returns_self(self):
        int_type = IntegerType(byte_size=8, signed=True)
        str_type = StringType()

        result = int_type.merge_with(str_type)

        self.assertIs(result, int_type)

    def test_int_merge_signed_with_unsigned_produces_signed(self):
        signed = IntegerType(byte_size=4, signed=True)
        unsigned = IntegerType(byte_size=4, signed=False)

        result = signed.merge_with(unsigned)

        self.assertTrue(result.signed)

    def test_int_merge_downcast_signed_requires_both_signed(self):
        signed = IntegerType(byte_size=4, signed=True)
        unsigned = IntegerType(byte_size=4, signed=False)

        result = signed.merge_with(unsigned, downcast=True)

        self.assertFalse(result.signed)
