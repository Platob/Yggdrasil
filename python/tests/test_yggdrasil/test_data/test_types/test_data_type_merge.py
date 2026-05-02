"""``DataType.merge_with`` semantics across primitive types.

The merge contract — exercised here — is the schema-reconciliation
backbone the rest of the library leans on:

* Same ``type_id`` → engine-specific narrowing / widening
  (``upcast``, ``downcast``, default no-op).
* Different ``type_id`` → ``NULL`` is a wildcard; otherwise ``self``
  wins unless ``upcast`` / ``downcast`` swings the choice.
* :attr:`Mode.OVERWRITE` flips the precedence so ``other`` wins.

These are the rules every cross-engine schema bridge relies on; if
they regress, anything that touches mixed sources breaks.
"""
from __future__ import annotations

import unittest

from yggdrasil.data.types.primitive import (
    DecimalType,
    FloatingPointType,
    IntegerType,
    NullType,
    StringType,
)
from yggdrasil.io.enums import Mode


class TestIntegerMerge(unittest.TestCase):

    def test_same_size_is_passthrough(self) -> None:
        a = IntegerType(byte_size=8, signed=True)
        b = IntegerType(byte_size=8, signed=True)

        result = a.merge_with(b)

        self.assertIsInstance(result, IntegerType)
        self.assertEqual(result.byte_size, 8)
        self.assertTrue(result.signed)

    def test_upcast_widens_to_larger(self) -> None:
        small = IntegerType(byte_size=4, signed=True)
        large = IntegerType(byte_size=8, signed=True)

        self.assertEqual(small.merge_with(large, upcast=True).byte_size, 8)

    def test_downcast_narrows_to_smaller(self) -> None:
        small = IntegerType(byte_size=4, signed=True)
        large = IntegerType(byte_size=8, signed=True)

        self.assertEqual(large.merge_with(small, downcast=True).byte_size, 4)

    def test_no_cast_default_keeps_self_size(self) -> None:
        a = IntegerType(byte_size=4, signed=True)
        b = IntegerType(byte_size=8, signed=True)

        self.assertEqual(a.merge_with(b).byte_size, 4)

    def test_signed_meets_unsigned_keeps_signed(self) -> None:
        signed = IntegerType(byte_size=4, signed=True)
        unsigned = IntegerType(byte_size=4, signed=False)

        self.assertTrue(signed.merge_with(unsigned).signed)

    def test_signed_meets_unsigned_under_downcast_keeps_signed(self) -> None:
        signed = IntegerType(byte_size=4, signed=True)
        unsigned = IntegerType(byte_size=4, signed=False)

        self.assertTrue(signed.merge_with(unsigned, downcast=True).signed)

    def test_overwrite_mode_returns_other(self) -> None:
        a = IntegerType(byte_size=4, signed=True)
        b = IntegerType(byte_size=8, signed=True)

        self.assertEqual(a.merge_with(b, mode=Mode.OVERWRITE).byte_size, 8)


class TestFloatMerge(unittest.TestCase):

    def test_upcast_widens(self) -> None:
        small = FloatingPointType(byte_size=4)
        large = FloatingPointType(byte_size=8)

        self.assertEqual(small.merge_with(large, upcast=True).byte_size, 8)

    def test_downcast_narrows(self) -> None:
        small = FloatingPointType(byte_size=4)
        large = FloatingPointType(byte_size=8)

        self.assertEqual(large.merge_with(small, downcast=True).byte_size, 4)

    def test_same_size_returns_self(self) -> None:
        a = FloatingPointType(byte_size=8)
        b = FloatingPointType(byte_size=8)

        self.assertIs(a.merge_with(b), a)


class TestDecimalMerge(unittest.TestCase):

    def test_default_keeps_self(self) -> None:
        a = DecimalType(precision=10, scale=2)
        b = DecimalType(precision=18, scale=4)

        result = a.merge_with(b)

        self.assertIsInstance(result, DecimalType)


class TestNullMerge(unittest.TestCase):

    def test_null_left_yields_other(self) -> None:
        result = NullType().merge_with(IntegerType(byte_size=8, signed=True))
        self.assertIsInstance(result, IntegerType)

    def test_null_right_yields_self(self) -> None:
        int_type = IntegerType(byte_size=8, signed=True)
        self.assertIs(int_type.merge_with(NullType()), int_type)


class TestCrossTypeMerge(unittest.TestCase):

    def test_same_type_string_returns_self(self) -> None:
        a = StringType()
        b = StringType()

        self.assertIs(a.merge_with(b), a)

    def test_int_and_string_keeps_self(self) -> None:
        int_type = IntegerType(byte_size=8, signed=True)
        str_type = StringType()

        self.assertIs(int_type.merge_with(str_type), int_type)
