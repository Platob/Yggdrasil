"""Tests for :class:`IntegerCurator` and :class:`FloatCurator`."""

from __future__ import annotations

import math
import unittest

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types import (
    Float32Type,
    Float64Type,
    Int8Type,
    Int16Type,
    Int32Type,
    NullType,
    UInt8Type,
    UInt32Type,
)
from yggdrasil.io.curation import (
    Curator,
    FloatCurator,
    IntegerCurator,
)


class TestIntegerShrinking(ArrowTestCase):
    """Pick the narrowest int that holds the observed range."""

    def test_pick_returns_integer_curator(self):
        arr = self.pa.array([1, 2, 3], type=self.pa.int64())
        self.assertIsInstance(Curator.pick(arr), IntegerCurator)

    def test_small_positive_values_collapse_to_uint8(self):
        arr = self.pa.array([0, 1, 200], type=self.pa.int64())
        result = IntegerCurator().curate(arr)
        self.assertEqual(result.dtype, UInt8Type())
        self.assertEqual(result.array.type, self.pa.uint8())

    def test_negative_values_pick_signed_int8(self):
        arr = self.pa.array([-128, 0, 127], type=self.pa.int64())
        result = IntegerCurator().curate(arr)
        self.assertEqual(result.dtype, Int8Type())

    def test_one_outside_int8_widens_to_int16(self):
        arr = self.pa.array([-129, 0, 127], type=self.pa.int64())
        result = IntegerCurator().curate(arr)
        self.assertEqual(result.dtype, Int16Type())

    def test_int32_range(self):
        arr = self.pa.array([-(2**31), 2**31 - 1], type=self.pa.int64())
        result = IntegerCurator().curate(arr)
        self.assertEqual(result.dtype, Int32Type())

    def test_uint32_range(self):
        arr = self.pa.array([0, 2**32 - 1], type=self.pa.int64())
        result = IntegerCurator().curate(arr)
        self.assertEqual(result.dtype, UInt32Type())

    def test_disable_unsigned_falls_back_to_signed(self):
        arr = self.pa.array([0, 200], type=self.pa.int64())
        result = IntegerCurator(allow_unsigned=False).curate(arr)
        self.assertEqual(result.dtype, Int16Type())

    def test_all_null_collapses_to_null_type(self):
        arr = self.pa.array([None, None], type=self.pa.int64())
        result = IntegerCurator().curate(arr)
        self.assertEqual(result.dtype, NullType())

    def test_already_narrowest_returns_same_array(self):
        arr = self.pa.array([1, 2, 3], type=self.pa.uint8())
        result = IntegerCurator().curate(arr)
        # No change → original array handed back, not a fresh cast.
        self.assertIs(result.array, arr)

    def test_max_width_cap(self):
        # Capping at int16 means a [0..200] payload still fits in
        # uint8 (smaller than int16, so unsigned wins).
        arr = self.pa.array([0, 200], type=self.pa.int64())
        result = IntegerCurator(max_width="int16").curate(arr)
        self.assertEqual(result.dtype, UInt8Type())


class TestFloatShrinking(ArrowTestCase):
    """Float32 wins when the round-trip preserves every cell."""

    def test_pick_returns_float_curator(self):
        arr = self.pa.array([1.5, 2.5], type=self.pa.float64())
        self.assertIsInstance(Curator.pick(arr), FloatCurator)

    def test_exact_representable_values_become_float32(self):
        arr = self.pa.array([1.5, 2.5, -3.25], type=self.pa.float64())
        result = FloatCurator().curate(arr)
        self.assertEqual(result.dtype, Float32Type())

    def test_high_precision_values_stay_float64(self):
        # ``math.pi`` needs more mantissa bits than float32 carries.
        arr = self.pa.array([math.pi, math.e], type=self.pa.float64())
        result = FloatCurator().curate(arr)
        self.assertEqual(result.dtype, Float64Type())

    def test_nan_and_inf_round_trip_cleanly(self):
        arr = self.pa.array(
            [1.5, math.nan, math.inf, -math.inf, 2.5], type=self.pa.float64()
        )
        result = FloatCurator().curate(arr)
        self.assertEqual(result.dtype, Float32Type())

    def test_all_null_collapses_to_null_type(self):
        arr = self.pa.array([None, None], type=self.pa.float64())
        result = FloatCurator().curate(arr)
        self.assertEqual(result.dtype, NullType())


class TestTabularRoutesToShrinkers(ArrowTestCase):
    """``curate_arrow_tabular`` picks up IntegerCurator / FloatCurator
    automatically for pretyped numeric columns."""

    def test_int_and_float_columns_get_shrunk(self):
        table = self.pa.table(
            {
                "id": self.pa.array([1, 2, 3], type=self.pa.int64()),
                "score": self.pa.array([1.5, 2.5, 3.5], type=self.pa.float64()),
                "label": ["a", "b", "c"],
            }
        )
        _, curated = Curator.curate_arrow_tabular(table)
        self.assertEqual(curated.schema.field("id").type, self.pa.uint8())
        self.assertEqual(curated.schema.field("score").type, self.pa.float32())


if __name__ == "__main__":
    unittest.main()
