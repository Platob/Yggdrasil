"""Arrow fast-cast null-count behavior, by source/target DataType pair.

The Arrow cast pipeline has type-specific fast paths sprinkled around:

* ``IntegerType._cast_arrow_array`` — signedness flip uses ``safe=False`` then
  re-fills nulls.
* ``NumericType._cast_arrow_array`` — empty string / empty binary become null
  before the numeric cast.
* ``TemporalType._cast_arrow_array`` — Arrow cast then ``fill_arrow_nulls``.
* ``DictionaryType._cast_arrow_array`` — index_in encodes unknowns to null.
* ``StructType._cast_arrow_array`` — child rebuild, parent null mask preserved.

What every fast path must promise:

1. **Null-free in → null-free out.** ``null_count == 0`` on the source
   never silently introduces nulls on a same-shape target.
2. **Null-bearing in → null mask preserved.** The validity buffer follows
   the values; the target's nullable flag controls whether the cast pads
   with the default scalar or keeps the null.
3. **Empty/sentinel source values become null on numeric casts.** Empty
   string and empty bytes are how the wider data plane encodes "missing
   number" coming in from CSV / web payloads — the numeric cast path
   normalizes them to null before pyarrow's int/float parser sees them.

These tests pin those guarantees down per DataType category. They're the
contract the parquet writer and downstream engines rely on — if we ever
swap a fast path for something faster, this is the regression net.
"""
from __future__ import annotations

import datetime

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions


class _NullCountMixin:
    """Helpers shared across per-type null-count fast-path tests."""

    def _cast(
        self,
        array,
        target_type,
        *,
        source_type=None,
        target_nullable: bool = True,
        source_nullable: bool = True,
    ):
        pa = self.pa
        src = Field.from_arrow(
            pa.field("x", source_type or array.type, nullable=source_nullable)
        )
        tgt = Field.from_arrow(
            pa.field("x", target_type, nullable=target_nullable)
        )
        return CastOptions(source=src, target=tgt).cast_arrow_array(array)


class TestIntegerCastNullCount(_NullCountMixin, ArrowTestCase):
    """Integer fast paths — signedness flips, width changes, chunked input."""

    def test_widen_null_free_preserves_null_free(self) -> None:
        arr = self.pa.array([1, 2, 3, 4], type=self.pa.int32())
        out = self._cast(arr, self.pa.int64())
        self.assertEqual(out.null_count, 0)
        self.assertEqual(out.type, self.pa.int64())

    def test_widen_with_nulls_preserves_null_mask(self) -> None:
        arr = self.pa.array([1, 2, None, 4], type=self.pa.int32())
        out = self._cast(arr, self.pa.int64())
        self.assertEqual(out.null_count, 1)
        self.assertFalse(out.is_valid()[2].as_py())

    def test_signedness_flip_null_free(self) -> None:
        # int32 → uint32: the signedness-flip fast path uses ``safe=False``
        # then re-fills nulls. A null-free input must stay null-free.
        arr = self.pa.array([1, 2, 3, 4], type=self.pa.int32())
        out = self._cast(arr, self.pa.uint32())
        self.assertEqual(out.null_count, 0)
        self.assertEqual(out.type, self.pa.uint32())

    def test_signedness_flip_with_nulls(self) -> None:
        arr = self.pa.array([1, 2, None, 4], type=self.pa.int32())
        out = self._cast(arr, self.pa.uint32(), target_nullable=True)
        self.assertEqual(out.null_count, 1)

    def test_signedness_flip_nonnullable_target_fills_default(self) -> None:
        # Non-nullable target on a null-bearing array forces fill_null.
        arr = self.pa.array([1, 2, None, 4], type=self.pa.int32())
        out = self._cast(arr, self.pa.int64(), target_nullable=False)
        self.assertEqual(out.null_count, 0)
        self.assertEqual(out.to_pylist(), [1, 2, 0, 4])

    def test_chunked_null_free_stays_null_free(self) -> None:
        pa = self.pa
        ca = pa.chunked_array(
            [pa.array([1, 2], type=pa.int32()), pa.array([3, 4], type=pa.int32())]
        )
        out = self._cast(ca, pa.int64())
        self.assertEqual(out.null_count, 0)
        self.assertEqual(out.num_chunks, 2)

    def test_chunked_partial_nulls_preserved(self) -> None:
        pa = self.pa
        ca = pa.chunked_array(
            [
                pa.array([1, 2], type=pa.int32()),
                pa.array([3, None, 5], type=pa.int32()),
            ]
        )
        out = self._cast(ca, pa.int64())
        self.assertEqual(out.null_count, 1)
        self.assertEqual(out.num_chunks, 2)


class TestFloatCastNullCount(_NullCountMixin, ArrowTestCase):
    """Float fast paths — width changes, NaN passthrough vs null."""

    def test_widen_null_free(self) -> None:
        arr = self.pa.array([1.5, 2.5, 3.0], type=self.pa.float32())
        out = self._cast(arr, self.pa.float64())
        self.assertEqual(out.null_count, 0)

    def test_widen_with_nulls(self) -> None:
        arr = self.pa.array([1.5, None, 3.0], type=self.pa.float32())
        out = self._cast(arr, self.pa.float64())
        self.assertEqual(out.null_count, 1)

    def test_nan_is_not_null(self) -> None:
        # NaN values are values, not nulls. Cast must keep them as values
        # — null_count tracks the validity buffer, not data sentinels.
        arr = self.pa.array([1.0, float("nan"), 3.0], type=self.pa.float64())
        out = self._cast(arr, self.pa.float32())
        self.assertEqual(out.null_count, 0)


class TestStringToNumericNullCount(_NullCountMixin, ArrowTestCase):
    """String → numeric fast path — empty string is normalized to null."""

    def test_string_to_int_all_present(self) -> None:
        arr = self.pa.array(["1", "2", "3"])
        out = self._cast(arr, self.pa.int64())
        self.assertEqual(out.null_count, 0)
        self.assertEqual(out.to_pylist(), [1, 2, 3])

    def test_empty_string_becomes_null(self) -> None:
        # Wider data plane convention: empty string in CSV/JSON-flat means
        # "no value". The numeric cast normalizes it before pyarrow's
        # int parser, which would otherwise raise.
        arr = self.pa.array(["1", "", "3"])
        out = self._cast(arr, self.pa.int64())
        self.assertEqual(out.null_count, 1)
        self.assertEqual(out.to_pylist(), [1, None, 3])

    def test_empty_binary_becomes_null(self) -> None:
        arr = self.pa.array([b"1", b"", b"3"], type=self.pa.binary())
        out = self._cast(arr, self.pa.int64())
        self.assertEqual(out.null_count, 1)
        self.assertEqual(out.to_pylist(), [1, None, 3])

    def test_string_to_float_null_free(self) -> None:
        arr = self.pa.array(["1.5", "2.25", "3.0"])
        out = self._cast(arr, self.pa.float64())
        self.assertEqual(out.null_count, 0)
        self.assertEqual(out.to_pylist(), [1.5, 2.25, 3.0])


class TestTemporalCastNullCount(_NullCountMixin, ArrowTestCase):
    """Temporal fast paths — unit/tz changes, timestamp ↔ date32."""

    def test_timestamp_unit_change_null_free(self) -> None:
        pa = self.pa
        arr = pa.array(
            [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 6, 1)],
            type=pa.timestamp("us"),
        )
        out = self._cast(arr, pa.timestamp("ms"))
        self.assertEqual(out.null_count, 0)
        self.assertEqual(out.type, pa.timestamp("ms"))

    def test_timestamp_unit_change_with_nulls(self) -> None:
        pa = self.pa
        arr = pa.array(
            [datetime.datetime(2024, 1, 1), None], type=pa.timestamp("us")
        )
        out = self._cast(arr, pa.timestamp("ms"))
        self.assertEqual(out.null_count, 1)

    def test_timestamp_to_date_null_free(self) -> None:
        pa = self.pa
        arr = pa.array(
            [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 6, 1)],
            type=pa.timestamp("us"),
        )
        out = self._cast(arr, pa.date32())
        self.assertEqual(out.null_count, 0)

    def test_timestamp_to_date_with_nulls(self) -> None:
        pa = self.pa
        arr = pa.array(
            [datetime.datetime(2024, 1, 1), None], type=pa.timestamp("us")
        )
        out = self._cast(arr, pa.date32())
        self.assertEqual(out.null_count, 1)


class TestDecimalCastNullCount(_NullCountMixin, ArrowTestCase):
    """Decimal fast paths — float → decimal, precision/scale changes."""

    def test_float_to_decimal_null_free(self) -> None:
        pa = self.pa
        arr = pa.array([1.5, 2.25, 3.0], type=pa.float64())
        out = self._cast(arr, pa.decimal128(10, 2))
        self.assertEqual(out.null_count, 0)

    def test_float_to_decimal_with_nulls(self) -> None:
        pa = self.pa
        arr = pa.array([1.5, None, 3.0], type=pa.float64())
        out = self._cast(arr, pa.decimal128(10, 2))
        self.assertEqual(out.null_count, 1)

    def test_decimal_precision_widen_null_free(self) -> None:
        import decimal as _decimal
        pa = self.pa
        arr = pa.array(
            [_decimal.Decimal("1.50"), _decimal.Decimal("2.25")],
            type=pa.decimal128(10, 2),
        )
        out = self._cast(arr, pa.decimal128(20, 4))
        self.assertEqual(out.null_count, 0)


class TestDictionaryCastNullCount(_NullCountMixin, ArrowTestCase):
    """Dictionary cast — index_in encodes unknowns as null.

    The pure pyarrow dictionary path (open-dictionary, no declared
    categories) is what we exercise here — the categorical path with
    declared categories lives in :class:`DictionaryType` and is covered
    by its own tests.
    """

    def test_string_to_dictionary_null_free(self) -> None:
        pa = self.pa
        arr = pa.array(["a", "b", "a", "c"])
        out = self._cast(arr, pa.dictionary(pa.int32(), pa.string()))
        self.assertEqual(out.null_count, 0)

    def test_string_to_dictionary_with_nulls(self) -> None:
        pa = self.pa
        arr = pa.array(["a", None, "a", "c"])
        out = self._cast(arr, pa.dictionary(pa.int32(), pa.string()))
        self.assertEqual(out.null_count, 1)


class TestStructCastNullCount(_NullCountMixin, ArrowTestCase):
    """Struct fast paths — per-child rebuild, parent validity preserved."""

    def test_struct_child_widen_null_free(self) -> None:
        pa = self.pa
        arr = pa.StructArray.from_arrays(
            [pa.array([1, 2, 3], type=pa.int32()), pa.array(["a", "b", "c"])],
            names=["x", "y"],
        )
        target = pa.struct(
            [pa.field("x", pa.int64()), pa.field("y", pa.string())]
        )
        out = self._cast(arr, target)
        self.assertEqual(out.null_count, 0)
        # children carry their own validity — pin those down explicitly.
        self.assertEqual(out.field("x").null_count, 0)
        self.assertEqual(out.field("y").null_count, 0)

    def test_struct_child_null_preserved(self) -> None:
        pa = self.pa
        arr = pa.StructArray.from_arrays(
            [pa.array([1, None, 3], type=pa.int32()), pa.array(["a", "b", "c"])],
            names=["x", "y"],
        )
        target = pa.struct(
            [pa.field("x", pa.int64()), pa.field("y", pa.string())]
        )
        out = self._cast(arr, target)
        # Parent struct stays valid (its own validity buffer isn't set)
        # but the child still reports a null.
        self.assertEqual(out.null_count, 0)
        self.assertEqual(out.field("x").null_count, 1)


class TestListCastNullCount(_NullCountMixin, ArrowTestCase):
    """List fast paths — element-type cast preserves outer validity."""

    def test_list_element_widen_null_free(self) -> None:
        pa = self.pa
        arr = pa.array(
            [[1, 2], [3, 4, 5]], type=pa.list_(pa.field("item", pa.int32()))
        )
        out = self._cast(arr, pa.list_(pa.field("item", pa.int64())))
        self.assertEqual(out.null_count, 0)

    def test_list_with_outer_null(self) -> None:
        pa = self.pa
        arr = pa.array(
            [[1, 2], None, [3, 4]],
            type=pa.list_(pa.field("item", pa.int32())),
        )
        out = self._cast(arr, pa.list_(pa.field("item", pa.int64())))
        self.assertEqual(out.null_count, 1)

    def test_list_with_inner_null_preserved(self) -> None:
        pa = self.pa
        arr = pa.array(
            [[1, None], [3, 4]], type=pa.list_(pa.field("item", pa.int32()))
        )
        out = self._cast(arr, pa.list_(pa.field("item", pa.int64())))
        # Outer list itself isn't null, but the inner values keep their null.
        self.assertEqual(out.null_count, 0)
        self.assertEqual(out.values.null_count, 1)
