"""``DataType.as_spark`` / ``Field.as_spark`` / ``Schema.as_spark``.

The contract:

* the result is a :class:`DataType` / :class:`Field` / :class:`Schema`
  on the yggdrasil side — never a pyspark object;
* it's a Spark-compatible *yggdrasil* shape, i.e. one whose
  ``to_spark()`` round-trips without a widening surprise;
* unsigned integers stay the same width and flip to signed (Spark has
  no native unsigned types — values reinterpret two's-complement so
  ``max(uint64) → -1`` as ``int64``);
* ``Float16`` widens to ``Float32`` (no native half-precision in
  Spark);
* ``TimeType`` becomes ``StringType``; ``DurationType`` becomes a
  64-bit signed int; non-UTC ``TimestampType`` drops to naive;
* nested types recurse via their child fields' ``as_spark``;
* types that are already Spark-compatible return ``self`` so the
  call is cheap to make defensively.
"""
from __future__ import annotations

import unittest

from yggdrasil.data.data_field import Field
from yggdrasil.data.enums.timezone import Timezone
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.nested import ArrayType, MapType, StructType
from yggdrasil.data.types.primitive import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DurationType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
    TimeType,
)


class TestPrimitiveAsSpark(unittest.TestCase):

    def test_signed_integer_returns_self(self) -> None:
        for size in (1, 2, 4, 8):
            with self.subTest(size=size):
                t = IntegerType(byte_size=size, signed=True)
                self.assertIs(t.as_spark(), t)

    def test_unsigned_integer_flips_signed_keeps_width(self) -> None:
        for size in (1, 2, 4, 8):
            with self.subTest(size=size):
                t = IntegerType(byte_size=size, signed=False)
                spark = t.as_spark()
                self.assertEqual(spark.byte_size, size)
                self.assertTrue(spark.signed)

    def test_unsigned_reinterpret_max_to_negative_one(self) -> None:
        # ``max(uintN)`` reinterprets as ``-1`` in the matching signed
        # width — two's-complement bit pattern.
        for size in (1, 2, 4, 8):
            with self.subTest(size=size):
                signed = IntegerType(byte_size=size, signed=True)
                max_unsigned = (1 << (size * 8)) - 1
                self.assertEqual(signed.reinterpret_pyobj(max_unsigned), -1)

    def test_signed_reinterpret_negative_one_to_max(self) -> None:
        for size in (1, 2, 4, 8):
            with self.subTest(size=size):
                unsigned = IntegerType(byte_size=size, signed=False)
                max_unsigned = (1 << (size * 8)) - 1
                self.assertEqual(unsigned.reinterpret_pyobj(-1), max_unsigned)

    def test_float16_widens_to_float32(self) -> None:
        spark = FloatingPointType(byte_size=2).as_spark()
        self.assertIsInstance(spark, FloatingPointType)
        self.assertEqual(spark.byte_size, 4)

    def test_float32_and_float64_unchanged(self) -> None:
        for size in (4, 8):
            with self.subTest(size=size):
                t = FloatingPointType(byte_size=size)
                self.assertIs(t.as_spark(), t)

    def test_time_widens_to_string(self) -> None:
        self.assertIsInstance(TimeType().as_spark(), StringType)

    def test_duration_widens_to_int64(self) -> None:
        spark = DurationType().as_spark()
        self.assertIsInstance(spark, IntegerType)
        self.assertEqual(spark.byte_size, 8)
        self.assertTrue(spark.signed)

    def test_timestamp_naive_unchanged(self) -> None:
        t = TimestampType()
        self.assertIs(t.as_spark(), t)

    def test_timestamp_utc_unchanged(self) -> None:
        t = TimestampType(tz="UTC")
        self.assertIs(t.as_spark(), t)

    def test_timestamp_non_utc_drops_to_naive(self) -> None:
        t = TimestampType(tz="Europe/Paris")
        spark = t.as_spark()
        self.assertIsInstance(spark, TimestampType)
        self.assertTrue(spark.tz.is_naive())
        self.assertEqual(spark.unit, t.unit)

    def test_pass_through_types_return_self(self) -> None:
        # Spark already represents these natively.
        for t in (
            BooleanType(),
            StringType(),
            BinaryType(),
            DateType(),
            DecimalType(precision=10, scale=2),
        ):
            with self.subTest(t=t):
                self.assertIs(t.as_spark(), t)


class TestNestedAsSpark(unittest.TestCase):

    def test_array_recurses_via_field_as_spark(self) -> None:
        arr = ArrayType.from_item(Field("item", TimeType()))
        spark = arr.as_spark()
        self.assertIsInstance(spark, ArrayType)
        self.assertIsInstance(spark.item_field.dtype, StringType)

    def test_array_already_spark_returns_self(self) -> None:
        arr = ArrayType.from_item(Field("item", IntegerType(byte_size=4, signed=True)))
        self.assertIs(arr.as_spark(), arr)

    def test_map_recurses_via_key_and_value_fields(self) -> None:
        map_type = MapType.from_key_value(
            key_field=Field("key", IntegerType(byte_size=4, signed=False)),
            value_field=Field("value", DurationType()),
        )
        spark = map_type.as_spark()
        self.assertIsInstance(spark, MapType)
        # Unsigned key flips to signed (same width).
        self.assertTrue(spark.key_field.dtype.signed)
        self.assertEqual(spark.key_field.dtype.byte_size, 4)
        # Duration becomes Int64.
        self.assertIsInstance(spark.value_field.dtype, IntegerType)
        self.assertEqual(spark.value_field.dtype.byte_size, 8)

    def test_struct_recurses_via_each_field(self) -> None:
        st = StructType(fields=[
            Field("a", IntegerType(byte_size=2, signed=False)),
            Field("b", TimestampType(tz="Asia/Tokyo")),
            Field("c", DateType()),
        ])
        spark = st.as_spark()
        self.assertIsInstance(spark, StructType)
        # uint16 → int16
        self.assertEqual(spark.fields[0].dtype.byte_size, 2)
        self.assertTrue(spark.fields[0].dtype.signed)
        # Asia/Tokyo → naive
        self.assertTrue(spark.fields[1].dtype.tz.is_naive())
        # date untouched
        self.assertIs(spark.fields[2].dtype, st.fields[2].dtype)

    def test_struct_already_spark_returns_self(self) -> None:
        st = StructType(fields=[
            Field("a", IntegerType(byte_size=4, signed=True)),
            Field("b", DateType()),
        ])
        self.assertIs(st.as_spark(), st)


class TestFieldAndSchemaAsSpark(unittest.TestCase):

    def test_field_returns_field_with_spark_dtype(self) -> None:
        f = Field("x", IntegerType(byte_size=2, signed=False))
        spark = f.as_spark()
        self.assertIsInstance(spark, Field)
        self.assertEqual(spark.name, "x")
        self.assertEqual(spark.nullable, f.nullable)
        self.assertTrue(spark.dtype.signed)
        self.assertEqual(spark.dtype.byte_size, 2)

    def test_field_already_spark_returns_self(self) -> None:
        f = Field("x", IntegerType(byte_size=4, signed=True))
        self.assertIs(f.as_spark(), f)

    def test_schema_returns_schema_with_spark_dtypes(self) -> None:
        s = Schema(inner_fields=[
            Field("x", IntegerType(byte_size=4, signed=False)),
            Field("y", TimeType()),
        ])
        spark = s.as_spark()
        self.assertIsInstance(spark, Schema)
        self.assertTrue(spark["x"].dtype.signed)
        self.assertIsInstance(spark["y"].dtype, StringType)

    def test_schema_already_spark_returns_self(self) -> None:
        s = Schema(inner_fields=[
            Field("x", IntegerType(byte_size=4, signed=True)),
            Field("y", StringType()),
        ])
        self.assertIs(s.as_spark(), s)


class TestSignFlipCastEngine(unittest.TestCase):
    """The cast engine round-trips ``as_spark`` rewrites lossless.

    ``IntegerType.as_spark`` flips ``signed`` on unsigned types
    while keeping the byte width. Pinning the cast engine here so
    the bit-reinterpret semantics stay tied to the type-side rewrite
    — if either drifts, these tests fail loudly rather than silently
    overflowing on round-trip.
    """

    def test_arrow_array_cast_uint64_to_int64_wraps_negative(self) -> None:
        import pyarrow as pa

        src = Field("x", IntegerType(byte_size=8, signed=False))
        tgt = Field("x", IntegerType(byte_size=8, signed=True))
        arr = pa.array([(1 << 64) - 1, 0, 100], type=pa.uint64())

        casted = tgt.cast_arrow_array(arr, source_field=src)

        self.assertEqual(casted.type, pa.int64())
        self.assertEqual(casted.to_pylist(), [-1, 0, 100])

    def test_arrow_array_cast_int64_to_uint64_wraps_to_max(self) -> None:
        import pyarrow as pa

        src = Field("x", IntegerType(byte_size=8, signed=True))
        tgt = Field("x", IntegerType(byte_size=8, signed=False))
        arr = pa.array([-1, 0, 100], type=pa.int64())

        casted = tgt.cast_arrow_array(arr, source_field=src)

        self.assertEqual(casted.type, pa.uint64())
        self.assertEqual(casted.to_pylist(), [(1 << 64) - 1, 0, 100])

    def test_arrow_tabular_cast_applies_per_column(self) -> None:
        import pyarrow as pa

        table = pa.table({
            "big": pa.array([(1 << 64) - 1, 0, 100], type=pa.uint64()),
            "small": pa.array([255, 0, 100], type=pa.uint8()),
        })
        s = Schema(inner_fields=[
            Field("big", IntegerType(byte_size=8, signed=True)),
            Field("small", IntegerType(byte_size=1, signed=True)),
        ])

        casted = s.cast_arrow_tabular(table)

        self.assertEqual(casted.column("big").type, pa.int64())
        self.assertEqual(casted.column("big").to_pylist(), [-1, 0, 100])
        self.assertEqual(casted.column("small").type, pa.int8())
        self.assertEqual(casted.column("small").to_pylist(), [-1, 0, 100])

    def test_polars_series_cast_uint64_to_int64_wraps_negative(self) -> None:
        import polars as pl

        src = Field("x", IntegerType(byte_size=8, signed=False))
        tgt = Field("x", IntegerType(byte_size=8, signed=True))
        series = pl.Series("x", [(1 << 64) - 1, 0, 100], dtype=pl.UInt64)

        casted = tgt.cast_polars_series(series, source_field=src)

        self.assertEqual(casted.dtype, pl.Int64)
        self.assertEqual(casted.to_list(), [-1, 0, 100])


class TestIntegerTypeNewAlwaysRedirects(unittest.TestCase):
    """``__new__`` redirects to the registered specialized class.

    The redirect fires regardless of which class the constructor was
    invoked through — ``IntegerType(byte_size=8, signed=True)``,
    ``Int8Type(byte_size=8)``, and ``UInt32Type(byte_size=8,
    signed=True)`` all collapse to ``Int64Type``. A specialized
    class can't quietly leave its declared width / signedness behind.
    """

    def test_abstract_redirects_to_fixed_per_byte_size_and_signed(self) -> None:
        from yggdrasil.data.types.primitive.numeric import (
            Int8Type, Int16Type, Int32Type, Int64Type,
            UInt8Type, UInt16Type, UInt32Type, UInt64Type,
        )
        cases = [
            (1, True, Int8Type), (2, True, Int16Type),
            (4, True, Int32Type), (8, True, Int64Type),
            (1, False, UInt8Type), (2, False, UInt16Type),
            (4, False, UInt32Type), (8, False, UInt64Type),
        ]
        for size, signed, expected in cases:
            with self.subTest(size=size, signed=signed):
                t = IntegerType(byte_size=size, signed=signed)
                self.assertIs(type(t), expected)

    def test_specialized_with_mismatched_args_redirects_to_canonical(self) -> None:
        from yggdrasil.data.types.primitive.numeric import (
            Int8Type, Int64Type, UInt64Type,
        )
        # Asking for an Int8Type with a wider byte_size lands on the
        # right specialized class instead of producing a malformed
        # ``Int8Type(byte_size=8)``.
        self.assertIs(type(Int8Type(byte_size=8)), Int64Type)
        self.assertIs(type(Int8Type(byte_size=8, signed=False)), UInt64Type)

    def test_specialized_no_args_uses_class_default(self) -> None:
        # ``Int8Type()`` with no args still produces ``Int8Type`` —
        # the class default ``byte_size=1`` lands during ``__init__``,
        # and ``__new__`` saw ``byte_size=None`` so the redirect
        # registry missed (no entry for ``(None, True)``).
        from yggdrasil.data.types.primitive.numeric import Int8Type, Int64Type

        self.assertIs(type(Int8Type()), Int8Type)
        self.assertEqual(Int8Type().byte_size, 1)
        self.assertIs(type(Int64Type()), Int64Type)
        self.assertEqual(Int64Type().byte_size, 8)

    def test_unknown_width_falls_through_to_abstract(self) -> None:
        # 16-byte / hugeint has no registered specialized class — the
        # call lands on plain ``IntegerType`` rather than crashing.
        t = IntegerType(byte_size=16, signed=True)
        self.assertIs(type(t), IntegerType)
        self.assertEqual(t.byte_size, 16)

    def test_pickle_round_trips_specialized_class(self) -> None:
        from yggdrasil.data.types.primitive.numeric import Int64Type, UInt32Type
        import pickle

        for original in (Int64Type(), UInt32Type()):
            with self.subTest(t=original):
                restored = pickle.loads(pickle.dumps(original))
                self.assertIs(type(restored), type(original))
                self.assertEqual(restored.byte_size, original.byte_size)
                self.assertEqual(restored.signed, original.signed)

    def test_floating_point_redirect_same_shape(self) -> None:
        from yggdrasil.data.types.primitive.numeric import (
            Float16Type, Float32Type, Float64Type,
        )
        # Abstract → fixed.
        self.assertIs(type(FloatingPointType(byte_size=2)), Float16Type)
        self.assertIs(type(FloatingPointType(byte_size=4)), Float32Type)
        self.assertIs(type(FloatingPointType(byte_size=8)), Float64Type)
        # Mismatched specialized → canonical.
        self.assertIs(type(Float32Type(byte_size=8)), Float64Type)
        self.assertIs(type(Float64Type(byte_size=2)), Float16Type)
        # No-args specialized stays on its default.
        self.assertIs(type(Float64Type()), Float64Type)
        self.assertEqual(Float64Type().byte_size, 8)
