"""Singleton + lazy engine-type cache on primitive DataTypes.

Final primitive DataTypes (``Int64Type``, ``Int32Type``, ``StringType``,
``BooleanType``, ``Float64Type`` and friends) construct to a single
process-wide instance — every default-arg ``Int64Type()`` call returns
the same object. The lazy ``to_arrow`` / ``to_polars`` / ``to_spark``
caches then survive across every caller without re-computing the
engine projection.

The abstract ``IntegerType`` redirect (``IntegerType(byte_size=8,
signed=True)``) also lands on the same singleton as ``Int64Type()`` —
so callers that route through either path share the cached engine
types.
"""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.primitive import (
    BinaryType,
    BooleanType,
    NullType,
    StringType,
)
from yggdrasil.data.types.primitive.numeric.floating_point import (
    Float8Type,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatingPointType,
)
from yggdrasil.data.types.primitive.numeric.integer import (
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    IntegerType,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
)


class TestPrimitiveSingletons(ArrowTestCase):

    def test_default_int_subclasses_are_singletons(self) -> None:
        for cls in (Int8Type, Int16Type, Int32Type, Int64Type,
                    UInt8Type, UInt16Type, UInt32Type, UInt64Type):
            self.assertIs(cls(), cls(), msg=f"{cls.__name__}")

    def test_default_float_subclasses_are_singletons(self) -> None:
        for cls in (Float8Type, Float16Type, Float32Type, Float64Type):
            self.assertIs(cls(), cls(), msg=f"{cls.__name__}")

    def test_parameterless_primitives_are_singletons(self) -> None:
        for cls in (StringType, BinaryType, BooleanType, NullType):
            self.assertIs(cls(), cls(), msg=f"{cls.__name__}")

    def test_integer_redirect_lands_on_singleton(self) -> None:
        # ``IntegerType(byte_size=8, signed=True)`` redirects to
        # ``Int64Type`` via ``IntegerType.__new__`` — and lands on the
        # same singleton instance ``Int64Type()`` would produce.
        self.assertIs(IntegerType(byte_size=8, signed=True), Int64Type())
        self.assertIs(IntegerType(byte_size=4, signed=True), Int32Type())
        self.assertIs(IntegerType(byte_size=1, signed=False), UInt8Type())

    def test_float_redirect_lands_on_singleton(self) -> None:
        self.assertIs(FloatingPointType(byte_size=8), Float64Type())
        self.assertIs(FloatingPointType(byte_size=4), Float32Type())

    def test_abstract_integer_with_default_args_is_singleton(self) -> None:
        # IntegerType() with no args has byte_size=None — there's no
        # specialized leaf to redirect to. The base ``__new__``
        # singleton path still fires and caches one ``IntegerType``
        # instance per call site.
        self.assertIs(IntegerType(), IntegerType())


class TestEngineTypeCache(ArrowTestCase):
    """``to_arrow`` / ``to_polars`` / ``to_spark`` cache the result on
    the instance via the ``__init_subclass__`` wrapper. Repeated calls
    on the same dtype return the same engine-type object (saves the
    rebuild on the hot cast path)."""

    def test_int64_to_arrow_is_cached(self) -> None:
        t = Int64Type()
        self.assertIs(t.to_arrow(), t.to_arrow())

    def test_int64_to_polars_is_cached(self) -> None:
        try:
            import polars  # noqa: F401
        except ImportError:
            self.skipTest("polars not installed")
        t = Int64Type()
        self.assertIs(t.to_polars(), t.to_polars())

    def test_int64_to_spark_is_cached(self) -> None:
        try:
            import pyspark.sql.types  # noqa: F401
        except ImportError:
            self.skipTest("pyspark not installed")
        t = Int64Type()
        self.assertIs(t.to_spark(), t.to_spark())

    def test_singleton_share_engine_type_cache(self) -> None:
        # Singleton means every ``Int64Type()`` is the same object —
        # so the cache filled by one caller is visible to the next.
        a = Int64Type().to_arrow()
        b = Int64Type().to_arrow()
        self.assertIs(a, b)

    def test_string_type_engine_cache(self) -> None:
        t = StringType()
        self.assertIs(t.to_arrow(), t.to_arrow())

    def test_decimal_instance_still_caches(self) -> None:
        # DecimalType isn't singletonized — every ``DecimalType(p, s)``
        # is a fresh instance — but the per-instance ``to_arrow``
        # cache still applies on the hot path.
        from yggdrasil.data.types.primitive.numeric.decimal import DecimalType
        t = DecimalType(precision=18, scale=4)
        self.assertIs(t.to_arrow(), t.to_arrow())
