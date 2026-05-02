"""``DataType.from_spark`` / ``to_spark`` and Spark-side cast helpers.

Coverage:

* Inbound — every primitive Spark type promotes to the matching
  yggdrasil subclass with the correct ``byte_size`` / precision /
  scale.
* Outbound — the reverse mapping per-primitive (with a parametrized
  matrix to keep regressions visible).
* Compute — ``cast_spark_column`` and ``fill_spark_column_nulls``
  produce the right values once the expression is bound to a real
  :class:`SparkSession`.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import (
    BooleanType,
    DateType,
    DecimalType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)
from yggdrasil.spark.tests import SparkTestCase


class TestFromSparkPrimitives(SparkTestCase):

    def test_integer_type(self) -> None:
        from pyspark.sql.types import IntegerType as SparkInt

        dtype = DataType.from_spark(SparkInt())

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.byte_size, 4)
        self.assertTrue(dtype.signed)

    def test_long_type(self) -> None:
        from pyspark.sql.types import LongType

        dtype = DataType.from_spark(LongType())

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.byte_size, 8)
        self.assertTrue(dtype.signed)

    def test_string_type(self) -> None:
        from pyspark.sql.types import StringType as SparkString

        self.assertIsInstance(DataType.from_spark(SparkString()), StringType)

    def test_boolean_type(self) -> None:
        from pyspark.sql.types import BooleanType as SparkBool

        self.assertIsInstance(DataType.from_spark(SparkBool()), BooleanType)

    def test_double_type(self) -> None:
        from pyspark.sql.types import DoubleType

        dtype = DataType.from_spark(DoubleType())

        self.assertIsInstance(dtype, FloatingPointType)
        self.assertEqual(dtype.byte_size, 8)

    def test_float_type(self) -> None:
        from pyspark.sql.types import FloatType

        dtype = DataType.from_spark(FloatType())

        self.assertIsInstance(dtype, FloatingPointType)
        self.assertEqual(dtype.byte_size, 4)

    def test_decimal_carries_precision_and_scale(self) -> None:
        from pyspark.sql.types import DecimalType as SparkDecimal

        dtype = DataType.from_spark(SparkDecimal(10, 2))

        self.assertIsInstance(dtype, DecimalType)
        self.assertEqual(dtype.precision, 10)
        self.assertEqual(dtype.scale, 2)

    def test_date_type(self) -> None:
        from pyspark.sql.types import DateType as SparkDate

        self.assertIsInstance(DataType.from_spark(SparkDate()), DateType)

    def test_timestamp_type(self) -> None:
        from pyspark.sql.types import TimestampType as SparkTs

        self.assertIsInstance(DataType.from_spark(SparkTs()), TimestampType)

    def test_struct_type_promotes_to_yggdrasil_struct(self) -> None:
        from pyspark.sql.types import (
            LongType,
            StringType as SparkString,
            StructField,
            StructType as SparkStruct,
        )

        spark_schema = SparkStruct(
            [
                StructField("id", LongType(), nullable=True),
                StructField("name", SparkString(), nullable=True),
            ]
        )

        self.assertIsInstance(DataType.from_spark(spark_schema), StructType)


class TestToSparkPrimitives(SparkTestCase):

    def test_to_spark_matrix(self) -> None:
        from pyspark.sql import types as T

        cases = [
            (StringType(), T.StringType),
            (BooleanType(), T.BooleanType),
            (IntegerType(byte_size=1, signed=True), T.ByteType),
            (IntegerType(byte_size=2, signed=True), T.ShortType),
            (IntegerType(byte_size=4, signed=True), T.IntegerType),
            (IntegerType(byte_size=8, signed=True), T.LongType),
            (FloatingPointType(byte_size=4), T.FloatType),
            (FloatingPointType(byte_size=8), T.DoubleType),
            (DecimalType(precision=10, scale=2), T.DecimalType),
            (DateType(), T.DateType),
            (TimestampType(tz="UTC"), T.TimestampType),
        ]
        for dtype, expected_spark_cls in cases:
            with self.subTest(dtype=str(dtype)):
                self.assertIsInstance(dtype.to_spark(), expected_spark_cls)


class TestSparkCompute(SparkTestCase):

    def test_cast_spark_column_returns_target_values(self) -> None:
        dtype = IntegerType(byte_size=8, signed=True)
        df = self.spark.createDataFrame([(1,), (2,), (3,)], ["x"])

        result = dtype.cast_spark_column(df["x"])
        rows = [r.x for r in df.withColumn("x", result).collect()]

        self.assertEqual(rows, [1, 2, 3])

    def test_fill_spark_column_nulls_with_explicit_default(self) -> None:
        dtype = IntegerType(byte_size=8, signed=True)
        df = self.spark.createDataFrame([(1,), (None,), (3,)], ["x"])

        result = dtype.fill_spark_column_nulls(
            df["x"],
            nullable=False,
            default_scalar=pa.scalar(0, type=pa.int64()),
        )
        rows = [r.x for r in df.withColumn("x", result).collect()]

        self.assertEqual(rows, [1, 0, 3])
