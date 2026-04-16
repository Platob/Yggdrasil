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


class TestDataTypeSpark(SparkTestCase):

    def test_from_spark_integer_type(self):
        from pyspark.sql.types import IntegerType as SparkInt

        dtype = DataType.from_spark(SparkInt())

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.byte_size, 4)
        self.assertTrue(dtype.signed)

    def test_from_spark_long_type(self):
        from pyspark.sql.types import LongType

        dtype = DataType.from_spark(LongType())

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.byte_size, 8)
        self.assertTrue(dtype.signed)

    def test_from_spark_string_type(self):
        from pyspark.sql.types import StringType as SparkString

        dtype = DataType.from_spark(SparkString())

        self.assertIsInstance(dtype, StringType)

    def test_from_spark_boolean_type(self):
        from pyspark.sql.types import BooleanType as SparkBool

        dtype = DataType.from_spark(SparkBool())

        self.assertIsInstance(dtype, BooleanType)

    def test_from_spark_double_type(self):
        from pyspark.sql.types import DoubleType

        dtype = DataType.from_spark(DoubleType())

        self.assertIsInstance(dtype, FloatingPointType)
        self.assertEqual(dtype.byte_size, 8)

    def test_from_spark_float_type(self):
        from pyspark.sql.types import FloatType

        dtype = DataType.from_spark(FloatType())

        self.assertIsInstance(dtype, FloatingPointType)
        self.assertEqual(dtype.byte_size, 4)

    def test_from_spark_decimal_type(self):
        from pyspark.sql.types import DecimalType as SparkDecimal

        dtype = DataType.from_spark(SparkDecimal(10, 2))

        self.assertIsInstance(dtype, DecimalType)
        self.assertEqual(dtype.precision, 10)
        self.assertEqual(dtype.scale, 2)

    def test_from_spark_date_type(self):
        from pyspark.sql.types import DateType as SparkDate

        dtype = DataType.from_spark(SparkDate())

        self.assertIsInstance(dtype, DateType)

    def test_from_spark_timestamp_type(self):
        from pyspark.sql.types import TimestampType as SparkTs

        dtype = DataType.from_spark(SparkTs())

        self.assertIsInstance(dtype, TimestampType)

    def test_from_spark_struct_type(self):
        from pyspark.sql.types import (
            LongType,
            StringType as SparkString,
            StructField,
            StructType as SparkStruct,
        )

        spark_schema = SparkStruct([
            StructField("id", LongType(), nullable=True),
            StructField("name", SparkString(), nullable=True),
        ])

        dtype = DataType.from_spark(spark_schema)

        self.assertIsInstance(dtype, StructType)

    def test_to_spark_all_primitives(self):
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
                spark_type = dtype.to_spark()
                self.assertIsInstance(spark_type, expected_spark_cls)

    def test_cast_spark_column_int(self):
        dtype = IntegerType(byte_size=8, signed=True)
        df = self.spark.createDataFrame([(1,), (2,), (3,)], ["x"])

        result = dtype.cast_spark_column(df["x"])
        result_df = df.withColumn("x", result)
        rows = [r.x for r in result_df.collect()]

        self.assertEqual(rows, [1, 2, 3])

    def test_fill_spark_column_nulls_non_nullable_int(self):
        dtype = IntegerType(byte_size=8, signed=True)
        df = self.spark.createDataFrame([(1,), (None,), (3,)], ["x"])

        result = dtype.fill_spark_column_nulls(
            df["x"],
            nullable=False,
            default_scalar=pa.scalar(0, type=pa.int64()),
        )
        result_df = df.withColumn("x", result)
        rows = [r.x for r in result_df.collect()]

        self.assertEqual(rows, [1, 0, 3])
