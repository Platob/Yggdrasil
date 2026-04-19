from __future__ import annotations

from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.nested import ArrayType, StructType
from yggdrasil.data.types.primitive import IntegerType, StringType
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase


class TestPolarsFlavor(PolarsTestCase):

    def test_datatype_flavor_returns_polars_dtype(self):
        dtype = IntegerType(byte_size=4, signed=True)
        self.assertEqual(dtype.to_polars_flavor(), self.pl.Int32)

    def test_datatype_flavor_nested_struct(self):
        struct = StructType(fields=[Field("a", IntegerType()), Field("b", StringType())])
        self.assertEqual(
            struct.to_polars_flavor(),
            self.pl.Struct([self.pl.Field("a", self.pl.Int64), self.pl.Field("b", self.pl.String)]),
        )

    def test_datatype_flavor_nested_array(self):
        arr = ArrayType.from_item_field(Field("item", IntegerType()))
        self.assertEqual(arr.to_polars_flavor(), self.pl.List(self.pl.Int64))

    def test_field_flavor_returns_polars_field(self):
        f = Field("qty", IntegerType(byte_size=4, signed=True), nullable=True)
        out = f.to_polars_flavor()
        self.assertIsInstance(out, self.pl.Field)
        self.assertEqual(out.name, "qty")
        self.assertEqual(out.dtype, self.pl.Int32)

    def test_schema_flavor_returns_polars_schema(self):
        s = Schema.from_any_fields([Field("a", IntegerType()), Field("b", StringType())])
        out = s.to_polars_flavor()
        self.assertIsInstance(out, self.pl.Schema)
        self.assertEqual(list(out.names()), ["a", "b"])
        self.assertEqual(out["a"], self.pl.Int64)
        self.assertEqual(out["b"], self.pl.String)


class TestSparkFlavor(SparkTestCase):

    def test_datatype_flavor_returns_spark_dtype(self):
        from pyspark.sql.types import IntegerType as SparkIntegerType

        dtype = IntegerType(byte_size=4, signed=True)
        self.assertIsInstance(dtype.to_spark_flavor(), SparkIntegerType)

    def test_datatype_flavor_nested_struct(self):
        from pyspark.sql.types import StructType as SparkStructType

        struct = StructType(fields=[Field("a", IntegerType()), Field("b", StringType())])
        out = struct.to_spark_flavor()
        self.assertIsInstance(out, SparkStructType)
        self.assertEqual([f.name for f in out.fields], ["a", "b"])

    def test_datatype_flavor_nested_array(self):
        from pyspark.sql.types import ArrayType as SparkArrayType, LongType

        arr = ArrayType.from_item_field(Field("item", IntegerType()))
        out = arr.to_spark_flavor()
        self.assertIsInstance(out, SparkArrayType)
        self.assertIsInstance(out.elementType, LongType)

    def test_field_flavor_returns_struct_field(self):
        from pyspark.sql.types import IntegerType as SparkIntegerType, StructField

        f = Field("qty", IntegerType(byte_size=4, signed=True), nullable=True)
        out = f.to_spark_flavor()
        self.assertIsInstance(out, StructField)
        self.assertEqual(out.name, "qty")
        self.assertIsInstance(out.dataType, SparkIntegerType)
        self.assertTrue(out.nullable)

    def test_schema_flavor_returns_spark_struct_type(self):
        from pyspark.sql.types import StructType as SparkStructType

        s = Schema.from_any_fields([Field("a", IntegerType()), Field("b", StringType())])
        out = s.to_spark_flavor()
        self.assertIsInstance(out, SparkStructType)
        self.assertEqual([f.name for f in out.fields], ["a", "b"])
