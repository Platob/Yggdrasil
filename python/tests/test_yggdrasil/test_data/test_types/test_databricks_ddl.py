from __future__ import annotations

import unittest

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.nested.map import MapType
from yggdrasil.data.types.nested.struct import StructType
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


class TestDatabricksDDL(unittest.TestCase):

    def test_primitive_to_databricks_ddl(self):
        self.assertEqual(NullType().to_databricks_ddl(), "VOID")
        self.assertEqual(BinaryType().to_databricks_ddl(), "BINARY")
        self.assertEqual(StringType().to_databricks_ddl(), "STRING")
        self.assertEqual(BooleanType().to_databricks_ddl(), "BOOLEAN")

        self.assertEqual(IntegerType(byte_size=1, signed=True).to_databricks_ddl(), "BYTE")
        self.assertEqual(IntegerType(byte_size=2, signed=True).to_databricks_ddl(), "SHORT")
        self.assertEqual(IntegerType(byte_size=4, signed=True).to_databricks_ddl(), "INT")
        self.assertEqual(IntegerType(byte_size=8, signed=True).to_databricks_ddl(), "BIGINT")

        self.assertEqual(FloatingPointType(byte_size=4).to_databricks_ddl(), "FLOAT")
        self.assertEqual(FloatingPointType(byte_size=8).to_databricks_ddl(), "DOUBLE")

        self.assertEqual(DecimalType(precision=10, scale=2).to_databricks_ddl(), "DECIMAL(10, 2)")

        self.assertEqual(DateType().to_databricks_ddl(), "DATE")
        self.assertEqual(TimeType().to_databricks_ddl(), "STRING")

        self.assertEqual(TimestampType(tz=None).to_databricks_ddl(), "TIMESTAMP_NTZ")
        self.assertEqual(TimestampType(tz="UTC").to_databricks_ddl(), "TIMESTAMP")

        self.assertEqual(DurationType().to_databricks_ddl(), "BIGINT")

    def test_unsigned_int_to_databricks_ddl(self):
        cases = [
            (IntegerType(byte_size=1, signed=False), "SHORT"),
            (IntegerType(byte_size=2, signed=False), "INT"),
            (IntegerType(byte_size=4, signed=False), "BIGINT"),
            (IntegerType(byte_size=8, signed=False), "DECIMAL(20, 0)"),
        ]
        for dtype, expected in cases:
            with self.subTest(dtype=dtype):
                self.assertEqual(dtype.to_databricks_ddl(), expected)

    def test_nested_to_databricks_ddl(self):
        arr = ArrayType(item_field=Field(name="item", dtype=StringType(), nullable=True))
        self.assertEqual(arr.to_databricks_ddl(), "ARRAY<STRING>")

        map_item_field = Field(
            name="entries",
            nullable=False,
            dtype=StructType(fields=[
                Field(name="key", dtype=StringType(), nullable=False),
                Field(name="value", dtype=IntegerType(byte_size=4), nullable=True),
            ]),
        )
        mp = MapType(item_field=map_item_field)
        self.assertEqual(mp.to_databricks_ddl(), "MAP<STRING, INT>")

        st = StructType(fields=[
            Field(name="a", dtype=StringType()),
            Field(name="b", dtype=IntegerType(byte_size=4)),
        ])
        self.assertEqual(st.to_databricks_ddl(), "STRUCT<`a`: STRING, `b`: INT>")

    def test_deeply_nested_to_databricks_ddl(self):
        inner_st = StructType(fields=[
            Field(name="f1", dtype=StringType()),
            Field(name="f2", dtype=BooleanType()),
        ])
        arr = ArrayType(item_field=Field(name="item", dtype=inner_st, nullable=True))
        mp = MapType(item_field=Field(
            name="entries",
            nullable=False,
            dtype=StructType(fields=[
                Field(name="key", dtype=StringType(), nullable=False),
                Field(name="value", dtype=arr, nullable=True),
            ]),
        ))
        expected = "MAP<STRING, ARRAY<STRUCT<`f1`: STRING, `f2`: BOOLEAN>>>"
        self.assertEqual(mp.to_databricks_ddl(), expected)

    def test_all_primitive_ddl_types_via_subtest(self):
        cases = [
            (NullType(), "VOID"),
            (BinaryType(), "BINARY"),
            (StringType(), "STRING"),
            (BooleanType(), "BOOLEAN"),
            (IntegerType(byte_size=1, signed=True), "BYTE"),
            (IntegerType(byte_size=2, signed=True), "SHORT"),
            (IntegerType(byte_size=4, signed=True), "INT"),
            (IntegerType(byte_size=8, signed=True), "BIGINT"),
            (FloatingPointType(byte_size=4), "FLOAT"),
            (FloatingPointType(byte_size=8), "DOUBLE"),
            (DecimalType(precision=38, scale=18), "DECIMAL(38, 18)"),
            (DateType(), "DATE"),
            (TimeType(), "STRING"),
            (TimestampType(tz=None), "TIMESTAMP_NTZ"),
            (TimestampType(tz="UTC"), "TIMESTAMP"),
            (DurationType(), "BIGINT"),
        ]
        for dtype, expected_ddl in cases:
            with self.subTest(dtype=str(dtype)):
                self.assertEqual(dtype.to_databricks_ddl(), expected_ddl)
