"""``to_spark_name`` — text rendering for the Databricks SQL dialect.

These DDL strings end up in ``CREATE TABLE`` and ``ALTER TABLE``
statements, so any drift in the format breaks live SQL output. Every
primitive and nested type is pinned, and the unsigned-integer mapping
gets its own test because Spark / Databricks have no native unsigned
types — we widen on the way out.
"""

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


class TestPrimitiveDDL(unittest.TestCase):

    def test_signed_int_widths(self) -> None:
        self.assertEqual(IntegerType(byte_size=1, signed=True).to_spark_name(), "BYTE")
        self.assertEqual(IntegerType(byte_size=2, signed=True).to_spark_name(), "SHORT")
        self.assertEqual(IntegerType(byte_size=4, signed=True).to_spark_name(), "INT")
        self.assertEqual(
            IntegerType(byte_size=8, signed=True).to_spark_name(), "BIGINT"
        )

    def test_floating_point_widths(self) -> None:
        self.assertEqual(FloatingPointType(byte_size=4).to_spark_name(), "FLOAT")
        self.assertEqual(FloatingPointType(byte_size=8).to_spark_name(), "DOUBLE")

    def test_decimal_carries_precision_and_scale(self) -> None:
        self.assertEqual(
            DecimalType(precision=10, scale=2).to_spark_name(),
            "DECIMAL(10, 2)",
        )

    def test_simple_primitives(self) -> None:
        self.assertEqual(NullType().to_spark_name(), "VOID")
        self.assertEqual(BinaryType().to_spark_name(), "BINARY")
        self.assertEqual(StringType().to_spark_name(), "STRING")
        self.assertEqual(BooleanType().to_spark_name(), "BOOLEAN")
        self.assertEqual(DateType().to_spark_name(), "DATE")
        # TimeType has no native Databricks counterpart — drops to STRING.
        self.assertEqual(TimeType().to_spark_name(), "STRING")

    def test_timestamp_naive_renders_as_ntz(self) -> None:
        self.assertEqual(TimestampType(tz=None).to_spark_name(), "TIMESTAMP_NTZ")

    def test_timestamp_utc_renders_as_timestamp(self) -> None:
        # All UTC-equivalent spellings collapse to ``TIMESTAMP`` — Databricks
        # stores it UTC-anchored regardless.
        for tz in ("UTC", "Etc/UTC", "GMT", "Z", "+00:00", "-00:00"):
            with self.subTest(tz=tz):
                self.assertEqual(TimestampType(tz=tz).to_spark_name(), "TIMESTAMP")

    def test_timestamp_non_utc_zone_renders_as_ntz(self) -> None:
        # Databricks ``TIMESTAMP`` would silently shift wall-clock values for
        # non-UTC zones; render naive instead so digits round-trip.
        for tz in ("Europe/Paris", "America/New_York", "Asia/Tokyo", "Etc/GMT+5"):
            with self.subTest(tz=tz):
                self.assertEqual(TimestampType(tz=tz).to_spark_name(), "TIMESTAMP_NTZ")

    def test_timestamp_unknown_zone_rejected_at_construction(self) -> None:
        # ``TimestampType.tz`` is :class:`Timezone | None`, so an
        # unparseable IANA string fails fast in ``__post_init__`` rather
        # than silently flowing through to DDL render.
        with self.assertRaises(ValueError):
            TimestampType(tz="Not/AZone")

    def test_duration_widens_to_bigint(self) -> None:
        # Duration has no native Databricks counterpart; carry as int64.
        self.assertEqual(DurationType().to_spark_name(), "BIGINT")

    def test_unsigned_int_widening(self) -> None:
        cases = [
            (IntegerType(byte_size=1, signed=False), "SHORT"),
            (IntegerType(byte_size=2, signed=False), "INT"),
            (IntegerType(byte_size=4, signed=False), "BIGINT"),
            (IntegerType(byte_size=8, signed=False), "DECIMAL(20, 0)"),
        ]
        for dtype, expected in cases:
            with self.subTest(dtype=dtype):
                self.assertEqual(dtype.to_spark_name(), expected)

    def test_full_primitive_matrix(self) -> None:
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
                self.assertEqual(dtype.to_spark_name(), expected_ddl)


class TestNestedDDL(unittest.TestCase):

    def test_array_of_string(self) -> None:
        arr = ArrayType(
            item_field=Field(name="item", dtype=StringType(), nullable=True)
        )
        self.assertEqual(arr.to_spark_name(), "ARRAY<STRING>")

    def test_map_renders_as_map_keytype_valuetype(self) -> None:
        item_field = Field(
            name="entries",
            nullable=False,
            dtype=StructType(
                fields=[
                    Field(name="key", dtype=StringType(), nullable=False),
                    Field(name="value", dtype=IntegerType(byte_size=4), nullable=True),
                ]
            ),
        )

        self.assertEqual(
            MapType(item_field=item_field).to_spark_name(), "MAP<STRING, INT>"
        )

    def test_struct_quotes_field_names(self) -> None:
        st = StructType(
            fields=[
                Field(name="a", dtype=StringType()),
                Field(name="b", dtype=IntegerType(byte_size=4)),
            ]
        )
        self.assertEqual(st.to_spark_name(), "STRUCT<`a`: STRING, `b`: INT>")

    def test_deeply_nested_round_trip(self) -> None:
        inner_st = StructType(
            fields=[
                Field(name="f1", dtype=StringType()),
                Field(name="f2", dtype=BooleanType()),
            ]
        )
        arr = ArrayType(item_field=Field(name="item", dtype=inner_st, nullable=True))
        mp = MapType(
            item_field=Field(
                name="entries",
                nullable=False,
                dtype=StructType(
                    fields=[
                        Field(name="key", dtype=StringType(), nullable=False),
                        Field(name="value", dtype=arr, nullable=True),
                    ]
                ),
            )
        )

        self.assertEqual(
            mp.to_spark_name(),
            "MAP<STRING, ARRAY<STRUCT<`f1`: STRING, `f2`: BOOLEAN>>>",
        )
