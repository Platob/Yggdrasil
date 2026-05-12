"""Equals protocol across :class:`DataType` / :class:`Field` / :class:`Schema`.

The protocol — pinned here — is the structural-comparison backbone the
rest of the library leans on for cache invalidation, schema drift
detection, and "do these two upstream sources agree?" checks.

* DataType: identity, distinct ``type_id``, parameterized differences
  (byte_size / signed / unit / tz / precision / scale), and the
  ``check_dtypes`` / ``check_metadata`` / ``check_names`` toggles.
* Field: name + nullable + dtype delegation, plus the coercion-failure
  path that quietly returns False instead of raising.
* Nested types: pairwise child comparison, missing / renamed / extra
  children, deep recursion through array-of-map-of-struct.
* Schema: name-keyed equality (so reorder is tolerated), missing/extra
  field detection, metadata gating, and recursion through several
  layers of nesting.

No engine dependency — pure-Python equality only.
"""
from __future__ import annotations

import unittest

import pyarrow as pa

from yggdrasil.data.data_field import Field
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
    NullType,
    ObjectType,
    StringType,
    TimestampType,
)


# ---------------------------------------------------------------------------
# DataType — primitives
# ---------------------------------------------------------------------------


class TestPrimitiveEquals(unittest.TestCase):

    def test_same_integer_equal(self) -> None:
        self.assertTrue(
            IntegerType(byte_size=8, signed=True).equals(
                IntegerType(byte_size=8, signed=True)
            )
        )

    def test_integer_byte_size_differs(self) -> None:
        self.assertFalse(
            IntegerType(byte_size=4, signed=True).equals(
                IntegerType(byte_size=8, signed=True)
            )
        )

    def test_integer_signed_differs(self) -> None:
        self.assertFalse(
            IntegerType(byte_size=8, signed=True).equals(
                IntegerType(byte_size=8, signed=False)
            )
        )

    def test_different_type_ids_unequal(self) -> None:
        self.assertFalse(IntegerType().equals(FloatingPointType()))
        self.assertFalse(StringType().equals(BinaryType()))
        self.assertFalse(BooleanType().equals(IntegerType(byte_size=1)))

    def test_null_equals_null(self) -> None:
        self.assertTrue(NullType().equals(NullType()))

    def test_object_equals_object(self) -> None:
        self.assertTrue(ObjectType().equals(ObjectType()))

    def test_object_unequal_to_primitive(self) -> None:
        self.assertFalse(ObjectType().equals(StringType()))

    def test_timestamp_unit_differs(self) -> None:
        self.assertFalse(
            TimestampType(unit="us", tz=None).equals(
                TimestampType(unit="ns", tz=None)
            )
        )

    def test_timestamp_tz_differs(self) -> None:
        self.assertFalse(
            TimestampType(unit="us", tz=None).equals(
                TimestampType(unit="us", tz="UTC")
            )
        )

    def test_timestamp_same_tz_equal(self) -> None:
        self.assertTrue(
            TimestampType(unit="us", tz="UTC").equals(
                TimestampType(unit="us", tz="UTC")
            )
        )

    def test_decimal_equality_carries_precision_and_scale(self) -> None:
        a = DecimalType(precision=10, scale=2)

        self.assertTrue(a.equals(DecimalType(precision=10, scale=2)))
        self.assertFalse(a.equals(DecimalType(precision=10, scale=3)))
        self.assertFalse(a.equals(DecimalType(precision=12, scale=2)))

    def test_string_and_date_default_equal_to_self(self) -> None:
        self.assertTrue(StringType().equals(StringType()))
        self.assertTrue(DateType().equals(DateType()))

    def test_duration_unit_differs(self) -> None:
        self.assertFalse(
            DurationType(unit="ms").equals(DurationType(unit="us"))
        )


# ---------------------------------------------------------------------------
# DataType — nested
# ---------------------------------------------------------------------------


class TestNestedEquals(unittest.TestCase):

    def test_array_of_same_inner_equal(self) -> None:
        a = ArrayType.from_item(IntegerType().to_field(name="item"))
        b = ArrayType.from_item(IntegerType().to_field(name="item"))

        self.assertTrue(a.equals(b))

    def test_array_inner_dtype_differs(self) -> None:
        a = ArrayType.from_item(IntegerType().to_field(name="item"))
        b = ArrayType.from_item(StringType().to_field(name="item"))

        self.assertFalse(a.equals(b))

    def test_array_unequal_to_struct(self) -> None:
        arr = ArrayType.from_item(IntegerType().to_field(name="item"))
        st = StructType(fields=[IntegerType().to_field(name="x")])

        self.assertFalse(arr.equals(st))

    def test_array_unequal_to_primitive(self) -> None:
        arr = ArrayType.from_item(IntegerType().to_field(name="item"))

        self.assertFalse(arr.equals(IntegerType()))

    def test_map_equal_when_kv_match(self) -> None:
        a = MapType.from_key_value(StringType(), IntegerType())
        b = MapType.from_key_value(StringType(), IntegerType())

        self.assertTrue(a.equals(b))

    def test_map_value_differs(self) -> None:
        a = MapType.from_key_value(StringType(), IntegerType())
        b = MapType.from_key_value(StringType(), StringType())

        self.assertFalse(a.equals(b))

    def test_struct_pairwise_equal(self) -> None:
        a = StructType(
            fields=[
                IntegerType().to_field(name="a"),
                StringType().to_field(name="b"),
            ]
        )
        b = StructType(
            fields=[
                IntegerType().to_field(name="a"),
                StringType().to_field(name="b"),
            ]
        )

        self.assertTrue(a.equals(b))

    def test_struct_missing_child_unequal(self) -> None:
        a = StructType(
            fields=[
                IntegerType().to_field(name="a"),
                StringType().to_field(name="b"),
            ]
        )
        b = StructType(fields=[IntegerType().to_field(name="a")])

        self.assertFalse(a.equals(b))

    def test_struct_renamed_child_unequal(self) -> None:
        a = StructType(fields=[IntegerType().to_field(name="a")])
        b = StructType(fields=[IntegerType().to_field(name="x")])

        self.assertFalse(a.equals(b))

    def test_struct_check_names_false_ignores_names(self) -> None:
        a = StructType(fields=[IntegerType().to_field(name="a")])
        b = StructType(fields=[IntegerType().to_field(name="x")])

        self.assertTrue(a.equals(b, check_names=False))

    def test_struct_check_names_false_compares_positionally(self) -> None:
        a = StructType(
            fields=[
                IntegerType().to_field(name="a"),
                StringType().to_field(name="b"),
            ]
        )
        b = StructType(
            fields=[
                IntegerType().to_field(name="x"),
                StringType().to_field(name="y"),
            ]
        )
        self.assertTrue(a.equals(b, check_names=False))

        # Swap the dtypes positionally → unequal even with names off.
        c = StructType(
            fields=[
                StringType().to_field(name="a"),
                IntegerType().to_field(name="b"),
            ]
        )
        self.assertFalse(a.equals(c, check_names=False))


class TestNestedRecursion(unittest.TestCase):

    def test_array_of_struct_equal(self) -> None:
        a = ArrayType.from_item(
            StructType(
                fields=[
                    IntegerType().to_field(name="x"),
                    StringType().to_field(name="y"),
                ]
            ).to_field(name="item")
        )
        b = ArrayType.from_item(
            StructType(
                fields=[
                    IntegerType().to_field(name="x"),
                    StringType().to_field(name="y"),
                ]
            ).to_field(name="item")
        )

        self.assertTrue(a.equals(b))

    def test_array_of_struct_inner_dtype_differs(self) -> None:
        a = ArrayType.from_item(
            StructType(fields=[IntegerType().to_field(name="x")]).to_field(name="item")
        )
        b = ArrayType.from_item(
            StructType(fields=[StringType().to_field(name="x")]).to_field(name="item")
        )

        self.assertFalse(a.equals(b))

    def test_deep_array_of_map_of_struct(self) -> None:
        def build() -> ArrayType:
            deep_struct = StructType(
                fields=[
                    ArrayType.from_item(
                        IntegerType().to_field(name="item")
                    ).to_field(name="a"),
                    StructType(
                        fields=[StringType().to_field(name="c")]
                    ).to_field(name="b"),
                ]
            )
            return ArrayType.from_item(
                MapType.from_key_value(StringType(), deep_struct).to_field(
                    name="item"
                )
            )

        self.assertTrue(build().equals(build()))

    def test_deep_difference_at_leaf(self) -> None:
        left = StructType(
            fields=[
                ArrayType.from_item(
                    StructType(
                        fields=[IntegerType().to_field(name="leaf")]
                    ).to_field(name="item")
                ).to_field(name="arr"),
            ]
        )
        right = StructType(
            fields=[
                ArrayType.from_item(
                    StructType(
                        fields=[FloatingPointType().to_field(name="leaf")]
                    ).to_field(name="item")
                ).to_field(name="arr"),
            ]
        )

        self.assertFalse(left.equals(right))


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


class TestFieldEquals(unittest.TestCase):

    def test_same_field_equal(self) -> None:
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)

        self.assertTrue(a.equals(b))

    def test_field_name_differs(self) -> None:
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="y", dtype=IntegerType(byte_size=8), nullable=True)

        self.assertFalse(a.equals(b))

    def test_field_check_names_false_ignores_name(self) -> None:
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="y", dtype=IntegerType(byte_size=8), nullable=True)

        self.assertTrue(a.equals(b, check_names=False))

    def test_field_dtype_differs(self) -> None:
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="x", dtype=StringType(), nullable=True)

        self.assertFalse(a.equals(b))

    def test_uncoercible_input_returns_false_without_raising(self) -> None:
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)

        self.assertFalse(a.equals("completely unparseable 😵"))


class TestFieldNestedEquals(unittest.TestCase):

    def test_field_with_struct_dtype(self) -> None:
        a = Field(
            name="s",
            dtype=StructType(fields=[IntegerType().to_field(name="x")]),
            nullable=True,
        )
        b = Field(
            name="s",
            dtype=StructType(fields=[IntegerType().to_field(name="x")]),
            nullable=True,
        )

        self.assertTrue(a.equals(b))

    def test_deep_field_equal(self) -> None:
        def build() -> Field:
            return Field(
                name="root",
                dtype=StructType(
                    fields=[
                        ArrayType.from_item(
                            StructType(
                                fields=[
                                    IntegerType().to_field(name="leaf"),
                                    MapType.from_key_value(
                                        StringType(), BooleanType()
                                    ).to_field(name="m"),
                                ]
                            ).to_field(name="item")
                        ).to_field(name="arr"),
                    ]
                ),
                nullable=True,
            )

        self.assertTrue(build().equals(build()))

    def test_deep_field_leaf_differs(self) -> None:
        left = Field(
            name="root",
            dtype=StructType(
                fields=[
                    ArrayType.from_item(
                        StructType(
                            fields=[IntegerType().to_field(name="leaf")]
                        ).to_field(name="item")
                    ).to_field(name="arr"),
                ]
            ),
            nullable=True,
        )
        right = Field(
            name="root",
            dtype=StructType(
                fields=[
                    ArrayType.from_item(
                        StructType(
                            fields=[StringType().to_field(name="leaf")]
                        ).to_field(name="item")
                    ).to_field(name="arr"),
                ]
            ),
            nullable=True,
        )

        self.assertFalse(left.equals(right))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchemaEquals(unittest.TestCase):

    @staticmethod
    def _xy_schema() -> Schema:
        return Schema.from_fields(
            [
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
                Field(name="y", dtype=StringType(), nullable=True),
            ]
        )

    def test_same_schema_equal(self) -> None:
        self.assertTrue(self._xy_schema().equals(self._xy_schema()))

    def test_reorder_tolerated_by_default(self) -> None:
        a = self._xy_schema()
        b = Schema.from_fields(
            [
                Field(name="y", dtype=StringType(), nullable=True),
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
            ]
        )

        self.assertTrue(a.equals(b))

    def test_reorder_inequal_when_check_names_false(self) -> None:
        a = self._xy_schema()
        b = Schema.from_fields(
            [
                Field(name="y", dtype=StringType(), nullable=True),
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
            ]
        )

        self.assertFalse(a.equals(b, check_names=False))

    def test_missing_field_in_either_direction(self) -> None:
        a = self._xy_schema()
        b = Schema.from_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)]
        )

        self.assertFalse(a.equals(b))
        self.assertFalse(b.equals(a))

    def test_extra_field_unequal(self) -> None:
        a = Schema.from_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)]
        )
        b = Schema.from_fields(
            [
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
                Field(name="z", dtype=StringType(), nullable=True),
            ]
        )

        self.assertFalse(a.equals(b))

    def test_renamed_field_unequal_unless_check_names_false(self) -> None:
        a = Schema.from_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)]
        )
        b = Schema.from_fields(
            [Field(name="z", dtype=IntegerType(byte_size=8), nullable=True)]
        )

        self.assertFalse(a.equals(b))
        self.assertTrue(a.equals(b, check_names=False))

    def test_metadata_difference_gated_by_check_metadata(self) -> None:
        a = Schema.from_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)],
            metadata={"origin": "A"},
        )
        b = Schema.from_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)],
            metadata={"origin": "B"},
        )

        self.assertFalse(a.equals(b))
        self.assertTrue(a.equals(b, check_metadata=False))

    def test_equals_none_is_false(self) -> None:
        self.assertFalse(self._xy_schema().equals(None))

    def test_coerces_arrow_schema_input(self) -> None:
        a = self._xy_schema()
        arrow_schema = pa.schema(
            [
                pa.field("x", pa.int64(), nullable=True),
                pa.field("y", pa.string(), nullable=True),
            ]
        )

        self.assertTrue(a.equals(arrow_schema))

    def test_uncoercible_input_returns_false(self) -> None:
        self.assertFalse(self._xy_schema().equals("not-a-schema-literal"))


class TestSchemaNestedEquals(unittest.TestCase):

    @staticmethod
    def _build_nested(tz: str | None = "UTC") -> Field:
        return Field(
            name="events",
            dtype=ArrayType.from_item(
                StructType(
                    fields=[
                        IntegerType().to_field(name="id"),
                        StringType().to_field(name="kind"),
                        MapType.from_key_value(
                            StringType(),
                            StructType(
                                fields=[
                                    TimestampType(
                                        unit="us", tz=tz
                                    ).to_field(name="ts"),
                                    BooleanType().to_field(name="ok"),
                                ]
                            ),
                        ).to_field(name="attrs"),
                    ]
                ).to_field(name="item")
            ),
            nullable=True,
        )

    def test_nested_clone_is_equal(self) -> None:
        a = Schema.from_fields([self._build_nested()])
        b = Schema.from_fields([self._build_nested()])

        self.assertTrue(a.equals(b))

    def test_nested_leaf_tz_change_breaks_equality(self) -> None:
        a = Schema.from_fields([self._build_nested(tz="UTC")])
        b = Schema.from_fields([self._build_nested(tz=None)])

        self.assertFalse(a.equals(b))


if __name__ == "__main__":
    unittest.main()
