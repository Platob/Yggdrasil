"""Full equals-protocol tests for DataType / Field / Schema.

Covers:

- Primitive dtypes: identity, distinct type_id, parameterized differences
  (byte_size / signed / unit / timezone / precision / scale), and the
  ``check_dtypes`` / ``check_metadata`` / ``check_names`` toggles.
- Field: coercion of non-Field inputs, name / nullable / metadata, and
  delegation to dtype.equals.
- Nested dtypes (ArrayType, MapType, StructType): pairwise child
  comparison, reorder tolerance, missing / renamed / extra children,
  and deep recursion.
- Schema: coercion, reorder tolerance, metadata scope, and recursive
  comparison through several layers of struct/array/map nesting.

The tests don't depend on any engine (polars/pandas/spark) — they only
exercise the pure-Python equality protocol.
"""

from __future__ import annotations

import unittest

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
# DataType.equals - primitive types
# ---------------------------------------------------------------------------


class TestPrimitiveEquals(unittest.TestCase):

    def test_same_integer_equal(self):
        a = IntegerType(byte_size=8, signed=True)
        b = IntegerType(byte_size=8, signed=True)
        self.assertTrue(a.equals(b))

    def test_integer_different_byte_size_not_equal(self):
        a = IntegerType(byte_size=4, signed=True)
        b = IntegerType(byte_size=8, signed=True)
        self.assertFalse(a.equals(b))

    def test_integer_different_signed_not_equal(self):
        a = IntegerType(byte_size=8, signed=True)
        b = IntegerType(byte_size=8, signed=False)
        self.assertFalse(a.equals(b))

    def test_different_type_id_not_equal(self):
        self.assertFalse(IntegerType().equals(FloatingPointType()))
        self.assertFalse(StringType().equals(BinaryType()))
        self.assertFalse(BooleanType().equals(IntegerType(byte_size=1)))

    def test_null_equals_null(self):
        self.assertTrue(NullType().equals(NullType()))

    def test_object_equals_object(self):
        self.assertTrue(ObjectType().equals(ObjectType()))

    def test_object_not_equal_primitive(self):
        self.assertFalse(ObjectType().equals(StringType()))

    def test_timestamp_unit_differs(self):
        a = TimestampType(unit="us", tz=None)
        b = TimestampType(unit="ns", tz=None)
        self.assertFalse(a.equals(b))

    def test_timestamp_tz_differs(self):
        a = TimestampType(unit="us", tz=None)
        b = TimestampType(unit="us", tz="UTC")
        self.assertFalse(a.equals(b))

    def test_timestamp_same_tz_equal(self):
        a = TimestampType(unit="us", tz="UTC")
        b = TimestampType(unit="us", tz="UTC")
        self.assertTrue(a.equals(b))

    def test_decimal_precision_scale(self):
        a = DecimalType(precision=10, scale=2)
        b = DecimalType(precision=10, scale=2)
        self.assertTrue(a.equals(b))

        self.assertFalse(a.equals(DecimalType(precision=10, scale=3)))
        self.assertFalse(a.equals(DecimalType(precision=12, scale=2)))

    def test_string_equals_string(self):
        self.assertTrue(StringType().equals(StringType()))

    def test_date_equals_date(self):
        self.assertTrue(DateType().equals(DateType()))

    def test_duration_unit_differs(self):
        self.assertFalse(
            DurationType(unit="ms").equals(DurationType(unit="us"))
        )


# ---------------------------------------------------------------------------
# DataType.equals - nested types
# ---------------------------------------------------------------------------


class TestNestedEquals(unittest.TestCase):

    def test_array_of_same_inner_equal(self):
        a = ArrayType.from_item_field(IntegerType().to_field(name="item"))
        b = ArrayType.from_item_field(IntegerType().to_field(name="item"))
        self.assertTrue(a.equals(b))

    def test_array_different_inner_not_equal(self):
        a = ArrayType.from_item_field(IntegerType().to_field(name="item"))
        b = ArrayType.from_item_field(StringType().to_field(name="item"))
        self.assertFalse(a.equals(b))

    def test_array_not_equal_to_struct(self):
        arr = ArrayType.from_item_field(IntegerType().to_field(name="item"))
        st = StructType(fields=[IntegerType().to_field(name="x")])
        self.assertFalse(arr.equals(st))

    def test_array_not_equal_to_primitive(self):
        arr = ArrayType.from_item_field(IntegerType().to_field(name="item"))
        self.assertFalse(arr.equals(IntegerType()))

    def test_map_same_equal(self):
        a = MapType.from_key_value(StringType(), IntegerType())
        b = MapType.from_key_value(StringType(), IntegerType())
        self.assertTrue(a.equals(b))

    def test_map_different_value_not_equal(self):
        a = MapType.from_key_value(StringType(), IntegerType())
        b = MapType.from_key_value(StringType(), StringType())
        self.assertFalse(a.equals(b))

    def test_struct_same_equal(self):
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

    def test_struct_reordered_still_equal(self):
        a = StructType(
            fields=[
                IntegerType().to_field(name="a"),
                StringType().to_field(name="b"),
            ]
        )
        b = StructType(
            fields=[
                StringType().to_field(name="b"),
                IntegerType().to_field(name="a"),
            ]
        )
        self.assertTrue(a.equals(b))

    def test_struct_missing_child_not_equal(self):
        a = StructType(
            fields=[
                IntegerType().to_field(name="a"),
                StringType().to_field(name="b"),
            ]
        )
        b = StructType(fields=[IntegerType().to_field(name="a")])
        self.assertFalse(a.equals(b))

    def test_struct_renamed_child_not_equal(self):
        a = StructType(fields=[IntegerType().to_field(name="a")])
        b = StructType(fields=[IntegerType().to_field(name="x")])
        self.assertFalse(a.equals(b))

    def test_struct_check_names_false_ignores_names(self):
        a = StructType(fields=[IntegerType().to_field(name="a")])
        b = StructType(fields=[IntegerType().to_field(name="x")])
        self.assertTrue(a.equals(b, check_names=False))

    def test_struct_check_names_false_compares_positionally(self):
        a = StructType(
            fields=[
                IntegerType().to_field(name="a"),
                StringType().to_field(name="b"),
            ]
        )
        # Same dtypes in same positions, different names — equal when
        # check_names=False.
        b = StructType(
            fields=[
                IntegerType().to_field(name="x"),
                StringType().to_field(name="y"),
            ]
        )
        self.assertTrue(a.equals(b, check_names=False))

        # Swapped dtypes positionally → not equal even with check_names=False.
        c = StructType(
            fields=[
                StringType().to_field(name="a"),
                IntegerType().to_field(name="b"),
            ]
        )
        self.assertFalse(a.equals(c, check_names=False))


class TestNestedRecurse(unittest.TestCase):

    def test_array_of_struct_equal(self):
        a = ArrayType.from_item_field(
            StructType(
                fields=[
                    IntegerType().to_field(name="x"),
                    StringType().to_field(name="y"),
                ]
            ).to_field(name="item")
        )
        b = ArrayType.from_item_field(
            StructType(
                fields=[
                    IntegerType().to_field(name="x"),
                    StringType().to_field(name="y"),
                ]
            ).to_field(name="item")
        )
        self.assertTrue(a.equals(b))

    def test_array_of_struct_diff_child(self):
        a = ArrayType.from_item_field(
            StructType(fields=[IntegerType().to_field(name="x")]).to_field(name="item")
        )
        b = ArrayType.from_item_field(
            StructType(fields=[StringType().to_field(name="x")]).to_field(name="item")
        )
        self.assertFalse(a.equals(b))

    def test_deep_struct_in_map_in_array(self):
        # Array<Map<String, Struct<a: Array<Int>, b: Struct<c: String>>>>
        deep_struct = StructType(
            fields=[
                ArrayType.from_item_field(
                    IntegerType().to_field(name="item")
                ).to_field(name="a"),
                StructType(
                    fields=[StringType().to_field(name="c")]
                ).to_field(name="b"),
            ]
        )
        map_type = MapType.from_key_value(StringType(), deep_struct)
        outer = ArrayType.from_item_field(map_type.to_field(name="item"))

        deep_struct_clone = StructType(
            fields=[
                ArrayType.from_item_field(
                    IntegerType().to_field(name="item")
                ).to_field(name="a"),
                StructType(
                    fields=[StringType().to_field(name="c")]
                ).to_field(name="b"),
            ]
        )
        outer_clone = ArrayType.from_item_field(
            MapType.from_key_value(StringType(), deep_struct_clone).to_field(
                name="item"
            )
        )

        self.assertTrue(outer.equals(outer_clone))

    def test_deep_difference_at_leaf(self):
        # Same structure but with IntegerType vs FloatingPointType at a leaf.
        left = StructType(
            fields=[
                ArrayType.from_item_field(
                    StructType(
                        fields=[IntegerType().to_field(name="leaf")]
                    ).to_field(name="item")
                ).to_field(name="arr"),
            ]
        )
        right = StructType(
            fields=[
                ArrayType.from_item_field(
                    StructType(
                        fields=[FloatingPointType().to_field(name="leaf")]
                    ).to_field(name="item")
                ).to_field(name="arr"),
            ]
        )
        self.assertFalse(left.equals(right))


# ---------------------------------------------------------------------------
# Field.equals
# ---------------------------------------------------------------------------


class TestFieldEquals(unittest.TestCase):

    def test_same_field_equal(self):
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        self.assertTrue(a.equals(b))

    def test_field_name_differs(self):
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="y", dtype=IntegerType(byte_size=8), nullable=True)
        self.assertFalse(a.equals(b))

    def test_field_name_differs_check_names_false(self):
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="y", dtype=IntegerType(byte_size=8), nullable=True)
        self.assertTrue(a.equals(b, check_names=False))

    def test_field_dtype_differs(self):
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="x", dtype=StringType(), nullable=True)
        self.assertFalse(a.equals(b))

    def test_field_nullable_differs(self):
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="x", dtype=IntegerType(byte_size=8), nullable=False)
        self.assertFalse(a.equals(b))

    def test_field_nullable_differs_check_dtypes_false(self):
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        b = Field(name="x", dtype=IntegerType(byte_size=8), nullable=False)
        # Nullable is gated by check_dtypes — skip structural checks.
        self.assertTrue(a.equals(b, check_dtypes=False))

    def test_field_metadata_differs(self):
        a = Field(
            name="x",
            dtype=IntegerType(byte_size=8),
            nullable=True,
            metadata={b"foo": b"bar"},
        )
        b = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        self.assertFalse(a.equals(b))

    def test_field_metadata_differs_check_metadata_false(self):
        a = Field(
            name="x",
            dtype=IntegerType(byte_size=8),
            nullable=True,
            metadata={b"foo": b"bar"},
        )
        b = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        self.assertTrue(a.equals(b, check_metadata=False))

    def test_field_equals_none_is_false(self):
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        self.assertFalse(a.equals(None))

    def test_field_coerces_arrow_field(self):
        import pyarrow as pa

        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        arrow_field = pa.field("x", pa.int64(), nullable=True)
        # Field.from_any accepts pa.Field — equals should too.
        self.assertTrue(a.equals(arrow_field))

    def test_field_coercion_failure_returns_false(self):
        a = Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)
        # An object that cannot be coerced to a Field returns False
        # instead of raising.
        self.assertFalse(a.equals("completely unparseable 😵"))


class TestFieldNestedEquals(unittest.TestCase):

    def test_field_with_struct_dtype(self):
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

    def test_field_with_struct_child_name_differs(self):
        a = Field(
            name="s",
            dtype=StructType(fields=[IntegerType().to_field(name="x")]),
            nullable=True,
        )
        b = Field(
            name="s",
            dtype=StructType(fields=[IntegerType().to_field(name="y")]),
            nullable=True,
        )
        self.assertFalse(a.equals(b))
        self.assertTrue(a.equals(b, check_names=False))

    def test_field_with_struct_child_nullable_differs(self):
        a = Field(
            name="s",
            dtype=StructType(
                fields=[
                    Field(
                        name="x",
                        dtype=IntegerType(byte_size=8),
                        nullable=True,
                    )
                ]
            ),
            nullable=True,
        )
        b = Field(
            name="s",
            dtype=StructType(
                fields=[
                    Field(
                        name="x",
                        dtype=IntegerType(byte_size=8),
                        nullable=False,
                    )
                ]
            ),
            nullable=True,
        )
        self.assertFalse(a.equals(b))
        self.assertTrue(a.equals(b, check_dtypes=False))

    def test_deeply_nested_field(self):
        def build() -> Field:
            return Field(
                name="root",
                dtype=StructType(
                    fields=[
                        ArrayType.from_item_field(
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

    def test_deeply_nested_field_inequal_at_leaf(self):
        left = Field(
            name="root",
            dtype=StructType(
                fields=[
                    ArrayType.from_item_field(
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
                    ArrayType.from_item_field(
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
# Schema.equals
# ---------------------------------------------------------------------------


class TestSchemaEquals(unittest.TestCase):

    def test_same_schema_equal(self):
        a = Schema.from_any_fields(
            [
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
                Field(name="y", dtype=StringType(), nullable=True),
            ]
        )
        b = Schema.from_any_fields(
            [
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
                Field(name="y", dtype=StringType(), nullable=True),
            ]
        )
        self.assertTrue(a.equals(b))

    def test_reordered_schema_equal_by_default(self):
        a = Schema.from_any_fields(
            [
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
                Field(name="y", dtype=StringType(), nullable=True),
            ]
        )
        b = Schema.from_any_fields(
            [
                Field(name="y", dtype=StringType(), nullable=True),
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
            ]
        )
        self.assertTrue(a.equals(b))

    def test_reordered_schema_inequal_when_check_names_false(self):
        a = Schema.from_any_fields(
            [
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
                Field(name="y", dtype=StringType(), nullable=True),
            ]
        )
        b = Schema.from_any_fields(
            [
                Field(name="y", dtype=StringType(), nullable=True),
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
            ]
        )
        # Without name matching, positional dtypes don't line up.
        self.assertFalse(a.equals(b, check_names=False))

    def test_missing_field_not_equal(self):
        a = Schema.from_any_fields(
            [
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
                Field(name="y", dtype=StringType(), nullable=True),
            ]
        )
        b = Schema.from_any_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)]
        )
        self.assertFalse(a.equals(b))
        self.assertFalse(b.equals(a))

    def test_extra_field_not_equal(self):
        a = Schema.from_any_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)]
        )
        b = Schema.from_any_fields(
            [
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
                Field(name="z", dtype=StringType(), nullable=True),
            ]
        )
        self.assertFalse(a.equals(b))

    def test_renamed_field_not_equal(self):
        a = Schema.from_any_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)]
        )
        b = Schema.from_any_fields(
            [Field(name="z", dtype=IntegerType(byte_size=8), nullable=True)]
        )
        self.assertFalse(a.equals(b))
        self.assertTrue(a.equals(b, check_names=False))

    def test_metadata_differs(self):
        a = Schema.from_any_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)],
            metadata={"origin": "A"},
        )
        b = Schema.from_any_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)],
            metadata={"origin": "B"},
        )
        self.assertFalse(a.equals(b))
        self.assertTrue(a.equals(b, check_metadata=False))

    def test_equals_none_is_false(self):
        a = Schema.from_any_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)]
        )
        self.assertFalse(a.equals(None))

    def test_coerces_arrow_schema(self):
        import pyarrow as pa

        a = Schema.from_any_fields(
            [
                Field(name="x", dtype=IntegerType(byte_size=8), nullable=True),
                Field(name="y", dtype=StringType(), nullable=True),
            ]
        )
        arrow_schema = pa.schema(
            [
                pa.field("x", pa.int64(), nullable=True),
                pa.field("y", pa.string(), nullable=True),
            ]
        )
        self.assertTrue(a.equals(arrow_schema))

    def test_coercion_failure_returns_false(self):
        a = Schema.from_any_fields(
            [Field(name="x", dtype=IntegerType(byte_size=8), nullable=True)]
        )
        self.assertFalse(a.equals("not-a-schema-literal"))


class TestSchemaNestedEquals(unittest.TestCase):

    def _build_nested_field(self) -> Field:
        return Field(
            name="events",
            dtype=ArrayType.from_item_field(
                StructType(
                    fields=[
                        IntegerType().to_field(name="id"),
                        StringType().to_field(name="kind"),
                        MapType.from_key_value(
                            StringType(),
                            StructType(
                                fields=[
                                    TimestampType(unit="us", tz="UTC").to_field(
                                        name="ts"
                                    ),
                                    BooleanType().to_field(name="ok"),
                                ]
                            ),
                        ).to_field(name="attrs"),
                    ]
                ).to_field(name="item")
            ),
            nullable=True,
        )

    def test_nested_schema_equals_clone(self):
        a = Schema.from_any_fields([self._build_nested_field()])
        b = Schema.from_any_fields([self._build_nested_field()])
        self.assertTrue(a.equals(b))

    def test_nested_schema_leaf_type_differs(self):
        a = Schema.from_any_fields([self._build_nested_field()])

        mutated = Field(
            name="events",
            dtype=ArrayType.from_item_field(
                StructType(
                    fields=[
                        IntegerType().to_field(name="id"),
                        StringType().to_field(name="kind"),
                        MapType.from_key_value(
                            StringType(),
                            StructType(
                                fields=[
                                    # unit changed ns -> us path covered above;
                                    # here we change timezone to None.
                                    TimestampType(unit="us", tz=None).to_field(
                                        name="ts"
                                    ),
                                    BooleanType().to_field(name="ok"),
                                ]
                            ),
                        ).to_field(name="attrs"),
                    ]
                ).to_field(name="item")
            ),
            nullable=True,
        )
        b = Schema.from_any_fields([mutated])
        self.assertFalse(a.equals(b))

    def test_nested_schema_deep_rename_requires_check_names_false(self):
        a = Schema.from_any_fields([self._build_nested_field()])

        # Rename a deep leaf ("ok" -> "okay") but keep dtypes identical.
        renamed = Field(
            name="events",
            dtype=ArrayType.from_item_field(
                StructType(
                    fields=[
                        IntegerType().to_field(name="id"),
                        StringType().to_field(name="kind"),
                        MapType.from_key_value(
                            StringType(),
                            StructType(
                                fields=[
                                    TimestampType(unit="us", tz="UTC").to_field(
                                        name="ts"
                                    ),
                                    BooleanType().to_field(name="okay"),
                                ]
                            ),
                        ).to_field(name="attrs"),
                    ]
                ).to_field(name="item")
            ),
            nullable=True,
        )
        b = Schema.from_any_fields([renamed])

        self.assertFalse(a.equals(b))
        self.assertTrue(a.equals(b, check_names=False))

    def test_nested_schema_reordered_siblings_still_equal(self):
        original = self._build_nested_field()

        reordered_inner_struct = StructType(
            fields=[
                BooleanType().to_field(name="ok"),
                TimestampType(unit="us", tz="UTC").to_field(name="ts"),
            ]
        )
        reordered = Field(
            name="events",
            dtype=ArrayType.from_item_field(
                StructType(
                    fields=[
                        IntegerType().to_field(name="id"),
                        StringType().to_field(name="kind"),
                        MapType.from_key_value(
                            StringType(), reordered_inner_struct
                        ).to_field(name="attrs"),
                    ]
                ).to_field(name="item")
            ),
            nullable=True,
        )

        a = Schema.from_any_fields([original])
        b = Schema.from_any_fields([reordered])
        self.assertTrue(a.equals(b))


if __name__ == "__main__":
    unittest.main()
