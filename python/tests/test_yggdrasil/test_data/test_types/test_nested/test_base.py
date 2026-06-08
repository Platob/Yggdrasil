"""``NestedType.equals`` — the structural-comparison protocol shared by
:class:`StructType` / :class:`ArrayType` / :class:`MapType`.

The contract under test:

* Same structure, same children, same dtypes → equal.
* Different *names* in the same position → unequal under
  ``check_names=True`` (default), equal under ``check_names=False``.
* Different *order* with the same names → unequal (struct equality is
  positional, not set-based).
* Different arity / mismatched child dtypes → unequal.
* Compared against a primitive or a different nested type ID →
  unequal.
* Empty struct equals empty struct.
* Metadata diff is gated behind ``check_metadata``.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested import NestedType
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.nested.map import MapType
from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType


def _struct(*names_and_types) -> StructType:
    return StructType(
        fields=[
            Field(name=name, dtype=dtype, nullable=True)
            for name, dtype in names_and_types
        ]
    )


# ---------------------------------------------------------------------------
# NestedType is abstract
# ---------------------------------------------------------------------------


class TestNestedAbstract:

    def test_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            NestedType()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# StructType.equals — name + dtype + arity
# ---------------------------------------------------------------------------


class TestStructEquals:

    def test_same_struct_equal(self) -> None:
        a = _struct(
            ("x", IntegerType(byte_size=8, signed=True)),
            ("y", StringType()),
        )
        b = _struct(
            ("x", IntegerType(byte_size=8, signed=True)),
            ("y", StringType()),
        )

        assert a.equals(b) is True

    def test_different_order_with_same_names_unequal(self) -> None:
        a = _struct(
            ("x", IntegerType(byte_size=8, signed=True)),
            ("y", StringType()),
        )
        b = _struct(
            ("y", StringType()),
            ("x", IntegerType(byte_size=8, signed=True)),
        )

        assert a.equals(b) is False

    def test_renamed_child_unequal_when_check_names_true(self) -> None:
        a = _struct(("x", IntegerType(byte_size=8, signed=True)))
        b = _struct(("y", IntegerType(byte_size=8, signed=True)))

        assert a.equals(b, check_names=True) is False

    def test_renamed_child_equal_when_check_names_false(self) -> None:
        a = _struct(("x", IntegerType(byte_size=8, signed=True)))
        b = _struct(("renamed", IntegerType(byte_size=8, signed=True)))

        assert a.equals(b, check_names=False) is True

    def test_different_arity_unequal(self) -> None:
        a = _struct(("x", IntegerType(byte_size=8, signed=True)))
        b = _struct(
            ("x", IntegerType(byte_size=8, signed=True)),
            ("y", StringType()),
        )

        assert a.equals(b) is False

    def test_mismatched_child_dtype_unequal(self) -> None:
        a = _struct(("x", IntegerType(byte_size=8, signed=True)))
        b = _struct(("x", StringType()))

        assert a.equals(b) is False

    def test_compared_to_primitive_is_unequal(self) -> None:
        a = _struct(("x", IntegerType(byte_size=8, signed=True)))

        assert a.equals(IntegerType(byte_size=8, signed=True)) is False

    def test_empty_struct_equal_to_empty_struct(self) -> None:
        assert StructType(fields=[]).equals(StructType(fields=[])) is True

    def test_empty_struct_unequal_to_populated_struct(self) -> None:
        a = StructType(fields=[])
        b = _struct(("x", IntegerType(byte_size=8, signed=True)))

        assert a.equals(b) is False
        assert b.equals(a) is False


# ---------------------------------------------------------------------------
# Cross-type-ID equality — never collapses, regardless of structure.
# ---------------------------------------------------------------------------


class TestCrossTypeIdEquals:

    def test_struct_vs_array_unequal(self) -> None:
        struct = _struct(("x", IntegerType(byte_size=8, signed=True)))
        arr = ArrayType.from_item(
            IntegerType(byte_size=8, signed=True).to_field()
        )

        assert struct.equals(arr) is False
        assert arr.equals(struct) is False


# ---------------------------------------------------------------------------
# MapType.equals
# ---------------------------------------------------------------------------


class TestMapEquals:

    def test_same_kv_equal(self) -> None:
        a = MapType.from_key_value(
            key_field=StringType(),
            value_field=IntegerType(byte_size=8, signed=True),
        )
        b = MapType.from_key_value(
            key_field=StringType(),
            value_field=IntegerType(byte_size=8, signed=True),
        )

        assert a.equals(b) is True

    def test_different_value_dtype_unequal(self) -> None:
        a = MapType.from_key_value(StringType(), IntegerType())
        b = MapType.from_key_value(StringType(), StringType())

        assert a.equals(b) is False


# ---------------------------------------------------------------------------
# ArrayType.equals
# ---------------------------------------------------------------------------


class TestArrayEquals:

    def test_same_item_equal(self) -> None:
        a = ArrayType.from_item(StringType().to_field())
        b = ArrayType.from_item(StringType().to_field())

        assert a.equals(b) is True

    def test_different_item_unequal(self) -> None:
        a = ArrayType.from_item(StringType().to_field())
        b = ArrayType.from_item(IntegerType().to_field())

        assert a.equals(b) is False


# ---------------------------------------------------------------------------
# check_metadata gate
# ---------------------------------------------------------------------------


class TestCheckMetadata:

    def test_metadata_diff_unequal_by_default(self) -> None:
        a = _struct(("x", StringType()))
        b_field = Field(
            name="x", dtype=StringType(), nullable=True, metadata={"k": "v"}
        )
        b = StructType(fields=[b_field])

        assert a.equals(b, check_metadata=True) is False

    def test_metadata_diff_equal_when_gate_disabled(self) -> None:
        a = _struct(("x", StringType()))
        b_field = Field(
            name="x", dtype=StringType(), nullable=True, metadata={"k": "v"}
        )
        b = StructType(fields=[b_field])

        assert a.equals(b, check_metadata=False) is True


# ---------------------------------------------------------------------------
# type_id sanity
# ---------------------------------------------------------------------------


class TestTypeIds:

    def test_each_nested_type_reports_its_id(self) -> None:
        assert _struct().type_id == DataTypeId.STRUCT
        assert (
            ArrayType.from_item(StringType().to_field()).type_id == DataTypeId.ARRAY
        )
        assert (
            MapType.from_key_value(StringType(), StringType()).type_id == DataTypeId.MAP
        )
