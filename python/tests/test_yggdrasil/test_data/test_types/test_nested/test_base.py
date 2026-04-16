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


def test_nested_type_is_abstract() -> None:
    with pytest.raises(TypeError):
        NestedType()  # type: ignore[abstract]


def test_nested_equals_same_struct_is_true() -> None:
    a = _struct(("x", IntegerType(byte_size=8, signed=True)), ("y", StringType()))
    b = _struct(("x", IntegerType(byte_size=8, signed=True)), ("y", StringType()))

    assert a.equals(b) is True


def test_nested_equals_different_order_matches_by_name() -> None:
    a = _struct(("x", IntegerType(byte_size=8, signed=True)), ("y", StringType()))
    b = _struct(("y", StringType()), ("x", IntegerType(byte_size=8, signed=True)))

    assert a.equals(b) is True


def test_nested_equals_different_names_is_false_when_check_names() -> None:
    a = _struct(("x", IntegerType(byte_size=8, signed=True)))
    b = _struct(("y", IntegerType(byte_size=8, signed=True)))

    assert a.equals(b, check_names=True) is False


def test_nested_equals_different_names_is_true_when_check_names_false() -> None:
    a = _struct(("x", IntegerType(byte_size=8, signed=True)))
    b = _struct(("renamed", IntegerType(byte_size=8, signed=True)))

    assert a.equals(b, check_names=False) is True


def test_nested_equals_returns_false_for_different_arity() -> None:
    a = _struct(("x", IntegerType(byte_size=8, signed=True)))
    b = _struct(
        ("x", IntegerType(byte_size=8, signed=True)),
        ("y", StringType()),
    )

    assert a.equals(b) is False


def test_nested_equals_returns_false_for_mismatched_dtype() -> None:
    a = _struct(("x", IntegerType(byte_size=8, signed=True)))
    b = _struct(("x", StringType()))

    assert a.equals(b) is False


def test_nested_equals_returns_false_when_compared_to_non_nested() -> None:
    a = _struct(("x", IntegerType(byte_size=8, signed=True)))
    assert a.equals(IntegerType(byte_size=8, signed=True)) is False


def test_nested_equals_returns_false_across_different_nested_type_ids() -> None:
    struct = _struct(("x", IntegerType(byte_size=8, signed=True)))
    arr = ArrayType.from_item_field(IntegerType(byte_size=8, signed=True).to_field())

    assert struct.equals(arr) is False
    assert arr.equals(struct) is False


def test_nested_equals_handles_empty_struct() -> None:
    a = StructType(fields=[])
    b = StructType(fields=[])

    assert a.equals(b) is True


def test_nested_equals_empty_vs_non_empty() -> None:
    a = StructType(fields=[])
    b = _struct(("x", IntegerType(byte_size=8, signed=True)))

    assert a.equals(b) is False
    assert b.equals(a) is False


def test_nested_equals_map_to_map() -> None:
    a = MapType.from_key_value(
        key_field=StringType(),
        value_field=IntegerType(byte_size=8, signed=True),
    )
    b = MapType.from_key_value(
        key_field=StringType(),
        value_field=IntegerType(byte_size=8, signed=True),
    )

    assert a.equals(b) is True


def test_nested_equals_map_different_value_dtype_is_false() -> None:
    a = MapType.from_key_value(key_field=StringType(), value_field=IntegerType())
    b = MapType.from_key_value(key_field=StringType(), value_field=StringType())

    assert a.equals(b) is False


def test_nested_equals_array_to_array_by_item() -> None:
    a = ArrayType.from_item_field(StringType().to_field())
    b = ArrayType.from_item_field(StringType().to_field())

    assert a.equals(b) is True


def test_nested_equals_array_different_item_dtype_is_false() -> None:
    a = ArrayType.from_item_field(StringType().to_field())
    b = ArrayType.from_item_field(IntegerType().to_field())

    assert a.equals(b) is False


def test_nested_equals_respects_check_metadata_flag() -> None:
    a = _struct(("x", StringType()))
    b_field = Field(name="x", dtype=StringType(), nullable=True, metadata={"k": "v"})
    b = StructType(fields=[b_field])

    assert a.equals(b, check_metadata=True) is False
    assert a.equals(b, check_metadata=False) is True


def test_nested_type_id_is_nested_variant() -> None:
    assert _struct().type_id == DataTypeId.STRUCT
    assert ArrayType.from_item_field(StringType().to_field()).type_id == DataTypeId.ARRAY
    assert MapType.from_key_value(StringType(), StringType()).type_id == DataTypeId.MAP
