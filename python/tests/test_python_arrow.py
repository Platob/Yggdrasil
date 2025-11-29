from dataclasses import dataclass
from typing import Optional, Tuple

import pyarrow as pa

from yggdrasil.types import arrow_field_from_hint


def test_primitive_default_name():
    field = arrow_field_from_hint(str)

    assert field.name == "str"
    assert field.type == pa.string()
    assert field.nullable is False


def test_optional_infers_nullable():
    field = arrow_field_from_hint(Optional[int])

    assert field.name == "int"
    assert field.type == pa.int64()
    assert field.nullable is True


def test_custom_field_name_override():
    field = arrow_field_from_hint(int, name="age")

    assert field.name == "age"
    assert field.type == pa.int64()
    assert field.nullable is False


def test_struct_from_dataclass_fields():
    @dataclass
    class Example:
        count: int
        note: Optional[str]

    field = arrow_field_from_hint(Example)

    assert field.name == "Example"
    assert field.type == pa.struct(
        [
            pa.field("count", pa.int64(), nullable=False),
            pa.field("note", pa.string(), nullable=True),
        ]
    )
    assert field.nullable is False


def test_tuple_fields_are_indexed():
    field = arrow_field_from_hint(Tuple[int, str])

    assert field.name == "Tuple"
    assert field.type == pa.struct(
        [
            pa.field("_1", pa.int64(), nullable=False),
            pa.field("_2", pa.string(), nullable=False),
        ]
    )
    assert field.nullable is False
