import datetime
import decimal
from dataclasses import dataclass
from typing import Annotated, Optional, Tuple

import pytest
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


def test_annotated_tuple_metadata_names():
    field = arrow_field_from_hint(
        Annotated[
            Tuple[int, Annotated[str, {"arrow_type": pa.large_string()}]],
            {"names": ["age", "comment"]},
        ]
    )

    assert field.type == pa.struct(
        [
            pa.field("age", pa.int64(), nullable=False),
            pa.field("comment", pa.large_string(), nullable=False),
        ]
    )
    assert field.nullable is False


def test_annotated_tuple_metadata_name_mismatch():
    with pytest.raises(TypeError):
        arrow_field_from_hint(Annotated[Tuple[int, str], {"names": ["only_one"]}])


def test_annotated_decimal_metadata():
    field = arrow_field_from_hint(
        Annotated[decimal.Decimal, {"precision": 10, "scale": 2}]
    )

    assert field.type == pa.decimal128(10, 2)
    assert field.nullable is False


def test_annotated_metadata_key_value_tuple():
    field = arrow_field_from_hint(
        Annotated[str, ("arrow_type", pa.large_string()), ("comment", "ignored")]
    )

    assert field.type == pa.large_string()
    assert field.nullable is False


def test_annotated_timestamp_metadata():
    field = arrow_field_from_hint(
        Annotated[
            datetime.datetime,
            {
                "unit": "ns",
                "tz": "America/New_York",
            },
        ]
    )

    assert field.type == pa.timestamp("ns", tz="America/New_York")
    assert field.nullable is False


def test_optional_annotated_nullable():
    field = arrow_field_from_hint(Annotated[Optional[int], {"arrow_type": pa.int32()}])

    assert field.type == pa.int32()
    assert field.nullable is True
