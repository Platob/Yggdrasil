from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field, field
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType


@dataclass
class TradeRow:
    ts: str
    qty: int


@dataclass
class DataclassFieldOwner:
    value: Optional[int] = None
    name: str = "x"


def test_field_factory_builds_field():
    out = field(
        "price",
        pa.int64(),
        nullable=False,
        metadata={"comment": "settlement price"},
    )

    assert isinstance(out, Field)
    assert out.name == "price"
    assert isinstance(out.dtype, IntegerType)
    assert out.nullable is False
    assert out.metadata is not None
    assert out.metadata[b"comment"] == b"settlement price"


def test_field_init_normalizes_dtype_from_arrow():
    out = Field("value", pa.string(), nullable=True)

    assert out.name == "value"
    assert isinstance(out.dtype, StringType)
    assert out.arrow_type == pa.string()


def test_field_has_default_and_default_property():
    out = Field("value", pa.int64(), default=7)

    assert out.has_default is True
    assert out.default == 7


def test_field_copy_overrides_selected_attributes():
    src = Field("value", pa.int64(), nullable=True, metadata={"comment": "a"})
    out = src.copy(name="value2", nullable=False)

    assert out.name == "value2"
    assert isinstance(out.dtype, IntegerType)
    assert out.nullable is False
    assert out.metadata is not None
    assert out.metadata[b"comment"] == b"a"


def test_from_pytype_optional_sets_nullable_true():
    out = Field.from_pytype(Optional[int], name="qty")

    assert out.name == "qty"
    assert isinstance(out.dtype, IntegerType)
    assert out.nullable is True


def test_from_pytype_plain_type_uses_default_name():
    out = Field.from_pytype(int)

    assert out.name == "int"
    assert isinstance(out.dtype, IntegerType)
    assert out.nullable is False


def test_from_pytype_string_rejects_field_shorthand_type_text():
    with pytest.raises(ValueError, match="Unexpected trailing tokens"):
        Field.from_pytype("qty:int32")


def test_from_str_shorthand_nullable_name_override_matches_current_parser_behavior():
    out = Field.from_str("qty!:int64")

    assert out.name == "qty"
    assert isinstance(out.dtype, StringType)
    assert out.nullable is False


def test_from_dataclass_field_optional_currently_drops_default_value():
    dc_field = dataclasses.fields(DataclassFieldOwner)[0]

    out = Field.from_dataclass_field(dc_field, owner=DataclassFieldOwner)

    assert out.name == "value"
    assert isinstance(out.dtype, IntegerType)
    assert out.nullable is True
    assert out.default is None


def test_from_str_json_payload():
    out = Field.from_str('{"name":"qty","dtype":{"id":3,"byte_size":4,"signed":true},"nullable":false}')

    assert out.name == "qty"
    assert isinstance(out.dtype, IntegerType)
    assert out.arrow_type == pa.int32()
    assert out.nullable is False


def test_from_any_accepts_field_instance():
    src = Field("value", pa.int64())
    out = Field.from_any(src)

    assert out is src


def test_from_any_accepts_arrow_field():
    out = Field.from_any(pa.field("name", pa.string(), nullable=False))

    assert out.name == "name"
    assert isinstance(out.dtype, StringType)
    assert out.nullable is False


def test_from_dataclass_builds_struct_field():
    out = Field.from_dataclass(TradeRow)

    assert out.name == "TradeRow"
    assert isinstance(out.dtype, StructType)
    assert out.nullable is False
    assert out.arrow_type == pa.struct(
        [
            pa.field("ts", pa.string(), nullable=True),
            pa.field("qty", pa.int32(), nullable=True),
        ]
    )


def test_from_dataclass_field_resolves_optional_and_default():
    dc_field = dataclasses.fields(DataclassFieldOwner)[0]

    out = Field.from_dataclass_field(dc_field, owner=DataclassFieldOwner)

    assert out.name == "value"
    assert isinstance(out.dtype, IntegerType)
    assert out.nullable is True
    assert out.default is None


def test_from_dict_builds_field():
    out = Field.from_dict(
        {
            "name": "value",
            "dtype": {"id": 11},
            "nullable": True,
            "metadata": {"comment": "hello"},
        }
    )

    assert out.name == "value"
    assert isinstance(out.dtype, StringType)
    assert out.nullable is True
    assert out.metadata is not None
    assert out.metadata[b"comment"] == b"hello"


def test_from_json_roundtrip():
    src = Field("value", pa.int64(), nullable=False, metadata={"comment": "x"})
    payload = src.to_json()

    out = Field.from_json(payload)

    assert out.name == "value"
    assert isinstance(out.dtype, IntegerType)
    assert out.nullable is False


def test_from_any_rejects_unknown_object():
    class Unsupported:
        pass

    with pytest.raises(TypeError, match="Cannot build Field from"):
        Field.from_any(Unsupported())
