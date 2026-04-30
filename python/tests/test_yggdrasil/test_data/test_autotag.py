from __future__ import annotations

from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import schema
from yggdrasil.data.types.nested import ArrayType, MapType, StructType
from yggdrasil.data.types.primitive import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DurationType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)


def test_datatype_autotag_base_kind():
    assert BooleanType().autotag() == {b"type_name": b"bool"}


def test_primitive_autotag_carries_byte_size():
    assert IntegerType(byte_size=4).autotag()[b"byte_size"] == b"4"
    assert BinaryType(byte_size=16).autotag()[b"byte_size"] == b"16"
    assert b"byte_size" not in StringType().autotag()


def test_integer_autotag_carries_signed():
    signed = IntegerType(byte_size=4, signed=True).autotag()
    assert signed == {b"type_name": b"integer", b"byte_size": b"4", b"signed": b"true"}

    unsigned = IntegerType(byte_size=2, signed=False).autotag()
    assert unsigned[b"signed"] == b"false"


def test_float_autotag_is_just_kind_and_size():
    assert FloatingPointType(byte_size=8).autotag() == {
        b"type_name": b"float",
        b"byte_size": b"8",
    }


def test_decimal_autotag_carries_precision_and_scale():
    tags = DecimalType(precision=18, scale=4).autotag()
    assert tags[b"type_name"] == b"decimal"
    assert tags[b"precision"] == b"18"
    assert tags[b"scale"] == b"4"


def test_timestamp_autotag_uses_tz_key():
    utc = TimestampType(unit="us", tz="UTC").autotag()
    assert utc[b"type_name"] == b"timestamp"
    assert utc[b"unit"] == b"us"
    assert utc[b"tz"] == b"UTC"

    naive = TimestampType(unit="ns").autotag()
    assert naive[b"unit"] == b"ns"
    assert b"tz" not in naive


def test_date_and_duration_autotag_carry_unit_only():
    assert DateType().autotag() == {b'byte_size': b'4', b"type_name": b"date", b"unit": b"d"}
    dur = DurationType(unit="ms").autotag()
    assert dur[b"type_name"] == b"duration"
    assert dur[b"unit"] == b"ms"
    assert b"tz" not in dur


def test_nested_autotag_is_just_kind():
    assert ArrayType(item_field=Field("item", IntegerType(byte_size=8))).autotag() == {
        b"type_name": b"array",
    }
    map_type = MapType.from_key_value(
        key_field=Field("key", StringType()),
        value_field=Field("value", IntegerType(byte_size=8)),
    )
    assert map_type.autotag() == {b"type_name": b"map"}
    assert StructType(fields=[Field("a", IntegerType(byte_size=8))]).autotag() == {
        b"type_name": b"struct",
    }


def test_field_autotag_sets_nullable_and_returns_self():
    f = Field("qty", IntegerType(byte_size=8), nullable=False)
    out = f.autotag()
    assert out is f

    tags = f.tags or {}
    assert tags[b"type_name"] == b"integer"
    assert tags[b"nullable"] == b"false"
    assert b"role" not in tags


def test_field_autotag_is_idempotent():
    f = Field("user_id", IntegerType(byte_size=8), nullable=False)
    first = dict(f.autotag().tags or {})
    second = dict(f.autotag().tags or {})
    assert first == second


def test_field_autotag_preserves_custom_tags():
    f = Field("email", StringType(), tags={"owner": "data-platform"})
    f.autotag()
    tags = f.tags or {}
    assert tags[b"owner"] == b"data-platform"


def test_schema_autotag_applies_per_field():
    s = schema(
        [
            Field("user_id", IntegerType(byte_size=8), nullable=False),
            Field("email", StringType()),
        ],
        metadata={"primary_key": "user_id"},
    )

    out = s.autotag()

    user_tags = out["user_id"].tags or {}
    assert user_tags[b"primary_key"] == b"true"

