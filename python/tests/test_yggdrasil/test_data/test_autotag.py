from __future__ import annotations

from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import schema
from yggdrasil.data.types.extensions.geography import GeographyType
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
    assert BooleanType().autotag() == {b"kind": b"bool"}


def test_primitive_autotag_carries_byte_size():
    assert IntegerType(byte_size=4).autotag()[b"byte_size"] == b"4"
    assert BinaryType(byte_size=16).autotag()[b"byte_size"] == b"16"
    assert b"byte_size" not in StringType().autotag()


def test_integer_autotag_carries_signed():
    signed = IntegerType(byte_size=4, signed=True).autotag()
    assert signed == {b"kind": b"integer", b"byte_size": b"4", b"signed": b"true"}

    unsigned = IntegerType(byte_size=2, signed=False).autotag()
    assert unsigned[b"signed"] == b"false"


def test_float_autotag_is_just_kind_and_size():
    assert FloatingPointType(byte_size=8).autotag() == {
        b"kind": b"float",
        b"byte_size": b"8",
    }


def test_decimal_autotag_carries_precision_and_scale():
    tags = DecimalType(precision=18, scale=4).autotag()
    assert tags[b"kind"] == b"decimal"
    assert tags[b"precision"] == b"18"
    assert tags[b"scale"] == b"4"


def test_timestamp_autotag_uses_tz_key():
    utc = TimestampType(unit="us", tz="UTC").autotag()
    assert utc[b"kind"] == b"timestamp"
    assert utc[b"unit"] == b"us"
    assert utc[b"tz"] == b"UTC"

    naive = TimestampType(unit="ns").autotag()
    assert naive[b"unit"] == b"ns"
    assert b"tz" not in naive


def test_date_and_duration_autotag_carry_unit_only():
    assert DateType().autotag() == {b"kind": b"date", b"unit": b"d"}
    dur = DurationType(unit="ms").autotag()
    assert dur[b"kind"] == b"duration"
    assert dur[b"unit"] == b"ms"
    assert b"tz" not in dur


def test_nested_autotag_is_just_kind():
    assert ArrayType(item_field=Field("item", IntegerType(byte_size=8))).autotag() == {
        b"kind": b"array",
    }
    map_type = MapType.from_key_value(
        key_field=Field("key", StringType()),
        value_field=Field("value", IntegerType(byte_size=8)),
    )
    assert map_type.autotag() == {b"kind": b"map"}
    assert StructType(fields=[Field("a", IntegerType(byte_size=8))]).autotag() == {
        b"kind": b"struct",
    }


def test_geography_autotag_carries_srid_and_model():
    tags = GeographyType(srid=4326, model="SPHERICAL").autotag()
    assert tags[b"kind"] == b"geography"
    assert tags[b"srid"] == b"4326"
    assert tags[b"model"] == b"SPHERICAL"


def test_field_autotag_sets_nullable_and_returns_self():
    f = Field("qty", IntegerType(byte_size=8), nullable=False)
    out = f.autotag()
    assert out is f

    tags = f.tags or {}
    assert tags[b"kind"] == b"integer"
    assert tags[b"nullable"] == b"false"
    assert b"role" not in tags


def test_field_autotag_identifier_heuristic():
    for name in ("id", "user_id", "order_uuid", "tenant_key"):
        f = Field(name, IntegerType(byte_size=8)).autotag()
        assert (f.tags or {})[b"role"] == b"identifier", name


def test_field_autotag_audit_timestamp_heuristic():
    for name in ("created_at", "updated_at", "deleted_at", "event_timestamp"):
        f = Field(name, TimestampType(unit="us", tz="UTC")).autotag()
        assert (f.tags or {})[b"role"] == b"audit_timestamp", name


def test_field_autotag_pii_email_heuristic():
    f = Field("contact_email", StringType()).autotag()
    assert (f.tags or {})[b"pii"] == b"email"


def test_field_autotag_sensitive_secret_heuristic():
    for name in ("password", "api_key", "refresh_token"):
        f = Field(name, StringType()).autotag()
        assert (f.tags or {})[b"sensitive"] == b"secret", name


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
    assert tags[b"pii"] == b"email"


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
    assert user_tags[b"role"] == b"identifier"
    assert user_tags[b"primary_key"] == b"true"

    email_tags = out["email"].tags or {}
    assert email_tags[b"pii"] == b"email"
