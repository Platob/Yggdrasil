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


def test_datatype_autotag_base_keys_always_present():
    tags = BooleanType().autotag()
    assert tags[b"type_id"] == b"BOOL"
    assert tags[b"type_class"] == b"boolean"


def test_integer_autotag_carries_width_and_signedness():
    tags = IntegerType(byte_size=4, signed=False).autotag()
    assert tags[b"type_class"] == b"numeric"
    assert tags[b"numeric_kind"] == b"integer"
    assert tags[b"byte_size"] == b"4"
    assert tags[b"signed"] == b"false"


def test_float_autotag_flags_numeric_kind():
    tags = FloatingPointType(byte_size=8).autotag()
    assert tags[b"numeric_kind"] == b"float"
    assert tags[b"byte_size"] == b"8"


def test_decimal_autotag_carries_precision_and_scale():
    tags = DecimalType(precision=18, scale=4).autotag()
    assert tags[b"numeric_kind"] == b"decimal"
    assert tags[b"precision"] == b"18"
    assert tags[b"scale"] == b"4"


def test_timestamp_autotag_flags_tz_awareness():
    utc = TimestampType(unit="us", tz="UTC").autotag()
    assert utc[b"temporal_kind"] == b"timestamp"
    assert utc[b"tz_aware"] == b"true"
    assert utc[b"timezone"] == b"UTC"

    naive = TimestampType(unit="ns").autotag()
    assert naive[b"tz_aware"] == b"false"
    assert b"timezone" not in naive


def test_date_and_time_autotag_set_temporal_kind_and_unit():
    date_tags = DateType().autotag()
    assert date_tags[b"temporal_kind"] == b"date"
    assert date_tags[b"unit"] == b"d"
    assert b"tz_aware" not in date_tags

    dur_tags = DurationType(unit="ms").autotag()
    assert dur_tags[b"temporal_kind"] == b"duration"
    assert dur_tags[b"unit"] == b"ms"


def test_string_autotag_flags_large_and_view():
    plain = StringType().autotag()
    assert plain[b"type_class"] == b"text"
    assert b"large" not in plain
    assert b"view" not in plain

    large = StringType(large=True).autotag()
    assert large[b"large"] == b"true"

    view = StringType(view=True).autotag()
    assert view[b"view"] == b"true"


def test_binary_autotag_flags_fixed_width():
    fixed = BinaryType(byte_size=16).autotag()
    assert fixed[b"type_class"] == b"binary"
    assert fixed[b"fixed_width"] == b"true"
    assert fixed[b"byte_size"] == b"16"


def test_nested_types_tag_kind_and_children():
    array = ArrayType(item_field=Field("item", IntegerType(byte_size=8))).autotag()
    assert array[b"nested_kind"] == b"array"
    assert array[b"element_type_id"] == b"INTEGER"

    map_type = MapType.from_key_value(
        key_field=Field("key", StringType()),
        value_field=Field("value", IntegerType(byte_size=8)),
    ).autotag()
    assert map_type[b"nested_kind"] == b"map"
    assert map_type[b"key_type_id"] == b"STRING"
    assert map_type[b"value_type_id"] == b"INTEGER"

    struct = StructType(
        fields=[
            Field("a", IntegerType(byte_size=8)),
            Field("b", StringType()),
        ]
    ).autotag()
    assert struct[b"nested_kind"] == b"struct"
    assert struct[b"num_fields"] == b"2"


def test_geography_autotag_carries_srid_and_model():
    tags = GeographyType(srid=4326, model="SPHERICAL").autotag()
    assert tags[b"type_class"] == b"geography"
    assert tags[b"srid"] == b"4326"
    assert tags[b"model"] == b"SPHERICAL"


def test_field_autotag_sets_nullable_and_returns_self():
    f = Field("qty", IntegerType(byte_size=8), nullable=False)
    out = f.autotag()
    assert out is f

    tags = f.tags or {}
    assert tags[b"type_class"] == b"numeric"
    assert tags[b"nullable"] == b"false"
    assert b"role" not in tags


def test_field_autotag_has_default_flag():
    f = Field("qty", IntegerType(byte_size=8), nullable=False, default=0)
    f.autotag()
    tags = f.tags or {}
    assert tags[b"has_default"] == b"true"


def test_field_autotag_identifier_heuristic():
    for name in ("id", "user_id", "order_uuid", "tenant_key"):
        f = Field(name, IntegerType(byte_size=8)).autotag()
        tags = f.tags or {}
        assert tags[b"role"] == b"identifier", name


def test_field_autotag_audit_timestamp_heuristic():
    for name in ("created_at", "updated_at", "deleted_at", "event_timestamp"):
        f = Field(name, TimestampType(unit="us", tz="UTC")).autotag()
        tags = f.tags or {}
        assert tags[b"role"] == b"audit_timestamp", name


def test_field_autotag_pii_email_heuristic():
    f = Field("contact_email", StringType()).autotag()
    tags = f.tags or {}
    assert tags[b"pii"] == b"email"


def test_field_autotag_sensitive_secret_heuristic():
    for name in ("password", "api_key", "refresh_token"):
        f = Field(name, StringType()).autotag()
        tags = f.tags or {}
        assert tags[b"sensitive"] == b"secret", name


def test_field_autotag_is_idempotent():
    f = Field("user_id", IntegerType(byte_size=8), nullable=False)
    first = dict(f.autotag().tags or {})
    second = dict(f.autotag().tags or {})
    assert first == second


def test_field_autotag_preserves_custom_tags():
    f = Field(
        "email",
        StringType(),
        tags={"owner": "data-platform"},
    )
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
