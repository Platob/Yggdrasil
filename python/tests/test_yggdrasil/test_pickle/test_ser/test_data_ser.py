"""Unit tests for yggdrasil.data Field / Schema serializers.

Covers:
- FieldSerialized  (tag YGG_FIELD = 303)
- SchemaSerialized (tag YGG_SCHEMA = 304)

All round-trips are exercised via:
1. Direct ``from_value`` / ``from_python_object`` constructors.
2. Top-level ``Serialized.from_python_object`` dispatch.
3. Wire round-trip through a ``BytesIO`` buffer (serialise → bytes → deserialise).
4. Public ``dumps`` / ``loads`` helpers.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.field import Field, field as make_field
from yggdrasil.data.schema import Schema, schema as make_schema
from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser import dumps, loads
from yggdrasil.pickle.ser.constants import CODEC_NONE
from yggdrasil.pickle.ser.data import (
    DataSerialized,
    FieldSerialized,
    SchemaSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _wire_roundtrip(ser: Serialized) -> Serialized:
    """Write *ser* to a buffer then read it back."""
    buf = BytesIO()
    ser.write_to(buf)
    return Serialized.read_from(buf, pos=0)


def _assert_fields_equal(a: Field, b: Field) -> None:
    assert a.name == b.name, f"name mismatch: {a.name!r} != {b.name!r}"
    assert a.arrow_type == b.arrow_type, f"type mismatch: {a.arrow_type} != {b.arrow_type}"
    assert a.nullable == b.nullable, f"nullable mismatch: {a.nullable} != {b.nullable}"
    assert (a.metadata or {}) == (b.metadata or {}), (
        f"metadata mismatch:\n  {a.metadata!r}\n  {b.metadata!r}"
    )


def _assert_schemas_equal(a: Schema, b: Schema) -> None:
    assert list(a.names) == list(b.names), (
        f"field names mismatch: {list(a.names)} != {list(b.names)}"
    )
    for name in a.names:
        _assert_fields_equal(a[name], b[name])
    assert (a.metadata or {}) == (b.metadata or {}), (
        f"schema metadata mismatch:\n  {a.metadata!r}\n  {b.metadata!r}"
    )


# ---------------------------------------------------------------------------
# tag registration
# ---------------------------------------------------------------------------

class TestTagRegistration:
    def test_ygg_field_tag_value(self):
        assert Tags.YGG_FIELD == 303

    def test_ygg_schema_tag_value(self):
        assert Tags.YGG_SCHEMA == 304

    def test_field_tag_registered(self):
        Tags._ensure_category_imported(Tags.FRAMEWORK_BASE)
        assert Tags.CLASSES.get(Tags.YGG_FIELD) is FieldSerialized

    def test_schema_tag_registered(self):
        Tags._ensure_category_imported(Tags.FRAMEWORK_BASE)
        assert Tags.CLASSES.get(Tags.YGG_SCHEMA) is SchemaSerialized

    def test_tags_are_in_framework_range(self):
        assert Tags.is_framework(Tags.YGG_FIELD)
        assert Tags.is_framework(Tags.YGG_SCHEMA)

    def test_tag_to_name_includes_ygg_field(self):
        assert Tags.TAG_TO_NAME.get(Tags.YGG_FIELD) == "YGG_FIELD"

    def test_tag_to_name_includes_ygg_schema(self):
        assert Tags.TAG_TO_NAME.get(Tags.YGG_SCHEMA) == "YGG_SCHEMA"


# ---------------------------------------------------------------------------
# FieldSerialized — basic construction
# ---------------------------------------------------------------------------

class TestFieldSerializedBasic:
    def test_from_value_returns_field_serialized(self):
        f = make_field("x", pa.int32())
        ser = FieldSerialized.from_value(f)
        assert isinstance(ser, FieldSerialized)

    def test_tag_is_ygg_field(self):
        f = make_field("x", pa.int32())
        ser = FieldSerialized.from_value(f)
        assert ser.tag == Tags.YGG_FIELD

    def test_wire_metadata_contains_ygg_object(self):
        f = make_field("x", pa.int32())
        ser = FieldSerialized.from_value(f)
        assert (ser.metadata or {}).get(b"ygg_object") == b"field"

    def test_codec_none_respected(self):
        f = make_field("x", pa.int32())
        ser = FieldSerialized.from_value(f, codec=CODEC_NONE)
        assert ser.codec == CODEC_NONE


# ---------------------------------------------------------------------------
# FieldSerialized — round-trips
# ---------------------------------------------------------------------------

class TestFieldSerializedRoundTrip:
    def test_simple_int_field(self):
        orig = make_field("count", pa.int64())
        ser = FieldSerialized.from_value(orig)
        result = ser.value
        _assert_fields_equal(orig, result)

    def test_nullable_false(self):
        orig = make_field("id", pa.int32(), nullable=False)
        ser = FieldSerialized.from_value(orig)
        result = ser.value
        assert result.nullable is False

    def test_string_field(self):
        orig = make_field("name", pa.large_utf8(), nullable=True)
        ser = FieldSerialized.from_value(orig)
        _assert_fields_equal(orig, ser.value)

    def test_timestamp_field(self):
        orig = make_field("created_at", pa.timestamp("us", tz="UTC"))
        ser = FieldSerialized.from_value(orig)
        _assert_fields_equal(orig, ser.value)

    def test_list_field(self):
        orig = make_field("tags", pa.list_(pa.utf8()))
        ser = FieldSerialized.from_value(orig)
        _assert_fields_equal(orig, ser.value)

    def test_map_field(self):
        orig = make_field("props", pa.map_(pa.utf8(), pa.int64()))
        ser = FieldSerialized.from_value(orig)
        _assert_fields_equal(orig, ser.value)

    def test_struct_field(self):
        orig = make_field(
            "address",
            pa.struct([pa.field("city", pa.utf8()), pa.field("zip", pa.utf8())]),
        )
        ser = FieldSerialized.from_value(orig)
        _assert_fields_equal(orig, ser.value)

    def test_with_metadata(self):
        orig = make_field(
            "user_id",
            pa.int64(),
            metadata={"description": "primary key", "source": "users"},
        )
        ser = FieldSerialized.from_value(orig)
        _assert_fields_equal(orig, ser.value)

    def test_with_tags(self):
        orig = make_field(
            "id",
            pa.int64(),
            tags={"primary_key": "true", "cluster_by": "false"},
        )
        ser = FieldSerialized.from_value(orig)
        result = ser.value
        _assert_fields_equal(orig, result)
        assert result.primary_key is True

    def test_as_python_returns_field(self):
        orig = make_field("val", pa.float64())
        ser = FieldSerialized.from_value(orig)
        result = ser.as_python()
        assert isinstance(result, Field)
        _assert_fields_equal(orig, result)

    def test_wire_roundtrip(self):
        orig = make_field("amount", pa.decimal128(18, 4))
        ser = FieldSerialized.from_value(orig, codec=CODEC_NONE)
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, FieldSerialized)
        _assert_fields_equal(orig, restored.value)

    def test_wire_roundtrip_with_metadata(self):
        orig = make_field(
            "score",
            pa.float32(),
            metadata={"unit": "percent"},
            tags={"partition_by": "true"},
        )
        ser = FieldSerialized.from_value(orig)
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, FieldSerialized)
        _assert_fields_equal(orig, restored.value)


# ---------------------------------------------------------------------------
# FieldSerialized — top-level dispatch
# ---------------------------------------------------------------------------

class TestFieldSerializedDispatch:
    def test_from_python_object_dispatch(self):
        f = make_field("x", pa.int16())
        ser = Serialized.from_python_object(f)
        assert isinstance(ser, FieldSerialized)

    def test_from_python_object_value(self):
        orig = make_field("x", pa.int16())
        ser = Serialized.from_python_object(orig)
        result = ser.as_python()
        assert isinstance(result, Field)
        _assert_fields_equal(orig, result)

    def test_data_serialized_dispatch(self):
        f = make_field("x", pa.bool_())
        ser = DataSerialized.from_python_object(f)
        assert isinstance(ser, FieldSerialized)

    def test_data_serialized_returns_none_for_unknown(self):
        result = DataSerialized.from_python_object("not a field")
        assert result is None

    def test_dumps_loads_roundtrip(self):
        orig = make_field("flag", pa.bool_())
        payload = dumps(orig)
        result = loads(payload)
        assert isinstance(result, Field)
        _assert_fields_equal(orig, result)

    def test_dumps_loads_b64_roundtrip(self):
        orig = make_field("ts", pa.timestamp("ms", tz="Europe/Berlin"))
        payload = dumps(orig, b64=True)
        assert isinstance(payload, str)
        result = loads(payload)
        assert isinstance(result, Field)
        _assert_fields_equal(orig, result)


# ---------------------------------------------------------------------------
# SchemaSerialized — basic construction
# ---------------------------------------------------------------------------

class TestSchemaSerializedBasic:
    def test_from_value_returns_schema_serialized(self):
        s = make_schema([make_field("a", pa.int32()), make_field("b", pa.utf8())])
        ser = SchemaSerialized.from_value(s)
        assert isinstance(ser, SchemaSerialized)

    def test_tag_is_ygg_schema(self):
        s = make_schema([make_field("a", pa.int32())])
        ser = SchemaSerialized.from_value(s)
        assert ser.tag == Tags.YGG_SCHEMA

    def test_wire_metadata_contains_ygg_object(self):
        s = make_schema([make_field("a", pa.int32())])
        ser = SchemaSerialized.from_value(s)
        assert (ser.metadata or {}).get(b"ygg_object") == b"schema"

    def test_codec_none_respected(self):
        s = make_schema([make_field("a", pa.int32())])
        ser = SchemaSerialized.from_value(s, codec=CODEC_NONE)
        assert ser.codec == CODEC_NONE


# ---------------------------------------------------------------------------
# SchemaSerialized — round-trips
# ---------------------------------------------------------------------------

class TestSchemaSerializedRoundTrip:
    def test_single_field(self):
        orig = make_schema([make_field("id", pa.int64())])
        ser = SchemaSerialized.from_value(orig)
        _assert_schemas_equal(orig, ser.value)

    def test_multi_field(self):
        orig = make_schema([
            make_field("id", pa.int64(), nullable=False),
            make_field("name", pa.utf8()),
            make_field("score", pa.float32()),
            make_field("active", pa.bool_()),
        ])
        ser = SchemaSerialized.from_value(orig)
        _assert_schemas_equal(orig, ser.value)

    def test_field_order_preserved(self):
        names = ["z", "a", "m", "b"]
        orig = make_schema([make_field(n, pa.utf8()) for n in names])
        result = SchemaSerialized.from_value(orig).value
        assert list(result.names) == names

    def test_schema_level_metadata(self):
        orig = make_schema(
            [make_field("x", pa.int32())],
            metadata={"table": "events", "version": "2"},
        )
        result = SchemaSerialized.from_value(orig).value
        _assert_schemas_equal(orig, result)

    def test_schema_level_tags(self):
        orig = make_schema(
            [make_field("x", pa.int32())],
            tags={"owner": "data-team"},
        )
        result = SchemaSerialized.from_value(orig).value
        _assert_schemas_equal(orig, result)

    def test_field_metadata_preserved(self):
        orig = make_schema([
            make_field("id", pa.int64(), tags={"primary_key": "true"}),
            make_field("ref", pa.int64(), tags={"foreign_key": "catalog.db.t.id"}),
        ])
        result = SchemaSerialized.from_value(orig).value
        _assert_schemas_equal(orig, result)
        assert result["id"].primary_key is True
        assert result["ref"].foreign_key == "catalog.db.t.id"

    def test_nullable_flags_preserved(self):
        orig = make_schema([
            make_field("required", pa.int32(), nullable=False),
            make_field("optional", pa.int32(), nullable=True),
        ])
        result = SchemaSerialized.from_value(orig).value
        assert result["required"].nullable is False
        assert result["optional"].nullable is True

    def test_nested_struct_field(self):
        inner = pa.struct([
            pa.field("street", pa.utf8()),
            pa.field("city", pa.utf8()),
        ])
        orig = make_schema([
            make_field("id", pa.int64()),
            make_field("address", inner),
        ])
        result = SchemaSerialized.from_value(orig).value
        _assert_schemas_equal(orig, result)

    def test_as_python_returns_schema(self):
        orig = make_schema([make_field("a", pa.int32()), make_field("b", pa.utf8())])
        ser = SchemaSerialized.from_value(orig)
        result = ser.as_python()
        assert isinstance(result, Schema)
        _assert_schemas_equal(orig, result)

    def test_wire_roundtrip(self):
        orig = make_schema([
            make_field("id", pa.int64()),
            make_field("ts", pa.timestamp("us", tz="UTC")),
        ])
        ser = SchemaSerialized.from_value(orig, codec=CODEC_NONE)
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, SchemaSerialized)
        _assert_schemas_equal(orig, restored.value)

    def test_wire_roundtrip_with_metadata(self):
        orig = make_schema(
            [
                make_field("id", pa.int64(), nullable=False, tags={"primary_key": "true"}),
                make_field("created_at", pa.timestamp("ms")),
            ],
            metadata={"source": "api"},
            tags={"partition_by": "created_at"},
        )
        ser = SchemaSerialized.from_value(orig)
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, SchemaSerialized)
        _assert_schemas_equal(orig, restored.value)

    def test_empty_schema(self):
        orig = Schema()
        ser = SchemaSerialized.from_value(orig)
        result = ser.value
        assert isinstance(result, Schema)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# SchemaSerialized — top-level dispatch
# ---------------------------------------------------------------------------

class TestSchemaSerializedDispatch:
    def test_from_python_object_dispatch(self):
        s = make_schema([make_field("a", pa.int32())])
        ser = Serialized.from_python_object(s)
        assert isinstance(ser, SchemaSerialized)

    def test_from_python_object_value(self):
        orig = make_schema([make_field("a", pa.float64()), make_field("b", pa.utf8())])
        ser = Serialized.from_python_object(orig)
        result = ser.as_python()
        assert isinstance(result, Schema)
        _assert_schemas_equal(orig, result)

    def test_data_serialized_dispatch(self):
        s = make_schema([make_field("x", pa.bool_())])
        ser = DataSerialized.from_python_object(s)
        assert isinstance(ser, SchemaSerialized)

    def test_dumps_loads_roundtrip(self):
        orig = make_schema([
            make_field("user_id", pa.int64()),
            make_field("email", pa.utf8()),
        ])
        payload = dumps(orig)
        result = loads(payload)
        assert isinstance(result, Schema)
        _assert_schemas_equal(orig, result)

    def test_dumps_loads_b64_roundtrip(self):
        orig = make_schema([make_field("price", pa.decimal128(18, 4))])
        payload = dumps(orig, b64=True)
        assert isinstance(payload, str)
        result = loads(payload)
        assert isinstance(result, Schema)
        _assert_schemas_equal(orig, result)

    def test_repeated_dispatch_uses_type_cache(self):
        """Second call must take the TYPES fast-path (no error or regression)."""
        orig = make_schema([make_field("n", pa.int32())])
        ser1 = Serialized.from_python_object(orig)
        ser2 = Serialized.from_python_object(orig)
        assert isinstance(ser1, SchemaSerialized)
        assert isinstance(ser2, SchemaSerialized)
        _assert_schemas_equal(ser1.value, ser2.value)


# ---------------------------------------------------------------------------
# cross-type correctness
# ---------------------------------------------------------------------------

class TestCrossTypeSafety:
    def test_field_and_schema_get_different_tags(self):
        f = make_field("x", pa.int32())
        s = make_schema([f])
        assert FieldSerialized.from_value(f).tag != SchemaSerialized.from_value(s).tag

    def test_field_payload_not_confused_with_schema(self):
        f = make_field("x", pa.int32())
        ser = FieldSerialized.from_value(f)
        # Tag-based dispatch must return FieldSerialized, not SchemaSerialized
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, FieldSerialized)

    def test_schema_payload_not_confused_with_field(self):
        s = make_schema([make_field("x", pa.int32())])
        ser = SchemaSerialized.from_value(s)
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, SchemaSerialized)
