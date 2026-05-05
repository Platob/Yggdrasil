"""``autotag`` — Databricks-friendly tag dicts derived from dtypes / fields / schemas.

The base contract is one ``type_name`` key holding a lowercase form of
``DataTypeId``. Subclasses extend with shape-specific keys: ``signed``
on integers, ``unit`` / ``tz`` on temporals, ``precision`` / ``scale``
on decimals. Field-level autotag adds ``nullable``; schema-level
autotag propagates per-field tags and pops table-scope metadata
(``partition_by`` / ``cluster_by``) onto the matching fields.
"""
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


# ---------------------------------------------------------------------------
# DataType.autotag — base + subclass extensions
# ---------------------------------------------------------------------------


class TestDataTypeAutotag:

    def test_boolean_carries_kind_only(self) -> None:
        assert BooleanType().autotag() == {b"type_name": b"bool"}

    def test_byte_size_present_when_set_absent_when_not(self) -> None:
        assert IntegerType(byte_size=4).autotag()[b"byte_size"] == b"4"
        assert BinaryType(byte_size=16).autotag()[b"byte_size"] == b"16"
        assert b"byte_size" not in StringType().autotag()

    def test_integer_carries_signed_flag(self) -> None:
        # Specialized integer subclasses (``Int32Type``, ``UInt16Type``,
        # ...) emit their own ``type_name`` so the tag carries width
        # straight to downstream filters.
        signed = IntegerType(byte_size=4, signed=True).autotag()
        assert signed == {
            b"type_name": b"int32",
            b"byte_size": b"4",
            b"signed": b"true",
        }

        unsigned = IntegerType(byte_size=2, signed=False).autotag()
        assert unsigned[b"type_name"] == b"uint16"
        assert unsigned[b"signed"] == b"false"

    def test_float_carries_kind_and_size(self) -> None:
        # ``Float64Type`` here — see :meth:`FloatingPointType.__new__`.
        assert FloatingPointType(byte_size=8).autotag() == {
            b"type_name": b"float64",
            b"byte_size": b"8",
        }

    def test_decimal_carries_precision_and_scale(self) -> None:
        tags = DecimalType(precision=18, scale=4).autotag()

        assert tags[b"type_name"] == b"decimal"
        assert tags[b"precision"] == b"18"
        assert tags[b"scale"] == b"4"

    def test_timestamp_carries_unit_and_tz_when_zoned(self) -> None:
        utc = TimestampType(unit="us", tz="UTC").autotag()

        assert utc[b"type_name"] == b"timestamp"
        assert utc[b"unit"] == b"us"
        assert utc[b"tz"] == b"UTC"

    def test_timestamp_omits_tz_when_naive(self) -> None:
        naive = TimestampType(unit="ns").autotag()

        assert naive[b"unit"] == b"ns"
        assert b"tz" not in naive

    def test_date_and_duration_carry_unit_only(self) -> None:
        assert DateType().autotag() == {
            b"byte_size": b"4",
            b"type_name": b"date",
            b"unit": b"d",
        }

        dur = DurationType(unit="ms").autotag()
        assert dur[b"type_name"] == b"duration"
        assert dur[b"unit"] == b"ms"
        assert b"tz" not in dur

    def test_nested_types_carry_kind_only(self) -> None:
        arr = ArrayType(item_field=Field("item", IntegerType(byte_size=8)))
        assert arr.autotag() == {b"type_name": b"array"}

        map_type = MapType.from_key_value(
            key_field=Field("key", StringType()),
            value_field=Field("value", IntegerType(byte_size=8)),
        )
        assert map_type.autotag() == {b"type_name": b"map"}

        st = StructType(fields=[Field("a", IntegerType(byte_size=8))])
        assert st.autotag() == {b"type_name": b"struct"}


# ---------------------------------------------------------------------------
# Field.autotag — mutates self, idempotent, preserves user tags.
# ---------------------------------------------------------------------------


class TestFieldAutotag:

    def test_returns_self_and_sets_nullable_tag(self) -> None:
        f = Field("qty", IntegerType(byte_size=8), nullable=False)

        out = f.autotag()

        assert out is f
        tags = f.tags or {}
        # Specialized id ``Int64Type`` ⇒ ``type_name=int64``.
        assert tags[b"type_name"] == b"int64"
        assert tags[b"nullable"] == b"false"
        assert b"role" not in tags

    def test_idempotent(self) -> None:
        f = Field("user_id", IntegerType(byte_size=8), nullable=False)

        first = dict(f.autotag().tags or {})
        second = dict(f.autotag().tags or {})

        assert first == second

    def test_preserves_caller_supplied_tags(self) -> None:
        f = Field("email", StringType(), tags={"owner": "data-platform"})

        f.autotag()

        tags = f.tags or {}
        assert tags[b"owner"] == b"data-platform"


# ---------------------------------------------------------------------------
# Schema.autotag — propagates per-field, pops partition_by / cluster_by.
# ---------------------------------------------------------------------------


class TestSchemaAutotag:

    def test_per_field_tags_carry_through(self) -> None:
        s = schema(
            [
                Field("user_id", IntegerType(byte_size=8), nullable=False),
                Field("email", StringType()),
                Field(
                    "created_at",
                    TimestampType(unit="us", tz="UTC"),
                    nullable=False,
                ),
            ],
            metadata={"primary_key": "user_id"},
        )

        out = s.autotag(tags={"layer": "silver"})

        id_tags = out["user_id"].tags or {}
        assert id_tags[b"type_name"] == b"int64"
        assert id_tags[b"signed"] == b"true"
        assert id_tags[b"nullable"] == b"false"
        assert id_tags[b"primary_key"] == b"true"

        assert (out["email"].tags or {})[b"type_name"] == b"string"

        ts_tags = out["created_at"].tags or {}
        assert ts_tags[b"type_name"] == b"timestamp"
        assert ts_tags[b"unit"] == b"us"
        assert ts_tags[b"tz"] == b"UTC"

        # Schema-level tag prefix gets the ``t:`` namespace.
        assert out.metadata is not None
        assert out.metadata[b"t:layer"] == b"silver"

    def test_partition_and_cluster_metadata_are_consumed(self) -> None:
        s = schema(
            [
                Field("trade_date", IntegerType(byte_size=8)),
                Field("book_id", IntegerType(byte_size=8)),
                Field("trade_id", IntegerType(byte_size=8)),
            ],
            metadata={
                "partition_by": "trade_date",
                "cluster_by": '["book_id", "trade_id"]',
            },
        )

        out = s.autotag()

        assert [f.name for f in out.partition_fields] == ["trade_date"]
        assert [f.name for f in out.cluster_fields] == ["book_id", "trade_id"]
        assert (out["trade_date"].tags or {})[b"partition_by"] == b"true"
        assert (out["book_id"].tags or {})[b"cluster_by"] == b"true"

        # Schema-level keys consumed (so they don't leak through Arrow / Delta).
        # ``out.metadata`` is allowed to be ``None`` once everything was
        # consumed off it.
        leftover = out.metadata or {}
        assert b"partition_by" not in leftover
        assert b"cluster_by" not in leftover
