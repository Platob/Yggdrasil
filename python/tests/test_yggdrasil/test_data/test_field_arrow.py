"""Arrow ↔ ``Field`` round-trips.

Two specific contracts get pinned here:

* ``to_arrow_field`` attaches an internal ``type_json`` metadata key
  so a Yggdrasil dtype can survive an Arrow IPC round-trip without
  losing dtype intent (Decimal precision, Timestamp tz, etc.).
* ``from_arrow_field`` strips that internal key on the way back so
  callers don't see implementation-detail metadata.

Plus the schema-level helpers and the default/fill behavior on
Arrow arrays.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.data.data_field import Field


class TestArrowFieldRoundTrip:

    def test_to_arrow_field_attaches_type_json_metadata(self) -> None:
        src = Field(
            "value", pa.int64(), nullable=False, metadata={"comment": "hello"}
        )

        out = src.to_arrow_field()

        assert out.name == "value"
        assert out.type == pa.int64()
        assert out.nullable is False
        assert out.metadata is not None
        assert b"comment" in out.metadata
        assert b"type_json" in out.metadata

    def test_from_arrow_field_strips_type_json_metadata(self) -> None:
        arrow_field = pa.field(
            "value",
            pa.int64(),
            nullable=True,
            metadata={
                b"comment": b"hello",
                b"type_json": b'{"id":3}',
            },
        )

        out = Field.from_arrow_field(arrow_field)

        assert out.name == "value"
        assert out.arrow_type == pa.int64()
        assert out.nullable is True
        # type_json gone, user metadata preserved.
        assert out.metadata == {b"comment": b"hello"}


class TestArrowSchemaPromotion:

    def test_arrow_schema_with_name_metadata_lifts_to_named_struct_field(self) -> None:
        schema = pa.schema(
            [
                pa.field("a", pa.int64(), nullable=True),
                pa.field("b", pa.string(), nullable=True),
            ],
            metadata={
                b"name": b"trade_row",
                b"comment": DEFAULT_FIELD_NAME.encode(),
            },
        )

        out = Field.from_arrow_schema(schema)

        assert out.name == "trade_row"
        assert out.nullable is False
        assert out.arrow_type == pa.struct(list(schema))
        assert out.metadata == {
            b"name": b"trade_row",
            b"comment": DEFAULT_FIELD_NAME.encode(),
        }

    def test_to_schema_carries_field_name_in_metadata_for_non_struct(self) -> None:
        src = Field("value", pa.int64(), nullable=False)

        schema = src.to_schema()

        # ``name`` is a Schema (= Field) attribute now, no longer
        # embedded in the metadata dict. Round-trips through
        # :meth:`Schema.to_arrow_schema`, which re-embeds it under
        # ``b"name"`` so the arrow schema carries it.
        assert schema.name == "value"
        assert schema.to_arrow_schema().metadata[b"name"] == b"value"


class TestArrowDefaultsAndFill:

    def test_fill_arrow_array_nulls_uses_field_default(self) -> None:
        src = Field("value", pa.int64(), nullable=False, default=7)
        arr = pa.array([1, None, 3], type=pa.int64())

        assert src.fill_arrow_array_nulls(arr).to_pylist() == [1, 7, 3]

    def test_default_arrow_array_uses_field_default(self) -> None:
        src = Field("value", pa.int64(), nullable=False, default=5)

        assert src.default_arrow_array(size=3).to_pylist() == [5, 5, 5]
