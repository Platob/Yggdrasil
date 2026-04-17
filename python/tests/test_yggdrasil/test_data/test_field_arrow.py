from __future__ import annotations

import pyarrow as pa
from yggdrasil.data.constants import DEFAULT_FIELD_NAME

from yggdrasil.data.data_field import Field


def test_to_arrow_field_attaches_internal_json_metadata():
    src = Field("value", pa.int64(), nullable=False, metadata={"comment": "hello"})

    out = src.to_arrow_field()

    assert out.name == "value"
    assert out.type == pa.int64()
    assert out.nullable is False
    assert out.metadata is not None
    assert b"comment" in out.metadata
    assert b"to_json" in out.metadata


def test_from_arrow_field_strips_internal_metadata():
    arrow_field = pa.field(
        "value",
        pa.int64(),
        nullable=True,
        metadata={
            b"comment": b"hello",
            b"to_json": b'{"id":3}',
        },
    )

    out = Field.from_arrow_field(arrow_field)

    assert out.name == "value"
    assert out.arrow_type == pa.int64()
    assert out.nullable is True
    assert out.metadata == {b"comment": b"hello"}


def test_from_arrow_schema_uses_schema_name_metadata_when_present():
    schema = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=True),
            pa.field("b", pa.string(), nullable=True),
        ],
        metadata={b"name": b"trade_row", b"comment": DEFAULT_FIELD_NAME.encode()},
    )

    out = Field.from_arrow_schema(schema)

    assert out.name == "trade_row"
    assert out.nullable is False
    assert out.arrow_type == pa.struct(list(schema))
    assert out.metadata == {b"name": b"trade_row", b"comment": DEFAULT_FIELD_NAME.encode()}


def test_to_schema_for_non_struct_field_preserves_name_in_schema_metadata():
    src = Field("value", pa.int64(), nullable=False)

    schema = src.to_schema()

    assert schema.metadata is not None
    assert schema.metadata[b"name"] == b"value"


def test_fill_arrow_array_nulls_uses_field_default():
    src = Field("value", pa.int64(), nullable=False, default=7)
    arr = pa.array([1, None, 3], type=pa.int64())

    out = src.fill_arrow_array_nulls(arr)

    assert out.to_pylist() == [1, 7, 3]


def test_default_arrow_array_uses_field_default():
    src = Field("value", pa.int64(), nullable=False, default=5)

    out = src.default_arrow_array(size=3)

    assert out.to_pylist() == [5, 5, 5]
