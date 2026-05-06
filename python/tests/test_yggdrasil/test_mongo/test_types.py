"""Tests for the BSON ↔ Arrow type bridge in :mod:`yggdrasil.mongo.types`.

Pure-Python tests: no pymongo / mongomock required — every helper in
``mongo.types`` works against bare Python documents.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from yggdrasil.mongo.types import (
    BSON_METADATA_KEY,
    OBJECT_ID_BYTES,
    arrow_table_to_documents,
    arrow_to_bson_type_name,
    bson_to_arrow_type,
    documents_to_arrow_table,
    infer_arrow_schema_from_documents,
    infer_schema_from_documents,
)


class TestBsonToArrow:
    def test_known_aliases(self):
        assert bson_to_arrow_type("bool") == pa.bool_()
        assert bson_to_arrow_type("int") == pa.int32()
        assert bson_to_arrow_type("long") == pa.int64()
        assert bson_to_arrow_type("double") == pa.float64()
        assert bson_to_arrow_type("string") == pa.string()
        assert bson_to_arrow_type("date") == pa.timestamp("ms", tz="UTC")
        assert bson_to_arrow_type("null") == pa.null()

    def test_objectid_is_fixed_binary_12(self):
        dtype = bson_to_arrow_type("objectId")
        assert pa.types.is_fixed_size_binary(dtype)
        assert dtype.byte_width == OBJECT_ID_BYTES

    def test_decimal_is_decimal128(self):
        dtype = bson_to_arrow_type("decimal")
        assert pa.types.is_decimal(dtype)
        assert dtype.precision == 38

    def test_unknown_falls_back_to_string(self):
        assert bson_to_arrow_type("definitely_not_a_bson_type") == pa.string()


class TestArrowToBson:
    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (pa.bool_(), "bool"),
            (pa.int32(), "int"),
            (pa.int64(), "long"),
            (pa.float64(), "double"),
            (pa.decimal128(38, 4), "decimal"),
            (pa.string(), "string"),
            (pa.binary(), "binData"),
            (pa.timestamp("ms", tz="UTC"), "date"),
            (pa.list_(pa.int32()), "array"),
            (pa.struct([pa.field("a", pa.int32())]), "object"),
            (pa.null(), "null"),
        ],
    )
    def test_canonical_aliases(self, dtype, expected):
        assert arrow_to_bson_type_name(dtype) == expected


class TestInferSchema:
    def test_empty_iterable_returns_empty_schema(self):
        schema = infer_arrow_schema_from_documents([])
        assert len(schema) == 0

    def test_promotes_int_to_long_on_overflow(self):
        docs = [{"x": 1}, {"x": 2 ** 40}]
        schema = infer_arrow_schema_from_documents(docs)
        assert schema.field("x").type == pa.int64()

    def test_promotes_int_to_double(self):
        docs = [{"x": 1}, {"x": 1.5}]
        schema = infer_arrow_schema_from_documents(docs)
        assert schema.field("x").type == pa.float64()

    def test_null_does_not_outrank_other_types(self):
        docs = [{"x": None}, {"x": "alice"}]
        schema = infer_arrow_schema_from_documents(docs)
        assert schema.field("x").type == pa.string()

    def test_records_bson_alias_in_metadata(self):
        docs = [{"name": "alice"}]
        schema = infer_arrow_schema_from_documents(docs)
        md = schema.field("name").metadata or {}
        assert md.get(BSON_METADATA_KEY) == b"string"

    def test_struct_inferred_from_subdocuments(self):
        docs = [
            {"address": {"city": "Paris", "zip": 75001}},
            {"address": {"city": "Lyon", "zip": 69000}},
        ]
        schema = infer_arrow_schema_from_documents(docs)
        addr = schema.field("address").type
        assert pa.types.is_struct(addr)
        names = [addr.field(i).name for i in range(addr.num_fields)]
        assert sorted(names) == ["city", "zip"]

    def test_array_inferred_recursively(self):
        docs = [{"tags": ["x", "y"]}, {"tags": ["z"]}]
        schema = infer_arrow_schema_from_documents(docs)
        tags = schema.field("tags").type
        assert pa.types.is_list(tags)
        assert tags.value_type == pa.string()

    def test_yggdrasil_schema_round_trip(self):
        docs = [{"x": 1, "y": "alpha"}]
        schema = infer_schema_from_documents(docs)
        # Schema inherits Field — check its names for shape.
        assert schema.names == ["x", "y"]


class TestDocumentTableRoundTrip:
    def test_documents_to_table_to_documents_preserves_values(self):
        docs = [
            {"a": 1, "b": "alice"},
            {"a": 2, "b": "bob"},
            {"a": 3, "b": None},
        ]
        table = documents_to_arrow_table(docs)
        round = arrow_table_to_documents(table)
        assert round == [
            {"a": 1, "b": "alice"},
            {"a": 2, "b": "bob"},
            {"a": 3, "b": None},
        ]

    def test_uses_explicit_schema_when_supplied(self):
        schema = pa.schema([
            pa.field("x", pa.int64()),
            pa.field("y", pa.string()),
        ])
        docs = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        table = documents_to_arrow_table(docs, schema=schema)
        assert table.schema == schema
        assert table.num_rows == 2

    def test_handles_datetime_via_round_trip(self):
        when = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        docs = [{"created": when}]
        table = documents_to_arrow_table(docs)
        assert pa.types.is_timestamp(table.schema.field("created").type)
        # Round-trip — pyarrow keeps datetime objects.
        out = arrow_table_to_documents(table)
        assert out[0]["created"].astimezone(dt.timezone.utc) == when
