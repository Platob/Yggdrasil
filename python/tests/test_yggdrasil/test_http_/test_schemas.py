"""Unit tests for :mod:`yggdrasil.http_.schemas`."""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.http_.schemas import (
    REQUEST_SCHEMA,
    REQUEST_URL_STRUCT,
    RESPONSE_SCHEMA,
)
from yggdrasil.url import URL_STRUCT


# -- helpers -----------------------------------------------------------------

def _field_names(schema) -> list[str]:
    return [f.name for f in schema.children]


def _arrow_field_map(schema) -> dict[str, pa.Field]:
    arrow = schema.to_arrow_schema()
    return {f.name: f for f in arrow}


# -- TestRequestSchema -------------------------------------------------------

class TestRequestSchema:

    EXPECTED_FIELDS = [
        "hash",
        "public_hash",
        "method",
        "url",
        "sender",
        "private_url_hash",
        "public_url_hash",
        "headers",
        "tags",
        "body",
        "body_size",
        "body_hash",
        "sent_at",
        "partition_key",
        "_pkl",
    ]

    def test_has_expected_fields(self) -> None:
        names = _field_names(REQUEST_SCHEMA)
        assert names == self.EXPECTED_FIELDS

    def test_to_arrow_schema_returns_pa_schema(self) -> None:
        result = REQUEST_SCHEMA.to_arrow_schema()
        assert isinstance(result, pa.Schema)

    def test_to_arrow_schema_field_count(self) -> None:
        arrow = REQUEST_SCHEMA.to_arrow_schema()
        assert len(arrow) == len(self.EXPECTED_FIELDS)

    def test_hash_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["hash"].type == pa.int64()
        assert fields["hash"].nullable is False

    def test_public_hash_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["public_hash"].type == pa.int64()
        assert fields["public_hash"].nullable is False

    def test_method_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["method"].type == pa.string()
        assert fields["method"].nullable is False

    def test_url_field_is_struct(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert isinstance(fields["url"].type, pa.StructType)
        assert fields["url"].nullable is False

    def test_sender_field_is_nullable_struct(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert isinstance(fields["sender"].type, pa.StructType)
        assert fields["sender"].nullable is True

    def test_private_url_hash_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["private_url_hash"].type == pa.int64()
        assert fields["private_url_hash"].nullable is False

    def test_public_url_hash_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["public_url_hash"].type == pa.int64()
        assert fields["public_url_hash"].nullable is False

    def test_headers_field_is_map(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert isinstance(fields["headers"].type, pa.MapType)
        assert fields["headers"].type.key_type == pa.string()
        assert fields["headers"].type.item_type == pa.string()
        assert fields["headers"].nullable is False

    def test_tags_field_is_map(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert isinstance(fields["tags"].type, pa.MapType)
        assert fields["tags"].type.key_type == pa.string()
        assert fields["tags"].type.item_type == pa.string()
        assert fields["tags"].nullable is False

    def test_body_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["body"].type == pa.large_binary()
        assert fields["body"].nullable is True

    def test_body_size_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["body_size"].type == pa.int64()
        assert fields["body_size"].nullable is False

    def test_body_hash_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["body_hash"].type == pa.int64()
        assert fields["body_hash"].nullable is False

    def test_sent_at_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["sent_at"].type == pa.timestamp("us", "UTC")
        assert fields["sent_at"].nullable is False

    def test_partition_key_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["partition_key"].type == pa.int64()
        assert fields["partition_key"].nullable is False

    def test_pkl_field_type(self) -> None:
        fields = _arrow_field_map(REQUEST_SCHEMA)
        assert fields["_pkl"].type == pa.large_binary()
        assert fields["_pkl"].nullable is True


# -- TestResponseSchema ------------------------------------------------------

class TestResponseSchema:

    # The response unnests every request field (except _pkl) with a
    # ``request_`` prefix, then adds its own top-level columns.
    EXPECTED_OWN_FIELDS = [
        "receiver",
        "hash",
        "public_hash",
        "status_code",
        "headers",
        "tags",
        "body",
        "body_size",
        "body_hash",
        "received_at",
        "partition_key",
        "_pkl",
    ]

    EXPECTED_REQUEST_PREFIX_FIELDS = [
        "request_hash",
        "request_public_hash",
        "request_method",
        "request_url",
        "request_sender",
        "request_private_url_hash",
        "request_public_url_hash",
        "request_headers",
        "request_tags",
        "request_body",
        "request_body_size",
        "request_body_hash",
        "request_sent_at",
        "request_partition_key",
    ]

    def test_has_all_expected_fields(self) -> None:
        names = _field_names(RESPONSE_SCHEMA)
        expected = self.EXPECTED_REQUEST_PREFIX_FIELDS + self.EXPECTED_OWN_FIELDS
        assert names == expected

    def test_to_arrow_schema_returns_pa_schema(self) -> None:
        result = RESPONSE_SCHEMA.to_arrow_schema()
        assert isinstance(result, pa.Schema)

    def test_to_arrow_schema_field_count(self) -> None:
        arrow = RESPONSE_SCHEMA.to_arrow_schema()
        expected_count = len(self.EXPECTED_REQUEST_PREFIX_FIELDS) + len(self.EXPECTED_OWN_FIELDS)
        assert len(arrow) == expected_count

    def test_unnested_request_fields_present(self) -> None:
        names = _field_names(RESPONSE_SCHEMA)
        for field_name in self.EXPECTED_REQUEST_PREFIX_FIELDS:
            assert field_name in names

    def test_request_pkl_not_unnested(self) -> None:
        names = _field_names(RESPONSE_SCHEMA)
        assert "request__pkl" not in names

    def test_hash_field_type(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert fields["hash"].type == pa.int64()
        assert fields["hash"].nullable is False

    def test_public_hash_field_type(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert fields["public_hash"].type == pa.int64()
        assert fields["public_hash"].nullable is False

    def test_status_code_field_type(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert fields["status_code"].type == pa.int32()
        assert fields["status_code"].nullable is False

    def test_headers_field_is_map(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert isinstance(fields["headers"].type, pa.MapType)
        assert fields["headers"].type.key_type == pa.string()
        assert fields["headers"].type.item_type == pa.string()
        assert fields["headers"].nullable is False

    def test_tags_field_is_nullable_map(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert isinstance(fields["tags"].type, pa.MapType)
        assert fields["tags"].nullable is True

    def test_body_field_type(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert fields["body"].type == pa.large_binary()
        assert fields["body"].nullable is True

    def test_body_size_field_type(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert fields["body_size"].type == pa.int64()
        assert fields["body_size"].nullable is False

    def test_body_hash_field_type(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert fields["body_hash"].type == pa.int64()
        assert fields["body_hash"].nullable is False

    def test_received_at_field_type(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert fields["received_at"].type == pa.timestamp("us", "UTC")
        assert fields["received_at"].nullable is False

    def test_partition_key_field_type(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert fields["partition_key"].type == pa.int64()
        assert fields["partition_key"].nullable is False

    def test_receiver_field_is_nullable_struct(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert isinstance(fields["receiver"].type, pa.StructType)
        assert fields["receiver"].nullable is True

    def test_pkl_field_type(self) -> None:
        fields = _arrow_field_map(RESPONSE_SCHEMA)
        assert fields["_pkl"].type == pa.large_binary()
        assert fields["_pkl"].nullable is True


# -- TestRequestUrlStruct ----------------------------------------------------

class TestRequestUrlStruct:

    def test_request_url_struct_is_url_struct(self) -> None:
        assert REQUEST_URL_STRUCT is URL_STRUCT

    def test_request_url_struct_is_pa_struct_type(self) -> None:
        assert isinstance(REQUEST_URL_STRUCT, pa.StructType)

    def test_url_struct_has_expected_subfields(self) -> None:
        subfield_names = [REQUEST_URL_STRUCT.field(i).name for i in range(REQUEST_URL_STRUCT.num_fields)]
        assert subfield_names == ["scheme", "userinfo", "host", "port", "path", "query", "fragment"]

    def test_url_field_in_request_schema_uses_url_struct(self) -> None:
        arrow = REQUEST_SCHEMA.to_arrow_schema()
        url_type = arrow.field("url").type
        assert isinstance(url_type, pa.StructType)
        # The Arrow struct type's fields should match URL_STRUCT's fields.
        assert url_type.num_fields == URL_STRUCT.num_fields
        for i in range(url_type.num_fields):
            assert url_type.field(i).name == URL_STRUCT.field(i).name
