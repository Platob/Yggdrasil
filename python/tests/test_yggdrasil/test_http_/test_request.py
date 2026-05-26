"""Unit tests for yggdrasil.http_.request.HTTPRequest."""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.io.headers import Headers
from yggdrasil.url import URL


def _make(
    method: str = "GET",
    url: str = "https://example.com/api",
    headers: dict | None = None,
    tags: dict | None = None,
    body: bytes | None = None,
) -> HTTPRequest:
    return HTTPRequest.prepare(
        method,
        url,
        headers=headers,
        tags=tags,
        body=body,
    )


# -- TestConstruction ---------------------------------------------------------

class TestConstruction:

    def test_prepare_get(self):
        r = HTTPRequest.prepare("GET", "https://example.com/api")
        assert r.method == "GET"
        assert r.url.host == "example.com"
        assert r.url.path == "/api"
        assert r.buffer is None

    def test_prepare_post_with_body(self):
        r = HTTPRequest.prepare("POST", "https://example.com/api", body=b"payload")
        assert r.method == "POST"
        assert r.buffer is not None
        assert r.buffer.to_bytes() == b"payload"
        assert r.content_length == 7

    def test_prepare_with_json(self):
        r = HTTPRequest.prepare("POST", "https://example.com/api", json={"k": "v"})
        assert r.headers.get("Content-Type") == "application/json"
        assert r.buffer.to_bytes() == b'{"k": "v"}'

    def test_prepare_with_headers(self):
        r = HTTPRequest.prepare(
            "GET",
            "https://example.com",
            headers={"X-Custom": "value"},
        )
        assert r.headers["X-Custom"] == "value"

    def test_prepare_with_tags(self):
        r = HTTPRequest.prepare("GET", "https://example.com", tags={"env": "prod"})
        assert r.tags == {"env": "prod"}

    def test_from_mapping_minimal(self):
        r = HTTPRequest.from_mapping({"url": "https://example.com/path"})
        assert r.method == "GET"
        assert r.url.host == "example.com"
        assert r.url.path == "/path"

    def test_from_mapping_with_method_and_body(self):
        r = HTTPRequest.from_mapping({
            "method": "PUT",
            "url": "https://api.com/items",
            "body": b"data",
        })
        assert r.method == "PUT"
        assert r.buffer is not None
        assert r.buffer.to_bytes() == b"data"

    def test_from_mapping_with_headers_and_tags(self):
        r = HTTPRequest.from_mapping({
            "url": "https://example.com",
            "headers": {"Accept": "text/html"},
            "tags": {"source": "test"},
        })
        assert r.headers["Accept"] == "text/html"
        assert r.tags == {"source": "test"}

    def test_from_string(self):
        r = HTTPRequest.from_("https://example.com/endpoint")
        assert r.method == "GET"
        assert r.url.host == "example.com"

    def test_default_method_is_get(self):
        r = HTTPRequest(
            method="",
            url="https://example.com",
            headers={},
            tags={},
            buffer=None,
            sent_at=None,
        )
        assert r.method == "GET"

    def test_headers_coerced_to_headers_type(self):
        r = _make(headers={"Foo": "bar"})
        assert isinstance(r.headers, Headers)

    def test_body_property_is_alias_for_buffer(self):
        r = _make(body=b"hello")
        assert r.body is r.buffer

    def test_holder_property_is_alias_for_buffer(self):
        r = _make(body=b"hello")
        assert r.holder is r.buffer

    def test_content_length_no_body(self):
        r = _make()
        assert r.content_length == 0

    def test_content_length_with_body(self):
        r = _make(body=b"12345")
        assert r.content_length == 5


# -- TestHashComputation ------------------------------------------------------

class TestHashComputation:

    def test_hash_is_int(self):
        r = _make()
        assert isinstance(r.hash, int)

    def test_public_hash_is_int(self):
        r = _make()
        assert isinstance(r.public_hash, int)

    def test_hash_stability(self):
        r1 = _make()
        r2 = _make()
        assert r1.hash == r2.hash
        assert r1.public_hash == r2.public_hash

    def test_hash_differs_by_method(self):
        r1 = _make(method="GET")
        r2 = _make(method="POST")
        assert r1.hash != r2.hash

    def test_hash_differs_by_url(self):
        r1 = _make(url="https://a.com/x")
        r2 = _make(url="https://b.com/x")
        assert r1.hash != r2.hash

    def test_hash_differs_by_body(self):
        r1 = _make(method="POST", body=b"aaa")
        r2 = _make(method="POST", body=b"bbb")
        assert r1.hash != r2.hash

    def test_body_hash_zero_when_no_body(self):
        r = _make()
        assert r.body_hash == 0

    def test_body_hash_nonzero_with_body(self):
        r = _make(body=b"content")
        assert r.body_hash != 0

    def test_body_size_zero_when_no_body(self):
        r = _make()
        assert r.body_size == 0

    def test_body_size_matches_body(self):
        r = _make(body=b"hello world")
        assert r.body_size == 11

    def test_public_hash_strips_userinfo(self):
        r = _make(url="https://user:pass@example.com/api")
        assert r.hash != r.public_hash

    def test_private_url_hash_includes_method(self):
        r1 = _make(method="GET")
        r2 = _make(method="POST")
        assert r1.private_url_hash != r2.private_url_hash

    def test_hash_cached(self):
        r = _make()
        h1 = r.hash
        h2 = r.hash
        assert h1 == h2


# -- TestUrlHandling ----------------------------------------------------------

class TestUrlHandling:

    def test_url_parsed(self):
        r = _make(url="https://example.com:8080/path?q=1#frag")
        assert r.url.scheme == "https"
        assert r.url.host == "example.com"
        assert r.url.port == 8080
        assert r.url.path == "/path"
        assert r.url.fragment == "frag"

    def test_query_normalization(self):
        r = _make(url="https://example.com/api?z=2&a=1")
        # URL normalizes query params alphabetically
        assert r.url.query_items() == (("a", "1"), ("z", "2"))

    def test_url_to_string(self):
        r = _make(url="https://example.com/api")
        assert "example.com" in r.url.to_string()
        assert "/api" in r.url.to_string()

    def test_anonymize_removes_userinfo(self):
        r = _make(url="https://user:pass@example.com/api")
        anon = r.anonymize(mode="remove")
        assert anon.url.userinfo is None
        assert "user" not in anon.url.to_string()

    def test_anonymize_redacts_userinfo(self):
        r = _make(url="https://user:pass@example.com/api")
        anon = r.anonymize(mode="redact")
        assert anon.url.userinfo == "<redacted>"

    def test_url_is_url_type(self):
        r = _make()
        assert isinstance(r.url, URL)

    def test_default_port_normalized_away(self):
        r = _make(url="https://example.com:443/api")
        assert r.url.port is None

    def test_non_default_port_preserved(self):
        r = _make(url="https://example.com:9090/api")
        assert r.url.port == 9090


# -- TestPartitionValues ------------------------------------------------------

class TestPartitionValues:

    def test_default_partition_values(self):
        r = _make(url="https://example.com/api/items")
        pv = r.partition_values()
        assert pv == {"host": "example.com", "path": "/api/items"}

    def test_partition_key_is_int(self):
        r = _make()
        assert isinstance(r.partition_key, int)

    def test_partition_key_same_endpoint(self):
        r1 = _make(method="GET", url="https://example.com/api")
        r2 = _make(method="POST", url="https://example.com/api")
        # Same host+path, different method => same partition
        assert r1.partition_key == r2.partition_key

    def test_partition_key_different_host(self):
        r1 = _make(url="https://a.com/api")
        r2 = _make(url="https://b.com/api")
        assert r1.partition_key != r2.partition_key

    def test_partition_key_different_path(self):
        r1 = _make(url="https://example.com/api/v1")
        r2 = _make(url="https://example.com/api/v2")
        assert r1.partition_key != r2.partition_key

    def test_partition_key_stable(self):
        r = _make(url="https://example.com/api")
        assert r.partition_key == r.partition_key


# -- TestArrowProjection ------------------------------------------------------

class TestArrowProjection:

    def test_arrow_values_has_expected_keys(self):
        r = _make()
        vals = r.arrow_values
        expected = {
            "hash", "public_hash", "method", "url", "sender",
            "private_url_hash", "public_url_hash", "headers", "tags",
            "body", "body_size", "body_hash", "sent_at", "partition_key",
            "_pkl",
        }
        assert set(vals.keys()) == expected

    def test_arrow_values_method(self):
        r = _make(method="DELETE")
        assert r.arrow_values["method"] == "DELETE"

    def test_arrow_values_body_none_when_no_body(self):
        r = _make()
        assert r.arrow_values["body"] is None
        assert r.arrow_values["body_size"] == 0

    def test_arrow_values_body_bytes(self):
        r = _make(body=b"payload")
        assert r.arrow_values["body"] == b"payload"
        assert r.arrow_values["body_size"] == 7

    def test_arrow_values_url_is_struct(self):
        r = _make(url="https://example.com/path")
        url_struct = r.arrow_values["url"]
        assert isinstance(url_struct, dict)
        assert url_struct["host"] == "example.com"
        assert url_struct["path"] == "/path"
        assert url_struct["scheme"] == "https"

    def test_arrow_values_tags_include_query_params(self):
        r = _make(url="https://example.com/api?foo=bar", tags={"env": "prod"})
        tags = r.arrow_values["tags"]
        assert tags["foo"] == "bar"
        assert tags["env"] == "prod"

    def test_to_arrow_batch_shape(self):
        r = _make()
        batch = r.to_arrow_batch()
        assert isinstance(batch, pa.RecordBatch)
        assert batch.num_rows == 1

    def test_values_to_arrow_batch_multiple(self):
        r1 = _make(method="GET", url="https://a.com/1")
        r2 = _make(method="POST", url="https://b.com/2", body=b"data")
        batch = HTTPRequest.values_to_arrow_batch([r1, r2])
        assert batch.num_rows == 2

    def test_arrow_round_trip_no_body(self):
        r = _make(method="GET", url="https://example.com/api")
        batch = r.to_arrow_batch()
        restored = list(HTTPRequest.from_arrow(batch))
        assert len(restored) == 1
        r2 = restored[0]
        assert r2.method == "GET"
        assert r2.url.host == "example.com"
        assert r2.url.path == "/api"

    def test_arrow_round_trip_with_body(self):
        r = _make(method="POST", url="https://example.com/api", body=b"test-body")
        batch = r.to_arrow_batch()
        restored = list(HTTPRequest.from_arrow(batch))
        assert len(restored) == 1
        r2 = restored[0]
        assert r2.method == "POST"
        assert r2.buffer.to_bytes() == b"test-body"
        assert r2.body_size == 9

    def test_arrow_round_trip_multiple(self):
        r1 = _make(method="GET", url="https://a.com/x")
        r2 = _make(method="PUT", url="https://b.com/y", body=b"update")
        batch = HTTPRequest.values_to_arrow_batch([r1, r2])
        restored = list(HTTPRequest.from_arrow(batch))
        assert len(restored) == 2
        assert restored[0].method == "GET"
        assert restored[0].url.host == "a.com"
        assert restored[1].method == "PUT"
        assert restored[1].buffer.to_bytes() == b"update"

    def test_from_arrow_table(self):
        r = _make()
        table = r.to_arrow_table()
        assert isinstance(table, pa.Table)
        restored = list(HTTPRequest.from_arrow(table))
        assert len(restored) == 1
        assert restored[0].method == r.method


# -- TestCopy -----------------------------------------------------------------

class TestCopy:

    def test_copy_preserves_all_fields(self):
        r = _make(
            method="POST",
            url="https://example.com/api",
            headers={"Accept": "text/html"},
            tags={"env": "prod"},
            body=b"body-data",
        )
        c = r.copy()
        assert c.method == r.method
        assert c.url == r.url
        assert c.tags == r.tags
        assert c.buffer is r.buffer
        assert c.hash == r.hash

    def test_copy_with_method_override(self):
        r = _make(method="GET")
        c = r.copy(method="DELETE")
        assert c.method == "DELETE"
        assert r.method == "GET"

    def test_copy_with_url_override(self):
        r = _make(url="https://example.com/old")
        c = r.copy(url="https://example.com/new")
        assert c.url.path == "/new"
        assert r.url.path == "/old"

    def test_copy_with_headers_override(self):
        r = _make(headers={"X-Old": "1"})
        c = r.copy(headers={"X-New": "2"})
        assert "X-New" in c.headers
        assert "X-Old" not in c.headers

    def test_copy_with_tags_override(self):
        r = _make(tags={"a": "1"})
        c = r.copy(tags={"b": "2"})
        assert c.tags == {"b": "2"}

    def test_copy_does_not_mutate_original(self):
        r = _make()
        original_hash = r.hash
        _ = r.copy(method="DELETE")
        assert r.hash == original_hash
        assert r.method == "GET"

    def test_copy_returns_same_class(self):
        r = _make()
        c = r.copy()
        assert type(c) is type(r)


# -- TestEquality -------------------------------------------------------------

class TestEquality:

    def test_identical_requests_same_hash(self):
        r1 = _make(method="GET", url="https://example.com/api")
        r2 = _make(method="GET", url="https://example.com/api")
        assert r1.hash == r2.hash

    def test_different_method_different_hash(self):
        r1 = _make(method="GET")
        r2 = _make(method="POST")
        assert r1.hash != r2.hash

    def test_different_url_different_hash(self):
        r1 = _make(url="https://a.com/x")
        r2 = _make(url="https://b.com/y")
        assert r1.hash != r2.hash

    def test_different_body_different_hash(self):
        r1 = _make(method="POST", body=b"one")
        r2 = _make(method="POST", body=b"two")
        assert r1.hash != r2.hash

    def test_no_body_vs_body_different_hash(self):
        r1 = _make(method="POST")
        r2 = _make(method="POST", body=b"x")
        assert r1.hash != r2.hash

    def test_public_hash_stable_for_same_request(self):
        r1 = _make()
        r2 = _make()
        assert r1.public_hash == r2.public_hash


# -- TestRepr -----------------------------------------------------------------

class TestRepr:

    def test_repr_includes_method(self):
        r = _make(method="GET")
        assert "GET" in repr(r)

    def test_repr_includes_url(self):
        r = _make(url="https://example.com/api")
        assert "example.com" in repr(r)

    def test_repr_includes_class_name(self):
        r = _make()
        assert "HTTPRequest" in repr(r)

    def test_str_returns_url_string(self):
        r = _make(url="https://example.com/api")
        assert str(r) == r.url.to_string()

    def test_repr_format(self):
        r = _make(method="POST", url="https://example.com/items")
        assert repr(r).startswith("HTTPRequest<POST ")
        assert repr(r).endswith(">")
