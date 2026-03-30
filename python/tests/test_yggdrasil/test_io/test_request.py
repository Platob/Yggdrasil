from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from yggdrasil.io import MediaType, MimeTypes
from yggdrasil.io.headers import DEFAULT_HOSTNAME
from yggdrasil.io.request import PreparedRequest, REQUEST_ARROW_SCHEMA


def _rb_row(batch) -> dict[str, Any]:
    return {name: batch.column(name)[0].as_py() for name in batch.schema.names}


def test_parse_dict_minimal_defaults() -> None:
    req = PreparedRequest.parse_mapping(
        {
            "request_url_str": "https://example.com/a?x=1",
        }
    )

    assert req.method == "GET"
    assert req.url.to_string() == "https://example.com/a?x=1"
    assert req.headers == {}
    assert req.tags == {}
    assert req.buffer is None
    assert req.sent_at == dt.datetime(1970, 1, 1, 0, 0, tzinfo=dt.timezone.utc)


def test_parse_dict_from_flattened_url_parts() -> None:
    req = PreparedRequest.parse_mapping(
        {
            "request_method": "POST",
            "request_url_scheme": "https",
            "request_url_host": "example.com",
            "request_url_port": 443,
            "request_url_path": "/api/v1/data",
            "request_url_query": "a=1&b=2",
            "request_url_fragment": "frag",
        }
    )

    assert req.method == "POST"
    assert req.url.scheme == "https"
    assert req.url.host == "example.com"
    assert req.url.port is None
    assert req.url.path == "/api/v1/data"
    assert req.url.query == "a=1&b=2"
    assert req.url.fragment == "frag"


def test_parse_dict_missing_url_raises() -> None:
    with pytest.raises(ValueError, match="missing url/url_str/request_url_str"):
        PreparedRequest.parse_mapping({"request_method": "GET"})


def test_parse_dict_headers_from_mapping() -> None:
    req = PreparedRequest.parse_mapping(
        {
            "request_url_str": "https://example.com",
            "request_headers": {
                "Content-Type": "application/json",
                "X-Test": "yes",
            },
        }
    )

    assert req.headers["Content-Type"] == "application/json"
    assert req.headers["X-Test"] == "yes"


def test_parse_dict_headers_from_promoted_fields() -> None:
    req = PreparedRequest.parse_mapping(
        {
            "request_url_str": "https://example.com",
            "request_content_type": "application/json",
            "request_content_length": 12,
            "request_content_encoding": "gzip",
            "request_transfer_encoding": "chunked",
        }
    )

    assert req.headers["Content-Type"] == "application/json"
    assert req.headers["Content-Length"] == "12"
    assert req.headers["Content-Encoding"] == "gzip"
    assert req.headers["Transfer-Encoding"] == "chunked"


def test_parse_dict_respects_custom_prefix() -> None:
    req = PreparedRequest.parse_mapping(
        {
            "foo_url_str": "https://example.com",
            "foo_headers": {"X-Test": "1"},
            "foo_tags": {"a": 1},
        },
        prefix="foo_",
    )

    assert req.url.to_string() == "https://example.com/"
    assert req.headers == {"X-Test": "1"}
    assert req.tags == {"a": "1"}


def test_parse_dict_tags_and_buffer() -> None:
    req = PreparedRequest.parse_mapping(
        {
            "request_url_str": "https://example.com",
            "request_tags": {"a": 1, "b": "two"},
            "request_body": b"hello",
        }
    )

    assert req.tags == {"a": "1", "b": "two"}
    assert req.buffer is not None
    assert req.buffer.to_bytes() == b"hello"


def test_parse_dict_sent_at_accepts_datetime() -> None:
    ts = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

    req = PreparedRequest.parse_mapping(
        {
            "request_url_str": "https://example.com",
            "request_sent_at": ts,
        }
    )

    assert req.sent_at == ts


def test_copy_reuses_buffer_by_default() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        body=b"abc",
    )

    req2 = req.copy()

    assert req2.method == req.method
    assert req2.url.to_string() == req.url.to_string()
    assert req2.headers == req.headers
    assert req2.buffer is req.buffer


def test_copy_can_deep_copy_buffer() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        body=b"abc",
    )

    req2 = req.copy(copy_buffer=True)

    assert req2.buffer is not req.buffer
    assert req2.buffer is not None
    assert req.buffer is not None
    assert req2.buffer.to_bytes() == req.buffer.to_bytes()


def test_copy_can_override_fields() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com/a",
        headers={"X-A": "1"},
        tags={"t": "1"},
    )

    req2 = req.copy(
        method="POST",
        url="https://example.com/b",
        headers={"X-B": "2"},
        tags={"t": "2"},
        sent_at=123,
    )

    assert req2.method == "POST"
    assert req2.url.to_string() == "https://example.com/b"
    assert req2.headers == {"X-B": "2"}
    assert req2.tags == {"t": "2"}
    assert req2.sent_at == dt.datetime(1970, 1, 1, 0, 2, 3, tzinfo=dt.timezone.utc)


def test_prepare_with_raw_body_sets_content_length() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/upload",
        body=b"payload",
    )

    assert req.method == "POST"
    assert req.buffer is not None
    assert req.buffer.to_bytes() == b"payload"
    assert req.headers["Content-Length"] == str(len(b"payload"))


def test_prepare_with_json_sets_json_content_type() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/items",
        json={"x": 1},
    )

    assert req.buffer is not None
    assert req.headers["Content-Type"] == MimeTypes.JSON.value
    assert req.headers["Content-Length"] == str(req.buffer.size)
    assert b'"x": 1' in req.buffer.to_bytes() or b'"x":1' in req.buffer.to_bytes()


def test_prepare_with_json_can_compress_when_threshold_exceeded() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/items",
        json={"data": "x" * 5000},
        compress_threshold=1,
    )

    assert req.buffer is not None
    assert req.headers["Content-Type"] == MimeTypes.JSON.value
    assert "Content-Encoding" in req.headers
    assert req.headers["Content-Length"] == str(req.buffer.size)


def test_prepare_normalizes_headers() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        headers={"content-type": "application/custom"},
        body=b"abc",
        normalize=True,
    )

    assert "Content-Type" in req.headers
    assert req.headers["Content-Type"] == "application/custom"
    assert req.headers["Content-Length"] == "3"


def test_prepare_to_send_applies_before_send() -> None:
    def before_send(req: PreparedRequest) -> PreparedRequest:
        req.headers["X-Test-Before-Send"] = "1"
        return req

    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com",
        before_send=before_send,
    )

    out = req.prepare_to_send(sent_at=1, headers=None)

    assert out.headers["X-Test-Before-Send"] == "1"
    assert out.sent_at == dt.datetime(1970, 1, 1, 0, 0, 1, tzinfo=dt.timezone.utc)


def test_prepare_to_send_merges_headers() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com",
    )

    out = req.prepare_to_send(
        sent_at=5,
        headers={"X-Injected": "1"},
    )

    assert out.headers["X-Injected"] == "1"
    assert out.sent_at == dt.datetime(1970, 1, 1, 0, 0, 5, tzinfo=dt.timezone.utc)


def test_body_property_alias() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        body=b"abc",
    )

    assert req.body is req.buffer
    assert req.body is not None
    assert req.body.to_bytes() == b"abc"


def test_authorization_and_x_api_key_properties() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com",
    )

    req.authorization = "Bearer abc"
    req.x_api_key = "secret"

    assert req.authorization == "Bearer abc"
    assert req.x_api_key == "secret"

    req.authorization = None
    req.x_api_key = None

    assert req.authorization is None
    assert req.x_api_key is None


def test_accept_media_type_roundtrip() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com",
    )

    req.accept_media_type = MediaType(MimeTypes.JSON, None)

    assert req.headers["Accept"] == MimeTypes.JSON.value
    assert req.accept_media_type.mime_type == MimeTypes.JSON


def test_update_headers_and_tags() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com?a=1",
        headers={"X-A": "1"},
        tags={"t1": "1"},
    )

    req.update_headers({"X-B": "2"}, normalize=False)
    req.update_tags({"t2": "2"})

    assert req.headers["X-A"] == "1"
    assert req.headers["X-B"] == "2"
    assert req.tags["t1"] == "1"
    assert req.tags["t2"] == "2"


def test_anonymize_redacts_url_and_sensitive_headers() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://user:pass@example.com/path?token=secret&ok=1",
        headers={
            "Authorization": "Bearer super-secret",
            "X-API-Key": "abcdef",
        },
    )

    anon = req.anonymize(mode="redact")

    assert anon is not req
    assert "Authorization" in anon.headers
    assert "X-API-Key" in anon.headers
    assert "<redacted>" in anon.headers["Authorization"] or anon.headers["Authorization"].startswith("Bearer ")
    assert anon.headers["X-API-Key"] == "<redacted>"


def test_arrow_values_contains_expected_projection() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1",
        headers={"User-Agent": "pytest"},
        body=b"hello",
        tags={"t": "2"},
    )

    values = req.arrow_values

    assert values["request_method"] == "POST"
    assert values["request_url_str"] == "https://example.com/api?a=1"
    assert values["request_url_host"] == "example.com"
    assert values["request_url_path"] == "/api"
    assert values["request_url_query"] == "a=1"
    assert values["request_user_agent"] == "pytest"
    assert values["request_content_length"] == 5
    assert values["request_body"] == b"hello"
    assert values["request_body_hash"] is not None
    assert values["request_tags"] == {"a": "1", "t": "2"}
    assert values["request_sent_at"] == dt.datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=dt.timezone.utc)


def test_match_value_supports_schema_fields() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://user@example.com/api?a=1#frag",
        headers={"Content-Type": "application/json"},
        body=b"hello",
    )

    assert req.match_value("request_method") == "POST"
    assert req.match_value("request_url_str") == "https://user@example.com/api?a=1#frag"
    assert req.match_value("request_url_scheme") == "https"
    assert req.match_value("request_url_userinfo") == "user"
    assert req.match_value("request_url_host") == "example.com"
    assert req.match_value("request_url_path") == "/api"
    assert req.match_value("request_url_query") == "a=1"
    assert req.match_value("request_url_fragment") == "frag"
    assert req.match_value("request_content_type") == "application/json"
    assert req.match_value("request_content_length") == 5
    assert req.match_value("request_body") == b"hello"
    assert req.match_value("request_body_hash") is not None


def test_match_value_invalid_key_raises() -> None:
    req = PreparedRequest.prepare(method="GET", url="https://example.com")

    with pytest.raises(ValueError, match="Unsupported request match key"):
        req.match_value("not_a_real_key")


def test_match_tuple() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?q=1",
        body=b"hello",
    )

    out = req.match_tuple(["request_method", "request_url_str", "request_body_hash"])

    assert out[0] == "POST"
    assert out[1] == "https://example.com/api?q=1"
    assert out[2] is not None


def test_to_arrow_batch_matches_schema() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?q=1",
        headers={
            "X-Test": "yes",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "Location": "/foo",
            "ETag": '"etag-1"',
            "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
        },
        body=b"hello world",
        tags={"explicit": "tag"},
    )
    req.sent_at = 123456789

    rb = req.to_arrow_batch()

    assert rb.schema == REQUEST_ARROW_SCHEMA
    assert rb.num_rows == 1


def test_to_arrow_batch_promotes_headers_and_keeps_remaining() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1&shared=url",
        headers={
            "Host": "example.com",
            "User-Agent": "pytest",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Accept-Language": "en",
            "Content-Type": "application/json",
            "Content-Length": "11",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "X-Request-ID": "rid-1",
            "X-Correlation-ID": "cid-1",
            "X-Other": "keep-me",
        },
        body=b"hello world",
        tags={"shared": "explicit", "t": "2"},
    )
    req.sent_at = 42

    row = _rb_row(req.to_arrow_batch())

    assert row["request_method"] == "POST"
    assert row["request_url_str"] == "https://example.com/api?a=1&shared=url"
    assert row["request_url_scheme"] == "https"
    assert row["request_url_host"] == "example.com"
    assert row["request_url_path"] == "/api"
    assert row["request_url_query"] == "a=1&shared=url"

    assert row["request_host"] == DEFAULT_HOSTNAME
    assert row["request_user_agent"] == "pytest"
    assert row["request_accept"] == "application/json"
    assert row["request_accept_encoding"] == "gzip"
    assert row["request_accept_language"] == "en"
    assert row["request_content_type"] == "application/json"
    assert row["request_content_length"] == 11
    assert row["request_content_encoding"] == "gzip"
    assert row["request_transfer_encoding"] == "chunked"

    headers_map = dict(row["request_headers"])
    assert headers_map["X-Other"] == "keep-me"

    assert dict(row["request_tags"]) == {
        "a": "1",
        "shared": "explicit",
        "t": "2",
    }
    assert row["request_body"] == b"hello world"
    assert row["request_body_hash"] == req.buffer.xxh3_int64()  # type: ignore[union-attr]
    assert row["request_sent_at"] == dt.datetime(1970, 1, 1, 0, 0, 0, 42, tzinfo=dt.timezone.utc)


def test_to_arrow_batch_uses_transfer_encoding_not_content_encoding() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        headers={
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
        },
        body=b"abc",
    )

    row = _rb_row(req.to_arrow_batch())

    assert row["request_content_encoding"] == "gzip"
    assert row["request_transfer_encoding"] == "chunked"


def test_to_arrow_batch_without_body_has_null_body_and_hash() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com",
    )

    row = _rb_row(req.to_arrow_batch())

    assert row["request_body"] is None
    assert row["request_body_hash"] is None


def test_from_arrow_roundtrip_record_batch() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1",
        headers={
            "User-Agent": "pytest",
            "Content-Type": "application/json",
            "X-Other": "keep",
        },
        body=b"hello",
        tags={"t": "2"},
    )
    req.sent_at = dt.datetime(1970, 1, 1, 0, 2, 3, tzinfo=dt.timezone.utc)

    rb = req.to_arrow_batch()
    out = list(PreparedRequest.from_arrow(rb))

    assert len(out) == 1
    rebuilt = out[0]

    assert rebuilt.method == req.method
    assert rebuilt.url.to_string() == req.url.to_string()
    assert rebuilt.buffer is not None
    assert rebuilt.buffer.to_bytes() == b"hello"
    assert rebuilt.tags == {"a": "1", "t": "2"}
    assert rebuilt.sent_at == dt.datetime(1970, 1, 1, 0, 2, 3, tzinfo=dt.timezone.utc)
    assert rebuilt.headers["User-Agent"] == "pytest"
    assert rebuilt.headers["Content-Type"] == "application/json"


def test_from_arrow_roundtrip_table() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com/path?q=1",
        headers={"Accept": "application/json"},
    )
    req.sent_at = dt.datetime(1970, 1, 1, 0, 0, 7, tzinfo=dt.timezone.utc)

    table = req.to_arrow_table()
    out = list(PreparedRequest.from_arrow(table))

    assert len(out) == 1
    rebuilt = out[0]

    assert rebuilt.method == "GET"
    assert rebuilt.url.to_string() == "https://example.com/path?q=1"
    assert rebuilt.sent_at == dt.datetime(1970, 1, 1, 0, 0, 7, tzinfo=dt.timezone.utc)