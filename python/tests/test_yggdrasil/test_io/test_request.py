# tests/io/test_request.py
from __future__ import annotations

import datetime
from typing import Any

import pytest

from yggdrasil.io.enums import MimeType
from yggdrasil.io.request import PreparedRequest, REQUEST_ARROW_SCHEMA


def _rb_row(batch) -> dict[str, Any]:
    return {name: batch.column(name)[0].as_py() for name in batch.schema.names}


def test_parse_dict_minimal_defaults() -> None:
    req = PreparedRequest.parse_dict(
        {
            "request_url_str": "https://example.com/a?x=1",
        }
    )

    assert req.method == "GET"
    assert req.url.to_string() == "https://example.com/a?x=1"
    assert req.headers == {}
    assert req.tags == {}
    assert req.buffer is None
    assert req.sent_at_timestamp == 0


def test_parse_dict_from_flattened_url_parts() -> None:
    req = PreparedRequest.parse_dict(
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
        PreparedRequest.parse_dict({"request_method": "GET"})


def test_parse_dict_headers_from_mapping() -> None:
    req = PreparedRequest.parse_dict(
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
    req = PreparedRequest.parse_dict(
        {
            "request_url_str": "https://example.com",
            "request_content_type": "application/json",
            "request_content_length": 12,
            "request_content_encoding": "gzip",
            "request_transfer_encoding": "chunked",
            "request_x_request_id": "rid-1",
            "request_x_correlation_id": "cid-1",
        }
    )

    assert req.headers["Content-Type"] == "application/json"
    assert req.headers["Content-Length"] == "12"
    assert req.headers["Content-Encoding"] == "gzip"
    assert req.headers["Transfer-Encoding"] == "chunked"
    assert req.headers["X-Request-ID"] == "rid-1"
    assert req.headers["X-Correlation-ID"] == "cid-1"


def test_parse_dict_tags_and_buffer() -> None:
    req = PreparedRequest.parse_dict(
        {
            "request_url_str": "https://example.com",
            "request_tags": {"a": 1, "b": "two"},
            "request_body": b"hello",
        }
    )

    assert req.tags == {"a": "1", "b": "two"}
    assert req.buffer is not None
    assert req.buffer.to_bytes() == b"hello"


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
    assert req.headers["Content-Type"] == MimeType.JSON.value
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
    assert req.headers["Content-Type"] == MimeType.JSON.value
    assert "Content-Encoding" in req.headers
    assert req.headers["Content-Length"] == str(req.buffer.size)


def test_prepare_to_send_without_sniff_keeps_timestamp_zero() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com",
    )

    out = req.prepare_to_send(normalize=False)

    assert out.sent_at_timestamp == 0


def test_prepare_to_send_with_sniff_sets_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyUser:
        product = None
        product_version = None
        email = None
        hostname = None
        url = None
        git_url = None

    monkeypatch.setattr("yggdrasil.environ.UserInfo.current", lambda: DummyUser())

    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com",
    )

    out = req.prepare_to_send(normalize=True)

    assert out.sent_at_timestamp > 0


def test_prepare_to_send_applies_before_send(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyUser:
        product = None
        product_version = None
        email = None
        hostname = None
        url = None
        git_url = None

    monkeypatch.setattr("yggdrasil.environ.UserInfo.current", lambda: DummyUser())

    def before_send(req: PreparedRequest) -> PreparedRequest:
        req.headers["X-Test-Before-Send"] = "1"
        return req

    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com",
        before_send=before_send,
    )

    out = req.prepare_to_send(normalize=True)

    assert out.headers["X-Test-Before-Send"] == "1"


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


def test_parse_query_params_handles_empty_and_bare_keys() -> None:
    assert PreparedRequest._parse_query_params(None) == {}
    assert PreparedRequest._parse_query_params("") == {}
    assert PreparedRequest._parse_query_params("a=1&b&c=3") == {
        "a": "1",
        "b": "",
        "c": "3",
    }


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
    req.sent_at_timestamp = 123456789

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
    req.sent_at_timestamp = 42

    row = _rb_row(req.to_arrow_batch())

    assert row["request_method"] == "POST"
    assert row["request_url_str"] == "https://example.com/api?a=1&shared=url"
    assert row["request_url_scheme"] == "https"
    assert row["request_url_host"] == "example.com"
    assert row["request_url_path"] == "/api"
    assert row["request_url_query"] == "a=1&shared=url"

    assert row["request_host"] == "example.com"
    assert row["request_user_agent"] == "pytest"
    assert row["request_accept"] == "application/json"
    assert row["request_accept_encoding"] == "gzip"
    assert row["request_accept_language"] == "en"
    assert row["request_content_type"] == "application/json"
    assert row["request_content_length"] == 11
    assert row["request_content_encoding"] == "gzip"
    assert row["request_transfer_encoding"] == "chunked"
    assert row["request_x_request_id"] == "rid-1"
    assert row["request_x_correlation_id"] == "cid-1"

    assert dict((k, v) for k, v in row["request_headers"] if k in ["X-Other"]) == {"X-Other": "keep-me"}
    assert dict(row["request_tags"]) == {
        "a": "1",
        "shared": "explicit",
        "t": "2",
    }
    assert row["request_body"] == b"hello world"
    assert row["request_body_hash"] is not None
    assert len(row["request_body_hash"]) == 32
    assert row["request_sent_at"] == datetime.datetime(1970, 1, 1, 0, 0, 0, 42, tzinfo=datetime.timezone.utc)
    assert row["request_sent_at_epoch"] == 42


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


def test_parse_accepts_json_string() -> None:
    req = PreparedRequest.parse(
        '{"request_method":"PUT","request_url_str":"https://example.com/x"}'
    )

    assert req.method == "PUT"
    assert req.url.to_string() == "https://example.com/x"


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
        sent_at_timestamp=123,
    )

    assert req2.method == "POST"
    assert req2.url.to_string() == "https://example.com/b"
    assert req2.headers == {"X-B": "2"}
    assert req2.tags == {"t": "2"}
    assert req2.sent_at_timestamp == 123


def test_body_property_alias() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        body=b"abc",
    )

    assert req.body is req.buffer
    assert req.body is not None
    assert req.body.to_bytes() == b"abc"


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