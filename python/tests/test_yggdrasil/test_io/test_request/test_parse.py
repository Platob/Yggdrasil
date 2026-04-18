"""`PreparedRequest.parse_mapping` / `parse_str` — flattened-dict ingestion."""

from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.io.request import PreparedRequest


def test_parse_mapping_minimal_defaults() -> None:
    req = PreparedRequest.parse_mapping({"request_url_str": "https://example.com/a?x=1"})

    assert req.method == "GET"
    assert req.url.to_string() == "https://example.com/a?x=1"
    assert req.headers == {}
    assert req.tags == {}
    assert req.buffer is None
    assert req.sent_at == dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def test_parse_mapping_from_flattened_url_parts() -> None:
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
    # Default https port is dropped.
    assert req.url.port is None
    assert req.url.path == "/api/v1/data"
    assert req.url.query == "a=1&b=2"
    assert req.url.fragment == "frag"


def test_parse_mapping_missing_url_raises() -> None:
    with pytest.raises(ValueError, match="missing url/url_str/request_url_str"):
        PreparedRequest.parse_mapping({"request_method": "GET"})


def test_parse_mapping_headers_from_mapping() -> None:
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


def test_parse_mapping_promotes_flattened_header_columns() -> None:
    """Flat columns like `request_content_type` should land in headers."""
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


def test_parse_mapping_respects_custom_prefix() -> None:
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
    assert req.tags == {"a": "1"}  # tags are stringified


def test_parse_mapping_tags_and_body() -> None:
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


def test_parse_mapping_sent_at_accepts_datetime() -> None:
    ts = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

    req = PreparedRequest.parse_mapping(
        {"request_url_str": "https://example.com", "request_sent_at": ts}
    )

    assert req.sent_at == ts
