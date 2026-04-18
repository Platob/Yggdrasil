"""`Response.parse_mapping` / `parse_str` — status-code + request linkage."""

from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.io.response import Response


def test_parse_mapping_minimal_defaults_to_octet_stream() -> None:
    resp = Response.parse_mapping(
        {
            "request_url_str": "https://example.com/a",
            "response_status_code": 200,
        }
    )

    assert resp.status_code == 200
    assert resp.request.method == "GET"
    assert resp.request.url.to_string() == "https://example.com/a"
    # When no Content-Type is given the default is octet-stream, length=0.
    assert resp.headers["Content-Type"] == "application/octet-stream"
    assert resp.headers["Content-Length"] == "0"
    assert resp.tags == {}
    assert resp.buffer.to_bytes() == b""
    assert resp.received_at_timestamp == 0


def test_parse_mapping_missing_status_code_raises() -> None:
    with pytest.raises(ValueError, match="missing status_code/status/code"):
        Response.parse_mapping({"request_url_str": "https://example.com/a"})


def test_parse_mapping_with_nested_request_object() -> None:
    resp = Response.parse_mapping(
        {
            "request": {
                "request_method": "POST",
                "request_url_str": "https://example.com/items",
                "request_headers": {"X-Test": "1"},
            },
            "response_status_code": 201,
            "response_headers": {"Content-Type": "application/json"},
            "response_body": b'{"ok":true}',
        }
    )

    assert resp.request.method == "POST"
    assert resp.request.url.to_string() == "https://example.com/items"
    assert resp.request.headers["X-Test"] == "1"
    assert resp.status_code == 201
    assert resp.buffer.to_bytes() == b'{"ok":true}'


def test_parse_mapping_respects_custom_prefix() -> None:
    resp = Response.parse_mapping(
        {
            "request_url_str": "https://example.com/a",
            "foo_status_code": 200,
            "foo_headers": {"Content-Type": "text/plain"},
            "foo_tags": {"a": 1},
            "foo_body": b"hello",
            "foo_received_at": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        },
        prefix="foo_",
    )

    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("text/plain")
    assert resp.tags == {"a": "1"}
    assert resp.buffer.to_bytes() == b"hello"
    assert resp.received_at_timestamp > 0


def test_parse_str_accepts_json_object_string() -> None:
    resp = Response.parse_str(
        '{"request_url_str":"https://example.com/a","response_status_code":200}'
    )

    assert resp.status_code == 200
    assert resp.request.url.to_string() == "https://example.com/a"


def test_parse_str_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty string"):
        Response.parse_str("   ")
