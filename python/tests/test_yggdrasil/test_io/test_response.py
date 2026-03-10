from __future__ import annotations

import datetime
from typing import Any

import pytest

from yggdrasil.io import BytesIO
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response, ARROW_SCHEMA, RESPONSE_ARROW_SCHEMA


def _rb_row(batch) -> dict[str, Any]:
    return {name: batch.column(name)[0].as_py() for name in batch.schema.names}


def _make_request() -> PreparedRequest:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com/api?q=1",
        headers={"X-Test": "req"},
        tags={"req_tag": "1"},
    )
    req.sent_at_timestamp = 111
    return req


def test_parse_dict_minimal() -> None:
    resp = Response.parse_dict(
        {
            "request_url_str": "https://example.com",
            "response_status_code": 200,
        }
    )

    assert resp.status_code == 200
    assert resp.request.url.to_string() == "https://example.com/"
    assert resp.headers["Content-Type"] == "application/octet-stream"
    assert resp.headers["Content-Length"] == "0"
    assert resp.buffer.to_bytes() == b""
    assert resp.tags == {}
    assert resp.received_at_timestamp == 0


def test_parse_dict_with_explicit_headers_and_body() -> None:
    resp = Response.parse_dict(
        {
            "request_url_str": "https://example.com",
            "response_status_code": 201,
            "response_headers": {
                "Content-Type": "application/json; charset=utf-8",
                "Content-Encoding": "gzip",
                "ETag": '"abc"',
            },
            "response_body": b'{"ok":true}',
            "response_tags": {"a": 1},
            "response_received_at_epoch": 123,
        }
    )

    assert resp.status_code == 201
    assert resp.headers["Content-Type"] == "application/json; charset=utf-8"
    assert resp.headers["Content-Encoding"] == "gzip"
    assert resp.headers["ETag"] == '"abc"'
    assert resp.buffer.to_bytes() == b'{"ok":true}'
    assert resp.tags == {"a": "1"}
    assert resp.received_at_timestamp == 123


def test_parse_dict_from_promoted_header_fields() -> None:
    resp = Response.parse_dict(
        {
            "request_url_str": "https://example.com",
            "response_status_code": 302,
            "response_content_type": "text/plain",
            "response_content_length": 4,
            "response_content_encoding": "gzip",
            "response_transfer_encoding": "chunked",
            "response_location": "/next",
            "response_etag": '"etag-1"',
            "response_last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            "response_x_request_id": "rid-1",
            "response_x_correlation_id": "cid-1",
            "response_body": b"test",
        }
    )

    assert resp.headers["Content-Type"] == "text/plain"
    assert resp.headers["Content-Length"] == "4"
    assert resp.headers["Content-Encoding"] == "gzip"
    assert resp.headers["Transfer-Encoding"] == "chunked"
    assert resp.headers["Location"] == "/next"
    assert resp.headers["ETag"] == '"etag-1"'
    assert resp.headers["Last-Modified"] == "Mon, 01 Jan 2024 00:00:00 GMT"
    assert resp.headers["X-Request-ID"] == "rid-1"
    assert resp.headers["X-Correlation-ID"] == "cid-1"


def test_parse_str_accepts_json_object_string() -> None:
    resp = Response.parse(
        '{"request_url_str":"https://example.com","response_status_code":204}'
    )

    assert resp.status_code == 204
    assert resp.request.url.to_string() == "https://example.com/"


def test_parse_dict_missing_status_raises() -> None:
    with pytest.raises(ValueError, match="missing status_code/status/code"):
        Response.parse_dict({"request_url_str": "https://example.com"})


def test_media_type_uses_declared_header() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={"Content-Type": "application/json; charset=utf-8"},
        buffer=BytesIO(b'{"a":1}'),
        tags={},
        received_at_timestamp=0,
    )

    assert resp.media_type.mime_type.value == "application/json"
    assert resp.text == '{"a":1}'


def test_json_parsing() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={"Content-Type": "application/json"},
        buffer=BytesIO(b'{"a":1}'),
        tags={},
        received_at_timestamp=0,
    )

    assert resp.json() == {"a": 1}


def test_ok_property() -> None:
    from yggdrasil.io.buffer import BytesIO

    ok_resp = Response(
        request=_make_request(),
        status_code=200,
        headers={},
        buffer=BytesIO(),
        tags={},
        received_at_timestamp=0,
    )
    bad_resp = Response(
        request=_make_request(),
        status_code=500,
        headers={},
        buffer=BytesIO(),
        tags={},
        received_at_timestamp=0,
    )

    assert ok_resp.ok is True
    assert bad_resp.ok is False


def test_anonymize_redacts_response_and_request() -> None:
    from yggdrasil.io.buffer import BytesIO

    req = PreparedRequest.prepare(
        method="GET",
        url="https://user:pass@example.com/path?token=secret",
        headers={"Authorization": "Bearer secret"},
    )
    resp = Response(
        request=req,
        status_code=200,
        headers={"Set-Cookie": "a=b", "X-API-Key": "secret"},
        buffer=BytesIO(),
        tags={},
        received_at_timestamp=0,
    )

    anon = resp.anonymize(mode="redact")

    assert anon is not resp
    assert "X-API-Key" in anon.headers
    assert anon.headers["X-API-Key"] == "<redacted>"
    assert anon.request is not req


def test_to_arrow_batch_matches_schema() -> None:
    from yggdrasil.io.buffer import BytesIO

    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={
            "Content-Type": "application/json",
            "Content-Length": "5",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "Location": "/next",
            "ETag": '"abc"',
            "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            "X-Request-ID": "rid-1",
            "X-Correlation-ID": "cid-1",
            "X-Other": "keep-me",
        },
        buffer=BytesIO(b"hello"),
        tags={"t": "1"},
        received_at_timestamp=222,
    )

    rb = resp.to_arrow_batch()

    assert rb.schema == RESPONSE_ARROW_SCHEMA
    assert rb.num_rows == 1
    assert ARROW_SCHEMA.names[-1] == "response_received_at_epoch"


def test_to_arrow_batch_promotes_headers_and_keeps_remaining() -> None:
    from yggdrasil.io.buffer import BytesIO

    resp = Response(
        request=_make_request(),
        status_code=206,
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
        buffer=BytesIO(b"hello world"),
        tags={"explicit": "tag"},
        received_at_timestamp=333,
    )

    row = _rb_row(resp.to_arrow_batch())

    assert row["response_status_code"] == 206
    assert row["response_host"] == "example.com"
    assert row["response_user_agent"] == "pytest"
    assert row["response_accept"] == "application/json"
    assert row["response_accept_encoding"] == "gzip"
    assert row["response_accept_language"] == "en"
    assert row["response_content_type"] == "application/json"
    assert row["response_content_length"] == 11
    assert row["response_content_encoding"] == "gzip"
    assert row["response_transfer_encoding"] == "chunked"
    assert row["response_x_request_id"] == "rid-1"
    assert row["response_x_correlation_id"] == "cid-1"
    assert dict(row["response_headers"]) == {"X-Other": "keep-me"}
    assert dict(row["response_tags"]) == {"explicit": "tag"}
    assert row["response_body"] == b"hello world"
    assert row["response_body_hash"] is not None
    assert len(row["response_body_hash"]) == 32
    assert row["response_received_at"] == datetime.datetime(1970, 1, 1, 0, 0, 0, 333, tzinfo=datetime.timezone.utc)
    assert row["response_received_at_epoch"] == 333


def test_from_arrow_round_trip_rebuilds_promoted_headers() -> None:
    from yggdrasil.io.buffer import BytesIO

    original = Response(
        request=_make_request(),
        status_code=200,
        headers={
            "Content-Type": "application/json",
            "Content-Length": "5",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "Location": "/next",
            "ETag": '"etag-1"',
            "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            "X-Request-ID": "rid-1",
            "X-Correlation-ID": "cid-1",
            "X-Other": "keep-me",
        },
        buffer=BytesIO(b"hello"),
        tags={"a": "1"},
        received_at_timestamp=444,
    )

    rb = original.to_arrow_batch()
    rebuilt = next(Response.from_arrow(rb))

    assert rebuilt.status_code == 200
    assert rebuilt.headers["Content-Type"] == "application/json"
    assert rebuilt.headers["Content-Length"] == "5"
    assert rebuilt.headers["Content-Encoding"] == "gzip"
    assert rebuilt.headers["Transfer-Encoding"] == "chunked"
    assert rebuilt.headers["Location"] == "/next"
    assert rebuilt.headers["ETag"] == '"etag-1"'
    assert rebuilt.headers["Last-Modified"] == "Mon, 01 Jan 2024 00:00:00 GMT"
    assert rebuilt.headers["X-Request-ID"] == "rid-1"
    assert rebuilt.headers["X-Correlation-ID"] == "cid-1"
    assert rebuilt.headers["X-Other"] == "keep-me"
    assert rebuilt.tags == {"a": "1"}
    assert rebuilt.buffer.to_bytes() == b"hello"
    assert rebuilt.received_at_timestamp == 444


def test_from_arrow_round_trip_rebuilds_request_promoted_headers_too() -> None:
    from yggdrasil.io.buffer import BytesIO

    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?q=1",
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "ETag": '"req-etag"',
            "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            "Location": "/req-next",
            "X-Other": "keep-me",
        },
        body=b"{}",
        tags={"x": "1"},
    )
    req.sent_at_timestamp = 999

    resp = Response(
        request=req,
        status_code=200,
        headers={"Content-Type": "application/json"},
        buffer=BytesIO(b"{}"),
        tags={},
        received_at_timestamp=555,
    )

    rebuilt = next(Response.from_arrow(resp.to_arrow_batch()))

    assert rebuilt.request.headers["Content-Type"] == "application/json"
    assert rebuilt.request.headers["Content-Encoding"] == "gzip"
    assert rebuilt.request.headers["Transfer-Encoding"] == "chunked"
    assert rebuilt.request.headers["ETag"] == '"req-etag"'
    assert rebuilt.request.headers["Last-Modified"] == "Mon, 01 Jan 2024 00:00:00 GMT"
    assert rebuilt.request.headers["Location"] == "/req-next"
    assert rebuilt.request.headers["X-Other"] == "keep-me"
    assert rebuilt.request.tags == {"q": "1", "x": "1"}


def test_to_arrow_batch_without_body_has_hash_for_empty_buffer() -> None:
    from yggdrasil.io.buffer import BytesIO

    resp = Response(
        request=_make_request(),
        status_code=204,
        headers={},
        buffer=BytesIO(),
        tags={},
        received_at_timestamp=0,
    )

    row = _rb_row(resp.to_arrow_batch())

    assert row["response_body"] == b""
    assert row["response_body_hash"] is not None
    assert len(row["response_body_hash"]) == 32


def test_to_starlette_strips_hop_by_hop_and_sets_media_type() -> None:
    from yggdrasil.io.buffer import BytesIO

    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Transfer-Encoding": "chunked",
            "X-Test": "1",
        },
        buffer=BytesIO(b'{"a":1}'),
        tags={},
        received_at_timestamp=0,
    )

    out = resp.to_starlette()

    assert out.status_code == 200
    assert out.headers["content-length"] == str(len(b'{"a":1}'))
    assert out.headers["x-test"] == "1"
    assert "transfer-encoding" not in out.headers
    assert out.media_type == "application/json"


def test_to_fastapi_falls_back_to_starlette_when_fastapi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    from yggdrasil.io.buffer import BytesIO

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastapi":
            raise ImportError("boom")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={"Content-Type": "text/plain"},
        buffer=BytesIO(b"ok"),
        tags={},
        received_at_timestamp=0,
    )

    out = resp.to_fastapi()

    assert out.status_code == 200
    assert out.headers["content-length"] == "2"