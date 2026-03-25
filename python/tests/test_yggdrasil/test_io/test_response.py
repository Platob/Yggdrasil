# tests/io/test_response.py
from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from yggdrasil.io.enums import MediaType, MimeType
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, Response


def _rb_row(batch) -> dict[str, Any]:
    return {name: batch.column(name)[0].as_py() for name in batch.schema.names}


def _make_request() -> PreparedRequest:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com/api?a=1",
        headers={"User-Agent": "pytest"},
        tags={"rq": "1"},
    )
    req.sent_at = dt.datetime.fromtimestamp(7, tz=dt.timezone.utc)
    return req


def test_parse_mapping_minimal() -> None:
    resp = Response.parse_mapping(
        {
            "request_url_str": "https://example.com/a",
            "response_status_code": 200,
        }
    )

    assert resp.status_code == 200
    assert resp.request.method == "GET"
    assert resp.request.url.to_string() == "https://example.com/a"
    assert resp.headers["Content-Type"] == "application/octet-stream"
    assert resp.headers["Content-Length"] == "0"
    assert resp.tags == {}
    assert resp.buffer.to_bytes() == b""
    assert resp.received_at_timestamp == 0


def test_parse_mapping_missing_status_code_raises() -> None:
    with pytest.raises(ValueError, match="missing status_code/status/code"):
        Response.parse_mapping(
            {
                "request_url_str": "https://example.com/a",
            }
        )


def test_parse_mapping_with_nested_request() -> None:
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


def test_parse_str_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty string"):
        Response.parse_str("   ")


def test_update_headers_and_tags() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={"Content-Type": "text/plain"},
        tags={"a": "1"},
        buffer=b"hello",  # type: ignore[arg-type]
        received_at=1,
    )

    resp.update_headers({"X-Test": "2"})
    resp.update_tags({"b": "2"})

    assert resp.headers["X-Test"] == "2"
    assert resp.tags == {"a": "1", "b": "2"}


def test_media_type_and_set_media_type() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={},
        tags={},
        buffer=b"hello",  # type: ignore[arg-type]
        received_at=1,
    )

    media = resp.media_type
    assert media.mime_type is not None

    resp.set_media_type(MediaType(MimeType.JSON, None))
    assert resp.headers["Content-Type"] == MimeType.JSON.value
    assert resp.request.headers["Accept"] == MimeType.JSON.value


def test_body_content_text_and_json() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={"Content-Type": "application/json; charset=utf-8"},
        tags={},
        buffer=b'{"x":1}',  # type: ignore[arg-type]
        received_at=1,
    )

    assert resp.body.to_bytes() == b'{"x":1}'
    assert resp.content == b'{"x":1}'
    assert resp.text == '{"x":1}'
    assert resp.json() == {"x": 1}


def test_ok_raise_for_status_warn_for_status() -> None:
    ok_resp = Response(
        request=_make_request(),
        status_code=204,
        headers={},
        tags={},
        buffer=b"",  # type: ignore[arg-type]
        received_at=1,
    )
    assert ok_resp.ok is True
    assert ok_resp.error() is None

    bad_resp = Response(
        request=_make_request(),
        status_code=500,
        headers={},
        tags={},
        buffer=b"",  # type: ignore[arg-type]
        received_at=1,
    )
    assert bad_resp.ok is False
    assert bad_resp.error() is not None

    with pytest.raises(Exception):
        bad_resp.raise_for_status()

    with pytest.warns(RuntimeWarning):
        bad_resp.warn_for_status()


def test_received_at_property() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={},
        tags={},
        buffer=b"",  # type: ignore[arg-type]
        received_at=42,
    )

    assert resp.received_at == dt.datetime(1970, 1, 1, 0, 0, 42, tzinfo=dt.timezone.utc)


def test_anonymize_redacts_request_and_response_headers() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://user:pass@example.com/a?token=secret",
        headers={"Authorization": "Bearer abc"},
    )
    resp = Response(
        request=req,
        status_code=200,
        headers={"X-API-Key": "secret"},
        tags={},
        buffer=b"hello",  # type: ignore[arg-type]
        received_at=1,
    )

    anon = resp.anonymize(mode="redact")

    assert anon is not resp
    assert "<redacted>" in anon.request.url.to_string().lower() or "secret" not in anon.request.url.to_string()
    assert "Authorization" in anon.request.headers
    assert "X-API-Key" in anon.headers


def test_arrow_values_contains_expected_projection() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1",
        headers={"User-Agent": "pytest"},
        body=b"rq",
        tags={"rt": "1"},
    )
    req.sent_at = dt.datetime.fromtimestamp(7, tz=dt.timezone.utc)

    resp = Response(
        request=req,
        status_code=201,
        headers={
            "User-Agent": "server",
            "Content-Type": "application/json",
            "X-Other": "keep",
        },
        tags={"t": "2"},
        buffer=b'{"ok":true}',  # type: ignore[arg-type]
        received_at=42,
    )

    values = resp.arrow_values

    assert values["request_method"] == "POST"
    assert values["request_url_str"] == "https://example.com/api?a=1"
    assert values["response_status_code"] == 201
    assert values["response_user_agent"] == "server"
    assert values["response_content_type"] == "application/json"
    assert values["response_body"] == b'{"ok":true}'
    assert values["response_body_hash"] is not None
    assert values["response_tags"] == {"t": "2"}
    assert values["response_received_at"] == dt.datetime(1970, 1, 1, 0, 0, 42, tzinfo=dt.timezone.utc)


def test_match_value_supports_request_and_response_fields() -> None:
    resp = Response(
        request=PreparedRequest.prepare(
            method="POST",
            url="https://example.com/api?a=1",
            body=b"rq",
        ),
        status_code=200,
        headers={"Content-Type": "application/json"},
        tags={"t": "2"},
        buffer=b'{"ok":true}',  # type: ignore[arg-type]
        received_at=42,
    )

    assert resp.match_value("request_method") == "POST"
    assert resp.match_value("request_url_str") == "https://example.com/api?a=1"
    assert resp.match_value("response_status_code") == 200
    assert resp.match_value("response_content_type") == "application/json"
    assert resp.match_value("response_body") == b'{"ok":true}'
    assert resp.match_value("response_received_at") == dt.datetime(1970, 1, 1, 0, 0, 42, tzinfo=dt.timezone.utc)


def test_match_value_invalid_key_raises() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={},
        tags={},
        buffer=b"",  # type: ignore[arg-type]
        received_at=1,
    )

    with pytest.raises(ValueError, match="Unsupported response match key"):
        resp.match_value("not_a_real_key")


def test_match_tuple() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={"Content-Type": "application/json"},
        tags={},
        buffer=b"hello",  # type: ignore[arg-type]
        received_at=42,
    )

    out = resp.match_tuple(
        ["request_method", "request_url_str", "response_status_code", "response_body_hash"]
    )

    assert out[0] == "GET"
    assert out[1] == "https://example.com/api?a=1"
    assert out[2] == 200
    assert out[3] is not None


def test_to_arrow_batch_matches_schema() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={
            "User-Agent": "server",
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "X-Other": "keep",
        },
        tags={"t": "2"},
        buffer=b'{"ok":true}',  # type: ignore[arg-type]
        received_at=42,
    )

    rb = resp.to_arrow_batch(parse=False)

    assert rb.schema == RESPONSE_ARROW_SCHEMA
    assert rb.num_rows == 1


def test_to_arrow_batch_promotes_headers_and_keeps_remaining() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1&shared=url",
        headers={"User-Agent": "pytest"},
        body=b"hello rq",
        tags={"shared": "rq"},
    )
    req.sent_at = dt.datetime.fromtimestamp(11, tz=dt.timezone.utc)

    resp = Response(
        request=req,
        status_code=200,
        headers={
            "Host": "example.com",
            "User-Agent": "server",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Accept-Language": "en",
            "Content-Type": "application/json",
            "Content-Length": "11",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "X-Other": "keep-me",
        },
        tags={"t": "2"},
        buffer=b"hello world",  # type: ignore[arg-type]
        received_at=42,
    )

    row = _rb_row(resp.to_arrow_batch(parse=False))

    assert row["request_method"] == "POST"
    assert row["request_url_str"] == "https://example.com/api?a=1&shared=url"
    assert row["response_status_code"] == 200

    assert row["response_host"] == "example.com"
    assert row["response_user_agent"] == "server"
    assert row["response_accept"] == "application/json"
    assert row["response_accept_encoding"] == "gzip"
    assert row["response_accept_language"] == "en"
    assert row["response_content_type"] == "application/json"
    assert row["response_content_length"] == 11
    assert row["response_content_encoding"] == "gzip"
    assert row["response_transfer_encoding"] == "chunked"

    headers_map = dict(row["response_headers"])
    assert headers_map["X-Other"] == "keep-me"

    assert dict(row["response_tags"]) == {"t": "2"}
    assert row["response_body"] == b"hello world"
    assert row["response_body_hash"] == resp.buffer.xxh3_int64()
    assert row["response_received_at"] == dt.datetime(1970, 1, 1, 0, 0, 42, 0, tzinfo=dt.timezone.utc)


def test_to_arrow_batch_without_body_has_empty_body_hash_from_empty_buffer() -> None:
    resp = Response(
        request=_make_request(),
        status_code=204,
        headers={},
        tags={},
        buffer=b"",  # type: ignore[arg-type]
        received_at=0,
    )

    row = _rb_row(resp.to_arrow_batch(parse=False))

    assert row["response_body"] == b""
    assert row["response_body_hash"] is not None


def test_from_arrow_roundtrip_record_batch() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1",
        headers={"User-Agent": "pytest"},
        body=b"rq",
        tags={"rt": "1"},
    )
    req.sent_at = dt.datetime.fromtimestamp(7, tz=dt.timezone.utc)

    resp = Response(
        request=req,
        status_code=201,
        headers={
            "User-Agent": "server",
            "Content-Type": "application/json",
            "X-Other": "keep",
        },
        tags={"t": "2"},
        buffer=b'{"ok":true}',  # type: ignore[arg-type]
        received_at=123,
    )

    rb = resp.to_arrow_batch(parse=False)
    out = list(Response.from_arrow_tabular(rb))

    assert len(out) == 1
    rebuilt = out[0]

    assert rebuilt.request.method == "POST"
    assert rebuilt.request.url.to_string() == "https://example.com/api?a=1"
    assert rebuilt.status_code == 201
    assert rebuilt.buffer.to_bytes() == b'{"ok":true}'
    assert rebuilt.tags == {"t": "2"}
    assert rebuilt.received_at_timestamp == 123000000
    assert rebuilt.headers["User-Agent"] == "server"
    assert rebuilt.headers["Content-Type"] == "application/json"


def test_from_arrow_roundtrip_table() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={"Content-Type": "text/plain"},
        tags={"t": "2"},
        buffer=b"hello",  # type: ignore[arg-type]
        received_at=7,
    )

    table = resp.to_arrow_table(parse=False)
    out = list(Response.from_arrow_tabular(table))

    assert len(out) == 1
    rebuilt = out[0]

    assert rebuilt.request.url.to_string() == "https://example.com/api?a=1"
    assert rebuilt.status_code == 200
    assert rebuilt.buffer.to_bytes() == b"hello"
    assert rebuilt.received_at_timestamp == 7000000


def test_to_asgi_payload_strips_hop_by_hop_and_sets_length() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={
            "Content-Type": "application/json",
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
            "X-Test": "1",
        },
        tags={},
        buffer=b'{"ok":true}',  # type: ignore[arg-type]
        received_at=1,
    )

    body, headers, media_type = resp._to_asgi_payload()

    assert body == b'{"ok":true}'
    assert headers["Content-Length"] == str(len(body))
    assert headers["X-Test"] == "1"
    assert "Connection" not in headers
    assert "Transfer-Encoding" not in headers
    assert media_type == "application/json"


def test_apply_returns_transformed_response() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={},
        tags={},
        buffer=b"",  # type: ignore[arg-type]
        received_at=1,
    )

    def transform(r: Response) -> Response:
        r.status_code = 201
        return r

    out = resp.apply(transform)
    assert out.status_code == 201


def test_response_repr_contains_status_and_url() -> None:
    resp = Response(
        request=_make_request(),
        status_code=200,
        headers={},
        tags={},
        buffer=b"hello",  # type: ignore[arg-type]
        received_at=1,
    )

    rep = repr(resp)
    assert "status_code=200" in rep
    assert "https://example.com/api?a=1" in rep