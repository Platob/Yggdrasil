"""Unit tests for PreparedRequest / Response serializers.

Covers:
- PreparedRequestSerialized  (tag PREPARED_REQUEST = 222)
- ResponseSerialized         (tag RESPONSE = 223)

All round-trips are exercised via:
1. Direct ``from_value`` / ``from_python_object`` constructors.
2. Top-level ``Serialized.from_python_object`` dispatch.
3. Wire round-trip through a ``BytesIO`` buffer (serialise → bytes → deserialise).
4. Public ``dumps`` / ``loads`` helpers.
"""
from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.io import BytesIO, URL
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.pickle.ser import dumps, loads
from yggdrasil.pickle.ser.constants import CODEC_NONE
from yggdrasil.pickle.ser.http_ import (
    HttpSerialized,
    PreparedRequestSerialized,
    ResponseSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

_URL = "https://api.example.com/v1/users?page=1&limit=10"
_NOW = dt.datetime(2025, 3, 15, 12, 0, 0, tzinfo=dt.timezone.utc)


def _make_request(
    method: str = "GET",
    url: str = _URL,
    headers: dict | None = None,
    body: bytes | None = None,
    tags: dict | None = None,
    sent_at: dt.datetime | None = None,
) -> PreparedRequest:
    return PreparedRequest.prepare(
        method=method,
        url=url,
        headers=headers or {},
        body=body,
        tags=tags or {},
    )


def _make_response(
    status_code: int = 200,
    body: bytes = b'{"ok": true}',
    headers: dict | None = None,
    tags: dict | None = None,
    request: PreparedRequest | None = None,
) -> Response:
    req = request or _make_request()
    all_headers = {"Content-Type": "application/json", **(headers or {})}
    buf = BytesIO(body, copy=False)
    return Response(
        request=req,
        status_code=status_code,
        headers=all_headers,
        tags=tags or {},
        buffer=buf,
        received_at=_NOW,
    )


def _wire_roundtrip(ser: Serialized) -> Serialized:
    buf = BytesIO()
    ser.write_to(buf)
    return Serialized.read_from(buf, pos=0)


def _assert_requests_equal(a: PreparedRequest, b: PreparedRequest) -> None:
    assert a.method == b.method, f"method: {a.method!r} != {b.method!r}"
    assert a.url.to_string() == b.url.to_string(), f"url: {a.url!r} != {b.url!r}"
    # The deserialization path (from_arrow) re-promotes stored column values back
    # into headers (Host, Content-Length, …).  The restored set is therefore a
    # superset of the original: every header that was in `a` must be in `b`.
    ah = {k.lower(): v for k, v in (a.headers or {}).items()}
    bh = {k.lower(): v for k, v in (b.headers or {}).items()}
    for k, v in ah.items():
        assert k in bh, f"header {k!r} missing from restored request (got {list(bh)})"
        assert bh[k] == v, f"header {k!r}: {v!r} != {bh[k]!r}"
    if a.buffer is not None or b.buffer is not None:
        ab = a.buffer.to_bytes() if a.buffer else b""
        bb = b.buffer.to_bytes() if b.buffer else b""
        assert ab == bb, f"body differs: {ab[:50]!r} != {bb[:50]!r}"


def _assert_responses_equal(a: Response, b: Response) -> None:
    assert a.status_code == b.status_code
    _assert_requests_equal(a.request, b.request)
    ab = a.buffer.to_bytes() if a.buffer else b""
    bb = b.buffer.to_bytes() if b.buffer else b""
    assert ab == bb, f"body: {ab[:50]!r} != {bb[:50]!r}"
    # Restored headers are a superset (normalization may add Content-Length etc.)
    ah = {k.lower(): v for k, v in (a.headers or {}).items()}
    bh = {k.lower(): v for k, v in (b.headers or {}).items()}
    for k, v in ah.items():
        assert k in bh, f"response header {k!r} missing (got {list(bh)})"
        assert bh[k] == v, f"response header {k!r}: {v!r} != {bh[k]!r}"


# ---------------------------------------------------------------------------
# tag registration
# ---------------------------------------------------------------------------

class TestTagRegistration:
    def test_prepared_request_tag_value(self):
        assert Tags.PREPARED_REQUEST == 222

    def test_response_tag_value(self):
        assert Tags.RESPONSE == 223

    def test_tags_in_system_range(self):
        assert Tags.is_system(Tags.PREPARED_REQUEST)
        assert Tags.is_system(Tags.RESPONSE)

    def test_prepared_request_tag_registered(self):
        Tags._ensure_category_imported(Tags.SYSTEM_BASE)
        assert Tags.CLASSES.get(Tags.PREPARED_REQUEST) is PreparedRequestSerialized

    def test_response_tag_registered(self):
        Tags._ensure_category_imported(Tags.SYSTEM_BASE)
        assert Tags.CLASSES.get(Tags.RESPONSE) is ResponseSerialized

    def test_tag_names(self):
        assert Tags.TAG_TO_NAME[Tags.PREPARED_REQUEST] == "PREPARED_REQUEST"
        assert Tags.TAG_TO_NAME[Tags.RESPONSE] == "RESPONSE"

    def test_tags_are_distinct(self):
        assert Tags.PREPARED_REQUEST != Tags.RESPONSE


# ---------------------------------------------------------------------------
# PreparedRequestSerialized — basic construction
# ---------------------------------------------------------------------------

class TestPreparedRequestBasic:
    def test_from_value_type(self):
        req = _make_request()
        ser = PreparedRequestSerialized.from_value(req)
        assert isinstance(ser, PreparedRequestSerialized)

    def test_tag(self):
        req = _make_request()
        ser = PreparedRequestSerialized.from_value(req)
        assert ser.tag == Tags.PREPARED_REQUEST

    def test_wire_metadata_ygg_object(self):
        req = _make_request()
        ser = PreparedRequestSerialized.from_value(req)
        assert (ser.metadata or {}).get(b"ygg_object") == b"prepared_request"

    def test_wire_metadata_method(self):
        req = _make_request(method="POST")
        ser = PreparedRequestSerialized.from_value(req)
        assert (ser.metadata or {}).get(b"method") == b"POST"

    def test_wire_metadata_url(self):
        req = _make_request()
        ser = PreparedRequestSerialized.from_value(req)
        url_bytes = (ser.metadata or {}).get(b"url", b"")
        assert b"api.example.com" in url_bytes

    def test_codec_none_respected(self):
        req = _make_request()
        ser = PreparedRequestSerialized.from_value(req, codec=CODEC_NONE)
        assert ser.codec == CODEC_NONE


# ---------------------------------------------------------------------------
# PreparedRequestSerialized — round-trips
# ---------------------------------------------------------------------------

class TestPreparedRequestRoundTrip:
    def test_get_no_body(self):
        orig = _make_request("GET", _URL)
        result = PreparedRequestSerialized.from_value(orig).value
        _assert_requests_equal(orig, result)

    def test_post_with_json_body(self):
        orig = _make_request(
            "POST",
            "https://api.example.com/items",
            body=b'{"name":"test"}',
            headers={"Content-Type": "application/json"},
        )
        result = PreparedRequestSerialized.from_value(orig).value
        _assert_requests_equal(orig, result)

    def test_method_preserved(self):
        for method in ("GET", "POST", "PUT", "DELETE", "PATCH"):
            orig = _make_request(method)
            result = PreparedRequestSerialized.from_value(orig).value
            assert result.method == method

    def test_url_parts_preserved(self):
        orig = _make_request("GET", "https://user:pass@host.example.com:8443/path?q=1#frag")
        result = PreparedRequestSerialized.from_value(orig).value
        assert result.url.host == "host.example.com"
        assert result.url.port == 8443
        assert result.url.path == "/path"

    def test_custom_headers_preserved(self):
        orig = _make_request(
            headers={"X-Custom": "value", "Authorization": "Bearer tok"},
        )
        result = PreparedRequestSerialized.from_value(orig).value
        rh = {k.lower(): v for k, v in result.headers.items()}
        assert rh.get("x-custom") == "value"
        assert rh.get("authorization") == "Bearer tok"

    def test_tags_preserved(self):
        orig = _make_request(tags={"env": "prod", "service": "users"})
        result = PreparedRequestSerialized.from_value(orig).value
        assert result.tags.get("env") == "prod"
        assert result.tags.get("service") == "users"

    def test_body_bytes_preserved(self):
        body = b"\x00\x01\x02\x03" * 100
        orig = _make_request("PUT", body=body)
        result = PreparedRequestSerialized.from_value(orig).value
        assert result.buffer.to_bytes() == body

    def test_as_python_returns_prepared_request(self):
        orig = _make_request()
        ser = PreparedRequestSerialized.from_value(orig)
        result = ser.as_python()
        assert isinstance(result, PreparedRequest)

    def test_wire_roundtrip(self):
        orig = _make_request("GET", _URL)
        ser = PreparedRequestSerialized.from_value(orig, codec=CODEC_NONE)
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, PreparedRequestSerialized)
        _assert_requests_equal(orig, restored.value)

    def test_wire_roundtrip_with_body(self):
        orig = _make_request("POST", body=b'{"a":1}',
                              headers={"Content-Type": "application/json"})
        ser = PreparedRequestSerialized.from_value(orig)
        restored = _wire_roundtrip(ser)
        _assert_requests_equal(orig, restored.value)

    def test_no_body_request(self):
        orig = PreparedRequest.prepare("HEAD", "https://example.com/")
        result = PreparedRequestSerialized.from_value(orig).value
        assert result.method == "HEAD"


# ---------------------------------------------------------------------------
# PreparedRequestSerialized — dispatch
# ---------------------------------------------------------------------------

class TestPreparedRequestDispatch:
    def test_from_python_object_dispatch(self):
        req = _make_request()
        ser = Serialized.from_python_object(req)
        assert isinstance(ser, PreparedRequestSerialized)

    def test_http_serialized_dispatch(self):
        req = _make_request()
        ser = HttpSerialized.from_python_object(req)
        assert isinstance(ser, PreparedRequestSerialized)

    def test_http_serialized_returns_none_for_unknown(self):
        assert HttpSerialized.from_python_object("not-a-request") is None

    def test_dumps_loads(self):
        orig = _make_request("GET", _URL)
        payload = dumps(orig)
        result = loads(payload)
        assert isinstance(result, PreparedRequest)
        _assert_requests_equal(orig, result)

    def test_dumps_loads_b64(self):
        orig = _make_request("POST", body=b"hello")
        payload = dumps(orig, b64=True)
        assert isinstance(payload, str)
        result = loads(payload)
        assert isinstance(result, PreparedRequest)
        _assert_requests_equal(orig, result)

    def test_repeated_dispatch_type_cache(self):
        req = _make_request()
        ser1 = Serialized.from_python_object(req)
        ser2 = Serialized.from_python_object(req)
        assert isinstance(ser1, PreparedRequestSerialized)
        assert isinstance(ser2, PreparedRequestSerialized)


# ---------------------------------------------------------------------------
# ResponseSerialized — basic construction
# ---------------------------------------------------------------------------

class TestResponseBasic:
    def test_from_value_type(self):
        resp = _make_response()
        ser = ResponseSerialized.from_value(resp)
        assert isinstance(ser, ResponseSerialized)

    def test_tag(self):
        resp = _make_response()
        ser = ResponseSerialized.from_value(resp)
        assert ser.tag == Tags.RESPONSE

    def test_wire_metadata_ygg_object(self):
        resp = _make_response()
        ser = ResponseSerialized.from_value(resp)
        assert (ser.metadata or {}).get(b"ygg_object") == b"response"

    def test_wire_metadata_status_code(self):
        resp = _make_response(status_code=404)
        ser = ResponseSerialized.from_value(resp)
        assert (ser.metadata or {}).get(b"status_code") == b"404"

    def test_codec_none_respected(self):
        resp = _make_response()
        ser = ResponseSerialized.from_value(resp, codec=CODEC_NONE)
        assert ser.codec == CODEC_NONE


# ---------------------------------------------------------------------------
# ResponseSerialized — round-trips
# ---------------------------------------------------------------------------

class TestResponseRoundTrip:
    def test_basic_200(self):
        orig = _make_response(200, b'{"ok":true}')
        result = ResponseSerialized.from_value(orig).value
        _assert_responses_equal(orig, result)

    def test_status_codes(self):
        for code in (200, 201, 204, 301, 400, 401, 404, 500, 503):
            orig = _make_response(status_code=code)
            result = ResponseSerialized.from_value(orig).value
            assert result.status_code == code

    def test_body_preserved(self):
        body = b"\xff\xfe" + b"x" * 512
        orig = _make_response(body=body, headers={"Content-Type": "application/octet-stream"})
        result = ResponseSerialized.from_value(orig).value
        assert result.buffer.to_bytes() == body

    def test_empty_body(self):
        orig = _make_response(status_code=204, body=b"")
        result = ResponseSerialized.from_value(orig).value
        assert result.status_code == 204
        assert result.buffer.to_bytes() == b""

    def test_response_headers_preserved(self):
        orig = _make_response(
            headers={"X-Request-Id": "abc123", "Cache-Control": "no-cache"},
        )
        result = ResponseSerialized.from_value(orig).value
        rh = {k.lower(): v for k, v in result.headers.items()}
        assert rh.get("x-request-id") == "abc123"

    def test_response_tags_preserved(self):
        orig = _make_response(tags={"cached": "false", "region": "eu-west-1"})
        result = ResponseSerialized.from_value(orig).value
        assert result.tags.get("region") == "eu-west-1"

    def test_embedded_request_method(self):
        req = _make_request("DELETE", "https://api.example.com/items/42")
        orig = _make_response(request=req)
        result = ResponseSerialized.from_value(orig).value
        assert result.request.method == "DELETE"

    def test_embedded_request_url(self):
        req = _make_request("GET", "https://api.example.com/search?q=foo")
        orig = _make_response(request=req)
        result = ResponseSerialized.from_value(orig).value
        assert "api.example.com" in result.request.url.to_string()
        assert result.request.url.path == "/search"

    def test_as_python_returns_response(self):
        orig = _make_response()
        ser = ResponseSerialized.from_value(orig)
        result = ser.as_python()
        assert isinstance(result, Response)

    def test_wire_roundtrip(self):
        orig = _make_response(200, b"hello world")
        ser = ResponseSerialized.from_value(orig, codec=CODEC_NONE)
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, ResponseSerialized)
        _assert_responses_equal(orig, restored.value)

    def test_wire_roundtrip_large_body(self):
        body = b"data" * 10_000
        orig = _make_response(
            body=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        ser = ResponseSerialized.from_value(orig)
        restored = _wire_roundtrip(ser)
        assert restored.value.buffer.to_bytes() == body

    def test_wire_roundtrip_post_with_body(self):
        req = _make_request("POST", body=b'{"id":1}',
                             headers={"Content-Type": "application/json"})
        orig = _make_response(200, b'{"created":true}', request=req)
        ser = ResponseSerialized.from_value(orig)
        restored = _wire_roundtrip(ser)
        r = restored.value
        _assert_responses_equal(orig, r)
        assert r.request.method == "POST"


# ---------------------------------------------------------------------------
# ResponseSerialized — dispatch
# ---------------------------------------------------------------------------

class TestResponseDispatch:
    def test_from_python_object_dispatch(self):
        resp = _make_response()
        ser = Serialized.from_python_object(resp)
        assert isinstance(ser, ResponseSerialized)

    def test_http_serialized_dispatch(self):
        resp = _make_response()
        ser = HttpSerialized.from_python_object(resp)
        assert isinstance(ser, ResponseSerialized)

    def test_response_preferred_over_request(self):
        """Response must dispatch to ResponseSerialized, not PreparedRequestSerialized."""
        resp = _make_response()
        ser = Serialized.from_python_object(resp)
        assert type(ser) is ResponseSerialized

    def test_dumps_loads(self):
        orig = _make_response(200, b'{"ok":true}')
        payload = dumps(orig)
        result = loads(payload)
        assert isinstance(result, Response)
        _assert_responses_equal(orig, result)

    def test_dumps_loads_b64(self):
        orig = _make_response(201, b'{"id":42}')
        payload = dumps(orig, b64=True)
        assert isinstance(payload, str)
        result = loads(payload)
        assert isinstance(result, Response)
        assert result.status_code == 201

    def test_repeated_dispatch_type_cache(self):
        resp = _make_response()
        ser1 = Serialized.from_python_object(resp)
        ser2 = Serialized.from_python_object(resp)
        assert isinstance(ser1, ResponseSerialized)
        assert isinstance(ser2, ResponseSerialized)


# ---------------------------------------------------------------------------
# cross-type safety
# ---------------------------------------------------------------------------

class TestCrossTypeSafety:
    def test_tags_are_distinct(self):
        req = _make_request()
        resp = _make_response()
        assert PreparedRequestSerialized.from_value(req).tag != ResponseSerialized.from_value(resp).tag

    def test_request_payload_stays_request(self):
        req = _make_request()
        ser = PreparedRequestSerialized.from_value(req)
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, PreparedRequestSerialized)
        assert isinstance(restored.value, PreparedRequest)

    def test_response_payload_stays_response(self):
        resp = _make_response()
        ser = ResponseSerialized.from_value(resp)
        restored = _wire_roundtrip(ser)
        assert isinstance(restored, ResponseSerialized)
        assert isinstance(restored.value, Response)

