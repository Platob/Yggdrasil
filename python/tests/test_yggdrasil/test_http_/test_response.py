"""Unit tests for :class:`yggdrasil.http_.response.HTTPResponse`."""
from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from yggdrasil.data.cast.datetime import _DT_MAX
from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse, RESPONSE_SCHEMA
from yggdrasil.path.memory import Memory
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    url: str = "https://example.com/data",
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> HTTPRequest:
    return HTTPRequest(
        method=method,
        url=URL.from_str(url),
        headers=headers or {},
        tags={},
        buffer=None,
        sent_at=0,
    )


def _make_response(
    body: bytes = b"",
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
    request: HTTPRequest | None = None,
) -> HTTPResponse:
    return HTTPResponse(
        request=request or _make_request(),
        status_code=status_code,
        headers=headers or {},
        tags=tags or {},
        buffer=Memory(binary=body),
        received_at=dt.datetime(2025, 1, 15, 12, 0, 0, tzinfo=dt.timezone.utc),
    )


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_basic_construction(self) -> None:
        req = _make_request()
        resp = _make_response(body=b"hello", status_code=200, request=req)
        assert resp.request is req
        assert resp.status_code == 200
        assert resp.content == b"hello"

    def test_status_code_coerced_to_int(self) -> None:
        resp = _make_response(status_code=201)
        assert isinstance(resp.status_code, int)
        assert resp.status_code == 201

    def test_headers_normalized_to_headers_type(self) -> None:
        resp = _make_response(headers={"X-Foo": "bar"})
        assert resp.headers.get("X-Foo") == "bar"

    def test_tags_stored(self) -> None:
        resp = _make_response(tags={"env": "prod"})
        assert resp.tags == {"env": "prod"}

    def test_received_at_stored(self) -> None:
        resp = _make_response()
        assert resp.received_at.year == 2025
        assert resp.received_at.month == 1

    def test_buffer_accessible_as_body(self) -> None:
        resp = _make_response(body=b"payload")
        assert resp.body is resp.buffer


# ---------------------------------------------------------------------------
# TestStatusHelpers
# ---------------------------------------------------------------------------

class TestStatusHelpers:

    @pytest.mark.parametrize("code", [200, 201, 204, 301, 302, 399])
    def test_ok_true_for_2xx_3xx(self, code: int) -> None:
        resp = _make_response(status_code=code)
        assert resp.ok is True

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 500, 502, 503])
    def test_ok_false_for_4xx_5xx(self, code: int) -> None:
        resp = _make_response(status_code=code)
        assert resp.ok is False

    def test_raise_for_status_noop_on_200(self) -> None:
        resp = _make_response(status_code=200)
        resp.raise_for_status()  # should not raise

    def test_raise_for_status_raises_on_404(self) -> None:
        resp = _make_response(status_code=404, body=b"not found")
        with pytest.raises(Exception):
            resp.raise_for_status()

    def test_raise_for_status_raises_on_500(self) -> None:
        resp = _make_response(status_code=500, body=b"server error")
        with pytest.raises(Exception):
            resp.raise_for_status()

    def test_error_returns_none_on_2xx(self) -> None:
        resp = _make_response(status_code=200)
        assert resp.error() is None

    def test_error_returns_exception_on_4xx(self) -> None:
        resp = _make_response(status_code=400)
        err = resp.error()
        assert isinstance(err, Exception)

    def test_error_returns_exception_on_5xx(self) -> None:
        resp = _make_response(status_code=503)
        err = resp.error()
        assert isinstance(err, Exception)

    def test_ok_boundary_at_400(self) -> None:
        assert _make_response(status_code=399).ok is True
        assert _make_response(status_code=400).ok is False


# ---------------------------------------------------------------------------
# TestBodyAccessors
# ---------------------------------------------------------------------------

class TestBodyAccessors:

    def test_content_returns_bytes(self) -> None:
        resp = _make_response(body=b"\x00\x01\x02")
        assert resp.content == b"\x00\x01\x02"

    def test_text_decodes_utf8_by_default(self) -> None:
        resp = _make_response(body="hello world".encode("utf-8"))
        assert resp.text == "hello world"

    def test_text_respects_charset_header(self) -> None:
        body = "café".encode("latin-1")
        resp = _make_response(
            body=body,
            headers={"Content-Type": "text/plain; charset=latin-1"},
        )
        assert resp.text == "café"

    def test_json_parses_object(self) -> None:
        resp = _make_response(
            body=b'{"name": "alice", "score": 42}',
            headers={"Content-Type": "application/json"},
        )
        data = resp.json()
        assert data["name"] == "alice"
        assert data["score"] == 42

    def test_json_parses_array(self) -> None:
        resp = _make_response(
            body=b'[1, 2, 3]',
            headers={"Content-Type": "application/json"},
        )
        data = resp.json()
        assert data == [1, 2, 3]

    def test_content_empty_body(self) -> None:
        resp = _make_response(body=b"")
        assert resp.content == b""

    def test_text_empty_body(self) -> None:
        resp = _make_response(body=b"")
        assert resp.text == ""

    def test_body_size_matches_content(self) -> None:
        payload = b"twelve bytes"
        resp = _make_response(body=payload)
        assert resp.body_size == len(payload)


# ---------------------------------------------------------------------------
# TestHeaders
# ---------------------------------------------------------------------------

class TestHeaders:

    def test_get_response_header(self) -> None:
        resp = _make_response(headers={"X-Request-Id": "abc123"})
        assert resp.headers.get("X-Request-Id") == "abc123"

    def test_missing_header_returns_none(self) -> None:
        resp = _make_response(headers={})
        assert resp.headers.get("X-Missing") is None

    def test_content_type_header_preserved(self) -> None:
        resp = _make_response(
            body=b"data",
            headers={"Content-Type": "application/xml"},
        )
        ct = resp.headers.get("Content-Type")
        assert ct is not None
        assert "application/xml" in ct

    def test_multiple_headers(self) -> None:
        resp = _make_response(
            headers={"X-A": "1", "X-B": "2", "X-C": "3"},
        )
        assert resp.headers.get("X-A") == "1"
        assert resp.headers.get("X-B") == "2"
        assert resp.headers.get("X-C") == "3"


# ---------------------------------------------------------------------------
# TestMediaType
# ---------------------------------------------------------------------------

class TestMediaType:

    def test_media_type_from_json_content_type(self) -> None:
        resp = _make_response(
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        assert resp.media_type.mime_type.value == "application/json"

    def test_media_type_from_text_plain(self) -> None:
        resp = _make_response(
            body=b"hi",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.media_type.mime_type.value == "text/plain"

    def test_media_type_strips_parameters(self) -> None:
        resp = _make_response(
            body=b"{}",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        assert resp.media_type.mime_type.value == "application/json"

    def test_media_type_codec_none_when_no_encoding(self) -> None:
        resp = _make_response(
            body=b"x",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.media_type.codec is None


# ---------------------------------------------------------------------------
# TestAnonymize
# ---------------------------------------------------------------------------

class TestAnonymize:

    def test_anonymize_strips_request_authorization(self) -> None:
        req = _make_request(headers={"Authorization": "Bearer secret"})
        resp = _make_response(request=req)
        anon = resp.anonymize(mode="remove")
        assert anon.request.headers.get("Authorization") is None

    def test_anonymize_strips_url_userinfo(self) -> None:
        req = _make_request(url="https://user:pass@example.com/path")
        resp = _make_response(request=req)
        anon = resp.anonymize(mode="remove")
        url_str = anon.request.url.to_string()
        assert "user" not in url_str
        assert "pass" not in url_str

    def test_anonymize_strips_sensitive_query_params(self) -> None:
        req = _make_request(url="https://example.com/path?api_key=secret123")
        resp = _make_response(request=req)
        anon = resp.anonymize(mode="remove")
        url_str = anon.request.url.to_string()
        assert "secret123" not in url_str

    def test_anonymize_preserves_status_code(self) -> None:
        resp = _make_response(status_code=201, body=b"created")
        anon = resp.anonymize(mode="remove")
        assert anon.status_code == 201

    def test_anonymize_preserves_body(self) -> None:
        resp = _make_response(body=b"body content")
        anon = resp.anonymize(mode="remove")
        assert anon.content == b"body content"

    def test_anonymize_returns_same_class(self) -> None:
        resp = _make_response()
        anon = resp.anonymize(mode="remove")
        assert type(anon) is HTTPResponse

    def test_anonymize_noop_when_empty_mode(self) -> None:
        resp = _make_response()
        result = resp.anonymize(mode="")
        assert result is resp


# ---------------------------------------------------------------------------
# TestArrowProjection
# ---------------------------------------------------------------------------

class TestArrowProjection:

    def test_arrow_values_contains_status_code(self) -> None:
        resp = _make_response(status_code=200)
        vals = resp.arrow_values
        assert vals["status_code"] == 200

    def test_arrow_values_contains_body_bytes(self) -> None:
        resp = _make_response(body=b"data")
        vals = resp.arrow_values
        assert vals["body"] == b"data"

    def test_arrow_values_contains_request_method(self) -> None:
        resp = _make_response()
        vals = resp.arrow_values
        assert vals["request_method"] == "GET"

    def test_arrow_values_keys_match_schema(self) -> None:
        resp = _make_response()
        schema_names = set(RESPONSE_SCHEMA.to_arrow_schema().names)
        assert set(resp.arrow_values.keys()) == schema_names

    def test_values_to_arrow_batch_single(self) -> None:
        resp = _make_response(body=b"one", status_code=200)
        batch = HTTPResponse.values_to_arrow_batch([resp])
        assert isinstance(batch, pa.RecordBatch)
        assert batch.num_rows == 1
        assert batch.column("status_code")[0].as_py() == 200

    def test_values_to_arrow_batch_multiple(self) -> None:
        r1 = _make_response(body=b"a", status_code=200)
        r2 = _make_response(body=b"b", status_code=404)
        batch = HTTPResponse.values_to_arrow_batch([r1, r2])
        assert batch.num_rows == 2
        assert batch.column("status_code")[0].as_py() == 200
        assert batch.column("status_code")[1].as_py() == 404

    def test_values_to_arrow_batch_schema_matches(self) -> None:
        resp = _make_response()
        batch = HTTPResponse.values_to_arrow_batch([resp])
        expected_names = set(RESPONSE_SCHEMA.to_arrow_schema().names)
        assert set(batch.schema.names) == expected_names

    def test_out_of_range_received_at_does_not_crash_read(self) -> None:
        # A corrupt/legacy cache cell can hold a timestamp int64 outside
        # Python's datetime range; reading the batch must not raise
        # OverflowError ("date value out of range"). The raw int64 is
        # recovered and, being irrecoverably out of range, throttled to
        # the int64-nanosecond upper edge (~2262) rather than dropped.
        resp = _make_response(body=b"{}", status_code=200)
        batch = HTTPResponse.values_to_arrow_batch([resp])
        bad = pa.array([9_000_000_000_000_000_000], type=pa.timestamp("us", "UTC"))
        cols = [bad if n == "received_at" else batch.column(n) for n in batch.schema.names]
        corrupt = pa.RecordBatch.from_arrays(cols, schema=batch.schema)

        out = list(HTTPResponse.from_arrow_tabular(corrupt))
        assert len(out) == 1
        assert out[0].status_code == 200
        assert out[0].received_at == _DT_MAX

    def test_in_range_received_at_round_trips(self) -> None:
        resp = _make_response(status_code=200)
        batch = HTTPResponse.values_to_arrow_batch([resp])
        out = list(HTTPResponse.from_arrow_tabular(batch))
        assert out[0].received_at == dt.datetime(2025, 1, 15, 12, 0, 0, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_includes_status_code(self) -> None:
        resp = _make_response(status_code=200)
        r = repr(resp)
        assert "200" in r

    def test_repr_includes_class_name(self) -> None:
        resp = _make_response()
        r = repr(resp)
        assert "HTTPResponse" in r

    def test_repr_includes_url(self) -> None:
        resp = _make_response()
        r = repr(resp)
        assert "example.com" in r

    def test_repr_different_status(self) -> None:
        r1 = repr(_make_response(status_code=200))
        r2 = repr(_make_response(status_code=500))
        assert "200" in r1
        assert "500" in r2
        assert r1 != r2
