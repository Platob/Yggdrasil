"""Tests for yggdrasil.io.response.Response."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.io.errors import (
    BadRequest,
    InternalServerError,
    NotFoundError,
    UnauthorizedError,
)
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response, RESPONSE_ARROW_SCHEMA

from ._helpers import EPOCH, make_request, make_response


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_buffer_wraps_bytes(self):
        resp = make_response(body=b"hello")
        assert resp.buffer.to_bytes() == b"hello"

    def test_status_code_coerced_to_int(self):
        resp = Response(
            request=make_request(),
            status_code="200",  # type: ignore[arg-type]
            headers={},
            tags={},
            buffer=b"",
            received_at=EPOCH,
        )
        assert resp.status_code == 200

    def test_routes_to_http_subclass_for_http_request(self):
        from yggdrasil.io.http_ import HTTPResponse

        resp = make_response()
        # Direct construction with http url stays as base Response only
        # if we explicitly call cls(...) — `parse_mapping` is the
        # routing entry point.
        parsed = Response.parse_mapping(
            {
                "request": {"url": "https://example.com/", "method": "GET"},
                "status_code": 200,
                "headers": {"Content-Type": "application/json"},
                "buffer": b"{}",
                "received_at": EPOCH,
            }
        )
        assert isinstance(parsed, HTTPResponse)


# ---------------------------------------------------------------------------
# parse / parse_mapping / parse_str
# ---------------------------------------------------------------------------


class TestParse:
    def test_parse_passthrough_for_existing(self):
        resp = make_response()
        assert Response.parse(resp) is resp

    def test_parse_str_round_trip(self):
        resp_str = (
            '{"request": {"url": "https://example.com/", "method": "GET"}, '
            '"status_code": 200, "headers": {"Content-Type": "application/json"}}'
        )
        parsed = Response.parse_str(resp_str)
        assert parsed.status_code == 200

    def test_parse_str_empty_raises(self):
        with pytest.raises(ValueError):
            Response.parse_str("   ")

    def test_parse_mapping_empty_raises(self):
        with pytest.raises(ValueError):
            Response.parse_mapping({})


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


class TestStatus:
    def test_ok_for_2xx(self):
        assert make_response(status_code=200).ok is True

    def test_ok_for_3xx(self):
        assert make_response(status_code=302).ok is True

    def test_not_ok_for_4xx(self):
        assert make_response(status_code=400).ok is False

    def test_raise_for_status_no_op_2xx(self):
        make_response(status_code=200).raise_for_status()

    def test_raise_for_status_4xx(self):
        with pytest.raises(BadRequest):
            make_response(status_code=400).raise_for_status()

    def test_raise_for_status_5xx(self):
        with pytest.raises(InternalServerError):
            make_response(status_code=500).raise_for_status()

    def test_specific_class_dispatch(self):
        with pytest.raises(NotFoundError):
            make_response(status_code=404).raise_for_status()
        with pytest.raises(UnauthorizedError):
            make_response(status_code=401).raise_for_status()

    def test_error_returns_none_when_ok(self):
        assert make_response(status_code=200).error() is None

    def test_error_returns_exception_when_failed(self):
        err = make_response(status_code=400).error()
        assert isinstance(err, BadRequest)

    def test_warn_for_status_emits_warning(self):
        with pytest.warns(RuntimeWarning):
            make_response(status_code=500).warn_for_status()


# ---------------------------------------------------------------------------
# Body / content
# ---------------------------------------------------------------------------


class TestBody:
    def test_content_returns_raw_bytes(self):
        resp = make_response(body=b'{"ok":true}')
        assert resp.content == b'{"ok":true}'

    def test_text_default_utf8(self):
        resp = make_response(body=b'{"ok":true}')
        assert "ok" in resp.text

    def test_json_returns_parsed_dict(self):
        resp = make_response(body=b'{"a":1}', content_type="application/json")
        assert resp.json() == {"a": 1}

    def test_body_property_is_buffer(self):
        resp = make_response(body=b"x")
        assert resp.body is resp.buffer


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


class TestUpdates:
    def test_update_headers(self):
        resp = make_response()
        resp.update_headers({"X-Custom": "v"})
        assert resp.headers.get("X-Custom") == "v"

    def test_update_tags(self):
        resp = make_response()
        resp.update_tags({"tag": "v"})
        assert resp.tags.get("tag") == "v"


# ---------------------------------------------------------------------------
# Anonymize
# ---------------------------------------------------------------------------


class TestAnonymize:
    def test_strips_request_userinfo(self):
        req = make_request(url="https://alice:pw@example.com/")
        resp = make_response(request=req)
        sanitized = resp.anonymize("remove")
        assert sanitized.request.url.userinfo is None


# ---------------------------------------------------------------------------
# Arrow
# ---------------------------------------------------------------------------


# ``Response.arrow_values`` hashes the body via xxh3 to produce the
# stable ``body_hash`` column; that hash is also what ``match_value``
# uses behind the scenes. The whole surface therefore depends on
# the optional ``xxhash`` package — skip these classes on a base
# install that hasn't pulled the ``pickle`` extra in.
class TestArrow:
    def test_to_arrow_batch_no_parse(self):
        pytest.importorskip("xxhash")
        resp = make_response()
        batch = resp.to_arrow_batch(parse=False)
        assert isinstance(batch, pa.RecordBatch)
        assert batch.schema == RESPONSE_ARROW_SCHEMA

    def test_to_arrow_table_no_parse(self):
        pytest.importorskip("xxhash")
        resp = make_response()
        table = resp.to_arrow_table(parse=False)
        assert isinstance(table, pa.Table)
        assert table.num_rows == 1

    def test_from_arrow_tabular_round_trip(self):
        pytest.importorskip("xxhash")
        resp = make_response()
        table = resp.to_arrow_table(parse=False)
        rebuilt = list(Response.from_arrow_tabular(table))
        assert len(rebuilt) == 1
        assert rebuilt[0].status_code == resp.status_code


# ---------------------------------------------------------------------------
# match_value
# ---------------------------------------------------------------------------


class TestMatchValue:
    def test_response_keys(self):
        pytest.importorskip("xxhash")
        resp = make_response(status_code=200)
        assert resp.match_value("status_code") == 200

    def test_request_keys_delegated(self):
        pytest.importorskip("xxhash")
        resp = make_response()
        assert resp.match_value("request.method") == "GET"

    def test_unsupported_key_raises(self):
        pytest.importorskip("xxhash")
        resp = make_response()
        with pytest.raises(ValueError):
            resp.match_value("not_a_real_key")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_runs_callable(self):
        resp = make_response()
        result = resp.apply(lambda r: make_response(status_code=999, request=r.request))
        assert result.status_code == 999


# ---------------------------------------------------------------------------
# HTTPResponse.drain_urllib3 — Content-Length resync
# ---------------------------------------------------------------------------


class _FakeUrllib3Response:
    """Minimal stand-in for ``urllib3.BaseHTTPResponse``."""

    def __init__(self, body: bytes, headers: dict[str, str], status: int = 200):
        self._body = body
        self.headers = headers
        self.status = status
        self.released = False

    def stream(self, amt: int = 65536):
        view = memoryview(self._body)
        for start in range(0, len(view), amt):
            yield bytes(view[start:start + amt])

    def read(self) -> bytes:
        return self._body

    def release_conn(self) -> None:
        self.released = True


class TestDrainContentLengthSync:
    def _fake(self, body: bytes, *, declared_length: str | None = None):
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if declared_length is not None:
            headers["Content-Length"] = declared_length
        return _FakeUrllib3Response(body, headers)

    def test_streamed_drain_syncs_content_length(self):
        from yggdrasil.io.http_ import HTTPResponse

        body = b'{"hello":"world"}'
        raw = self._fake(body, declared_length=str(len(body)))
        resp = HTTPResponse.from_urllib3(
            request=make_request(),
            response=raw,
            tags=None,
            received_at=EPOCH,
        )

        # Pre-drain: ``Response.__init__`` ran with an empty buffer, so
        # the header should currently report 0.
        assert resp.headers["Content-Length"] == "0"

        resp.drain_urllib3(raw, stream=True)

        assert resp.buffer.size == len(body)
        assert resp.headers["Content-Length"] == str(len(body))
        assert raw.released is True

    def test_nonstreamed_drain_syncs_content_length(self):
        from yggdrasil.io.http_ import HTTPResponse

        body = b"x" * 4096
        raw = self._fake(body, declared_length=str(len(body)))
        resp = HTTPResponse.from_urllib3(
            request=make_request(),
            response=raw,
            tags=None,
            received_at=EPOCH,
        )

        resp.drain_urllib3(raw, stream=False)

        assert resp.headers["Content-Length"] == str(len(body))
        assert int(resp.headers["Content-Length"]) == resp.buffer.size

    def test_drain_overwrites_stale_remote_length(self):
        from yggdrasil.io.http_ import HTTPResponse

        body = b'{"a":1}'
        # Remote claims a wrong length — we still trust the bytes that
        # actually landed.
        raw = self._fake(body, declared_length="999999")
        resp = HTTPResponse.from_urllib3(
            request=make_request(),
            response=raw,
            tags=None,
            received_at=EPOCH,
        )

        resp.drain_urllib3(raw, stream=True)

        assert resp.headers["Content-Length"] == str(len(body))

    def test_drain_backfills_missing_content_type_from_buffer(self):
        from yggdrasil.io.http_ import HTTPResponse

        # JSON body, but the server didn't bother to send Content-Type.
        body = b'{"hello":"world"}'
        raw = _FakeUrllib3Response(body, headers={})
        resp = HTTPResponse.from_urllib3(
            request=make_request(),
            response=raw,
            tags=None,
            received_at=EPOCH,
        )

        resp.drain_urllib3(raw, stream=True)

        # Content-Type should now be present and reflect what the
        # buffer actually contains (json, not octet-stream).
        ct = resp.headers.get("Content-Type", "")
        assert ct, "Content-Type should be backfilled after drain"
        assert "json" in ct.lower(), f"expected json content-type, got {ct!r}"
        assert resp.headers["Content-Length"] == str(len(body))

    def test_drain_backfills_placeholder_content_type(self):
        from yggdrasil.io.http_ import HTTPResponse

        body = b'{"k":1}'
        raw = _FakeUrllib3Response(
            body,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp = HTTPResponse.from_urllib3(
            request=make_request(),
            response=raw,
            tags=None,
            received_at=EPOCH,
        )

        resp.drain_urllib3(raw, stream=True)

        # ``application/octet-stream`` is treated as a placeholder and
        # gets replaced with the sniffed media after drain.
        ct = resp.headers["Content-Type"]
        assert "octet-stream" not in ct.lower()
        assert "json" in ct.lower()

    def test_drain_preserves_explicit_content_type(self):
        from yggdrasil.io.http_ import HTTPResponse

        # Server picked a non-placeholder Content-Type — the drain
        # resync must not override it (the server's declaration wins).
        body = b'{"hello":"world"}'
        raw = _FakeUrllib3Response(
            body,
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )
        resp = HTTPResponse.from_urllib3(
            request=make_request(),
            response=raw,
            tags=None,
            received_at=EPOCH,
        )

        resp.drain_urllib3(raw, stream=True)

        assert resp.headers["Content-Type"].startswith("text/plain")
        assert resp.headers["Content-Length"] == str(len(body))
