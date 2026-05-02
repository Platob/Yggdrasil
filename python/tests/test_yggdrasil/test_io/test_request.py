"""Tests for yggdrasil.io.request.PreparedRequest."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.request import PreparedRequest, REQUEST_ARROW_SCHEMA
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_minimum_required_args(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        assert req.method == "GET"
        assert isinstance(req.url, URL)
        assert req.url.host == "example.com"

    def test_url_is_normalized(self):
        req = PreparedRequest.prepare(method="GET", url="HTTPS://EXAMPLE.com/")
        assert req.url.scheme == "https"
        assert req.url.host == "example.com"

    def test_body_is_wrapped_as_bytesio(self):
        req = PreparedRequest.prepare(
            method="POST", url="https://example.com/", body=b"payload"
        )
        assert isinstance(req.buffer, BytesIO)
        assert req.buffer.to_bytes() == b"payload"
        # Content-Length is added from body
        assert req.headers.get("Content-Length") == str(len(b"payload"))

    def test_json_sets_content_type(self):
        req = PreparedRequest.prepare(
            method="POST", url="https://example.com/", json={"a": 1}
        )
        assert req.headers["Content-Type"] == "application/json"
        assert req.buffer is not None
        assert b'"a"' in req.buffer.to_bytes()

    def test_routes_to_http_subclass_for_http_url(self):
        from yggdrasil.io.http_ import HTTPRequest

        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        assert isinstance(req, HTTPRequest)

    def test_default_method_when_post_init(self):
        # Direct construction via __post_init__ defaults blank method to GET.
        req = PreparedRequest(method="", url="https://example.com/", headers={}, tags={}, buffer=None, sent_at=None)
        assert req.method == "GET"


# ---------------------------------------------------------------------------
# parse / parse_mapping
# ---------------------------------------------------------------------------


class TestParse:
    def test_parse_string_url(self):
        req = PreparedRequest.parse("https://example.com/")
        assert req.url.host == "example.com"

    def test_parse_mapping_with_url_str(self):
        req = PreparedRequest.parse_mapping(
            {"url_str": "https://example.com/x", "method": "POST"}
        )
        assert req.method == "POST"
        assert req.url.path == "/x"

    def test_parse_mapping_missing_url_raises(self):
        with pytest.raises(ValueError):
            PreparedRequest.parse_mapping({"method": "GET"})


# ---------------------------------------------------------------------------
# Properties / accessors
# ---------------------------------------------------------------------------


class TestProperties:
    def test_body_property_alias(self):
        req = PreparedRequest.prepare(
            method="POST", url="https://example.com/", body=b"x"
        )
        assert req.body is req.buffer

    def test_content_length(self):
        req = PreparedRequest.prepare(
            method="POST", url="https://example.com/", body=b"abcd"
        )
        assert req.content_length == 4

    def test_authorization_setter_getter(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        req.authorization = "Bearer xyz"
        assert req.authorization == "Bearer xyz"
        assert req.headers.get("Authorization") == "Bearer xyz"

    def test_authorization_clear(self):
        req = PreparedRequest.prepare(
            method="GET",
            url="https://example.com/",
            headers={"Authorization": "Bearer xyz"},
        )
        req.authorization = None
        assert req.authorization is None

    def test_x_api_key(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        req.x_api_key = "abc"
        assert req.x_api_key == "abc"


# ---------------------------------------------------------------------------
# Mutation: copy / update
# ---------------------------------------------------------------------------


class TestCopy:
    def test_copy_preserves_method_and_url(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        clone = req.copy()
        assert clone.method == req.method
        assert clone.url == req.url

    def test_copy_with_method_override(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        clone = req.copy(method="POST")
        assert clone.method == "POST"
        assert req.method == "GET"

    def test_copy_with_url_override(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        clone = req.copy(url="https://example.com/y")
        assert clone.url.path == "/y"

    def test_copy_buffer_deep(self):
        req = PreparedRequest.prepare(
            method="POST", url="https://example.com/", body=b"original"
        )
        clone = req.copy(copy_buffer=True)
        clone.buffer.seek(0)
        clone.buffer.write(b"X")
        # Original buffer is untouched
        assert req.buffer.to_bytes() == b"original"


class TestUpdates:
    def test_update_headers(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        req.update_headers({"X-Custom": "v"})
        assert req.headers.get("X-Custom") == "v"

    def test_update_tags(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        req.update_tags({"tag": "v"})
        assert req.tags.get("tag") == "v"

    def test_update_headers_empty_no_op(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        prev = dict(req.headers)
        req.update_headers({})
        assert req.headers == prev


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestHashing:
    def test_xxh3_64_deterministic(self):
        a = PreparedRequest.prepare(method="GET", url="https://example.com/")
        b = PreparedRequest.prepare(method="GET", url="https://example.com/")
        assert a.xxh3_64().intdigest() == b.xxh3_64().intdigest()

    def test_xxh3_b64_returns_str(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        assert isinstance(req.xxh3_b64(), str)


# ---------------------------------------------------------------------------
# Anonymize
# ---------------------------------------------------------------------------


class TestAnonymize:
    def test_strips_authorization_in_remove_mode(self):
        req = PreparedRequest.prepare(
            method="GET",
            url="https://example.com/",
            headers={"Authorization": "Bearer xyz"},
        )
        sanitized = req.anonymize("remove")
        assert "Authorization" not in sanitized.headers

    def test_redacts_authorization(self):
        req = PreparedRequest.prepare(
            method="GET",
            url="https://example.com/",
            headers={"Authorization": "Bearer xyz"},
        )
        sanitized = req.anonymize("redact")
        assert sanitized.headers.get("Authorization") == "Bearer <redacted>"

    def test_strips_userinfo_from_url(self):
        req = PreparedRequest.prepare(
            method="GET", url="https://alice:pw@example.com/"
        )
        sanitized = req.anonymize("remove")
        assert sanitized.url.userinfo is None


# ---------------------------------------------------------------------------
# Arrow round-trip
# ---------------------------------------------------------------------------


class TestArrowRoundtrip:
    def test_to_arrow_batch_uses_request_schema(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        batch = req.to_arrow_batch()
        assert isinstance(batch, pa.RecordBatch)
        assert batch.schema == REQUEST_ARROW_SCHEMA

    def test_to_arrow_table(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        tbl = req.to_arrow_table()
        assert isinstance(tbl, pa.Table)
        assert tbl.num_rows == 1

    def test_from_arrow_round_trip(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        tbl = req.to_arrow_table()
        rebuilt = list(PreparedRequest.from_arrow(tbl))
        assert len(rebuilt) == 1
        assert rebuilt[0].method == "GET"
        assert rebuilt[0].url.host == "example.com"


# ---------------------------------------------------------------------------
# match_value / match_values / match_tuple
# ---------------------------------------------------------------------------


class TestMatchValue:
    def test_basic_lookup(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        assert req.match_value("request_method") == "GET"
        assert req.match_value("request_url_host") == "example.com"

    def test_unsupported_key_raises(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        with pytest.raises(ValueError):
            req.match_value("not_a_real_key")

    def test_match_values_returns_dict(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        values = req.match_values(["request_method", "request_url_host"])
        assert values == {"request_method": "GET", "request_url_host": "example.com"}

    def test_match_tuple_preserves_order(self):
        req = PreparedRequest.prepare(method="POST", url="https://example.com/x")
        tup = req.match_tuple(["request_method", "request_url_path"])
        assert tup == ("POST", "/x")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_runs_callable(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/")
        result = req.apply(lambda r: r.copy(method="POST"))
        assert result.method == "POST"


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_url_and_method(self):
        req = PreparedRequest.prepare(method="GET", url="https://example.com/x")
        text = repr(req)
        assert "GET" in text
        assert "example.com" in text
