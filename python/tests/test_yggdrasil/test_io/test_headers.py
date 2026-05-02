"""Tests for yggdrasil.io.headers."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.headers import (
    DEFAULT_HOSTNAME,
    PromotedHeaders,
    SENSITIVE_HEADER_KEYS,
    get_default_headers,
    get_default_user_agent,
    normalize_headers,
)


# ---------------------------------------------------------------------------
# Default helpers
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_hostname_is_string(self):
        assert isinstance(DEFAULT_HOSTNAME, str)
        assert DEFAULT_HOSTNAME

    def test_get_default_headers_returns_dict(self):
        d = get_default_headers()
        assert isinstance(d, dict)
        assert "X-Ygg-Version" in d
        assert "X-Py-Version" in d

    def test_get_default_user_agent_starts_with_yggdrasil(self):
        ua = get_default_user_agent()
        assert ua.startswith("yggdrasil/") or "yggdrasil/" in ua


# ---------------------------------------------------------------------------
# PromotedHeaders.extract
# ---------------------------------------------------------------------------


class TestPromotedExtract:
    def test_promotes_known_headers(self):
        promoted = PromotedHeaders.extract(
            {"Content-Type": "application/json", "Content-Length": "12"}
        )
        assert promoted.content_type == "application/json"
        assert promoted.content_length == 12

    def test_remaining_holds_unrecognized(self):
        promoted = PromotedHeaders.extract({"X-Custom": "v"})
        assert promoted.remaining.get("X-Custom") == "v"

    def test_normalizes_header_name(self):
        promoted = PromotedHeaders.extract({"content-type": "text/plain"})
        assert promoted.content_type == "text/plain"

    def test_invalid_content_length_falls_back_to_zero(self):
        promoted = PromotedHeaders.extract({"Content-Length": "abc"})
        assert promoted.content_length == 0

    def test_host_kwarg_seeds_remaining(self):
        promoted = PromotedHeaders.extract({}, host="example.com")
        assert promoted.host == "example.com"

    def test_values_property(self):
        promoted = PromotedHeaders.extract({"User-Agent": "test"})
        values = promoted.values
        assert values["user_agent"] == "test"


# ---------------------------------------------------------------------------
# normalize_headers
# ---------------------------------------------------------------------------


class TestNormalizeHeaders:
    def test_canonicalizes_keys(self):
        result = normalize_headers(
            {"content-type": "text/plain"},
            is_request=False,
            add_missing=False,
        )
        assert "Content-Type" in result

    def test_strips_sensitive_in_remove_mode(self):
        result = normalize_headers(
            {"Authorization": "Bearer abc", "X-Other": "v"},
            is_request=False,
            mode="remove",
            anonymize=True,
            add_missing=False,
        )
        assert "Authorization" not in result
        assert result["X-Other"] == "v"

    def test_redacts_sensitive_in_redact_mode(self):
        result = normalize_headers(
            {"Authorization": "Bearer abc"},
            is_request=False,
            mode="redact",
            anonymize=True,
            add_missing=False,
        )
        assert result["Authorization"] == "Bearer <redacted>"

    def test_basic_auth_redacted(self):
        result = normalize_headers(
            {"Authorization": "Basic ZXhhbXBsZQ=="},
            is_request=False,
            mode="redact",
            anonymize=True,
            add_missing=False,
        )
        assert result["Authorization"] == "Basic <redacted>"

    def test_request_adds_default_user_agent(self):
        result = normalize_headers(
            {},
            is_request=True,
            add_missing=True,
        )
        assert result.get("User-Agent")

    def test_request_adds_accept_default(self):
        result = normalize_headers(
            {},
            is_request=True,
            add_missing=True,
        )
        assert result.get("Accept") == "*/*"

    def test_request_with_existing_user_agent_preserved(self):
        result = normalize_headers(
            {"User-Agent": "custom/1.0"},
            is_request=True,
            add_missing=True,
        )
        assert result["User-Agent"] == "custom/1.0"

    def test_body_drives_content_length_when_missing(self):
        body = BytesIO(b"hello world")
        result = normalize_headers(
            {},
            is_request=False,
            body=body,
            add_missing=True,
        )
        assert int(result["Content-Length"]) == body.size

    def test_explicit_content_length_preserved(self):
        body = BytesIO(b"hello")
        result = normalize_headers(
            {"Content-Length": "42"},
            is_request=False,
            body=body,
            add_missing=True,
        )
        assert result["Content-Length"] == "42"


class TestSensitiveHeaderRegistry:
    def test_authorization_listed(self):
        assert "authorization" in SENSITIVE_HEADER_KEYS

    def test_cookie_listed(self):
        assert "cookie" in SENSITIVE_HEADER_KEYS
