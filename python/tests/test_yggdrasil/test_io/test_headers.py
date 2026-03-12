# tests/test_headers.py
from __future__ import annotations

from yggdrasil.io.enums import MediaType, GZIP
from yggdrasil.io.headers import PromotedHeaders, normalize_headers


class _DummyBody:
    def __init__(self, size: int, media_type: MediaType):
        self.size = size
        self.media_type = media_type


def test_promoted_headers_extract_common_headers():
    headers = {
        "host": "api.example.com",
        "USER-AGENT": "pytest/1.0",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, br",
        "Accept-Language": "en-US",
        "Content-Type": "application/json",
        "Content-Length": "123",
        "Content-Encoding": "gzip",
        "X-Request-ID": "req-1",
        "X-Correlation-ID": "corr-1",
        "X-Custom": "value",
    }

    promoted = PromotedHeaders.extract(headers)

    assert promoted.host == "api.example.com"
    assert promoted.user_agent == "pytest/1.0"
    assert promoted.accept == "application/json"
    assert promoted.accept_encoding == "gzip, br"
    assert promoted.accept_language == "en-US"
    assert promoted.content_type == "application/json"
    assert promoted.content_length == 123
    assert promoted.content_encoding == "gzip"
    assert promoted.x_request_id == "req-1"
    assert promoted.x_correlation_id == "corr-1"
    assert promoted.remaining == {"X-Custom": "value"}


def test_promoted_headers_extract_is_case_insensitive():
    headers = {
        "HoSt": "example.com",
        "x-custom-header": "abc",
    }

    promoted = PromotedHeaders.extract(headers)

    assert promoted.host == "example.com"
    assert promoted.remaining == {"x-custom-header": "abc"}


def test_promoted_headers_invalid_content_length_becomes_none():
    headers = {
        "Content-Length": "nope",
        "X-Test": "1",
    }

    promoted = PromotedHeaders.extract(headers)

    assert promoted.content_length == 0
    assert promoted.remaining == {"X-Test": "1"}


def test_promoted_headers_values_mapping():
    headers = {
        "Host": "example.com",
        "Content-Length": "42",
    }

    promoted = PromotedHeaders.extract(headers)

    assert promoted.values["host"] == "example.com"
    assert promoted.values["content_length"] == 42
    assert promoted.values["content_type"] is None


def test_normalize_headers_canonicalizes_names_without_anonymizing():
    headers = {
        "content-type": "application/json",
        "content-length": "12",
        "content-encoding": "gzip",
        "user-agent": "pytest",
    }

    out = normalize_headers(headers)

    assert out == {
        "Content-Type": "application/json",
        "Content-Length": "12",
        "Content-Encoding": "gzip",
        "User-Agent": "pytest",
    }


def test_normalize_headers_removes_sensitive_headers_in_remove_mode():
    headers = {
        "Authorization": "Bearer secret-token",
        "Cookie": "session=abc",
        "X-Test": "ok",
    }

    out = normalize_headers(headers, anonymize=True, mode="remove")

    assert out == {"X-Test": "ok"}


def test_normalize_headers_redacts_bearer_authorization():
    headers = {
        "Authorization": "Bearer very-secret-token",
        "X-Test": "ok",
    }

    out = normalize_headers(headers, anonymize=True, mode="redact")

    assert out["Authorization"] == "Bearer <redacted>"
    assert out["X-Test"] == "ok"


def test_normalize_headers_redacts_basic_authorization():
    headers = {
        "Authorization": "Basic dXNlcjpwYXNz",
    }

    out = normalize_headers(headers, anonymize=True, mode="redact")

    assert out["Authorization"] == "Basic <redacted>"


def test_normalize_headers_redacts_unknown_authorization_scheme():
    headers = {
        "Authorization": "Digest something-secret",
    }

    out = normalize_headers(headers, anonymize=True, mode="redact")

    assert out["Authorization"] == "<redacted>"


def test_normalize_headers_redacts_jwt_like_values():
    jwt_like = "aaaaabbbbbccccccdddddeeeee.fffffggggghhhhhiiiii.jjjjjkkkkklllllmmmm"
    headers = {
        "X-Custom": jwt_like,
    }

    out = normalize_headers(headers, anonymize=True, mode="redact")

    assert out["X-Custom"] == "<redacted>"


def test_normalize_headers_backfills_body_headers():
    body = _DummyBody(
        size=99,
        media_type=MediaType.parse("application/json"),
    )

    out = normalize_headers({}, body=body)

    assert out["Content-Type"] == "application/json"
    assert out["Content-Length"] == "99"
    assert "Content-Encoding" not in out


def test_normalize_headers_backfills_content_encoding_when_codec_exists():
    media_type = MediaType.parse("application/json")
    media_type = MediaType(mime_type=media_type.mime_type, codec=GZIP)

    body = _DummyBody(
        size=99,
        media_type=media_type,
    )

    out = normalize_headers({}, body=body)

    assert out["Content-Type"] == "application/json"
    assert out["Content-Length"] == "99"
    assert out["Content-Encoding"] == "gzip"


def test_normalize_headers_does_not_override_existing_body_headers():
    media_type = MediaType.parse("application/json")
    media_type = MediaType(mime_type=media_type.mime_type, codec=GZIP)

    body = _DummyBody(
        size=99,
        media_type=media_type,
    )

    out = normalize_headers(
        {
            "Content-Type": "text/plain",
            "Content-Length": "7",
            "Content-Encoding": "chunked",
        },
        body=body,
    )

    assert out["Content-Type"] == "text/plain"
    assert out["Content-Length"] == "7"
    assert out["Content-Encoding"] == "chunked"


def test_normalize_headers_bytes_input_is_supported():
    headers = {
        b"content-type": b"application/json",
        b"user-agent": b"pytest",
    }

    out = normalize_headers(headers)

    assert out == {
        "Content-Type": "application/json",
        "User-Agent": "pytest",
    }


def test_hash_mode_currently_matches_redact_behavior():
    headers = {
        "Cookie": "secret=1",
    }

    out = normalize_headers(headers, anonymize=True, mode="hash")

    assert out["Cookie"] == "<redacted>"