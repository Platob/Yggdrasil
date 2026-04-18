"""`Response.anonymize` — redact URLs, auth headers, and API keys before caching."""

from __future__ import annotations

from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response


def test_anonymize_redact_keeps_headers_but_scrubs_values() -> None:
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
    # userinfo must not survive
    assert "user:pass" not in anon.request.url.to_string()
    assert "Authorization" in anon.request.headers
    # API key is scrubbed but the header remains.
    assert "X-API-Key" in anon.headers
    assert anon.headers["X-API-Key"] == "<redacted>"


def test_anonymize_remove_drops_sensitive_headers_entirely() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com/a",
        headers={"Authorization": "Bearer abc"},
    )
    resp = Response(
        request=req,
        status_code=200,
        headers={"X-API-Key": "secret", "Content-Type": "application/json"},
        tags={},
        buffer=b'{"ok":true}',  # type: ignore[arg-type]
        received_at=1,
    )

    anon = resp.anonymize(mode="remove")

    assert "Authorization" not in anon.request.headers
    assert "X-API-Key" not in anon.headers
    # Non-sensitive response headers are preserved.
    assert anon.headers["Content-Type"] == "application/json"
