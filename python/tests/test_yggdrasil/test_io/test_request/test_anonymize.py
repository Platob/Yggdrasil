"""`PreparedRequest.anonymize` — URL/credential redaction modes."""

from __future__ import annotations

from yggdrasil.io.request import PreparedRequest


def test_anonymize_redact_preserves_headers_but_scrubs_secrets() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://user:pass@example.com/path?token=secret&ok=1",
        headers={
            "Authorization": "Bearer super-secret",
            "X-API-Key": "abcdef",
        },
    )

    anon = req.anonymize(mode="redact")

    # anonymize must not mutate the original.
    assert anon is not req

    assert "Authorization" in anon.headers
    assert "X-API-Key" in anon.headers
    # The literal secret values are scrubbed.
    assert "super-secret" not in anon.headers["Authorization"]
    assert anon.headers["X-API-Key"] == "<redacted>"


def test_anonymize_remove_drops_credential_headers_entirely() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://user:pass@example.com/path",
        headers={
            "Authorization": "Bearer super-secret",
            "X-API-Key": "abcdef",
            "Accept": "application/json",
        },
    )

    anon = req.anonymize(mode="remove")

    assert "Authorization" not in anon.headers
    assert "X-API-Key" not in anon.headers
    # Non-sensitive headers are kept.
    assert anon.headers.get("Accept") == "application/json"


def test_anonymize_redact_scrubs_userinfo_from_url() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://user:pass@example.com/api",
    )

    anon = req.anonymize(mode="redact")

    # Username/password must not survive into the redacted URL.
    assert "user:pass" not in anon.url.to_string()
