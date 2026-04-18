"""`PreparedRequest` convenience properties — body, auth, api key, media type."""

from __future__ import annotations

from yggdrasil.io import MediaType, MimeTypes
from yggdrasil.io.request import PreparedRequest


def test_body_is_alias_for_buffer() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        body=b"abc",
    )

    assert req.body is req.buffer
    assert req.body is not None
    assert req.body.to_bytes() == b"abc"


def test_authorization_and_x_api_key_roundtrip() -> None:
    req = PreparedRequest.prepare(method="GET", url="https://example.com")

    req.authorization = "Bearer abc"
    req.x_api_key = "secret"

    assert req.authorization == "Bearer abc"
    assert req.x_api_key == "secret"

    req.authorization = None
    req.x_api_key = None

    assert req.authorization is None
    assert req.x_api_key is None


def test_accept_media_type_roundtrips_through_headers() -> None:
    req = PreparedRequest.prepare(method="GET", url="https://example.com")

    req.accept_media_type = MediaType(MimeTypes.JSON, None)

    assert req.headers["Accept"] == MimeTypes.JSON.value
    assert req.accept_media_type.mime_type == MimeTypes.JSON
