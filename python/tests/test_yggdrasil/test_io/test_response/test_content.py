"""`Response` body / content / text / json / media_type."""

from __future__ import annotations

from yggdrasil.io import MimeTypes
from yggdrasil.io.enums import MediaType

from .._helpers import make_response


def test_body_content_text_and_json_round_trip() -> None:
    resp = make_response(
        body=b'{"x":1}',
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    assert resp.body.to_bytes() == b'{"x":1}'
    assert resp.content == b'{"x":1}'
    assert resp.text == '{"x":1}'
    assert resp.json() == {"x": 1}


def test_media_type_and_set_media_type_mirrors_onto_accept() -> None:
    resp = make_response(body=b"hello", headers={"Content-Type": "application/octet-stream"})

    # The default media type is readable as a MediaType.
    assert resp.media_type.mime_type is not None

    resp.set_media_type(MediaType(MimeTypes.JSON, None))

    assert resp.headers["Content-Type"] == MimeTypes.JSON.value
    # Setting Content-Type on the response also reflects into the request's
    # Accept header so follow-up requests remain consistent.
    assert resp.request.headers["Accept"] == MimeTypes.JSON.value
