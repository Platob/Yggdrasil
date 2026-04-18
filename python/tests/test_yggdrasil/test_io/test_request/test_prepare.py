"""`PreparedRequest.prepare` — body/json encoding, compression, `before_send`."""

from __future__ import annotations

import datetime as dt

from yggdrasil.io import MimeTypes
from yggdrasil.io.request import PreparedRequest


def test_prepare_with_raw_body_sets_content_length() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/upload",
        body=b"payload",
    )

    assert req.method == "POST"
    assert req.buffer is not None
    assert req.buffer.to_bytes() == b"payload"
    assert req.headers["Content-Length"] == str(len(b"payload"))


def test_prepare_with_json_sets_content_type_and_serialized_body() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/items",
        json={"x": 1},
    )

    assert req.buffer is not None
    assert req.headers["Content-Type"] == MimeTypes.JSON.value
    assert req.headers["Content-Length"] == str(req.buffer.size)

    body = req.buffer.to_bytes()
    # Accept either compact or indented JSON layout.
    assert b'"x": 1' in body or b'"x":1' in body


def test_prepare_compresses_when_threshold_exceeded() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/items",
        json={"data": "x" * 5000},
        compress_threshold=1,
    )

    assert req.buffer is not None
    assert req.headers["Content-Type"] == MimeTypes.JSON.value
    assert "Content-Encoding" in req.headers
    # Content-Length must track the *compressed* buffer.
    assert req.headers["Content-Length"] == str(req.buffer.size)


def test_prepare_normalizes_header_case() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        headers={"content-type": "application/custom"},
        body=b"abc",
        normalize=True,
    )

    assert "Content-Type" in req.headers
    assert req.headers["Content-Type"] == "application/custom"
    assert req.headers["Content-Length"] == "3"


def test_prepare_to_send_applies_before_send_hook() -> None:
    def before_send(req: PreparedRequest) -> PreparedRequest:
        req.headers["X-Test-Before-Send"] = "1"
        return req

    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com",
        before_send=before_send,
    )

    out = req.prepare_to_send(sent_at=1, headers=None)

    assert out.headers["X-Test-Before-Send"] == "1"
    assert out.sent_at == dt.datetime(1970, 1, 1, 0, 0, 1, tzinfo=dt.timezone.utc)


def test_prepare_to_send_merges_header_overrides() -> None:
    req = PreparedRequest.prepare(method="GET", url="https://example.com")

    out = req.prepare_to_send(sent_at=5, headers={"X-Injected": "1"})

    assert out.headers["X-Injected"] == "1"
    assert out.sent_at == dt.datetime(1970, 1, 1, 0, 0, 5, tzinfo=dt.timezone.utc)
