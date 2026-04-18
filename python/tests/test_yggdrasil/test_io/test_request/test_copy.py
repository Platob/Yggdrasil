"""`PreparedRequest.copy` — field overrides, buffer aliasing, `update_*`."""

from __future__ import annotations

import datetime as dt

from yggdrasil.io.request import PreparedRequest


def test_copy_reuses_buffer_by_default() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        body=b"abc",
    )

    req2 = req.copy()

    assert req2.method == req.method
    assert req2.url.to_string() == req.url.to_string()
    assert req2.headers == req.headers
    # Default copy shares the underlying buffer — cheaper for hot send paths.
    assert req2.buffer is req.buffer


def test_copy_buffer_true_duplicates_payload() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        body=b"abc",
    )

    req2 = req.copy(copy_buffer=True)

    assert req2.buffer is not req.buffer
    assert req2.buffer is not None
    assert req.buffer is not None
    assert req2.buffer.to_bytes() == req.buffer.to_bytes()


def test_copy_can_override_core_fields() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com/a",
        headers={"X-A": "1"},
        tags={"t": "1"},
    )

    req2 = req.copy(
        method="POST",
        url="https://example.com/b",
        headers={"X-B": "2"},
        tags={"t": "2"},
        sent_at=123,
    )

    assert req2.method == "POST"
    assert req2.url.to_string() == "https://example.com/b"
    assert req2.headers == {"X-B": "2"}
    assert req2.tags == {"t": "2"}
    # int sent_at is interpreted as seconds since epoch.
    assert req2.sent_at == dt.datetime(1970, 1, 1, 0, 2, 3, tzinfo=dt.timezone.utc)


def test_update_headers_and_tags_mutate_in_place() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com?a=1",
        headers={"X-A": "1"},
        tags={"t1": "1"},
    )

    req.update_headers({"X-B": "2"}, normalize=False)
    req.update_tags({"t2": "2"})

    assert req.headers["X-A"] == "1"
    assert req.headers["X-B"] == "2"
    assert req.tags == {"t1": "1", "t2": "2"}
