"""Regression tests for the memoised response media-type resolution.

``HTTPResponse`` construction normalises ``Content-Type`` /
``Content-Encoding`` into a :class:`MediaType` on every response. That
resolution is the same pure function over two header strings, so it is
memoised (``_media_from_mime_strings`` + the hoisted ``_OCTET_MEDIA``
fallback). These tests pin that the cache is *correct*: the right
MediaType per header pair, distinct pairs don't collide, and — because
:class:`MediaType` is frozen — repeated lookups return the identical
shared instance rather than a per-call rebuild.
"""
from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.enums import MediaType, MimeTypes
from yggdrasil.http_.response import (
    HTTPResponse,
    _media_from_mime_strings,
    _media_type_from_headers,
    _OCTET_MEDIA,
)
from yggdrasil.http_.request import HTTPRequest
from yggdrasil.path.memory import Memory
from yggdrasil.url import URL


def _resp(headers: dict[str, str], body: bytes = b"{}") -> HTTPResponse:
    req = HTTPRequest(method="GET", url=URL.from_str("https://x/y"), headers={},
                      tags={}, buffer=None, sent_at=0)
    return HTTPResponse(
        request=req,
        status_code=200,
        headers=headers,
        tags={},
        buffer=Memory(binary=body),
        received_at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
    )


# ---------------------------------------------------------------------------
# Correctness of the memoised resolver
# ---------------------------------------------------------------------------


def test_octet_fallback_is_shared_constant():
    assert _media_from_mime_strings(None, None) is _OCTET_MEDIA
    assert _OCTET_MEDIA.mime_type is MimeTypes.OCTET_STREAM


def test_json_content_type_resolves():
    mt = _media_from_mime_strings("application/json", None)
    assert mt.mime_type is MimeTypes.JSON


def test_arrow_stream_content_type_resolves():
    mt = _media_from_mime_strings("application/vnd.apache.arrow.stream", None)
    assert mt.mime_type is MimeTypes.ARROW_STREAM


def test_repeated_lookup_returns_identical_instance():
    # MediaType is frozen, so the memoised lookup hands back the same
    # object — not a fresh-but-equal one — on every call.
    a = _media_from_mime_strings("application/json", None)
    b = _media_from_mime_strings("application/json", None)
    assert a is b


def test_distinct_header_pairs_do_not_collide():
    js = _media_from_mime_strings("application/json", None)
    csv = _media_from_mime_strings("text/csv", None)
    gz = _media_from_mime_strings("application/json", "gzip")
    assert js.mime_type is MimeTypes.JSON
    assert csv.mime_type is MimeTypes.CSV
    # The encoding arm participates in the key — same content-type, but a
    # codec changes the resolved MediaType.
    assert gz != js
    assert gz.codec is not None


def test_placeholder_content_type_is_not_sniffed_from_headers():
    # octet-stream / unknown placeholders return None from the
    # header-only resolver so the body sniff can take over.
    assert _media_type_from_headers({"Content-Type": "application/octet-stream"}) is None
    assert _media_type_from_headers({"Content-Type": "unknown/unknown"}) is None
    assert _media_type_from_headers({}) is None


def test_content_encoding_alone_resolves_from_headers():
    mt = _media_type_from_headers({"Content-Encoding": "gzip"})
    assert mt is not None
    assert mt.codec is not None


# ---------------------------------------------------------------------------
# End-to-end through HTTPResponse construction
# ---------------------------------------------------------------------------


def test_response_stamps_media_type_from_json_header():
    r = _resp({"Content-Type": "application/json"})
    assert r.media_type.mime_type is MimeTypes.JSON


def test_response_two_constructions_share_media_instance():
    a = _resp({"Content-Type": "application/json"})
    b = _resp({"Content-Type": "application/json"})
    # Same memoised resolution underneath both responses.
    assert a.media_type is b.media_type


def test_response_placeholder_content_type_stays_octet_without_magic():
    # An octet-stream placeholder over a JSON byte body stays octet-stream:
    # JSON has no magic signature to sniff, so there's nothing to promote
    # it to. (Binary formats with magic bytes — parquet, gzip — would be
    # detected; plain JSON text is not.) This pins that the memoised
    # resolver doesn't accidentally "upgrade" a placeholder.
    r = _resp({"Content-Type": "application/octet-stream"}, body=b'{"a": 1}')
    assert r.media_type.mime_type is MimeTypes.OCTET_STREAM
