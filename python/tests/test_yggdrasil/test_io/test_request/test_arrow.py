"""Arrow projection / roundtrip for `PreparedRequest`.

`arrow_values`, `to_arrow_batch/table`, `from_arrow`, and `match_value`
all rely on the same `REQUEST_ARROW_SCHEMA` — keep them verified together
so schema drift breaks loudly.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from yggdrasil.io.headers import DEFAULT_HOSTNAME
from yggdrasil.io.request import REQUEST_ARROW_SCHEMA, PreparedRequest


def _row(batch) -> dict[str, Any]:
    return {name: batch.column(name)[0].as_py() for name in batch.schema.names}


# ---------------------------------------------------------------------------
# arrow_values — used by CacheConfig.match_value, so keep projection stable
# ---------------------------------------------------------------------------

def test_arrow_values_projects_expected_columns() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1",
        headers={"User-Agent": "pytest"},
        body=b"hello",
        tags={"t": "2"},
    )

    values = req.arrow_values

    assert values["request_method"] == "POST"
    assert values["request_url_str"] == "https://example.com/api?a=1"
    assert values["request_url_host"] == "example.com"
    assert values["request_url_path"] == "/api"
    assert values["request_url_query"] == "a=1"
    assert values["request_user_agent"] == "pytest"
    assert values["request_content_length"] == 5
    assert values["request_body"] == b"hello"
    assert values["request_body_hash"] is not None
    assert values["request_tags"] == {"a": "1", "t": "2"}
    assert values["request_sent_at"] == dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# match_value / match_tuple — single source of keys for CacheConfig
# ---------------------------------------------------------------------------

def test_match_value_supports_every_schema_field() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://user@example.com/api?a=1#frag",
        headers={"Content-Type": "application/json"},
        body=b"hello",
    )

    assert req.match_value("request_method") == "POST"
    assert req.match_value("request_url_str") == "https://user@example.com/api?a=1#frag"
    assert req.match_value("request_url_scheme") == "https"
    assert req.match_value("request_url_userinfo") == "user"
    assert req.match_value("request_url_host") == "example.com"
    assert req.match_value("request_url_path") == "/api"
    assert req.match_value("request_url_query") == "a=1"
    assert req.match_value("request_url_fragment") == "frag"
    assert req.match_value("request_content_type") == "application/json"
    assert req.match_value("request_content_length") == 5
    assert req.match_value("request_body") == b"hello"
    assert req.match_value("request_body_hash") is not None


def test_match_value_rejects_unknown_key() -> None:
    req = PreparedRequest.prepare(method="GET", url="https://example.com")

    with pytest.raises(ValueError, match="Unsupported request match key"):
        req.match_value("not_a_real_key")


def test_match_tuple_returns_values_in_argument_order() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?q=1",
        body=b"hello",
    )

    out = req.match_tuple(["request_method", "request_url_str", "request_body_hash"])

    assert out[0] == "POST"
    assert out[1] == "https://example.com/api?q=1"
    assert out[2] is not None


# ---------------------------------------------------------------------------
# to_arrow_batch — schema parity and header promotion
# ---------------------------------------------------------------------------

def test_to_arrow_batch_matches_schema() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?q=1",
        headers={
            "X-Test": "yes",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "Location": "/foo",
            "ETag": '"etag-1"',
            "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
        },
        body=b"hello world",
        tags={"explicit": "tag"},
    )
    req.sent_at = 123456789

    rb = req.to_arrow_batch()

    assert rb.schema == REQUEST_ARROW_SCHEMA
    assert rb.num_rows == 1


def test_to_arrow_batch_promotes_well_known_headers_and_keeps_the_rest() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1&shared=url",
        headers={
            "Host": "example.com",
            "User-Agent": "pytest",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Accept-Language": "en",
            "Content-Type": "application/json",
            "Content-Length": "11",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "X-Request-ID": "rid-1",
            "X-Correlation-ID": "cid-1",
            "X-Other": "keep-me",
        },
        body=b"hello world",
        tags={"shared": "explicit", "t": "2"},
    )
    req.sent_at = 42

    row = _row(req.to_arrow_batch())

    # Promoted columns
    assert row["request_method"] == "POST"
    assert row["request_url_str"] == "https://example.com/api?a=1&shared=url"
    assert row["request_url_scheme"] == "https"
    assert row["request_url_host"] == "example.com"
    assert row["request_url_path"] == "/api"
    assert row["request_url_query"] == "a=1&shared=url"
    assert row["request_host"] == DEFAULT_HOSTNAME
    assert row["request_user_agent"] == "pytest"
    assert row["request_accept"] == "application/json"
    assert row["request_accept_encoding"] == "gzip"
    assert row["request_accept_language"] == "en"
    assert row["request_content_type"] == "application/json"
    assert row["request_content_length"] == 11
    assert row["request_content_encoding"] == "gzip"
    assert row["request_transfer_encoding"] == "chunked"

    # Un-promoted headers land in the "remaining" map.
    assert dict(row["request_headers"])["X-Other"] == "keep-me"

    # Tags merged with the flattened URL query parameters.
    assert dict(row["request_tags"]) == {"a": "1", "shared": "explicit", "t": "2"}

    assert row["request_body"] == b"hello world"
    assert row["request_body_hash"] == req.buffer.xxh3_int64()  # type: ignore[union-attr]
    assert row["request_sent_at"] == dt.datetime(1970, 1, 1, 0, 0, 0, 42, tzinfo=dt.timezone.utc)


def test_to_arrow_batch_separates_content_encoding_from_transfer_encoding() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com",
        headers={"Content-Encoding": "gzip", "Transfer-Encoding": "chunked"},
        body=b"abc",
    )

    row = _row(req.to_arrow_batch())

    assert row["request_content_encoding"] == "gzip"
    assert row["request_transfer_encoding"] == "chunked"


def test_to_arrow_batch_without_body_has_null_body_and_hash() -> None:
    req = PreparedRequest.prepare(method="GET", url="https://example.com")

    row = _row(req.to_arrow_batch())

    assert row["request_body"] is None
    assert row["request_body_hash"] is None


# ---------------------------------------------------------------------------
# from_arrow — roundtrip parity (record batch + table)
# ---------------------------------------------------------------------------

def test_from_arrow_roundtrip_record_batch() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1",
        headers={
            "User-Agent": "pytest",
            "Content-Type": "application/json",
            "X-Other": "keep",
        },
        body=b"hello",
        tags={"t": "2"},
    )
    req.sent_at = dt.datetime(1970, 1, 1, 0, 2, 3, tzinfo=dt.timezone.utc)

    rebuilt_list = list(PreparedRequest.from_arrow(req.to_arrow_batch()))

    assert len(rebuilt_list) == 1
    rebuilt = rebuilt_list[0]

    assert rebuilt.method == req.method
    assert rebuilt.url.to_string() == req.url.to_string()
    assert rebuilt.buffer is not None
    assert rebuilt.buffer.to_bytes() == b"hello"
    assert rebuilt.tags == {"a": "1", "t": "2"}
    assert rebuilt.sent_at == dt.datetime(1970, 1, 1, 0, 2, 3, tzinfo=dt.timezone.utc)
    assert rebuilt.headers["User-Agent"] == "pytest"
    assert rebuilt.headers["Content-Type"] == "application/json"


def test_from_arrow_roundtrip_table() -> None:
    req = PreparedRequest.prepare(
        method="GET",
        url="https://example.com/path?q=1",
        headers={"Accept": "application/json"},
    )
    req.sent_at = dt.datetime(1970, 1, 1, 0, 0, 7, tzinfo=dt.timezone.utc)

    rebuilt_list = list(PreparedRequest.from_arrow(req.to_arrow_table()))

    assert len(rebuilt_list) == 1
    rebuilt = rebuilt_list[0]

    assert rebuilt.method == "GET"
    assert rebuilt.url.to_string() == "https://example.com/path?q=1"
    assert rebuilt.sent_at == dt.datetime(1970, 1, 1, 0, 0, 7, tzinfo=dt.timezone.utc)
