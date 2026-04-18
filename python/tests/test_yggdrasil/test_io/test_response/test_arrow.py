"""Arrow projection / roundtrip for `Response`.

Same pattern as `test_request/test_arrow.py`: keep `arrow_values`,
`to_arrow_batch`, `match_value`, and `from_arrow_tabular` verified together
so schema drift trips this file first.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import RESPONSE_ARROW_SCHEMA, Response

from .._helpers import make_request, make_response


def _row(batch) -> dict[str, Any]:
    return {name: batch.column(name)[0].as_py() for name in batch.schema.names}


# ---------------------------------------------------------------------------
# arrow_values — every cache match_value lookup funnels through here
# ---------------------------------------------------------------------------

def test_arrow_values_projects_request_and_response_columns() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1",
        headers={"User-Agent": "pytest"},
        body=b"rq",
        tags={"rt": "1"},
    )
    req.sent_at = dt.datetime.fromtimestamp(7, tz=dt.timezone.utc)

    resp = Response(
        request=req,
        status_code=201,
        headers={
            "User-Agent": "server",
            "Content-Type": "application/json",
            "X-Other": "keep",
        },
        tags={"t": "2"},
        buffer=b'{"ok":true}',  # type: ignore[arg-type]
        received_at=42,
    )

    values = resp.arrow_values

    assert values["request_method"] == "POST"
    assert values["request_url_str"] == "https://example.com/api?a=1"
    assert values["response_status_code"] == 201
    assert values["response_user_agent"] == "server"
    assert values["response_content_type"] == "application/json"
    assert values["response_body"] == b'{"ok":true}'
    assert values["response_body_hash"] is not None
    assert values["response_tags"] == {"t": "2"}
    assert values["response_received_at"] == dt.datetime(1970, 1, 1, 0, 0, 42, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# match_value / match_tuple
# ---------------------------------------------------------------------------

def test_match_value_covers_request_and_response_keys() -> None:
    resp = Response(
        request=PreparedRequest.prepare(
            method="POST",
            url="https://example.com/api?a=1",
            body=b"rq",
        ),
        status_code=200,
        headers={"Content-Type": "application/json"},
        tags={"t": "2"},
        buffer=b'{"ok":true}',  # type: ignore[arg-type]
        received_at=42,
    )

    assert resp.match_value("request_method") == "POST"
    assert resp.match_value("request_url_str") == "https://example.com/api?a=1"
    assert resp.match_value("response_status_code") == 200
    assert resp.match_value("response_content_type") == "application/json"
    assert resp.match_value("response_body") == b'{"ok":true}'
    assert resp.match_value("response_received_at") == dt.datetime(1970, 1, 1, 0, 0, 42, tzinfo=dt.timezone.utc)


def test_match_value_rejects_unknown_key() -> None:
    resp = make_response()
    with pytest.raises(ValueError, match="Unsupported response match key"):
        resp.match_value("not_a_real_key")


def test_match_tuple_preserves_argument_order() -> None:
    resp = make_response(
        request=make_request(),
        headers={"Content-Type": "application/json"},
        body=b"hello",
        received_at=42,
    )

    out = resp.match_tuple(
        ["request_method", "request_url_str", "response_status_code", "response_body_hash"]
    )

    assert out[0] == "GET"
    assert out[1] == "https://example.com/a"
    assert out[2] == 200
    assert out[3] is not None


# ---------------------------------------------------------------------------
# to_arrow_batch — schema parity + header promotion
# ---------------------------------------------------------------------------

def test_to_arrow_batch_matches_schema() -> None:
    resp = make_response(
        headers={
            "User-Agent": "server",
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "X-Other": "keep",
        },
        tags={"t": "2"},
        body=b'{"ok":true}',
        received_at=42,
    )

    rb = resp.to_arrow_batch(parse=False)

    assert rb.schema == RESPONSE_ARROW_SCHEMA
    assert rb.num_rows == 1


def test_to_arrow_batch_promotes_response_headers() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1&shared=url",
        headers={"User-Agent": "pytest"},
        body=b"hello rq",
        tags={"shared": "rq"},
    )
    req.sent_at = dt.datetime.fromtimestamp(11, tz=dt.timezone.utc)

    resp = Response(
        request=req,
        status_code=200,
        headers={
            "Host": "example.com",
            "User-Agent": "server",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Accept-Language": "en",
            "Content-Type": "application/json",
            "Content-Length": "11",
            "Content-Encoding": "gzip",
            "Transfer-Encoding": "chunked",
            "X-Other": "keep-me",
        },
        tags={"t": "2"},
        buffer=b"hello world",  # type: ignore[arg-type]
        received_at=42,
    )

    row = _row(resp.to_arrow_batch(parse=False))

    assert row["request_method"] == "POST"
    assert row["request_url_str"] == "https://example.com/api?a=1&shared=url"
    assert row["response_status_code"] == 200

    assert row["response_host"] == "example.com"
    assert row["response_user_agent"] == "server"
    assert row["response_accept"] == "application/json"
    assert row["response_accept_encoding"] == "gzip"
    assert row["response_accept_language"] == "en"
    assert row["response_content_type"] == "application/json"
    assert row["response_content_length"] == 11
    assert row["response_content_encoding"] == "gzip"
    assert row["response_transfer_encoding"] == "chunked"

    assert dict(row["response_headers"])["X-Other"] == "keep-me"
    assert dict(row["response_tags"]) == {"t": "2"}
    assert row["response_body"] == b"hello world"
    assert row["response_body_hash"] == resp.buffer.xxh3_int64()
    assert row["response_received_at"] == dt.datetime(1970, 1, 1, 0, 0, 42, tzinfo=dt.timezone.utc)


def test_to_arrow_batch_empty_body_still_hashes() -> None:
    resp = make_response(status_code=204, body=b"", received_at=0)

    row = _row(resp.to_arrow_batch(parse=False))

    assert row["response_body"] == b""
    assert row["response_body_hash"] is not None


# ---------------------------------------------------------------------------
# from_arrow_tabular — roundtrip parity (record batch + table)
# ---------------------------------------------------------------------------

def test_from_arrow_tabular_roundtrip_record_batch() -> None:
    req = PreparedRequest.prepare(
        method="POST",
        url="https://example.com/api?a=1",
        headers={"User-Agent": "pytest"},
        body=b"rq",
        tags={"rt": "1"},
    )
    req.sent_at = dt.datetime.fromtimestamp(7, tz=dt.timezone.utc)

    resp = Response(
        request=req,
        status_code=201,
        headers={
            "User-Agent": "server",
            "Content-Type": "application/json",
            "X-Other": "keep",
        },
        tags={"t": "2"},
        buffer=b'{"ok":true}',  # type: ignore[arg-type]
        received_at=123,
    )

    rebuilt_list = list(Response.from_arrow_tabular(resp.to_arrow_batch(parse=False)))

    assert len(rebuilt_list) == 1
    rebuilt = rebuilt_list[0]

    assert rebuilt.request.method == "POST"
    assert rebuilt.request.url.to_string() == "https://example.com/api?a=1"
    assert rebuilt.status_code == 201
    assert rebuilt.buffer.to_bytes() == b'{"ok":true}'
    assert rebuilt.tags == {"t": "2"}
    # microseconds since epoch — 123 seconds → 123_000_000
    assert rebuilt.received_at_timestamp == 123_000_000
    assert rebuilt.headers["User-Agent"] == "server"
    assert rebuilt.headers["Content-Type"] == "application/json"


def test_from_arrow_tabular_roundtrip_table() -> None:
    resp = make_response(
        request=make_request(),
        headers={"Content-Type": "text/plain"},
        body=b"hello",
        received_at=7,
    )

    rebuilt_list = list(Response.from_arrow_tabular(resp.to_arrow_table(parse=False)))

    assert len(rebuilt_list) == 1
    rebuilt = rebuilt_list[0]

    assert rebuilt.request.url.to_string() == "https://example.com/a"
    assert rebuilt.status_code == 200
    assert rebuilt.buffer.to_bytes() == b"hello"
    assert rebuilt.received_at_timestamp == 7_000_000
