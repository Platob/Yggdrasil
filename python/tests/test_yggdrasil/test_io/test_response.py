import datetime
import json

import pytest

from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.io import BytesIO
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response, RESPONSE_ARROW_SCHEMA


def map_as_dict(x):
    # Arrow map scalars may become list[tuple] via as_py()
    if x is None:
        return {}
    if isinstance(x, dict):
        return {str(k): str(v) for k, v in x.items() if k is not None and v is not None}
    if isinstance(x, list):
        return {str(k): str(v) for k, v in x if k is not None and v is not None}
    return {}


@pytest.fixture
def sample_url_str():
    return "https://user:pass@example.com:8443/path/to?q=1#frag"


@pytest.fixture
def request_obj(sample_url_str):
    return PreparedRequest.prepare(
        method="POST",
        url=sample_url_str,
        headers={"A": "1"},
        body=b"hello",
        tags={"rt": "rv"},
    )


@pytest.fixture
def response_obj(request_obj):
    # note: content-type intentionally included for deterministic text/json tests
    return Response(
        request=request_obj,
        status_code=200,
        headers={"Content-Type": "application/json; charset=utf-8", "X": "1"},
        buffer=request_obj.buffer,  # reuse "hello" bytes (not json, but fine for many tests)
        received_at_timestamp=123,
        tags={"t": "v"},
    )


def test_parse_str_json_object(sample_url_str):
    # JSON can't serialize bytes, so body must be JSON-friendly (string)
    payload = {
        "request": {
            "method": "GET",
            "url_str": sample_url_str,
            "headers": {"X": "1"},
        },
        "status_code": 201,
        "headers": {"Content-Type": "text/plain; charset=utf-8"},
        "body": "hello",
        "received_at_timestamp": "456",
        "tags": {"k": "v"},
    }

    raw = json.dumps(payload)
    r = Response.parse(raw)

    assert isinstance(r, Response)
    assert r.request.method == "GET"
    assert r.status_code == 201
    assert r.headers["Content-Type"].startswith("text/plain")
    assert r.buffer.to_bytes() == b"hello"
    assert r.received_at_timestamp == 456
    assert r.tags == {"k": "v"}


def test_parse_dict_status_aliases(sample_url_str):
    base = {
        "request": {"method": "GET", "url_str": sample_url_str},
        "headers": {},
        "body": "x",
        "received_at_timestamp": 1,
        "tags": {"a": 1},
    }

    r1 = Response.parse_dict({**base, "status_code": 200})
    assert r1.status_code == 200

    r2 = Response.parse_dict({**base, "status": "404"})
    assert r2.status_code == 404

    r3 = Response.parse_dict({**base, "code": 500.0})
    assert r3.status_code == 500


def test_parse_dict_missing_status_raises(sample_url_str):
    with pytest.raises(ValueError):
        Response.parse_dict({"request": {"method": "GET", "url_str": sample_url_str}})


def test_properties_ok_text_json(sample_url_str):
    # Proper JSON payload, so json() works
    req = PreparedRequest.prepare(method="GET", url=sample_url_str)
    r = Response(
        request=req,
        status_code=200,
        headers={"Content-Type": "application/json; charset=utf-8"},
        buffer=BytesIO.parse(b'{"a": 1}'),  # BytesIO.parse via class
        received_at_timestamp=1,
        tags={},
    )

    assert r.ok is True
    assert r.text.strip() == '{"a": 1}'
    assert r.json() == {"a": 1}


def test_raise_for_status_raises(sample_url_str):
    req = PreparedRequest.prepare(method="GET", url=sample_url_str)
    r = Response(
        request=req,
        status_code=404,
        headers={"Content-Type": "text/plain"},
        buffer=BytesIO.parse(b"nope"),
        received_at_timestamp=1,
        tags={},
    )

    with pytest.raises(Exception):
        r.raise_for_status()


def test_to_arrow_batch_schema_and_values(response_obj):
    rb = response_obj.to_arrow_batch(parse=False)
    assert isinstance(rb, pa.RecordBatch)
    assert rb.schema == RESPONSE_ARROW_SCHEMA
    assert rb.num_rows == 1

    cols = {name: rb.column(name) for name in rb.schema.names}

    assert cols["request_method"][0].as_py() == response_obj.request.method
    assert cols["response_status_code"][0].as_py() == response_obj.status_code

    headers_py = map_as_dict(cols["response_headers"][0].as_py())
    assert headers_py["X"] == "1"
    assert headers_py["Content-Type"].startswith("application/json")

    tags_py = map_as_dict(cols["response_tags"][0].as_py())
    assert tags_py == {"t": "v"}

    body = cols["response_body"][0].as_py()
    assert body == response_obj.buffer.to_bytes()

    body_hash = cols["response_body_hash"][0].as_py()
    assert isinstance(body_hash, (bytes, bytearray))
    assert len(body_hash) == 32

    # both columns store epoch micros per current behavior
    assert cols["response_received_at"][0].as_py() == datetime.datetime(1970, 1, 1, 0, 0, 0, 123, tzinfo=datetime.timezone.utc)
    assert cols["response_received_at_epoch"][0].as_py() == response_obj.received_at_timestamp


def test_from_arrow_roundtrip(response_obj):
    rb = response_obj.to_arrow_batch(parse=False)
    out = list(Response.from_arrow(rb, parse=False))

    assert len(out) == 1
    r2 = out[0]

    assert r2.status_code == response_obj.status_code
    assert r2.request.method == response_obj.request.method

    assert r2.buffer.to_bytes() == response_obj.buffer.to_bytes()
    assert r2.received_at_timestamp == response_obj.received_at_timestamp

    # headers/tags equality (normalized to strings)
    assert dict(r2.headers) == {str(k): str(v) for k, v in response_obj.headers.items()}
    assert dict(r2.tags or {}) == {str(k): str(v) for k, v in (response_obj.tags or {}).items()}


def test_to_starlette_filters_hop_by_hop_and_sets_content_length(sample_url_str):
    req = PreparedRequest.prepare(method="GET", url=sample_url_str)
    r = Response(
        request=req,
        status_code=200,
        headers={
            "Content-Type": "text/plain",
            "Connection": "keep-alive",  # hop-by-hop
            "Transfer-Encoding": "chunked",  # hop-by-hop
            "X": "1",
        },
        buffer=BytesIO.parse(b"abc"),
        received_at_timestamp=1,
        tags={},
    )

    s = r.to_starlette()
    # Starlette stores headers in a case-insensitive structure; easiest is access raw:
    raw = dict(s.headers)
    assert "connection" not in raw
    assert "transfer-encoding" not in raw
    assert raw["content-length"] == "3"
    assert raw["x"] == "1"


def test_to_fastapi_falls_back_to_starlette_or_returns_response(sample_url_str):
    # This test is intentionally tolerant:
    # - if FastAPI is installed, you get a fastapi.Response
    # - if not, you get a starlette Response from fallback
    req = PreparedRequest.prepare(method="GET", url=sample_url_str)
    r = Response(
        request=req,
        status_code=200,
        headers={"Content-Type": "text/plain"},
        buffer=BytesIO.parse(b"abc"),
        received_at_timestamp=1,
        tags={},
    )

    resp = r.to_fastapi()
    assert hasattr(resp, "headers")
    assert hasattr(resp, "status_code")
    assert resp.status_code == 200
    assert dict(resp.headers).get("content-length") == "3"