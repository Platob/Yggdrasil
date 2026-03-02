import datetime
import json
import pytest

from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.io.request import PreparedRequest, REQUEST_ARROW_SCHEMA


def map_as_dict(x):
    # Arrow map scalars commonly become list[tuple[str,str]] via as_py()
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, list):
        return dict(x)
    raise TypeError(f"Unexpected map type: {type(x)}")


@pytest.fixture
def sample_url_str():
    return "https://user:pass@example.com:8443/path/to?q=1#frag"


@pytest.fixture
def minimal_dict_jsonable(sample_url_str):
    # JSON can't represent bytes, so use a string body for parse(str/bytes) tests.
    return {
        "method": "POST",
        "url_str": sample_url_str,
        "headers": {"X": 1, "Y": "2"},
        "tags": {"a": 10},
        "body": "hello",          # <- JSONable
        "sent_at_timestamp": "123",
    }


@pytest.fixture
def minimal_dict_raw(sample_url_str):
    # For parse(dict) we can use bytes directly
    return {
        "method": "POST",
        "url_str": sample_url_str,
        "headers": {"X": 1, "Y": "2"},
        "tags": {"a": 10},
        "body": b"hello",
        "sent_at_timestamp": "123",
    }


def test_parse_from_str(minimal_dict_jsonable):
    s = json.dumps(minimal_dict_jsonable)
    req = PreparedRequest.parse(s)
    assert req.method == "POST"
    assert req.headers == {"X": "1", "Y": "2"}
    assert req.tags == {"a": "10"}
    assert req.sent_at_timestamp == 123
    assert req.buffer is not None
    assert req.buffer.to_bytes() == b"hello"  # BytesIO.parse("hello") -> bytes of the string


def test_parse_from_bytes(minimal_dict_jsonable):
    b = json.dumps(minimal_dict_jsonable).encode("utf-8")
    req = PreparedRequest.parse(b)
    assert req.method == "POST"
    assert req.url.to_string()
    assert req.sent_at_timestamp == 123
    assert req.buffer is not None
    assert req.buffer.to_bytes() == b"hello"


def test_parse_from_dict(minimal_dict_raw):
    req = PreparedRequest.parse(minimal_dict_raw)
    assert req.method == "POST"
    assert req.buffer is not None
    assert req.buffer.to_bytes() == b"hello"


def test_parse_rejects_unknown_type():
    with pytest.raises(ValueError):
        PreparedRequest.parse(12345)


def test_parse_dict_default_method_and_headers_tags_cast(sample_url_str):
    obj = {
        "url_str": sample_url_str,
        "headers": {"A": 1, 2: 3},
        "tags": {"k": None},
    }
    req = PreparedRequest.parse_dict(obj)
    assert req.method == "GET"
    assert req.headers == {"A": "1", "2": "3"}
    assert req.tags == {"k": "None"}


def test_parse_dict_prefers_url_str_over_struct(sample_url_str):
    obj = {
        "url_str": sample_url_str,
        "url": {
            "scheme": "http",
            "host": "wrong.example",
            "port": 80,
            "path": "/wrong",
            "query": "",
            "fragment": "",
            "userinfo": "",
        },
    }
    req = PreparedRequest.parse_dict(obj)
    s = req.url.to_string()
    assert "example.com" in s
    assert "wrong.example" not in s


def test_parse_dict_accepts_url_struct_when_no_url_str():
    obj = {
        "method": "GET",
        "url": {
            "scheme": "https",
            "userinfo": "",
            "host": "example.com",
            "port": 443,
            "path": "/p",
            "query": "x=1",
            "fragment": "f",
        },
    }
    req = PreparedRequest.parse_dict(obj)
    s = req.url.to_string()
    assert "example.com" in s
    assert "/p" in s


def test_parse_dict_falls_back_to_exploded_fields():
    obj = {
        "method": "PUT",
        "url_scheme": "https",
        "url_host": "example.com",
        "url_port": 443,
        "url_path": "/z",
        "url_query": "a=b",
        "url_fragment": "c",
        "url_userinfo": "",
    }
    req = PreparedRequest.parse_dict(obj, prefix="request_")
    assert req.method == "PUT"
    s = req.url.to_string()
    assert "example.com" in s
    assert "/z" in s


def test_parse_dict_missing_url_raises():
    with pytest.raises(ValueError):
        PreparedRequest.parse_dict({"method": "GET"})


def test_parse_dict_buffer_aliases_and_missing_buffer(sample_url_str):
    obj = {"url_str": sample_url_str, "body": b"abc"}
    req = PreparedRequest.parse_dict(obj)
    assert req.buffer is not None
    assert req.buffer.to_bytes() == b"abc"

    obj2 = {"url_str": sample_url_str}
    req2 = PreparedRequest.parse_dict(obj2)
    assert req2.buffer is None


def test_parse_dict_sent_at_timestamp_aliases(sample_url_str):
    obj = {"url_str": sample_url_str, "request_sent_at_epoch": "999"}
    assert PreparedRequest.parse_dict(obj).sent_at_timestamp == 999

    obj2 = {"url_str": sample_url_str, "request_sent_at": "888"}
    assert PreparedRequest.parse_dict(obj2).sent_at_timestamp == 888

    obj3 = {"url_str": sample_url_str, "sent_at": "777"}
    assert PreparedRequest.parse_dict(obj3).sent_at_timestamp == 777

    obj4 = {"url_str": sample_url_str, "sent_at_timestamp": None}
    assert PreparedRequest.parse_dict(obj4).sent_at_timestamp == 0


def test_copy_overrides_and_copy_buffer(sample_url_str):
    req = PreparedRequest.prepare(
        method="POST",
        url=sample_url_str,
        headers={"A": "1"},
        body=b"hello",
    )

    req2 = req.copy(copy_buffer=True)
    assert req2.buffer is not None and req.buffer is not None
    assert req2.buffer is not req.buffer
    assert req2.buffer.to_bytes() == req.buffer.to_bytes()

    req3 = req.copy(headers={"B": 2})
    assert req3.headers == {"B": "2"}
    assert "A" not in req3.headers

    req4 = req.copy(method="GET", url="https://example.org/x")
    assert req4.method == "GET"
    assert "example.org" in req4.url.to_string()


def test_copy_buffer_param_allows_explicit_none(sample_url_str):
    req = PreparedRequest.prepare(method="GET", url=sample_url_str, body=b"x")
    req2 = req.copy(buffer=None)
    assert req2.buffer is None


def test_prepare_with_json_sets_content_type_and_body(sample_url_str):
    req = PreparedRequest.prepare(
        method="POST",
        url=sample_url_str,
        headers={},
        json={"a": 1},
    )
    assert req.buffer is not None
    body = req.buffer.to_bytes()
    assert b'"a"' in body and b"1" in body
    assert req.headers.get("Content-Type") is not None
    assert "Content-Length" in req.headers


def test_prepare_with_body_sets_content_length(sample_url_str):
    req = PreparedRequest.prepare(
        method="POST",
        url=sample_url_str,
        headers={"X": "1"},
        body=b"abcd",
    )
    assert req.headers["X"] == "1"
    assert req.buffer is not None
    assert req.headers["Content-Length"] == req.buffer.size


def test_prepare_to_send_sets_timestamp_only_when_requested(sample_url_str):
    req = PreparedRequest.prepare(method="GET", url=sample_url_str)

    out = req.prepare_to_send(add_statistics=True)
    assert out.sent_at_timestamp > 0

    out2 = req.prepare_to_send(add_statistics=False)
    assert out2.sent_at_timestamp == 0


def test_prepare_to_send_applies_before_send(sample_url_str):
    req = PreparedRequest.prepare(method="GET", url=sample_url_str, headers={"A": "1"})

    def before_send(r: PreparedRequest) -> PreparedRequest:
        return r.copy(headers={"X": "y"})

    req.before_send = before_send

    out = req.prepare_to_send(add_statistics=True)
    assert out.headers == {"X": "y"}
    assert out.sent_at_timestamp > 0


def test_anonymize_does_not_crash_and_removes_obvious_secrets(sample_url_str):
    req = PreparedRequest.prepare(
        method="GET",
        url=sample_url_str,
        headers={"Authorization": "Bearer SUPERSECRET", "X": "1"},
    )

    out = req.anonymize(mode="remove")
    assert isinstance(out, PreparedRequest)
    assert out.headers != req.headers

    joined = " ".join(f"{k}:{v}" for k, v in out.headers.items())
    assert "SUPERSECRET" not in joined


def test_to_arrow_batch_schema_and_values(sample_url_str):
    req = PreparedRequest.prepare(
        method="POST",
        url=sample_url_str,
        headers={"A": "1"},
        body=b"hello",
        tags={"t": "v"},
    )
    req.sent_at_timestamp = 123

    batch = req.to_arrow_batch()
    assert isinstance(batch, pa.RecordBatch)
    assert batch.schema == REQUEST_ARROW_SCHEMA

    cols = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}

    assert cols["request_method"][0].as_py() == "POST"
    assert cols["request_url_str"][0].as_py() == req.url.to_string()

    url_struct = cols["request_url"][0].as_py()
    assert isinstance(url_struct, dict)
    assert url_struct.get("host") is not None

    headers_map = map_as_dict(cols["request_headers"][0].as_py())
    assert headers_map["A"] == "1"
    assert headers_map["Content-Length"] == "5"  # prepare() adds it

    tags_map = map_as_dict(cols["request_tags"][0].as_py())
    assert tags_map == {"t": "v"}

    assert cols["request_body"][0].as_py() == b"hello"

    body_hash = cols["request_body_hash"][0].as_py()
    assert isinstance(body_hash, (bytes, bytearray))
    assert len(body_hash) == 32

    assert cols["request_sent_at"][0].as_py() == datetime.datetime(1970, 1, 1, 0, 0, 0, 123, tzinfo=datetime.timezone.utc)
    assert cols["request_sent_at_epoch"][0].as_py() == 123


def test_to_arrow_batch_without_body(sample_url_str):
    req = PreparedRequest.prepare(method="GET", url=sample_url_str, headers={"A": "1"})
    batch = req.to_arrow_batch()
    cols = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
    assert cols["request_body"][0].as_py() is None
    assert cols["request_body_hash"][0].as_py() is None