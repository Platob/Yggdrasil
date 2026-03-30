# tests/io/test_session_config.py
from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.dataclasses.waiting import WaitingConfig
from yggdrasil.io import SaveMode
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.io.send_config import CacheConfig, SendConfig, SendManyConfig


def _make_request(
    *,
    method: str = "GET",
    url: str = "https://example.com/a",
    body: bytes | None = None,
) -> PreparedRequest:
    req = PreparedRequest.prepare(
        method=method,
        url=url,
        body=body,
    )
    req.sent_at = dt.datetime.fromtimestamp(7)
    return req


def _make_response(
    *,
    request: PreparedRequest | None = None,
    status_code: int = 200,
    content_type: str = "application/json",
    body: bytes = b'{"ok":true}',
    received_at_timestamp: int = 1767225600000000,
) -> Response:
    return Response(
        request=request or _make_request(),
        status_code=status_code,
        headers={"Content-Type": content_type},
        tags={"rt": "1"},
        buffer=body,  # type: ignore[arg-type]
        received_at=received_at_timestamp,
    )


def test_cache_config_defaults() -> None:
    cfg = CacheConfig()

    assert cfg.request_by == [
        "request_method",
        "request_url_scheme",
        "request_url_host",
        "request_url_path",
        "request_url_port",
        "request_url_query",
        "request_content_length",
        "request_body_hash",
    ]
    assert cfg.response_by is None
    assert cfg.mode == SaveMode.APPEND
    assert cfg.anonymize == "remove"
    assert cfg.received_from is None
    assert cfg.received_to is None
    assert cfg.wait == WaitingConfig.check_arg(False)


def test_cache_config_coerces_received_datetimes_from_strings() -> None:
    cfg = CacheConfig(
        received_from="2026-01-01T00:00:00Z",
        received_to="2026-01-31T12:00:00Z",
    )

    assert isinstance(cfg.received_from, dt.datetime)
    assert isinstance(cfg.received_to, dt.datetime)


def test_cache_config_parse_mapping_coerces_received_datetimes() -> None:
    cfg = CacheConfig.parse_mapping(
        {
            "received_from": "2026-02-01T00:00:00Z",
            "received_to": "2026-02-28T23:59:59Z",
        }
    )

    assert isinstance(cfg.received_from, dt.datetime)
    assert isinstance(cfg.received_to, dt.datetime)


def test_cache_config_invalid_request_by_raises() -> None:
    with pytest.raises(ValueError, match="Invalid request_by key"):
        CacheConfig(request_by=["request_method", "not_a_real_column"])


def test_cache_config_invalid_response_by_raises() -> None:
    with pytest.raises(ValueError, match="Invalid response_by key"):
        CacheConfig(response_by=["not_a_real_response_column"])


def test_cache_config_by_combines_request_and_response_keys() -> None:
    cfg = CacheConfig(
        request_by=["request_method"],
        response_by=["response_status_code"],
    )

    assert cfg.by == ["request_method", "response_status_code"]


def test_cache_config_sql_literal() -> None:
    now = dt.datetime(2026, 1, 1, 12, 30, 15, 123456)

    assert CacheConfig.sql_literal(None) == "null"
    assert CacheConfig.sql_literal(True) == "true"
    assert CacheConfig.sql_literal(False) == "false"
    assert CacheConfig.sql_literal(12) == "12"
    assert CacheConfig.sql_literal("ab'cd") == "'ab''cd'"
    assert CacheConfig.sql_literal(now) == "timestamp '2026-01-01 12:30:15.123456'"


def test_cache_config_request_values_and_tuple() -> None:
    req = _make_request(
        method="POST",
        url="https://example.com/api?q=1",
        body=b"hello",
    )
    cfg = CacheConfig(
        request_by=["request_method", "request_url_str", "request_body_hash"],
    )

    values = cfg.request_values(req)
    assert values["request_method"] == "POST"
    assert values["request_url_str"] == "https://example.com/api?q=1"
    assert values["request_body_hash"] is not None

    as_tuple = cfg.request_tuple(req)
    assert len(as_tuple) == 3
    assert as_tuple[0] == "POST"


def test_cache_config_response_values_and_tuple() -> None:
    resp = _make_response(
        status_code=200,
        content_type="application/json",
    )
    cfg = CacheConfig(response_by=["response_status_code", "response_content_type"])

    values = cfg.response_values(resp)
    assert values == {
        "response_status_code": 200,
        "response_content_type": "application/json",
    }
    assert cfg.response_tuple(resp) == (200, "application/json")


def test_cache_config_filter_request_returns_true_for_valid_request() -> None:
    req = _make_request(
        method="GET",
        url="https://example.com",
    )
    cfg = CacheConfig(request_by=["request_method", "request_url_str"])

    assert cfg.filter_request(req) is True


def test_cache_config_filter_response_matches_request_and_time_window() -> None:
    req = _make_request(
        method="GET",
        url="https://example.com/a",
    )
    resp = _make_response(
        request=req,
        status_code=200,
        received_at_timestamp="2026-01-01T01:00:00Z",
    )
    cfg = CacheConfig(
        request_by=["request_method", "request_url_str"],
        response_by=["response_status_code"],
        received_from="2026-01-01T00:00:00Z",
        received_to="2026-12-31T23:59:59Z",
    )

    assert cfg.filter_response(resp, request=req) is True


def test_cache_config_filter_response_rejects_request_mismatch() -> None:
    req = _make_request(
        method="GET",
        url="https://example.com/a",
    )
    other_req = _make_request(
        method="GET",
        url="https://example.com/b",
    )
    resp = _make_response(
        request=other_req,
        status_code=200,
        received_at_timestamp=1767225600000000,
    )
    cfg = CacheConfig(request_by=["request_method", "request_url_str"])

    assert cfg.filter_response(resp, request=req) is False


def test_cache_config_filter_response_rejects_received_from() -> None:
    resp = _make_response(
        received_at_timestamp=1,
    )
    cfg = CacheConfig(received_from="2026-01-01T00:00:00Z")

    assert cfg.filter_response(resp) is False


def test_cache_config_filter_response_rejects_received_to() -> None:
    resp = _make_response(
        received_at_timestamp=9999999999999999,
    )
    cfg = CacheConfig(received_to="2026-01-01T00:00:00Z")

    assert cfg.filter_response(resp) is False


def test_cache_config_identity_tuple() -> None:
    req = _make_request(
        method="GET",
        url="https://example.com/a",
    )
    resp = _make_response(
        request=req,
        status_code=200,
    )
    cfg = CacheConfig(
        request_by=["request_method", "request_url_str"],
        response_by=["response_status_code"],
    )

    out = cfg.identity_tuple(resp, request=req)
    assert out == ("GET", "https://example.com/a", 200)


def test_cache_config_sql_request_clause() -> None:
    req = _make_request(
        method="POST",
        url="https://example.com/api?q=1",
        body=b"hello",
    )
    cfg = CacheConfig(
        request_by=["request_method", "request_url_str", "request_body_hash"],
    )

    clause = cfg.sql_request_clause(req)

    assert "request_method = 'POST'" in clause
    assert "request_url_str = 'https://example.com/api?q=1'" in clause
    assert "request_body_hash =" in clause


def test_cache_config_sql_response_clause() -> None:
    resp = _make_response(
        status_code=200,
    )
    cfg = CacheConfig(
        response_by=["response_status_code"],
        received_from="2026-01-01T00:00:00Z",
        received_to="2026-01-31T00:00:00Z",
    )

    clause = cfg.sql_response_clause(resp)

    assert "response_status_code = 200" in clause
    assert "response_received_at >=" in clause
    assert "response_received_at <" in clause


def test_cache_config_sql_clause_combines_request_and_response() -> None:
    req = _make_request(method="GET", url="https://example.com/a")
    resp = _make_response(
        request=req,
        status_code=200,
    )
    cfg = CacheConfig(
        request_by=["request_method", "request_url_str"],
        response_by=["response_status_code"],
    )

    clause = cfg.sql_clause(request=req, response=resp)

    assert "(request_method = 'GET' AND request_url_str = 'https://example.com/a')" in clause
    assert "(response_status_code = 200)" in clause


def test_cache_config_make_lookup_sql() -> None:
    req = _make_request(method="GET", url="https://example.com/a")
    cfg = CacheConfig(
        request_by=["request_method", "request_url_str"],
        response_by=["response_status_code"],
    )

    sql = cfg.make_lookup_sql(
        table_name="cache_table",
        request=req,
    )

    assert "SELECT * FROM (" in sql
    assert "FROM (SELECT * FROM cache_table WHERE" in sql
    assert "PARTITION BY request_method, request_url_str, response_status_code" in sql


def test_cache_config_make_batch_lookup_sql() -> None:
    r1 = _make_request(method="GET", url="https://example.com/a")
    r2 = _make_request(method="GET", url="https://example.com/b")
    cfg = CacheConfig(
        request_by=["request_method", "request_url_str"],
        response_by=["response_status_code"],
        received_from="2026-01-01T00:00:00Z",
    )

    sql = cfg.make_batch_lookup_sql(
        table_name="cache_table",
        requests=[r1, r2],
    )

    assert "request_url_str = 'https://example.com/a'" in sql
    assert "request_url_str = 'https://example.com/b'" in sql
    assert "response_received_at >=" in sql
    assert "PARTITION BY request_method, request_url_str, response_status_code" in sql


def test_send_config_check_arg_from_mapping() -> None:
    cfg = SendConfig.check_arg(
        {
            "raise_error": False,
            "remote_cache": {"received_from": "2026-01-01T00:00:00Z"},
        }
    )

    assert isinstance(cfg, SendConfig)
    assert cfg.raise_error is False
    assert isinstance(cfg.remote_cache, CacheConfig)
    assert isinstance(cfg.remote_cache.received_from, dt.datetime)


def test_send_many_config_from_send_config() -> None:
    send_cfg = SendConfig(
        raise_error=False,
        remote_cache=CacheConfig(received_from="2026-01-01T00:00:00Z"),
    )

    many_cfg = SendManyConfig.check_arg(send_cfg, batch_size=10)

    assert isinstance(many_cfg, SendManyConfig)
    assert many_cfg.raise_error is False
    assert many_cfg.batch_size == 10
    assert isinstance(many_cfg.remote_cache.received_from, dt.datetime)


def test_send_many_to_send_config() -> None:
    many_cfg = SendManyConfig(
        wait=False,
        raise_error=False,
        stream=False,
        remote_cache=CacheConfig(received_from="2026-01-01T00:00:00Z"),
        local_cache=CacheConfig(received_to="2026-01-31T00:00:00Z"),
    )

    send_cfg = many_cfg.to_send_config()

    assert isinstance(send_cfg, SendConfig)
    assert send_cfg.raise_error is False
    assert send_cfg.stream is False
    assert isinstance(send_cfg.remote_cache.received_from, dt.datetime)
    assert isinstance(send_cfg.local_cache.received_to, dt.datetime)


def test_merge_rejects_unknown_field() -> None:
    cfg = CacheConfig()
    with pytest.raises(TypeError, match="unexpected field"):
        cfg.merge(not_a_field=1)


def test_parse_mapping_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="expects a Mapping"):
        CacheConfig.parse_mapping(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_wait_is_coerced() -> None:
    cfg = CacheConfig(wait=WaitingConfig(5))
    assert isinstance(cfg.wait, WaitingConfig)
    assert cfg.wait.timeout_total_seconds == 5