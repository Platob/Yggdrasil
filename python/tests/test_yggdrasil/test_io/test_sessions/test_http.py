# tests/test_http_session_real_world.py
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import ssl
import time

import pytest

from yggdrasil.io.http_ import HTTPSession

# ---------------------------
# Real-world / no-mock tests
# ---------------------------

REAL_HTTP = os.getenv("REAL_HTTP", "1") == "1"
NETWORK_TIMEOUT_SEC = float(os.getenv("NETWORK_TIMEOUT_SEC", "5.0"))

pytestmark = pytest.mark.skipif(
    not REAL_HTTP,
    reason="Real network tests disabled (set REAL_HTTP=1 to enable).",
)


def _tcp_can_connect(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _tls_can_handshake(host: str, port: int, timeout: float) -> bool:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                return True
    except OSError:
        return False


def _require_https(host: str) -> None:
    if not _tcp_can_connect(host, 443, NETWORK_TIMEOUT_SEC):
        pytest.skip(f"Cannot connect to {host}:443")
    if not _tls_can_handshake(host, 443, NETWORK_TIMEOUT_SEC):
        pytest.skip(f"Cannot complete TLS handshake with {host}:443")


@pytest.fixture(scope="session", autouse=True)
def _network_precheck():
    # Light precheck so failures are mostly meaningful test failures, not
    # "no outbound network in this CI runner".
    _require_https("example.com")


def test_send_get_stream_true_reads_bytes():
    _require_https("example.com")

    s = HTTPSession()
    req = s.prepare_request(
        "GET",
        "https://example.com",
        headers={"User-Agent": "real-http-test"},
    )
    resp = s.send(req, stream=True)

    assert 200 <= resp.status_code < 400

    data = resp.read(256) if hasattr(resp, "read") else resp.content[:256]
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0


def test_send_get_stream_false_preloads_content():
    _require_https("example.com")

    s = HTTPSession()
    req = s.prepare_request(
        "GET",
        "https://example.com",
        headers={"User-Agent": "real-http-test"},
    )
    resp = s.send(req, stream=False, raise_error=False)

    assert 200 <= resp.status_code < 400

    content = resp.content
    assert isinstance(content, (bytes, bytearray))
    assert len(content) > 0
    assert b"<html" in content.lower()


def test_prepare_request_with_base_url_and_params():
    _require_https("httpbin.org")

    s = HTTPSession(base_url="https://httpbin.org")
    req = s.prepare_request(
        "GET",
        "/get",
        params={"a": "1", "b": "two"},
        headers={"User-Agent": "real-http-test"},
    )
    resp = s.send(req, stream=False)

    assert resp.status_code == 200

    payload = json.loads(resp.content.decode("utf-8"))
    assert payload["args"] == {"a": "1", "b": "two"}


def test_redirect_followed():
    _require_https("httpbin.org")

    s = HTTPSession()
    req = s.prepare_request(
        "GET",
        "https://httpbin.org/redirect/1",
        headers={"User-Agent": "real-http-test"},
    )
    resp = s.send(req, stream=False)

    assert resp.status_code == 200
    assert b"httpbin" in resp.content.lower()


def test_post_json_roundtrip():
    _require_https("httpbin.org")

    s = HTTPSession()

    payload = {"hello": "world", "ts": time.time()}

    req = s.prepare_request(
        "POST",
        "https://httpbin.org/post",
        headers={"User-Agent": "real-http-test"},
        json=payload,
    )

    resp = s.send(req, stream=False)
    assert resp.status_code == 200

    data = json.loads(resp.content.decode("utf-8"))
    assert data["json"] == payload
    assert data["headers"]["User-Agent"] == "real-http-test"


def test_post_raw_body_roundtrip():
    _require_https("httpbin.org")

    s = HTTPSession()
    body = b"hello-real-world"

    req = s.prepare_request(
        "POST",
        "https://httpbin.org/post",
        headers={
            "User-Agent": "real-http-test",
            "Content-Type": "application/octet-stream",
        },
        body=body,
    )

    resp = s.send(req, stream=False)
    assert resp.status_code == 200

    data = json.loads(resp.content.decode("utf-8"))
    assert data["data"] == body.decode("utf-8")


def test_status_404_with_raise_error_false():
    _require_https("httpbin.org")

    s = HTTPSession()
    req = s.prepare_request(
        "GET",
        "https://httpbin.org/status/404",
        headers={"User-Agent": "real-http-test"},
    )
    resp = s.send(req, stream=False, raise_error=False)

    assert resp.status_code == 404
    assert resp.ok is False


def test_status_404_with_raise_error_true_raises():
    _require_https("httpbin.org")

    s = HTTPSession()
    req = s.prepare_request(
        "GET",
        "https://httpbin.org/status/404",
        headers={"User-Agent": "real-http-test"},
    )

    with pytest.raises(Exception):
        s.send(req, stream=False, raise_error=True)


def test_head_request():
    _require_https("example.com")

    s = HTTPSession()
    resp = s.head(
        "https://example.com",
        headers={"User-Agent": "real-http-test"},
        raise_error=False,
    )

    assert 200 <= resp.status_code < 400


def test_session_get_helper():
    _require_https("httpbin.org")

    s = HTTPSession()
    resp = s.get(
        "https://httpbin.org/get",
        headers={"User-Agent": "real-http-test"},
        raise_error=False,
    )

    assert resp.status_code == 200
    data = json.loads(resp.content.decode("utf-8"))
    assert data["headers"]["User-Agent"] == "real-http-test"


def test_session_post_helper_with_json():
    _require_https("httpbin.org")

    s = HTTPSession()
    payload = {"x": 1, "y": "z"}

    resp = s.post(
        "https://httpbin.org/post",
        headers={"User-Agent": "real-http-test"},
        json=payload,
        raise_error=False,
    )

    assert resp.status_code == 200
    data = json.loads(resp.content.decode("utf-8"))
    assert data["json"] == payload


def test_response_text_and_json_helpers():
    _require_https("httpbin.org")

    s = HTTPSession()

    req = s.prepare_request(
        "GET",
        "https://httpbin.org/json",
        headers={"User-Agent": "real-http-test"},
    )
    resp = s.send(req, stream=False)

    assert resp.status_code == 200
    assert isinstance(resp.text, str)
    decoded = resp.json()
    assert isinstance(decoded, dict)
    assert "slideshow" in decoded


def test_repeated_requests_reuse_same_session():
    _require_https("example.com")

    s = HTTPSession()

    req1 = s.prepare_request(
        "GET",
        "https://example.com",
        headers={"User-Agent": "real-http-test"},
    )
    req2 = s.prepare_request(
        "GET",
        "https://example.com",
        headers={"User-Agent": "real-http-test"},
    )

    resp1 = s.send(req1, stream=False, raise_error=False)
    resp2 = s.send(req2, stream=False, raise_error=False)

    assert 200 <= resp1.status_code < 400
    assert 200 <= resp2.status_code < 400
    assert len(resp1.content) > 0
    assert len(resp2.content) > 0


def test_send_many_real_requests_ordered():
    _require_https("httpbin.org")

    s = HTTPSession()

    requests = iter(
        [
            s.prepare_request("GET", "https://httpbin.org/get?a=1", headers={"User-Agent": "real-http-test"}),
            s.prepare_request("GET", "https://httpbin.org/get?a=2", headers={"User-Agent": "real-http-test"}),
            s.prepare_request("GET", "https://httpbin.org/get?a=3", headers={"User-Agent": "real-http-test"}),
        ]
    )

    responses = list(
        s.send_many(
            requests,
            ordered=True,
            raise_error=False,
            stream=False,
            batch_size=3,
            max_in_flight=3,
        )
    )

    assert len(responses) == 3
    payloads = [json.loads(r.content.decode("utf-8")) for r in responses]
    assert payloads[0]["args"] == {"a": "1"}
    assert payloads[1]["args"] == {"a": "2"}
    assert payloads[2]["args"] == {"a": "3"}


from pathlib import Path

from yggdrasil.io.send_config import CacheConfig


def _wait_for_file(path: Path, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(0.05)
    pytest.fail(f"Timed out waiting for cache file: {path}")


def _local_cache_path(root: Path, request) -> Path:
    cache_folder = root / "cache"
    url = request.url

    if url.host:
        cache_folder = cache_folder / url.host

    if url.path:
        path_parts = [part for part in url.path.split("/") if part]
        if path_parts:
            cache_folder = cache_folder.joinpath(*path_parts)

    return cache_folder / f"{request.xxh3_b64(url_safe=True)}.ypkl"


def test_send_persists_local_cache_and_reuses_it(tmp_path: Path):
    _require_https("example.com")

    s = HTTPSession()
    req = s.prepare_request(
        "GET",
        "https://example.com",
        headers={"User-Agent": "real-http-test"},
    )

    local_cache = CacheConfig(
        path=tmp_path,
        received_from=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
    )

    resp1 = s.send(
        req,
        stream=False,
        raise_error=False,
        local_cache=local_cache,
    )
    assert 200 <= resp1.status_code < 400

    expected_file = _local_cache_path(tmp_path, req)
    _wait_for_file(expected_file)

    # Second call should hit local cache.
    resp2 = s.send(
        req,
        stream=False,
        raise_error=False,
        local_cache=local_cache,
    )

    assert 200 <= resp2.status_code < 400
    assert resp2.request.url.to_string() == req.url.to_string()
    assert resp2.status_code == resp1.status_code
    assert resp2.content == resp1.content


def test_send_local_cache_respects_received_from_window(tmp_path: Path):
    _require_https("example.com")

    s = HTTPSession()
    req = s.prepare_request(
        "GET",
        "https://example.com",
        headers={"User-Agent": "real-http-test"},
    )

    warm_cache = CacheConfig(
        path=tmp_path,
        received_from=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
    )

    resp1 = s.send(
        req,
        stream=False,
        raise_error=False,
        local_cache=warm_cache,
    )
    assert 200 <= resp1.status_code < 400

    expected_file = _local_cache_path(tmp_path, req)
    _wait_for_file(expected_file)

    # Move the freshness window into the future so the existing file is stale.
    stale_cutoff = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    strict_cache = CacheConfig(
        path=tmp_path,
        received_from=stale_cutoff,
    )

    cached_file = strict_cache.local_cache_file(req, suffix=".ypkl")
    assert cached_file is None

    # A fresh network request should still succeed.
    resp2 = s.send(
        req,
        stream=False,
        raise_error=False,
        local_cache=strict_cache,
    )
    assert 200 <= resp2.status_code < 400
    assert len(resp2.content) > 0


def test_send_local_cache_with_ttl(tmp_path: Path):
    _require_https("example.com")

    s = HTTPSession()
    req = s.prepare_request(
        "GET",
        "https://example.com",
        headers={"User-Agent": "real-http-test"},
    )

    local_cache = CacheConfig(
        path=tmp_path,
        received_ttl=dt.timedelta(minutes=10),
    )

    resp1 = s.send(
        req,
        stream=False,
        raise_error=False,
        local_cache=local_cache,
    )
    assert 200 <= resp1.status_code < 400

    expected_file = _local_cache_path(tmp_path, req)
    _wait_for_file(expected_file)

    resp2 = s.send(
        req,
        stream=False,
        raise_error=False,
        local_cache=local_cache,
    )
    assert 200 <= resp2.status_code < 400
    assert resp2.content == resp1.content


def test_send_many_with_local_cache_warm_then_reuse(tmp_path: Path):
    _require_https("httpbin.org")

    s = HTTPSession()
    local_cache = CacheConfig(
        path=tmp_path,
        received_from=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
    )

    reqs = [
        s.prepare_request("GET", "https://httpbin.org/get?a=1", headers={"User-Agent": "real-http-test"}),
        s.prepare_request("GET", "https://httpbin.org/get?a=2", headers={"User-Agent": "real-http-test"}),
    ]

    first = list(
        s.send_many(
            iter(reqs),
            ordered=True,
            raise_error=False,
            stream=False,
            batch_size=2,
            max_in_flight=2,
            local_cache=local_cache,
        )
    )
    assert len(first) == 2
    assert all(r.status_code == 200 for r in first)

    for req in reqs:
        _wait_for_file(local_cache.local_cache_file(req, suffix=".ypkl", force=True))  # Ensure cache files are written before proceeding.

    second = list(
        s.send_many(
            iter(reqs),
            ordered=True,
            raise_error=False,
            stream=False,
            batch_size=2,
            max_in_flight=2,
            local_cache=local_cache,
        )
    )
    assert len(second) == 2

    first_payloads = [json.loads(r.content.decode("utf-8")) for r in first]
    second_payloads = [json.loads(r.content.decode("utf-8")) for r in second]

    assert first_payloads[0]["args"] == second_payloads[0]["args"] == {"a": "1"}
    assert first_payloads[1]["args"] == second_payloads[1]["args"] == {"a": "2"}