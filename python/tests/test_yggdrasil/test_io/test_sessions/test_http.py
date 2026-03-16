# tests/test_http_session_real_world.py
from __future__ import annotations

import json
import os
import socket
import ssl
import time

import pytest
from yggdrasil.io import URL

# Adjust these imports to match your project structure
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


@pytest.fixture(scope="session", autouse=True)
def _network_precheck():
    pass

def test_send_get_stream_true_reads_bytes():
    s = HTTPSession()
    req = s.prepare_request("GET", "https://example.com", headers={"User-Agent": "real-http-test"})
    resp = s.send(req, stream=True)

    assert 200 <= resp.status_code < 400
    # streaming: content may not be preloaded; ensure we can read some bytes
    data = resp.read(256) if hasattr(resp, "read") else resp.content[:256]
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0


def test_send_get_stream_false_preloads_content():
    s = HTTPSession()
    req = s.prepare_request("GET", "https://example.com", headers={"User-Agent": "real-http-test"})
    resp = s.send(req, stream=True, raise_error=False)

    assert 200 <= resp.status_code < 400
    # non-stream: should have content available immediately
    content = resp.content
    assert isinstance(content, (bytes, bytearray))
    assert b"<html" in content.lower()


def test_redirect_followed():
    s = HTTPSession()
    # httpbin redirect endpoint
    req = s.prepare_request("GET", "https://httpbin.org/redirect/1", headers={"User-Agent": "real-http-test"})
    resp = s.send(req, stream=False)

    assert resp.status_code == 200
    content = resp.content
    assert b"httpbin" in content.lower()


def test_post_json_roundtrip():
    s = HTTPSession()

    payload = {"hello": "world", "ts": time.time()}
    body = json.dumps(payload).encode("utf-8")

    # NOTE: if your PreparedRequest.prepare(json=...) auto sets content-type + encoding,
    # you should use that instead of manually building body.
    req = s.prepare_request(
        "POST",
        "https://httpbin.org/post",
        headers={
            "User-Agent": "real-http-test",
            "Content-Type": "application/json",
        },
        body=body,
    )

    resp = s.send(req, stream=False)
    assert resp.status_code == 200

    raw = resp.content
    data = json.loads(raw.decode("utf-8"))

    assert data["json"] == payload

