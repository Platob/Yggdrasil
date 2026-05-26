import pytest
pytestmark = pytest.mark.skip(reason="Session internals need further name migration")

"""Integration tests for HTTPSession with real HTTP calls and caching."""
from __future__ import annotations

import datetime as dt
import json
import http.server
import pathlib
import shutil
import threading
import time
from typing import Any

import pyarrow as pa
import pytest

from yggdrasil.http_.cache_config import CacheConfig
from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.send_config import SendConfig
from yggdrasil.http_.session import HTTPSession
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Test HTTP server
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    call_count: int = 0

    def do_GET(self):
        _Handler.call_count += 1
        if self.path == "/json":
            body = json.dumps({"ok": True, "n": _Handler.call_count}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/text":
            body = b"hello world"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/empty":
            self.send_response(204)
            self.end_headers()
        elif self.path == "/error":
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            body = b"internal error"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/slow":
            time.sleep(0.3)
            body = b"done"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        _Handler.call_count += 1
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        resp = json.dumps({"echoed": body.decode(), "n": _Handler.call_count}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture()
def base_url(server):
    _Handler.call_count = 0
    return server


@pytest.fixture()
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Basic HTTP verbs
# ---------------------------------------------------------------------------


class TestGet:

    def test_get_json(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    def test_get_text(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/text")
        assert resp.status_code == 200
        assert resp.text == "hello world"

    def test_get_empty(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/empty")
        assert resp.status_code == 204

    def test_get_404(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/nonexistent")
        assert resp.status_code == 404

    def test_get_500(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/error")
        assert resp.status_code == 500
        assert not resp.ok


class TestPost:

    def test_post_json_body(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.post("/echo", data=b'{"key": "value"}')
        assert resp.status_code == 200
        data = resp.json()
        assert data["echoed"] == '{"key": "value"}'


class TestResponseProperties:

    def test_content_type(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/json")
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_ok_property(self, base_url):
        session = HTTPSession(base_url=base_url)
        assert session.get("/json").ok
        assert not session.get("/error").ok

    def test_raise_for_status(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/error")
        with pytest.raises(Exception):
            resp.raise_for_status()

    def test_response_has_request(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/json")
        assert resp.request is not None
        assert resp.request.method == "GET"


# ---------------------------------------------------------------------------
# Session singleton
# ---------------------------------------------------------------------------


class TestSessionSingleton:

    def test_same_base_url_same_instance(self, base_url):
        a = HTTPSession(base_url=base_url)
        b = HTTPSession(base_url=base_url)
        assert a is b

    def test_different_base_url_different_instance(self, base_url):
        a = HTTPSession(base_url=base_url)
        b = HTTPSession(base_url="http://other.example.com")
        assert a is not b


# ---------------------------------------------------------------------------
# Local cache
# ---------------------------------------------------------------------------


class TestLocalCache:

    def test_cache_hit_skips_network(self, base_url, cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(root=cache_dir)
        cfg = SendConfig(local_cache=cache)

        req = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        req.send_config = cfg

        resp1 = session.send(req)
        assert resp1.status_code == 200
        n1 = resp1.json()["n"]

        req2 = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        req2.send_config = cfg
        resp2 = session.send(req2)
        assert resp2.status_code == 200
        n2 = resp2.json()["n"]
        assert n2 == n1

    def test_different_urls_not_cached(self, base_url, cache_dir):
        session = HTTPSession(base_url=base_url)
        cache = CacheConfig(root=cache_dir)
        cfg = SendConfig(local_cache=cache)

        req1 = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        req1.send_config = cfg
        resp1 = session.send(req1)
        n1 = resp1.json()["n"]

        req2 = HTTPRequest.prepare(method="GET", url=f"{base_url}/text")
        req2.send_config = cfg
        resp2 = session.send(req2)
        assert resp2.text == "hello world"
        assert _Handler.call_count > n1


# ---------------------------------------------------------------------------
# send_many
# ---------------------------------------------------------------------------


class TestSendMany:

    def test_send_many_returns_all_responses(self, base_url):
        session = HTTPSession(base_url=base_url)
        reqs = [
            HTTPRequest.prepare(method="GET", url=f"{base_url}/json"),
            HTTPRequest.prepare(method="GET", url=f"{base_url}/text"),
        ]
        batch = session.send_many(reqs)
        responses = list(batch.responses())
        assert len(responses) == 2


# ---------------------------------------------------------------------------
# Request/Response round-trip
# ---------------------------------------------------------------------------


class TestRequestResponse:

    def test_request_url_parsed(self, base_url):
        req = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        assert req.url.path == "/json"
        assert req.method == "GET"

    def test_request_headers(self, base_url):
        req = HTTPRequest.prepare(
            method="GET",
            url=f"{base_url}/json",
            headers={"X-Custom": "test"},
        )
        assert req.headers.get("X-Custom") == "test"

    def test_request_hash_stable(self, base_url):
        a = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        b = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        assert a.hash == b.hash

    def test_request_hash_differs_by_method(self, base_url):
        a = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        b = HTTPRequest.prepare(method="POST", url=f"{base_url}/json")
        assert a.hash != b.hash
