"""Integration tests for HTTPSession with real HTTP calls."""
from __future__ import annotations

import json
import http.server
import threading

import pytest

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.session import HTTPSession


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
            body = b"internal error"
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
    HTTPSession._INSTANCES.clear()
    return server


class TestGet:

    def test_get_json(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/json")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

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
        resp = session.get("/nonexistent", raise_error=False)
        assert resp.status_code == 404

    def test_get_500(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/error", raise_error=False)
        assert resp.status_code == 500
        assert not resp.ok


class TestPost:

    def test_post_body(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.post("/echo", data=b'{"key": "value"}')
        assert resp.status_code == 200
        assert resp.json()["echoed"] == '{"key": "value"}'


class TestResponseProperties:

    def test_content_type(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/json")
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_ok_property(self, base_url):
        session = HTTPSession(base_url=base_url)
        assert session.get("/json").ok
        assert not session.get("/error", raise_error=False).ok

    def test_raise_for_status(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/error", raise_error=False)
        with pytest.raises(Exception):
            resp.raise_for_status()

    def test_response_has_request(self, base_url):
        session = HTTPSession(base_url=base_url)
        resp = session.get("/json")
        assert resp.request is not None
        assert resp.request.method == "GET"


class TestSessionSingleton:

    def test_same_base_url_same_instance(self, base_url):
        a = HTTPSession(base_url=base_url)
        b = HTTPSession(base_url=base_url)
        assert a is b

    def test_different_base_url_different_instance(self, base_url):
        a = HTTPSession(base_url=base_url)
        b = HTTPSession(base_url="http://other.example.com")
        assert a is not b


class TestRequestResponse:

    def test_request_url_parsed(self, base_url):
        req = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        assert req.url.path == "/json"

    def test_request_hash_stable(self, base_url):
        a = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        b = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        assert a.hash == b.hash

    def test_request_hash_differs_by_method(self, base_url):
        a = HTTPRequest.prepare(method="GET", url=f"{base_url}/json")
        b = HTTPRequest.prepare(method="POST", url=f"{base_url}/json")
        assert a.hash != b.hash
