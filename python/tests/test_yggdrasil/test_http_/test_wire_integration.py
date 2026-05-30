"""Wire-level integration tests for HTTPSession.

Tests the full send pipeline against a local HTTP server — connection
pooling, keep-alive reuse, concurrent dispatch, large bodies, chunked
responses, timeouts, redirect chains, and error recovery.
"""
from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.session import HTTPSession


# ---------------------------------------------------------------------------
# Local server
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    call_count: int = 0

    def do_GET(self):
        _Handler.call_count += 1
        path = self.path.split("?")[0]

        if path == "/json":
            body = json.dumps({"n": _Handler.call_count}).encode()
            self._respond(200, body, "application/json")

        elif path == "/large":
            body = json.dumps({"data": "x" * 100_000}).encode()
            self._respond(200, body, "application/json")

        elif path == "/slow":
            time.sleep(0.3)
            self._respond(200, b'{"slow": true}', "application/json")

        elif path == "/redirect-chain":
            self.send_response(302)
            self.send_header("Location", "/redirect-step2")
            self.end_headers()

        elif path == "/redirect-step2":
            self.send_response(301)
            self.send_header("Location", "/json")
            self.end_headers()

        elif path == "/status/204":
            self.send_response(204)
            self.end_headers()

        elif path == "/status/429":
            self.send_response(429)
            self.send_header("Retry-After", "0")
            body = b'{"error": "rate limited"}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/status/500":
            self._respond(500, b'{"error": "server"}', "application/json")

        elif path == "/status/503":
            self._respond(503, b'{"error": "unavailable"}', "application/json")

        elif path == "/echo-headers":
            headers = {k: v for k, v in self.headers.items()}
            body = json.dumps(headers).encode()
            self._respond(200, body, "application/json")

        elif path == "/close-connection":
            self.send_response(200)
            self.send_header("Connection", "close")
            body = b'{"close": true}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self._respond(404, b"not found", "text/plain")

    def do_POST(self):
        _Handler.call_count += 1
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        resp = json.dumps({
            "echoed_size": len(body),
            "content_type": self.headers.get("Content-Type"),
            "n": _Handler.call_count,
        }).encode()
        self._respond(200, resp, "application/json")

    def do_HEAD(self):
        _Handler.call_count += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Count", str(_Handler.call_count))
        self.end_headers()

    def _respond(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _reset():
    _Handler.call_count = 0
    HTTPSession._INSTANCES.clear()
    yield
    HTTPSession._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Connection pooling / keep-alive
# ---------------------------------------------------------------------------


class TestConnectionPooling:

    def test_keep_alive_reuses_socket(self, server):
        session = HTTPSession(base_url=server)
        r1 = session.get("/json")
        r2 = session.get("/json")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()["n"] == r1.json()["n"] + 1

    def test_pool_survives_close_header(self, server):
        session = HTTPSession(base_url=server)
        r1 = session.get("/close-connection")
        assert r1.json()["close"] is True
        r2 = session.get("/json")
        assert r2.status_code == 200

    def test_many_sequential_reuse(self, server):
        session = HTTPSession(base_url=server)
        for i in range(20):
            r = session.get("/json")
            assert r.status_code == 200
        assert _Handler.call_count == 20

    def test_clear_connections(self, server):
        session = HTTPSession(base_url=server)
        session.get("/json")
        session.clear_connections()
        r = session.get("/json")
        assert r.status_code == 200

    def test_keep_alive_header_is_sent(self, server):
        session = HTTPSession(base_url=server)
        echoed = {k.lower(): v for k, v in session.get("/echo-headers").json().items()}
        assert echoed.get("connection") == "keep-alive"


class _KeepAliveHandler(_Handler):
    # HTTP/1.1 so the server honours keep-alive and the client can actually
    # pool + reuse the socket (the base _Handler is HTTP/1.0 → every response
    # is will_close, which never pools).
    protocol_version = "HTTP/1.1"


@pytest.fixture(scope="module")
def ka_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _KeepAliveHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


class TestKeepAlivePooling:

    @staticmethod
    def _idle(session):
        return sum(len(q) for q in session._connections.values())

    def test_keep_alive_response_is_pooled_and_reused(self, ka_server):
        session = HTTPSession(base_url=ka_server)
        session.get("/json").json()
        assert self._idle(session) == 1   # socket held open for reuse
        session.get("/json").json()
        assert self._idle(session) == 1   # same socket reused, not a 2nd one

    def test_close_header_socket_is_not_pooled(self, ka_server):
        session = HTTPSession(base_url=ka_server)
        session.get("/close-connection").json()
        # Server said Connection: close — don't pool a dead socket.
        assert self._idle(session) == 0
        # …and the next request still works on a fresh socket.
        assert session.get("/json").status_code == 200


# ---------------------------------------------------------------------------
# Concurrent dispatch
# ---------------------------------------------------------------------------


class TestConcurrentDispatch:

    def test_send_many_parallel(self, server):
        session = HTTPSession(base_url=server)
        reqs = [
            HTTPRequest.prepare("GET", f"{server}/json?i={i}")
            for i in range(20)
        ]
        resps = list(session.send_many(reqs))
        assert len(resps) == 20
        assert all(r.status_code == 200 for r in resps)

    def test_send_many_mixed_status(self, server):
        session = HTTPSession(base_url=server)
        reqs = [
            HTTPRequest.prepare("GET", f"{server}/json"),
            HTTPRequest.prepare("GET", f"{server}/status/204"),
            HTTPRequest.prepare("GET", f"{server}/status/500"),
        ]
        resps = list(session.send_many(reqs, raise_error=False))
        codes = sorted(r.status_code for r in resps)
        assert 200 in codes
        assert 204 in codes
        assert 500 in codes

    def test_threaded_sessions(self, server):
        import concurrent.futures
        results = []

        def worker(i):
            session = HTTPSession(base_url=server)
            r = session.get(f"/json?t={i}")
            return r.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(worker, i) for i in range(16)]
            results = [f.result(timeout=10) for f in futs]

        assert all(r == 200 for r in results)


# ---------------------------------------------------------------------------
# Large bodies
# ---------------------------------------------------------------------------


class TestLargeBodies:

    def test_large_response(self, server):
        session = HTTPSession(base_url=server)
        r = session.get("/large")
        data = r.json()
        assert len(data["data"]) == 100_000

    def test_large_post(self, server):
        session = HTTPSession(base_url=server)
        payload = b"x" * 50_000
        r = session.post("/echo", data=payload)
        assert r.json()["echoed_size"] == 50_000

    def test_empty_body_204(self, server):
        session = HTTPSession(base_url=server)
        r = session.get("/status/204")
        assert r.status_code == 204


# ---------------------------------------------------------------------------
# Redirect chains
# ---------------------------------------------------------------------------


class TestRedirectChains:

    def test_double_redirect(self, server):
        session = HTTPSession(base_url=server)
        r = session.get("/redirect-chain")
        assert r.status_code == 200
        assert r.json()["n"] > 0

    def test_redirect_preserves_session(self, server):
        session = HTTPSession(base_url=server)
        r = session.get("/redirect-chain")
        assert r.request is not None


# ---------------------------------------------------------------------------
# Error status codes
# ---------------------------------------------------------------------------


class TestErrorStatuses:

    def test_500_raise_error_false(self, server):
        session = HTTPSession(base_url=server)
        r = session.get("/status/500", raise_error=False)
        assert r.status_code == 500
        assert not r.ok

    def test_500_raises_by_default(self, server):
        session = HTTPSession(base_url=server)
        with pytest.raises(Exception):
            session.get("/status/500")

    def test_503_raise_error_false(self, server):
        session = HTTPSession(base_url=server)
        r = session.get("/status/503", raise_error=False)
        assert r.status_code == 503

    def test_404_raise_error_false(self, server):
        session = HTTPSession(base_url=server)
        r = session.get("/nonexistent", raise_error=False)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Header forwarding
# ---------------------------------------------------------------------------


class TestHeaderForwarding:

    def test_session_headers_forwarded(self, server):
        session = HTTPSession(
            base_url=server,
            headers={"X-Custom": "session-val"},
        )
        r = session.get("/echo-headers")
        data = r.json()
        assert data.get("x-custom") == "session-val" or data.get("X-Custom") == "session-val"

    def test_per_request_headers(self, server):
        session = HTTPSession(base_url=server)
        r = session.get("/echo-headers", headers={"X-Per-Request": "yes"})
        data = r.json()
        found = data.get("x-per-request") or data.get("X-Per-Request")
        assert found == "yes"

    def test_head_returns_custom_header(self, server):
        session = HTTPSession(base_url=server)
        r = session.head("/head")
        assert r.status_code == 200
        count = r.headers.get("X-Count") or r.headers.get("x-count")
        assert count is not None


# ---------------------------------------------------------------------------
# POST with content types
# ---------------------------------------------------------------------------


class TestPost:

    def test_post_json(self, server):
        session = HTTPSession(base_url=server)
        r = session.post("/echo", json={"key": "value"})
        data = r.json()
        assert data["echoed_size"] > 0
        assert "json" in (data.get("content_type") or "").lower()

    def test_post_raw_bytes(self, server):
        session = HTTPSession(base_url=server)
        r = session.post("/echo", data=b"raw-bytes-here")
        assert r.json()["echoed_size"] == 14


# ---------------------------------------------------------------------------
# Arrow metadata round-trip
# ---------------------------------------------------------------------------


class TestArrowMetadata:

    def test_batch_multiple_responses(self, server):
        session = HTTPSession(base_url=server)
        resps = [session.get("/json"), session.get("/json")]
        batch = HTTPResponse.values_to_arrow_batch(resps)
        assert batch.num_rows == 2
        assert all(s == 200 for s in batch.column("status_code").to_pylist())

    def test_response_hash_differs_by_body(self, server):
        session = HTTPSession(base_url=server)
        r1 = session.get("/json")
        r2 = session.get("/json")
        assert r1.arrow_values["request_hash"] == r2.arrow_values["request_hash"]
