"""Wire-level HTTP session integration tests.

Exercises the real :meth:`HTTPSession._send_once` code path (stdlib
``http.client``) against a local ``http.server`` stub. Covers:

* Fresh connection + response round-trip.
* Connection reuse (keep-alive pool).
* Stale pooled connection recovery (server closes socket between requests).
* Timeout propagation — connect timeout fires, read timeout fires.
* Content-Length auto-stamping from body size.
* Redirect following (301/302/303 → GET, 307 → preserve method).
* Retry on 5xx/429 with backoff.
"""
from __future__ import annotations

import http.server
import socket
import ssl
import threading
import time
from typing import Any

import pytest

from yggdrasil.dataclasses.waiting import WaitingConfig
from yggdrasil.http_ import HTTPSession
from yggdrasil.http_.exceptions import MaxRetryError, ReadTimeoutError
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# Test server infrastructure
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    """Minimal handler that dispatches to per-path callables on the server."""

    def log_message(self, format, *args):
        pass  # silence stderr

    def _dispatch(self):
        path = self.path.split("?", 1)[0]
        handler = self.server._routes.get(path)  # type: ignore[attr-defined]
        if handler is None:
            self.send_error(404)
            return
        handler(self)

    do_GET = _dispatch
    do_POST = _dispatch
    do_PUT = _dispatch


class _TestServer(http.server.HTTPServer):
    """Threaded HTTP server with per-path route registration."""

    def __init__(self):
        # Bind to a free port on localhost.
        super().__init__(("127.0.0.1", 0), _Handler)
        self._routes: dict[str, Any] = {}
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"

    def route(self, path: str):
        """Decorator to register a handler for *path*."""
        def decorator(fn):
            self._routes[path] = fn
            return fn
        return decorator

    def stop(self):
        self.shutdown()
        self._thread.join(timeout=2)


@pytest.fixture()
def server():
    srv = _TestServer()
    yield srv
    srv.stop()


def _session(base_url: str, **kwargs) -> HTTPSession:
    """Build a non-singleton session for testing."""
    # Bypass the singleton cache by using a unique base_url each time
    # (the port is already unique per fixture).
    return HTTPSession(base_url=URL.from_(base_url), **kwargs)


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------


class TestWireRoundTrip:

    def test_get_200(self, server: _TestServer) -> None:
        @server.route("/hello")
        def handler(req: _Handler):
            req.send_response(200)
            req.send_header("Content-Type", "application/json")
            body = b'{"msg":"hi"}'
            req.send_header("Content-Length", str(len(body)))
            req.end_headers()
            req.wfile.write(body)

        session = _session(server.base_url)
        resp = session.get(f"{server.base_url}/hello", raise_error=False)
        assert resp.status_code == 200
        assert resp.json() == {"msg": "hi"}

    def test_post_with_body(self, server: _TestServer) -> None:
        received: dict[str, Any] = {}

        @server.route("/echo")
        def handler(req: _Handler):
            length = int(req.headers.get("Content-Length", 0))
            received["body"] = req.rfile.read(length)
            received["content_length"] = req.headers.get("Content-Length")
            req.send_response(200)
            req.send_header("Content-Length", "0")
            req.end_headers()

        session = _session(server.base_url)
        session.post(f"{server.base_url}/echo", body=b"hello world")
        assert received["body"] == b"hello world"
        assert received["content_length"] == "11"

    def test_empty_body_no_content_length(self, server: _TestServer) -> None:
        received_headers: dict[str, str | None] = {}

        @server.route("/no-body")
        def handler(req: _Handler):
            received_headers["content_length"] = req.headers.get("Content-Length")
            req.send_response(204)
            req.send_header("Content-Length", "0")
            req.end_headers()

        session = _session(server.base_url)
        session.get(f"{server.base_url}/no-body", raise_error=False)
        assert received_headers["content_length"] is None


# ---------------------------------------------------------------------------
# Connection reuse (keep-alive)
# ---------------------------------------------------------------------------


class TestConnectionReuse:

    def test_reuses_connection_across_requests(self, server: _TestServer) -> None:
        call_count = {"n": 0}

        @server.route("/count")
        def handler(req: _Handler):
            call_count["n"] += 1
            req.send_response(200)
            req.send_header("Content-Length", "2")
            req.end_headers()
            req.wfile.write(b"ok")

        session = _session(server.base_url)
        session.get(f"{server.base_url}/count", raise_error=False)
        session.get(f"{server.base_url}/count", raise_error=False)
        session.get(f"{server.base_url}/count", raise_error=False)

        assert call_count["n"] == 3
        # Only one connection should be in the pool (reused).
        host, port = server.server_address
        key = ("http", host, port)
        cached = session._connections.get(key)
        assert cached is not None
        assert len(cached) == 1


# ---------------------------------------------------------------------------
# Stale connection recovery
# ---------------------------------------------------------------------------


class TestStaleConnectionRecovery:

    def test_recovers_from_stale_pooled_connection(self, server: _TestServer) -> None:
        """A server that closes keep-alive sockets between requests."""
        call_count = {"n": 0}

        @server.route("/fragile")
        def handler(req: _Handler):
            call_count["n"] += 1
            body = b'{"attempt":' + str(call_count["n"]).encode() + b'}'
            req.send_response(200)
            req.send_header("Content-Type", "application/json")
            req.send_header("Content-Length", str(len(body)))
            req.send_header("Connection", "close")
            req.end_headers()
            req.wfile.write(body)

        session = _session(server.base_url)

        # First request succeeds and connection goes back to pool.
        resp1 = session.get(f"{server.base_url}/fragile", raise_error=False)
        assert resp1.status_code == 200

        # Server sent Connection: close, so the connection is closed
        # server-side. A naive pool would try to reuse it and fail.
        # Our implementation should transparently retry on a fresh connection.
        resp2 = session.get(f"{server.base_url}/fragile", raise_error=False)
        assert resp2.status_code == 200
        assert call_count["n"] == 2

    def test_stale_connection_does_not_exhaust_retries(self, server: _TestServer) -> None:
        """Stale connection recovery must not count against the retry budget."""
        call_count = {"n": 0}

        @server.route("/stale-test")
        def handler(req: _Handler):
            call_count["n"] += 1
            req.send_response(200)
            req.send_header("Content-Length", "2")
            req.end_headers()
            req.wfile.write(b"ok")

        session = _session(server.base_url)

        # Prime the pool with a connection, then kill it server-side.
        resp = session.get(f"{server.base_url}/stale-test", raise_error=False)
        assert resp.status_code == 200

        # Manually close the pooled connection's socket to simulate a
        # server-side close that the client hasn't noticed yet.
        host, port = server.server_address
        key = ("http", host, port)
        cached = session._connections.get(key)
        if cached:
            stale_conn = cached[0]
            if stale_conn.sock:
                stale_conn.sock.close()

        # The next request should succeed — the stale socket is detected
        # and a fresh connection is made without consuming retry budget.
        resp2 = session.get(f"{server.base_url}/stale-test", raise_error=False)
        assert resp2.status_code == 200
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Timeout propagation
# ---------------------------------------------------------------------------


class TestTimeoutPropagation:

    def test_connect_timeout_fires(self) -> None:
        """Connecting to a non-routable address times out quickly."""
        # 192.0.2.1 is TEST-NET-1 (RFC 5737) — guaranteed non-routable,
        # so connect() will hang until timeout fires.
        session = HTTPSession(
            base_url=URL.from_("http://192.0.2.1"),
            waiting=WaitingConfig(timeout=0.5, retries=0),
        )
        req = PreparedRequest.prepare("GET", "http://192.0.2.1/never")
        t0 = time.monotonic()
        with pytest.raises((MaxRetryError, OSError, TimeoutError)):
            session.send(req, raise_error=False)
        elapsed = time.monotonic() - t0
        # Should have timed out in ~0.5s, not 20 minutes.
        assert elapsed < 5.0

    def test_read_timeout_fires(self, server: _TestServer) -> None:
        """Server accepts connection but delays the response past the timeout."""
        @server.route("/hang")
        def handler(req: _Handler):
            time.sleep(3)  # longer than the client timeout
            req.send_response(200)
            req.send_header("Content-Length", "0")
            req.end_headers()

        session = _session(
            server.base_url,
            waiting=WaitingConfig(timeout=0.3, retries=0),
        )
        req = PreparedRequest.prepare("GET", f"{server.base_url}/hang")
        t0 = time.monotonic()
        with pytest.raises((MaxRetryError, ReadTimeoutError, socket.timeout, TimeoutError)):
            session.send(req, raise_error=False)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0

    def test_default_connect_timeout_capped_at_30s(self) -> None:
        """Default WaitingConfig caps connect at 30s, not 20 min."""
        from yggdrasil.dataclasses.waiting import DEFAULT_WAITING_CONFIG
        tp = DEFAULT_WAITING_CONFIG.timeout_pool
        assert tp.connect == 30.0
        assert tp.read == 1200.0


# ---------------------------------------------------------------------------
# Redirect following
# ---------------------------------------------------------------------------


class TestRedirects:

    def test_301_follows_redirect(self, server: _TestServer) -> None:
        @server.route("/old")
        def redirect(req: _Handler):
            req.send_response(301)
            req.send_header("Location", "/new")
            req.send_header("Content-Length", "0")
            req.end_headers()

        @server.route("/new")
        def target(req: _Handler):
            req.send_response(200)
            body = b"arrived"
            req.send_header("Content-Length", str(len(body)))
            req.end_headers()
            req.wfile.write(body)

        session = _session(server.base_url)
        resp = session.get(f"{server.base_url}/old", raise_error=False)
        assert resp.status_code == 200
        assert resp.content == b"arrived"

    def test_302_post_becomes_get(self, server: _TestServer) -> None:
        received_methods: list[str] = []

        @server.route("/submit")
        def redirect(req: _Handler):
            received_methods.append("POST")
            req.send_response(302)
            req.send_header("Location", "/result")
            req.send_header("Content-Length", "0")
            req.end_headers()

        @server.route("/result")
        def target(req: _Handler):
            received_methods.append("GET")
            req.send_response(200)
            req.send_header("Content-Length", "4")
            req.end_headers()
            req.wfile.write(b"done")

        session = _session(server.base_url)
        resp = session.post(f"{server.base_url}/submit", body=b"data", raise_error=False)
        assert resp.status_code == 200
        assert received_methods == ["POST", "GET"]

    def test_307_preserves_method(self, server: _TestServer) -> None:
        received_methods: list[str] = []

        @server.route("/temp")
        def redirect(req: _Handler):
            received_methods.append(req.command)
            req.send_response(307)
            req.send_header("Location", "/dest")
            req.send_header("Content-Length", "0")
            req.end_headers()

        @server.route("/dest")
        def target(req: _Handler):
            received_methods.append(req.command)
            req.send_response(200)
            req.send_header("Content-Length", "2")
            req.end_headers()
            req.wfile.write(b"ok")

        session = _session(server.base_url)
        resp = session.post(f"{server.base_url}/temp", body=b"x", raise_error=False)
        assert resp.status_code == 200
        assert received_methods == ["POST", "POST"]


# ---------------------------------------------------------------------------
# Retry on 5xx / 429
# ---------------------------------------------------------------------------


class TestRetry:

    def test_retries_on_503(self, server: _TestServer) -> None:
        attempts = {"n": 0}

        @server.route("/flaky")
        def handler(req: _Handler):
            attempts["n"] += 1
            if attempts["n"] < 3:
                req.send_response(503)
                req.send_header("Content-Length", "0")
                req.end_headers()
            else:
                req.send_response(200)
                body = b"recovered"
                req.send_header("Content-Length", str(len(body)))
                req.end_headers()
                req.wfile.write(body)

        session = _session(server.base_url)
        resp = session.get(f"{server.base_url}/flaky", raise_error=False)
        assert resp.status_code == 200
        assert resp.content == b"recovered"
        assert attempts["n"] == 3

    def test_max_retries_returns_last_response(self, server: _TestServer) -> None:
        attempts = {"n": 0}

        @server.route("/always-503")
        def handler(req: _Handler):
            attempts["n"] += 1
            req.send_response(503)
            req.send_header("Content-Length", "0")
            req.end_headers()

        session = _session(server.base_url)
        resp = session.get(f"{server.base_url}/always-503", raise_error=False)
        assert resp.status_code == 503
        # total=3 retries + 1 initial = 4 attempts
        assert attempts["n"] == 4

    def test_429_respects_retry_after(self, server: _TestServer) -> None:
        attempts = {"n": 0}
        timestamps: list[float] = []

        @server.route("/rate-limit")
        def handler(req: _Handler):
            attempts["n"] += 1
            timestamps.append(time.monotonic())
            if attempts["n"] == 1:
                req.send_response(429)
                req.send_header("Retry-After", "1")
                req.send_header("Content-Length", "0")
                req.end_headers()
            else:
                req.send_response(200)
                req.send_header("Content-Length", "2")
                req.end_headers()
                req.wfile.write(b"ok")

        session = _session(server.base_url)
        resp = session.get(f"{server.base_url}/rate-limit", raise_error=False)
        assert resp.status_code == 200
        assert attempts["n"] == 2
        # Should have waited ~1 second between attempts.
        assert timestamps[1] - timestamps[0] >= 0.9


# ---------------------------------------------------------------------------
# URL parsing — _send_once uses request.url directly
# ---------------------------------------------------------------------------


class TestURLParsing:

    def test_query_string_preserved(self, server: _TestServer) -> None:
        received_paths: list[str] = []

        @server.route("/search")
        def handler(req: _Handler):
            received_paths.append(req.path)
            req.send_response(200)
            req.send_header("Content-Length", "2")
            req.end_headers()
            req.wfile.write(b"ok")

        session = _session(server.base_url)
        session.get(
            f"{server.base_url}/search",
            params={"q": "test", "page": "2"},
            raise_error=False,
        )
        assert "q=test" in received_paths[0]
        assert "page=2" in received_paths[0]

    def test_non_default_port(self, server: _TestServer) -> None:
        """Port from the URL is used for the connection."""
        received_hosts: list[str] = []

        @server.route("/port-check")
        def handler(req: _Handler):
            received_hosts.append(req.headers.get("Host", ""))
            req.send_response(200)
            req.send_header("Content-Length", "2")
            req.end_headers()
            req.wfile.write(b"ok")

        session = _session(server.base_url)
        session.get(f"{server.base_url}/port-check", raise_error=False)
        host, port = server.server_address
        assert received_hosts[0] == f"{host}:{port}"
