"""Send-path resilience: connection-level failures (stale keep-alive, TLS EOF,
reset) retry up to the full budget, each on a *fresh* connection.

This is the ``ssl.SSLEOFError`` mid-upload case — a Databricks Files PUT whose
pooled TLS socket the peer silently dropped. The fix bumps the retry budget to
8 (including the ``other`` tier that classifies SSL-EOF / reset) and evicts the
host's idle pool before each retry so we never hand back another about-to-fail
socket.
"""
from __future__ import annotations

import http.server
import threading

import pytest

from yggdrasil.http_ import retry as retry_mod
from yggdrasil.http_.exceptions import MaxRetryError, NewConnectionError
from yggdrasil.http_.session import HTTPSession


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    # Keep the retry schedule from actually sleeping during the test.
    monkeypatch.setattr(retry_mod.time, "sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def _fresh_singletons():
    HTTPSession._INSTANCES.clear()
    yield
    HTTPSession._INSTANCES.clear()


# -- budget --------------------------------------------------------------

def test_retry_budget_raised_to_eight():
    r = HTTPSession()._retry
    assert r.total == 8
    # "other" (SSL EOF / connection reset on send) tracks the full total so
    # the reported failure mode gets every retry, not a smaller sub-budget.
    assert r.other == 8
    assert r.status == 8


def test_connection_errors_exhaust_only_after_eight():
    session = HTTPSession()
    r = session._retry.new()
    n = 0
    with pytest.raises(MaxRetryError):
        while True:
            r = r.increment(
                method="POST", url="https://peer/api",
                error=NewConnectionError(session, "EOF in violation of protocol"),
                _pool=session,
            )
            n += 1
    assert n == 8  # 8 retries succeed; the 9th increment exhausts the budget


# -- new connection per retry (integration) ------------------------------

class _FlakyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fail_first = 0   # drop the first N connections before responding
    conns = 0        # connections the server accepted

    def setup(self):
        super().setup()
        type(self).conns += 1
        self._kill = type(self).conns <= type(self).fail_first

    def do_GET(self):
        if self._kill:
            # Close without responding → the client's getresponse hits EOF and
            # raises RemoteDisconnected (a ConnectionResetError), the same shape
            # as a peer dropping a pooled socket.
            self.close_connection = True
            return
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FlakyHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _reset_handler():
    _FlakyHandler.fail_first = 0
    _FlakyHandler.conns = 0
    yield


def test_recovers_after_dropped_connections_each_retry_dials_fresh(server):
    # First three connections are dropped mid-handshake; the fourth answers.
    _FlakyHandler.fail_first = 3
    session = HTTPSession(base_url=server)
    r = session.get("/x")
    assert r.status_code == 200
    assert r.content == b"ok"
    # One TCP connection per attempt → the retries used fresh sockets, not a
    # re-popped (already-dead) pooled one.
    assert _FlakyHandler.conns == 4


def test_exhausts_budget_and_raises_after_eight(server):
    # The peer never recovers — give up after the 8-retry budget, having tried
    # a fresh connection each time.
    _FlakyHandler.fail_first = 99
    session = HTTPSession(base_url=server)
    with pytest.raises(MaxRetryError):
        session.get("/x")
    assert _FlakyHandler.conns == 9  # initial attempt + 8 retries, all fresh
