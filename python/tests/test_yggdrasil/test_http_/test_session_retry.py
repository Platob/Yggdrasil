"""Send-path resilience: connection-level failures (stale keep-alive, TLS EOF,
reset) retry up to the full budget, each on a *fresh* connection.

This is the ``ssl.SSLEOFError`` mid-upload case — a Databricks Files PUT whose
pooled TLS socket the peer silently dropped. The fix bumps the retry budget to
8 (including the ``other`` tier that classifies SSL-EOF / reset) and evicts the
host's idle pool before each retry so we never hand back another about-to-fail
socket.
"""
from __future__ import annotations

import contextlib
import http.server
import logging
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
    fail_first = 0     # drop the first N connections before responding
    throttle_first = 0  # answer the first N requests with 429 before 200
    conns = 0          # connections the server accepted
    requests = 0       # requests the server answered (any status)

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
        type(self).requests += 1
        if type(self).requests <= type(self).throttle_first:
            body = b"slow down"
            self.send_response(429)
            self.send_header("Retry-After", "0")  # no real wait in the test
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
    _FlakyHandler.throttle_first = 0
    _FlakyHandler.conns = 0
    _FlakyHandler.requests = 0
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


def test_429_retries_on_a_fresh_connection(server):
    # First three requests are throttled (429), the fourth is 200. Each 429
    # retry must dial a fresh socket rather than reuse the throttled pooled
    # one — so connection count tracks request count.
    _FlakyHandler.throttle_first = 3
    session = HTTPSession(base_url=server)
    r = session.get("/x")
    assert r.status_code == 200
    assert r.content == b"ok"
    assert _FlakyHandler.requests == 4   # 3 × 429 + 1 × 200
    assert _FlakyHandler.conns == 4      # a fresh connection per attempt


# -- connection-error logging: quiet until past the 4th retry ------------


@contextlib.contextmanager
def _capture_conn_logs():
    """Capture the session logger's connection-drop retry records (debug +
    warning) directly off ``yggdrasil.http_.session`` — the project attaches its
    own handler and the ``yggdrasil`` logger doesn't propagate to root, so
    pytest's ``caplog`` (a root handler) never sees these."""
    logger = logging.getLogger("yggdrasil.http_.session")
    records: list[logging.LogRecord] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "Connection error on" in record.getMessage():
                records.append(record)

    sink = _Sink(level=logging.DEBUG)
    prev_level = logger.level
    logger.addHandler(sink)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(sink)
        logger.setLevel(prev_level)


def test_early_connection_drops_stay_quiet(server):
    # Three transient drops then success: every retry is within the first
    # four, so none escalate to WARNING — they're logged at DEBUG only.
    _FlakyHandler.fail_first = 3
    with _capture_conn_logs() as records:
        session = HTTPSession(base_url=server)
        assert session.get("/x").status_code == 200

    assert len(records) == 3                                   # one per drop
    assert all(r.levelno == logging.DEBUG for r in records)    # none warned


def test_persistent_connection_drops_warn_after_fourth_retry(server):
    # Five drops then success: retries 1–3 stay DEBUG, the 4th and 5th escalate
    # to WARNING — only persistent drops surface as warnings.
    _FlakyHandler.fail_first = 5
    with _capture_conn_logs() as records:
        session = HTTPSession(base_url=server)
        assert session.get("/x").status_code == 200

    assert len(records) == 5
    debugs = [r for r in records if r.levelno == logging.DEBUG]
    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert len(debugs) == 3       # the first three retries stay quiet
    assert len(warnings) == 2     # the 4th and 5th retries warn


# -- 429 identity rotation: only for unauthenticated traffic -------------
#
# Browser-identity rotation on a 429 is a scraping tactic. Against an
# *authenticated* API (a request carrying ``Authorization``) it is actively
# harmful: services like the Databricks Files API key the authenticated rate
# limit on a stable, attributable User-Agent, so swapping in a random browser
# UA mid-retry gets the request reclassified as anonymous and throttled harder
# — turning one 429 into a storm. The session must leave an authenticated
# request's User-Agent untouched and only rotate the unauthenticated case.


class _UACapturingHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    throttle_first = 0          # answer the first N requests with 429 before 200
    requests = 0
    user_agents: list[str] = []

    def do_GET(self):
        type(self).requests += 1
        type(self).user_agents.append(self.headers.get("User-Agent", ""))
        if type(self).requests <= type(self).throttle_first:
            body = b"slow down"
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture(scope="module")
def ua_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _UACapturingHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _reset_ua_handler():
    _UACapturingHandler.throttle_first = 0
    _UACapturingHandler.requests = 0
    _UACapturingHandler.user_agents = []
    yield


def test_429_keeps_user_agent_for_authenticated_requests(ua_server):
    # An authenticated request (Authorization header set) must keep its
    # User-Agent verbatim across every 429 retry — the edge attributes the
    # authenticated rate limit to it.
    _UACapturingHandler.throttle_first = 3
    session = HTTPSession(base_url=ua_server)
    stable_ua = "yggdrasil-sdk/1.2.3 (databricks-files)"
    r = session.get(
        "/x",
        headers={"Authorization": "Bearer tok", "User-Agent": stable_ua},
    )
    assert r.status_code == 200
    assert _UACapturingHandler.requests == 4         # 3 × 429 + 1 × 200
    # Every attempt — initial and all three retries — carried the same UA.
    assert _UACapturingHandler.user_agents == [stable_ua] * 4


def test_429_rotates_user_agent_for_unauthenticated_requests(ua_server):
    # No Authorization → the scraping rotation still applies: at least one
    # retry presents a different (browser-shaped) User-Agent.
    _UACapturingHandler.throttle_first = 3
    session = HTTPSession(base_url=ua_server)
    stable_ua = "yggdrasil-sdk/1.2.3"
    r = session.get("/x", headers={"User-Agent": stable_ua})
    assert r.status_code == 200
    assert _UACapturingHandler.requests == 4
    seen = _UACapturingHandler.user_agents
    assert seen[0] == stable_ua                       # initial attempt unchanged
    assert any(ua != stable_ua for ua in seen[1:])    # retries rotated identity
