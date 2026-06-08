"""Connection-pool semantics + thread-safety for :class:`HTTPSession`.

``HTTPSession`` *is* its own connection pool: ``_get_connection`` pops an idle
socket for ``(scheme, host, port)`` (or dials a fresh one) and
``_release_connection`` parks a drained socket back — capped at
``pool_maxsize`` per host, closing the overflow so a runaway caller can't leak
file descriptors. Both paths are deliberately lock-free, leaning on
GIL-atomic ``deque.popleft`` / ``deque.append``.

These tests pin that contract:

* idle sockets are reused across sequential sends (keep-alive);
* per-host deques are isolated by ``(scheme, host, port)``;
* the per-host idle cache is bounded — overflow sockets are *closed*, not
  parked;
* ``_evict_host`` / ``clear_connections`` close every socket they drop;
* under heavy concurrent get/release the pool stays bounded and never loses a
  socket (every dialed conn is either pooled or closed — never leaked, never
  double-closed), which is the property that keeps the lock-free design safe.
"""
from __future__ import annotations

import collections
import http.server
import os
import threading
import time
from socketserver import ThreadingMixIn

import pytest

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.session import HTTPSession
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeConn:
    """Stand-in for ``http.client.HTTPConnection`` that records closes.

    ``close`` is the only behaviour the pool exercises. A shared lock-guarded
    counter lets the thread-safety test prove every dialed conn is closed at
    most once (no double-free) and accounted for (no leak).
    """

    _lock = threading.Lock()
    closed_total = 0

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls.closed_total = 0

    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        with _FakeConn._lock:
            _FakeConn.closed_total += 1


# ---------------------------------------------------------------------------
# Real server (for keep-alive reuse + leak checks)
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive so sockets are reusable

    def do_GET(self):
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class _Server(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


@pytest.fixture
def base_url():
    srv = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    HTTPSession._INSTANCES.clear()
    try:
        yield url
    finally:
        HTTPSession._INSTANCES.clear()
        srv.shutdown()


def _pooled(session: HTTPSession) -> int:
    return sum(len(q) for q in session._connections.values())


# ---------------------------------------------------------------------------
# Reuse + per-host isolation (real server)
# ---------------------------------------------------------------------------


def test_idle_socket_is_reused_across_sends(base_url):
    session = HTTPSession(base_url=base_url)
    session.send(HTTPRequest.prepare("GET", f"{base_url}/a"))

    key = next(iter(session._connections))
    first = session._connections[key][0]
    assert _pooled(session) == 1

    session.send(HTTPRequest.prepare("GET", f"{base_url}/b"))
    # The keep-alive socket is popped, reused, and parked back — same object,
    # not a freshly dialed one.
    assert session._connections[key][0] is first
    assert _pooled(session) == 1


def test_per_host_pools_are_isolated():
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url="http://example.invalid")
    a, b, c = _FakeConn(), _FakeConn(), _FakeConn()
    session._release_connection(("https", "host-a", 443), a)
    session._release_connection(("https", "host-b", 443), b)
    session._release_connection(("http", "host-a", 80), c)
    # Distinct (scheme, host, port) tuples never share a deque.
    assert set(session._connections) == {
        ("https", "host-a", 443),
        ("https", "host-b", 443),
        ("http", "host-a", 80),
    }
    assert all(len(q) == 1 for q in session._connections.values())
    HTTPSession._INSTANCES.clear()


# ---------------------------------------------------------------------------
# _get_connection / _release_connection unit behaviour
# ---------------------------------------------------------------------------


def test_get_connection_pops_cached_then_builds(monkeypatch):
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url="http://example.invalid")
    built: list[_FakeConn] = []
    monkeypatch.setattr(
        session, "_build_connection",
        lambda *a, **k: built.append(_FakeConn()) or built[-1],
    )

    cached = _FakeConn()
    session._connections[("http", "h", 80)] = collections.deque([cached])
    # First call drains the cached socket without dialing...
    assert session._get_connection("http", "h", 80, None) is cached
    assert built == []
    # ...the now-empty deque forces a fresh dial.
    fresh = session._get_connection("http", "h", 80, None)
    assert fresh is built[0]
    HTTPSession._INSTANCES.clear()


def test_release_creates_deque_for_unseen_key():
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url="http://example.invalid")
    conn = _FakeConn()
    session._release_connection(("http", "new", 80), conn)
    assert list(session._connections[("http", "new", 80)]) == [conn]
    assert conn.close_count == 0  # parked, not closed
    HTTPSession._INSTANCES.clear()


def test_release_caps_at_pool_maxsize_and_closes_overflow():
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url="http://example.invalid", pool_maxsize=4)
    assert session.pool_maxsize == 4
    key = ("http", "h", 80)
    conns = [_FakeConn() for _ in range(7)]
    for c in conns:
        session._release_connection(key, c)
    # Exactly pool_maxsize parked; the 3 overflow sockets are closed, not leaked.
    assert _pooled(session) == 4
    parked = list(session._connections[key])
    assert conns[:4] == parked
    assert all(c.close_count == 0 for c in conns[:4])
    assert all(c.close_count == 1 for c in conns[4:])
    HTTPSession._INSTANCES.clear()


def test_evict_host_closes_and_drops_only_that_host():
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url="http://example.invalid")
    target = [_FakeConn() for _ in range(3)]
    other = _FakeConn()
    for c in target:
        session._release_connection(("https", "evict.me", 443), c)
    session._release_connection(("https", "keep.me", 443), other)

    session._evict_host(URL.from_("https://evict.me/path"))
    assert ("https", "evict.me", 443) not in session._connections
    assert all(c.close_count == 1 for c in target)
    # A sibling host's idle sockets are untouched.
    assert ("https", "keep.me", 443) in session._connections
    assert other.close_count == 0
    HTTPSession._INSTANCES.clear()


def test_evict_host_on_unknown_host_is_noop():
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url="http://example.invalid")
    # No deque for this host — must not raise.
    session._evict_host(URL.from_("http://never.seen/x"))
    assert session._connections == {}
    HTTPSession._INSTANCES.clear()


def test_clear_connections_closes_everything():
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url="http://example.invalid")
    conns = [_FakeConn() for _ in range(5)]
    session._release_connection(("http", "a", 80), conns[0])
    session._release_connection(("http", "a", 80), conns[1])
    session._release_connection(("https", "b", 443), conns[2])
    session._release_connection(("https", "b", 443), conns[3])
    session._release_connection(("http", "c", 80), conns[4])

    session.clear_connections()
    assert session._connections == {}
    assert all(c.close_count == 1 for c in conns)
    HTTPSession._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Thread-safety: lock-free get/release under contention
# ---------------------------------------------------------------------------


def test_concurrent_get_release_stays_bounded_no_leak(monkeypatch):
    """Hammer the lock-free pool from many threads; prove the invariant.

    Every conn the pool hands out is eventually either parked (idle, bounded)
    or closed — never lost, never double-closed. The ``len < pool_maxsize``
    cap is a racy approximation, so the parked count may briefly overshoot by
    at most one-per-thread; it must never grow without bound.
    """
    HTTPSession._INSTANCES.clear()
    pool_maxsize = 6
    session = HTTPSession(base_url="http://example.invalid", pool_maxsize=pool_maxsize)
    key = ("http", "h", 80)

    built_lock = threading.Lock()
    built: list[_FakeConn] = []
    _FakeConn.reset()

    def fake_build(*_a, **_k):
        c = _FakeConn()
        with built_lock:
            built.append(c)
        return c

    monkeypatch.setattr(session, "_build_connection", fake_build)

    n_threads = 16
    iters = 400
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker():
        try:
            barrier.wait()
            for _ in range(iters):
                conn = session._get_connection("http", "h", 80, None)
                session._release_connection(key, conn)
        except BaseException as exc:  # noqa: BLE001 — surface any race crash
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"lock-free pool raced into an error: {errors[:3]}"

    pooled = _pooled(session)
    # Bounded: never a socket-per-iteration. Allow the documented small
    # per-thread overshoot of the racy cap.
    assert pooled <= pool_maxsize + n_threads
    # Conservation: every dialed conn is accounted for as parked-or-closed,
    # and nothing was closed twice.
    assert len(built) == pooled + _FakeConn.closed_total
    assert all(c.close_count <= 1 for c in built)

    # Whatever stayed parked is still usable.
    session.clear_connections()
    assert _pooled(session) == 0
    HTTPSession._INSTANCES.clear()


def test_concurrent_real_sends_keep_pool_bounded(base_url):
    """Many threads sharing one session keep the idle cache bounded + warm."""
    session = HTTPSession(base_url=base_url)
    n_threads = 12
    per_thread = 20
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []
    oks = []
    oks_lock = threading.Lock()

    def worker():
        try:
            barrier.wait()
            local_ok = 0
            for i in range(per_thread):
                r = session.send(HTTPRequest.prepare("GET", f"{base_url}/c?i={i}"))
                if r.status_code == 200:
                    local_ok += 1
            with oks_lock:
                oks.append(local_ok)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent sends raced: {errors[:3]}"
    assert sum(oks) == n_threads * per_thread

    pooled = _pooled(session)
    # 240 requests across one host must not leave a socket-per-request parked;
    # idle keep-alives are bounded by pool_maxsize (+ small racy overshoot).
    assert 1 <= pooled <= session.pool_maxsize + n_threads
    HTTPSession._INSTANCES.clear()


@pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd"),
    reason="fd accounting needs /proc (Linux)",
)
def test_no_fd_leak_under_concurrent_fanout(base_url):
    """A large concurrent fan-out must not grow the process FD table per request."""
    session = HTTPSession(base_url=base_url)

    # Warm the pool first so steady-state FD count is established.
    list(session.send_many(
        iter([HTTPRequest.prepare("GET", f"{base_url}/w?i={i}") for i in range(10)]),
        raise_error=False,
    ))
    time.sleep(0.05)
    before = len(os.listdir("/proc/self/fd"))

    for _ in range(5):
        got = list(session.send_many(
            iter([HTTPRequest.prepare("GET", f"{base_url}/f?i={i}") for i in range(40)]),
            raise_error=False,
        ))
        assert len(got) == 40
    time.sleep(0.05)
    after = len(os.listdir("/proc/self/fd"))

    # 200 requests over a bounded pool may add a handful of pooled sockets but
    # nothing proportional to the request count.
    assert after - before <= session.pool_maxsize + 8, (
        f"fd table grew {before}->{after}; suspected socket leak"
    )
    HTTPSession._INSTANCES.clear()
