"""Concurrency benchmarks for ``HTTPSession.send_many``.

The fast path fans out across the session job pool
(``JobPoolExecutor(max_workers=pool_maxsize)``). ``pool_maxsize`` is clamped to
8, so achieved concurrency is ``min(max_in_flight, 8, n)``. These tests measure
the *observed* concurrency (the server records its peak simultaneous handlers,
under a lock) and the wall-time speedup — not just "faster than serial" — and
check the connection pool stays bounded (no socket leak) under fan-out.
"""
from __future__ import annotations

import http.server
import threading
import time
from socketserver import ThreadingMixIn

import pytest

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.session import HTTPSession

_LATENCY = 0.05  # per-request server delay so overlap is measurable
_POOL = 8        # HTTPSession clamps pool_maxsize (and thus pool workers) to 8


class _ConcHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    _lock = threading.Lock()
    active = 0
    peak = 0

    @classmethod
    def reset(cls):
        with cls._lock:
            cls.active = 0
            cls.peak = 0

    def do_GET(self):
        cls = type(self)
        with cls._lock:
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
        try:
            time.sleep(_LATENCY)
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with cls._lock:
                cls.active -= 1

    def log_message(self, *a):
        pass


class _Server(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


@pytest.fixture
def base_url():
    srv = _Server(("127.0.0.1", 0), _ConcHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    HTTPSession._INSTANCES.clear()
    _ConcHandler.reset()
    try:
        yield url
    finally:
        HTTPSession._INSTANCES.clear()
        srv.shutdown()


def _reqs(base: str, n: int):
    return [HTTPRequest.prepare("GET", f"{base}/c?i={i}") for i in range(n)]


def _drain(session, base, n, **kw):
    got = list(session.send_many(iter(_reqs(base, n)), raise_error=False, **kw))
    assert len(got) == n
    assert all(r.status_code == 200 for r in got)
    return got


# -- observed concurrency ---------------------------------------------------

def test_fanout_saturates_the_pool(base_url):
    session = HTTPSession(base_url=base_url)
    _ConcHandler.reset()
    _drain(session, base_url, 40)
    # Default fan-out runs up to pool_maxsize requests at once — never more,
    # and it genuinely reaches the ceiling rather than dribbling.
    assert _ConcHandler.peak <= _POOL
    assert _ConcHandler.peak >= _POOL - 2


def test_max_in_flight_throttles_concurrency(base_url):
    session = HTTPSession(base_url=base_url)
    _ConcHandler.reset()
    _drain(session, base_url, 24, max_in_flight=3)
    # The window caps simultaneous requests below the pool size.
    assert _ConcHandler.peak <= 3
    assert _ConcHandler.peak >= 2


# -- speedup scales with concurrency ---------------------------------------

def test_speedup_scales_with_concurrency(base_url):
    session = HTTPSession(base_url=base_url)
    n = 24

    def timed(max_in_flight):
        _ConcHandler.reset()
        t0 = time.perf_counter()
        _drain(session, base_url, n, max_in_flight=max_in_flight)
        return time.perf_counter() - t0

    # Warm the pool/sockets so the first measured run isn't penalised.
    timed(_POOL)

    slow = timed(2)   # ~ceil(24/2)=12 waves of latency
    fast = timed(8)   # ~ceil(24/8)=3  waves of latency
    # 8-way should be well under half the 2-way wall time (4x fewer waves);
    # generous bound to stay robust on a loaded box.
    assert fast < slow * 0.6, f"8-way {fast:.3f}s not far below 2-way {slow:.3f}s"

    # And the absolute floor: 8-way can't beat the 3-wave latency lower bound.
    assert fast >= 3 * _LATENCY * 0.5


# -- connection pool stays bounded -----------------------------------------

def test_pool_bounded_and_warm_after_fanout(base_url):
    session = HTTPSession(base_url=base_url)
    _drain(session, base_url, 40)
    pooled = sum(len(q) for q in session._connections.values())
    # A 40-request fan-out must not leave a socket per request parked in the
    # pool — idle keep-alives are capped at pool_maxsize.
    assert pooled <= session.pool_maxsize
    assert pooled >= 1  # ...but warm sockets are retained for reuse

    # The warm pool serves a follow-up batch correctly.
    _drain(session, base_url, 5)
