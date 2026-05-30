"""Thread-safety of the shared :class:`HTTPSession` (singleton + job pool).

``HTTPSession`` is a process-wide singleton keyed by its construction knobs, so
many call sites end up sharing one instance — and therefore one lock-free
connection pool and one lazily-built :class:`JobPoolExecutor`. That sharing is
the whole point (pool reuse), but it only pays off if it's safe under
concurrency:

* constructing the same-keyed session from many threads must converge on a
  single instance (no torn half-initialised object);
* the lazy ``job_pool`` build must hand every racing caller the *same* pool and
  not leak the losers' throwaway pools;
* fanning out ``send_many`` from several threads against one shared session must
  complete every request correctly while keeping the idle pool bounded.
"""
from __future__ import annotations

import http.server
import threading
import time
from socketserver import ThreadingMixIn

import pytest

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.session import HTTPSession


# ---------------------------------------------------------------------------
# Real keep-alive server
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        from urllib.parse import urlsplit, parse_qs

        qs = parse_qs(urlsplit(self.path).query)
        i = int(qs.get("i", ["0"])[0])
        time.sleep(0.01)
        body = f'{{"i": {i}}}'.encode()
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


def _run_threads(fn, n: int):
    """Start *n* threads on ``fn(i)`` behind a barrier; return (results, errors)."""
    barrier = threading.Barrier(n)
    results: list = [None] * n
    errors: list[BaseException] = []
    lock = threading.Lock()

    def wrap(i: int):
        try:
            barrier.wait()
            results[i] = fn(i)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=wrap, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


# ---------------------------------------------------------------------------
# Singleton construction is race-safe
# ---------------------------------------------------------------------------


def test_concurrent_construction_converges_on_one_instance(base_url):
    HTTPSession._INSTANCES.clear()
    instances, errors = _run_threads(
        lambda _i: HTTPSession(base_url=base_url), 24
    )
    assert not errors, f"construction raced: {errors[:3]}"
    ids = {id(s) for s in instances}
    # Same construction key → exactly one shared, fully-initialised instance.
    assert len(ids) == 1
    only = instances[0]
    assert only._initialized is True
    assert only._lock is not None
    assert only.pool_maxsize >= 1
    HTTPSession._INSTANCES.clear()


def test_distinct_keys_make_distinct_instances(base_url):
    HTTPSession._INSTANCES.clear()
    a = HTTPSession(base_url=base_url, pool_maxsize=4)
    b = HTTPSession(base_url=base_url, pool_maxsize=7)
    # pool_maxsize is part of the singleton key, so these don't collapse.
    assert a is not b
    assert a.pool_maxsize == 4
    assert b.pool_maxsize == 7
    HTTPSession._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Lazy job_pool build is race-safe
# ---------------------------------------------------------------------------


def test_concurrent_job_pool_access_returns_single_pool(base_url):
    session = HTTPSession(base_url=base_url)
    session._job_pool = None  # force the lazy-build path for every thread

    pools, errors = _run_threads(lambda _i: session.job_pool, 24)
    assert not errors, f"job_pool build raced: {errors[:3]}"
    pool_ids = {id(p) for p in pools}
    # Every racing caller observes the one winning pool; losers' throwaway
    # pools are shut down inside the property, not handed back.
    assert len(pool_ids) == 1
    assert session._job_pool is pools[0]
    HTTPSession._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Concurrent send_many across threads on one shared session
# ---------------------------------------------------------------------------


def test_concurrent_send_many_on_shared_session(base_url):
    session = HTTPSession(base_url=base_url)
    n_threads = 10
    per_call = 15

    def fan(_i: int):
        reqs = [
            HTTPRequest.prepare("GET", f"{base_url}/x?i={j}")
            for j in range(per_call)
        ]
        got = list(session.send_many(iter(reqs), raise_error=False))
        assert len(got) == per_call
        assert all(r.status_code == 200 for r in got)
        return sorted(r.json()["i"] for r in got)

    results, errors = _run_threads(fan, n_threads)
    assert not errors, f"shared-session fan-out raced: {errors[:3]}"
    # Every thread got a complete, correct result set.
    assert all(r == list(range(per_call)) for r in results)

    # The shared idle pool stayed bounded despite N concurrent fan-outs.
    pooled = sum(len(q) for q in session._connections.values())
    assert pooled <= session.pool_maxsize + n_threads
    HTTPSession._INSTANCES.clear()


def test_mixed_send_and_send_many_share_pool_safely(base_url):
    session = HTTPSession(base_url=base_url)

    def worker(i: int):
        if i % 2 == 0:
            r = session.send(HTTPRequest.prepare("GET", f"{base_url}/s?i={i}"))
            assert r.status_code == 200
            return r.json()["i"]
        reqs = [HTTPRequest.prepare("GET", f"{base_url}/m?i={j}") for j in range(8)]
        got = list(session.send_many(iter(reqs), raise_error=False))
        assert len(got) == 8
        return -1

    _results, errors = _run_threads(worker, 16)
    assert not errors, f"mixed send/send_many raced: {errors[:3]}"

    pooled = sum(len(q) for q in session._connections.values())
    assert pooled <= session.pool_maxsize + 16
    HTTPSession._INSTANCES.clear()
