"""Tests for the concurrent ``send_many`` fast path.

``_send_many`` routes uncached batches through ``_send_many_fast``, which
fans out across the session job pool (blocking socket I/O releases the
GIL, so the wait overlaps). These tests pin the semantics that change
matters for:

* completeness — every request yields a response;
* ``ordered=False`` (the default) yields in completion order, while
  ``ordered=True`` preserves submission order;
* a single request runs inline (no pool overhead, still correct);
* the fan-out actually overlaps latency (faster than a serial loop);
* an exception from one request propagates out of the generator.
"""
from __future__ import annotations

import http.server
import json
import socket
import threading
import time
from socketserver import ThreadingMixIn

import pytest

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.session import HTTPSession


_LATENCY_S = 0.05  # per-request server delay — big enough to see overlap


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_GET(self):
        from urllib.parse import urlsplit, parse_qs

        path = urlsplit(self.path)
        qs = parse_qs(path.query)
        i = int(qs.get("i", ["0"])[0])
        if path.path == "/slow":
            time.sleep(_LATENCY_S)
        body = json.dumps({"i": i}).encode()
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        self.wfile.write(head + body)

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


def _reqs(base_url: str, n: int, path: str = "/json") -> list[HTTPRequest]:
    return [HTTPRequest.prepare("GET", f"{base_url}{path}?i={i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Completeness + ordering
# ---------------------------------------------------------------------------


def test_all_responses_yielded(base_url):
    session = HTTPSession(base_url=base_url)
    got = list(session.send_many(iter(_reqs(base_url, 25)), raise_error=False))
    assert len(got) == 25
    assert all(r.status_code == 200 for r in got)
    assert sorted(r.json()["i"] for r in got) == list(range(25))


def test_unordered_default_returns_all(base_url):
    session = HTTPSession(base_url=base_url)
    # /slow with a per-request id staggers completion, so completion order
    # diverges from submission order — but every id must still be present.
    reqs = _reqs(base_url, 12, path="/slow")
    got = list(session.send_many(iter(reqs), raise_error=False))
    assert sorted(r.json()["i"] for r in got) == list(range(12))


def test_ordered_preserves_submission_order(base_url):
    session = HTTPSession(base_url=base_url)
    reqs = _reqs(base_url, 12, path="/slow")
    got = list(session.send_many(iter(reqs), raise_error=False, ordered=True))
    assert [r.json()["i"] for r in got] == list(range(12))


def test_single_request_inline_path(base_url):
    session = HTTPSession(base_url=base_url)
    got = list(session.send_many(iter(_reqs(base_url, 1)), raise_error=False))
    assert len(got) == 1
    assert got[0].json()["i"] == 0


def test_empty_batch_yields_nothing(base_url):
    session = HTTPSession(base_url=base_url)
    assert list(session.send_many(iter([]), raise_error=False)) == []


# ---------------------------------------------------------------------------
# Concurrency actually overlaps latency
# ---------------------------------------------------------------------------


def test_fanout_faster_than_serial_under_latency(base_url):
    session = HTTPSession(base_url=base_url)
    n = 16
    reqs = _reqs(base_url, n, path="/slow")

    t0 = time.perf_counter()
    got = list(session.send_many(iter(reqs), raise_error=False))
    fanout = time.perf_counter() - t0
    assert len(got) == n

    serial_lower_bound = n * _LATENCY_S
    # With pool_maxsize workers overlapping, the wall time must be well
    # under a strict serial run (n * latency). Generous bound to stay
    # robust on a loaded CI box.
    assert fanout < serial_lower_bound * 0.7, (
        f"fan-out {fanout:.3f}s not below serial bound {serial_lower_bound:.3f}s"
    )


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


def test_connection_error_propagates(base_url):
    session = HTTPSession(base_url=base_url)
    # One request points at a dead port — the wire send raises, and that
    # must surface out of the generator rather than be silently dropped.
    good = _reqs(base_url, 3)
    bad = HTTPRequest.prepare("GET", "http://127.0.0.1:1/nope")
    reqs = good + [bad]
    with pytest.raises(Exception):
        list(session.send_many(iter(reqs), raise_error=False))
