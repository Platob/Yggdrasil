"""Wire tests for opt-in body streaming (``SendConfig.stream`` /
``session.get(..., stream=True)``).

With ``stream=True`` the wire send leaves the response body
un-preloaded (``preload_content=False``): the buffer keeps the live
socket as its source so a consumer reading via ``.stream()`` pulls the
body incrementally instead of materialising the whole payload up front.
These tests pin the behaviour that matters:

* the streamed bytes equal the full body (correctness),
* the default (non-stream) path is unchanged,
* the connection is returned to the pool once the body is drained,
* and the streamed peak heap is below the preload double-copy peak.
"""
from __future__ import annotations

import gc
import http.server
import socket
import threading
import tracemalloc
from socketserver import ThreadingMixIn

import pytest

from yggdrasil.http_.session import HTTPSession
from yggdrasil.http_.send_config import SendConfig


_BIG = b"D" * (4 * 1024 * 1024)  # 4 MiB — big enough to see the copy gap


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_GET(self):
        body = _BIG
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/octet-stream\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        self.wfile.write(head + body)

    def log_message(self, *a):
        pass


class _Server(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


@pytest.fixture
def session():
    """Fresh server + isolated session per test.

    The singleton cache is cleared on entry *and* exit and the session's
    pooled keep-alive sockets are dropped on teardown, so a streamed
    (``preload_content=False``) response that leaves a live socket in one
    test can never be reused by the next — full-suite ordering stays
    hermetic.
    """
    srv = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    HTTPSession._INSTANCES.clear()
    sess = HTTPSession(base_url=base)
    try:
        yield sess
    finally:
        sess.clear_connections()
        HTTPSession._INSTANCES.clear()
        srv.shutdown()


# --- SendConfig field contract --------------------------------------------


def test_send_config_stream_field():
    assert SendConfig().stream is False
    assert SendConfig(stream=True).stream is True
    assert SendConfig(stream=True) == SendConfig(stream=True)
    assert SendConfig(stream=True) != SendConfig()
    assert hash(SendConfig(stream=True)) != hash(SendConfig())
    assert "stream=True" in repr(SendConfig(stream=True))


def test_send_config_stream_pickle_roundtrip():
    import pickle
    assert pickle.loads(pickle.dumps(SendConfig(stream=True))).stream is True


def test_send_config_stream_from_options():
    assert SendConfig.from_(None, stream=True).stream is True
    assert SendConfig.from_(SendConfig(), stream=True).stream is True


# --- wire behaviour --------------------------------------------------------


def test_stream_get_body_matches_full(session):
    r = session.get("/big", stream=True)
    chunks = list(r.stream(64 * 1024))
    assert b"".join(chunks) == _BIG


def test_stream_content_accessor_still_works(session):
    r = session.get("/big", stream=True)
    assert r.content == _BIG  # .content drains the un-preloaded body


def test_non_stream_path_unchanged(session):
    r = session.get("/big")
    assert r.content == _BIG
    assert len(r.content) == len(_BIG)


def test_stream_releases_connection(session):
    r = session.get("/big", stream=True)
    for _ in r.stream(64 * 1024):
        pass
    pooled = sum(len(q) for q in session._connections.values())
    assert pooled >= 1, "socket should return to the pool after the body drains"

    # And the warm socket is reusable for a follow-up request.
    r2 = session.get("/big", stream=True)
    assert b"".join(r2.stream(64 * 1024)) == _BIG


def test_stream_peak_below_preload(session):

    def stream_consume():
        r = session.get("/big", stream=True)
        total = 0
        for c in r.stream(64 * 1024):
            total += len(c)
        return total

    def preload_content():
        return len(session.get("/big").content)

    # warm
    stream_consume()
    preload_content()

    def peak(fn):
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn()
        _cur, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()
        return pk

    peak_stream = peak(stream_consume)
    peak_preload = peak(preload_content)
    # Streaming without retaining chunks peaks at ~1 body; preload .content
    # holds the buffer plus a full-body copy (~2 bodies). Allow margin.
    assert peak_stream < peak_preload * 0.8, (
        f"stream peak {peak_stream} not below preload {peak_preload}"
    )
