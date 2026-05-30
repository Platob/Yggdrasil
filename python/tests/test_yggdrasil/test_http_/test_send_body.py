"""Zero-copy, bounded request-body streaming via ``Holder.iter_mv``.

The wire send no longer materialises the body with ``read_mv(-1, 0)``; it hands
``http.client`` the holder's ``iter_mv`` — a generator of zero-copy
``memoryview`` chunks. These tests pin the IO enrichment itself (chunking,
reassembly, zero-copy, window, replayable iteration) and the wire behaviour it
enables: a body round-trips byte-exact through a real ``HTTPSession`` POST —
both an in-memory :class:`Memory` and a real on-disk file — replays intact
across a connection retry, and uploads a large file in bounded memory.
"""
from __future__ import annotations

import gc
import hashlib
import http.server
import threading
import tracemalloc
from socketserver import ThreadingMixIn

import pytest

from yggdrasil.http_.session import HTTPSession
from yggdrasil.path.local_path import LocalPath
from yggdrasil.path.memory import Memory

_MIB = 1024 * 1024


def _file_io(path, data):
    """A real on-disk file opened as a readable IO (``.size`` + ``read_mv``)."""
    path.write_bytes(data)
    return LocalPath.from_(str(path)).open(mode="rb")


# -- unit: Holder.iter_mv ---------------------------------------------------

def test_iter_mv_chunks_reassemble():
    data = bytes(range(256)) * 400  # 102_400 bytes
    chunks = list(Memory(binary=data).iter_mv(4096))
    assert all(isinstance(c, memoryview) for c in chunks)
    assert all(len(c) <= 4096 for c in chunks)
    assert b"".join(bytes(c) for c in chunks) == data


def test_iter_mv_yields_zero_copy_views():
    data = b"abcdefgh" * 1000
    holder = Memory(binary=data)
    first = next(holder.iter_mv(100))
    assert isinstance(first, memoryview)        # not a bytes copy
    assert bytes(first) == data[:100]


def test_iter_mv_is_replayable():
    # Positional reads don't move a cursor, so the same holder streams again —
    # exactly what a connection retry relies on.
    holder = Memory(binary=b"0123456789" * 5000)
    once = b"".join(bytes(c) for c in holder.iter_mv(8192))
    twice = b"".join(bytes(c) for c in holder.iter_mv(8192))
    assert once == twice == holder.read_bytes()


def test_iter_mv_window_with_start_and_length():
    data = bytes(range(256))
    chunks = list(Memory(binary=data).iter_mv(16, start=100, length=50))
    assert b"".join(bytes(c) for c in chunks) == data[100:150]


def test_iter_mv_empty_holder_yields_nothing():
    assert list(Memory(binary=b"").iter_mv()) == []


def test_iter_mv_rejects_bad_chunk_size():
    with pytest.raises(ValueError):
        list(Memory(binary=b"x").iter_mv(0))


def test_iter_mv_over_a_real_file(tmp_path):
    data = bytes((i * 5) % 256 for i in range(200_000))
    io_ = _file_io(tmp_path / "b.bin", data)
    assert b"".join(bytes(c) for c in io_.iter_mv(8192)) == data


# -- wire: round-trip through a real session --------------------------------

class _EchoHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fail_first = 0
    conns = 0

    def setup(self):
        super().setup()
        type(self).conns += 1
        self._kill = type(self).conns <= type(self).fail_first

    def do_POST(self):
        if self._kill:
            self.close_connection = True   # drop → client retries on a fresh socket
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        digest = hashlib.sha256(body).hexdigest().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("X-Body-Len", str(len(body)))
        self.send_header("Content-Length", str(len(digest)))
        self.end_headers()
        self.wfile.write(digest)

    def log_message(self, *a):
        pass


class _Server(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    _EchoHandler.fail_first = 0
    _EchoHandler.conns = 0
    HTTPSession._INSTANCES.clear()
    try:
        yield base
    finally:
        HTTPSession._INSTANCES.clear()
        srv.shutdown()


def _post_and_check(session, holder, data):
    r = session.post("/echo", body=holder)
    assert r.status_code == 200
    assert r.headers.get("X-Body-Len") == str(len(data))
    assert r.content.decode() == hashlib.sha256(data).hexdigest()


def test_wire_post_in_memory_body_roundtrips(server):
    session = HTTPSession(base_url=server)
    data = bytes((i * 31) % 256 for i in range(3 * _MIB))
    _post_and_check(session, Memory(binary=data), data)


def test_wire_post_file_body_roundtrips(server, tmp_path):
    session = HTTPSession(base_url=server)
    data = bytes((i * 17) % 256 for i in range(4 * _MIB))
    _post_and_check(session, _file_io(tmp_path / "body.bin", data), data)


def test_wire_retry_replays_full_body_on_fresh_socket(server, tmp_path):
    # First connection is dropped before responding → the send retries on a
    # fresh socket with a fresh iter_mv that must re-send the whole file.
    _EchoHandler.fail_first = 1
    session = HTTPSession(base_url=server)
    data = bytes((i * 7) % 256 for i in range(4 * _MIB))
    r = session.post("/echo", body=_file_io(tmp_path / "body.bin", data))
    assert r.status_code == 200
    assert r.content.decode() == hashlib.sha256(data).hexdigest()
    assert _EchoHandler.conns == 2  # dropped once, succeeded on the retry


# -- benchmark: bounded memory ----------------------------------------------

def test_iter_mv_streams_a_file_in_bounded_memory(tmp_path):
    # Produce the whole body two ways and compare peak heap:
    #   (a) iter_mv chunked views (what the wire send now does), vs
    #   (b) read_mv(-1, 0) — the old path that materialises the file whole.
    size = 32 * _MIB
    data = bytes((i * 13) % 256 for i in range(size))
    path = tmp_path / "big.bin"
    path.write_bytes(data)

    def peak(fn):
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn()
        _cur, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()
        return pk

    def via_iter_mv():
        total = 0
        for chunk in LocalPath.from_(str(path)).open(mode="rb").iter_mv():
            total += len(chunk)  # not retained → bounded resident set
        assert total == size

    def via_full_read():
        mv = LocalPath.from_(str(path)).open(mode="rb").read_mv(-1, 0)
        assert len(mv) == size

    via_iter_mv()      # warm
    via_full_read()

    peak_iter = peak(via_iter_mv)
    peak_full = peak(via_full_read)

    assert peak_full > size * 0.5, f"control didn't materialise the file: {peak_full}"
    assert peak_iter < peak_full * 0.25, (
        f"iter_mv peak {peak_iter} not well below full-read {peak_full}"
    )
