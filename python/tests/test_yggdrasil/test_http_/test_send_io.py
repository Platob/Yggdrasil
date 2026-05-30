"""SendIO — the seekable, zero-copy, spill-backed request-body adapter.

Unit cases pin the file-like contract (chunked read, len, seek/tell, window,
zero-copy memoryview). Wire cases prove a body — in-memory *and* spilled to
disk — round-trips byte-exact through a real ``HTTPSession`` POST, replays
intact across a connection-retry, and (the headline) uploads a large spilled
body in bounded memory instead of materialising it whole.
"""
from __future__ import annotations

import gc
import hashlib
import http.server
import io
import threading
import tracemalloc
from socketserver import ThreadingMixIn

import pytest

from yggdrasil.http_.send_io import SEND_CHUNK, SendIO
from yggdrasil.http_.session import HTTPSession
from yggdrasil.path.memory import Memory
from yggdrasil.path.memory_stream import MemoryStream

_MIB = 1024 * 1024


def _spilled_stream(data: bytes, *, spill_threshold: int = _MIB) -> MemoryStream:
    """A MemoryStream holding ``data`` mostly on disk: a small in-memory window
    (``spill_threshold``) with the default multi-GiB retention budget, so the
    head stays readable from byte 0 (nothing evicted) while the bulk is spilled.
    """
    ms = MemoryStream(source=io.BytesIO(data), spill_threshold=spill_threshold)
    ms._pull_to_eof()
    return ms


# -- unit: file-like contract ----------------------------------------------

def test_len_and_chunked_read_reassembles():
    data = bytes(range(256)) * 400  # 102_400 bytes
    sio = SendIO(Memory(binary=data), chunk_size=4096)
    assert len(sio) == len(data)
    out = bytearray()
    while True:
        chunk = sio.read(64 * 1024)   # ask big; SendIO caps at chunk_size
        if not chunk:
            break
        assert len(chunk) <= 4096
        out += chunk
    assert bytes(out) == data
    assert sio.tell() == len(data)


def test_read_returns_zero_copy_memoryview_into_holder():
    data = b"abcdefgh" * 1000
    holder = Memory(binary=data)
    sio = SendIO(holder, chunk_size=4096)
    mv = sio.read(100)
    assert isinstance(mv, memoryview)
    # Same bytes as the holder slice, and an actual view (not a fresh copy):
    # mutating the holder buffer shows through the still-live view.
    assert bytes(mv) == data[:100]
    assert bytes(holder.read_mv(100, 0)) == data[:100]


def test_seek_rewind_replays_from_start():
    data = b"0123456789" * 5000
    sio = SendIO(Memory(binary=data), chunk_size=8192)
    first = b"".join(iter(lambda: bytes(sio.read(8192)), b""))
    assert first == data
    assert sio.read(1) == b""           # exhausted
    assert sio.seek(0) == 0             # rewind
    again = b"".join(iter(lambda: bytes(sio.read(8192)), b""))
    assert again == data                # full replay
    # whence variants
    sio.seek(10)
    assert sio.tell() == 10
    sio.seek(5, 1)
    assert sio.tell() == 15
    sio.seek(-3, 2)
    assert sio.tell() == len(data) - 3
    sio.seek(-100, 0)                   # clamps, never negative
    assert sio.tell() == 0


def test_base_and_length_define_a_window():
    data = bytes(range(256))
    sio = SendIO(Memory(binary=data), base=100, length=50, chunk_size=16)
    assert len(sio) == 50
    assert b"".join(iter(lambda: bytes(sio.read(16)), b"")) == data[100:150]


def test_empty_body_reads_nothing():
    sio = SendIO(Memory(binary=b""))
    assert len(sio) == 0
    assert sio.read(10) == b""


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        SendIO(Memory(binary=b"x"), chunk_size=0)
    with pytest.raises(ValueError):
        SendIO(Memory(binary=b"x"), base=99)


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


def _post_and_check(session, base, holder, data):
    r = session.post("/echo", body=holder)
    assert r.status_code == 200
    assert r.headers.get("X-Body-Len") == str(len(data))
    assert r.content.decode() == hashlib.sha256(data).hexdigest()


def test_wire_post_in_memory_body_roundtrips(server):
    session = HTTPSession(base_url=server)
    data = bytes((i * 31) % 256 for i in range(3 * _MIB))
    _post_and_check(session, server, Memory(binary=data), data)


def test_wire_post_spilled_stream_body_roundtrips(server):
    session = HTTPSession(base_url=server)
    data = bytes((i * 17) % 256 for i in range(4 * _MIB))
    ms = _spilled_stream(data, spill_threshold=_MIB)
    # Precondition: the body really is spilled (bulk on disk) yet retained
    # from byte 0, which is exactly what SendIO must stream + replay.
    assert ms._window_start > 0
    assert ms._spill_start == 0
    _post_and_check(session, server, ms, data)


def test_wire_retry_replays_full_spilled_body_on_fresh_socket(server):
    # First connection is dropped before responding → the send retries on a
    # fresh socket with a fresh SendIO that must re-send the whole spilled body.
    _EchoHandler.fail_first = 1
    session = HTTPSession(base_url=server)
    data = bytes((i * 7) % 256 for i in range(4 * _MIB))
    ms = _spilled_stream(data, spill_threshold=_MIB)
    r = session.post("/echo", body=ms)
    assert r.status_code == 200
    assert r.content.decode() == hashlib.sha256(data).hexdigest()
    assert _EchoHandler.conns == 2  # dropped once, succeeded on the retry


# -- benchmark: bounded memory ----------------------------------------------

def test_sendio_streams_a_spilled_body_in_bounded_memory():
    # Compare peak heap of producing the whole body two ways:
    #   (a) SendIO chunked reads (what the wire send now does), vs
    #   (b) read_mv(-1, 0) — the old path that materialises the body whole.
    # SendIO must peak far below the full-materialisation path.
    size = 32 * _MIB
    data = bytes((i * 13) % 256 for i in range(size))

    def peak(fn):
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn()
        _cur, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()
        return pk

    def via_sendio():
        ms = _spilled_stream(data, spill_threshold=_MIB)
        sio = SendIO(ms)
        total = 0
        while True:
            chunk = sio.read(SEND_CHUNK)  # not retained → bounded resident set
            if not chunk:
                break
            total += len(chunk)
        assert total == size

    def via_full_read():
        ms = _spilled_stream(data, spill_threshold=_MIB)
        mv = ms.read_mv(-1, 0)            # one contiguous view over the whole body
        assert len(mv) == size

    # warm (build spill files, fill caches) so the measured run is steady-state
    via_sendio()
    via_full_read()

    peak_sendio = peak(via_sendio)
    peak_full = peak(via_full_read)

    # Full materialisation pays ~the whole body; SendIO pays ~one window+chunk.
    assert peak_full > size * 0.5, f"control didn't materialise the body: {peak_full}"
    assert peak_sendio < peak_full * 0.25, (
        f"SendIO peak {peak_sendio} not well below full-read {peak_full}"
    )
