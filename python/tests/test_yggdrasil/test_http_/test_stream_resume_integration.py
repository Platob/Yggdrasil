"""End-to-end resilience: drive :class:`HTTPSession` against a real local
HTTP server that drops the connection mid-body, and prove the download both
completes byte-for-byte AND resumes from the exact offset it stopped at.

Unlike ``test_stream_resume.py`` (which fakes the socket), this exercises the
full wire path — :meth:`HTTPResponse.from_wire` wraps the live socket in an
:class:`HTTPStream`, so a real ``RST`` mid-stream surfaces as a transient
``ConnectionResetError`` that the stream recovers from by re-dialing the origin
with a ``Range: bytes=<received>-`` header.

The body is a few MiB so it spans several 1 MiB pull chunks: the client
consumes 2 MiB, the server forces an abortive close (``SO_LINGER`` 0) mid-third
chunk, and the resume re-requests ``bytes=2097152-`` as ``206``. The server
records the Range offset so the test can assert the resume happened *at the
received position*, not via a full re-fetch.
"""
from __future__ import annotations

import http.server
import socket
import struct
import threading
import time

import pytest

from yggdrasil.http_.session import HTTPSession

_MIB = 1024 * 1024
# > pull_chunk (1 MiB) so the transfer spans multiple reads; the cut lands
# after two whole chunks, giving a deterministic 2 MiB resume offset.
_PAYLOAD = bytes((i * 7) % 256 for i in range(3 * _MIB))
_CUT_AT = 2 * _MIB


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive + Range support
    cut_done = False               # cut the body exactly once per test
    range_starts: list[int] = []   # offsets the client resumed from

    def do_GET(self):
        rng = self.headers.get("Range")
        start = 0
        if rng and rng.startswith("bytes="):
            start = int(rng[len("bytes="):].split("-", 1)[0] or 0)
            _Handler.range_starts.append(start)

        if self.path == "/resumable" or rng or _Handler.cut_done:
            self._serve(start)
        else:
            _Handler.cut_done = True
            self._cut()

    def _serve(self, start):
        body = _PAYLOAD[start:]
        if start:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(_PAYLOAD) - 1}/{len(_PAYLOAD)}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cut(self):
        # Promise the whole body, deliver only the head, then yank the socket
        # with an abortive close so the client's body read raises
        # ConnectionResetError after it has consumed the head.
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(_PAYLOAD)))
        self.end_headers()
        self.wfile.write(_PAYLOAD[:_CUT_AT])
        self.wfile.flush()
        time.sleep(0.3)  # let the client drain the two whole chunks first
        # Drop the makefile refs so the fd really closes; SO_LINGER 0 makes
        # close() send an RST (not a graceful FIN → that would be a
        # non-transient IncompleteRead).
        self.close_connection = True
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        for f in (self.rfile, self.wfile):
            try:
                f.close()
            except Exception:
                pass
        self.connection.close()

    def log_message(self, fmt, *args):
        pass


class _QuietServer(http.server.ThreadingHTTPServer):
    # The abortive close leaves BaseHTTPRequestHandler.finish flushing a closed
    # socket — expected here, so don't spew the traceback to stderr.
    def handle_error(self, request, client_address):
        pass


@pytest.fixture(scope="module")
def server():
    srv = _QuietServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _reset():
    _Handler.cut_done = False
    _Handler.range_starts = []
    HTTPSession._INSTANCES.clear()
    yield
    HTTPSession._INSTANCES.clear()


def test_range_download_is_byte_exact(server):
    # Baseline: the server honors Range, so a plain GET and an explicit slice
    # both come back correct — the precondition the resume relies on.
    session = HTTPSession(base_url=server)
    assert session.get("/resumable").content == _PAYLOAD
    assert session.get("/resumable", headers={"Range": "bytes=1000-"}).content == _PAYLOAD[1000:]


def test_midstream_rst_resumes_at_received_offset(server):
    # The server RSTs mid-stream after _CUT_AT bytes; the stream must reconnect
    # with a Range from the byte it stopped at and deliver the whole body.
    session = HTTPSession(base_url=server)
    r = session.get("/cut-once")
    assert r.status_code == 200          # status is from the first (cut) response
    assert r.content == _PAYLOAD         # ...but the body is whole, no gap/dup
    # Recovery was a real range-resume from where it stopped, not a re-fetch
    # from zero — and exactly one resume was needed.
    assert _Handler.range_starts == [_CUT_AT]
