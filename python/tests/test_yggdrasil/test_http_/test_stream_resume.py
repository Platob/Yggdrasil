"""Mid-stream resilience for :class:`HTTPStream`: a socket that times out or
is cut mid-download must reconnect with a ``Range`` header *at the exact byte
it stopped at* and continue, so the reassembled body is byte-for-byte complete
with no gap and no duplication.

``test_stream.py`` only inspects the ``Range`` header a single reconnect emits.
These cases drive the whole loop — read until a transient failure fires,
reconnect, keep pulling — and assert on the *delivered bytes*, which is the
property that actually matters to a caller.

The harness models a flaky origin: an original socket body plus a fake
connection the stream re-dials on failure. ``cuts`` scripts how many bytes each
successive connection serves before raising — ``cuts[0]`` is the original
source, ``cuts[1:]`` the reconnects, in order. ``None`` serves cleanly to EOF.
"""
from __future__ import annotations

import socket
import ssl

import pytest

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.stream import HTTPStream

_URL = "https://host.example/v/large.bin"
# Small chunks so a single body spans many reads — a cut lands mid-stream, the
# realistic case, rather than on a clean chunk boundary.
_CHUNK = 16


class _FlakyReader:
    """A socket-body ``.read(n)`` over ``payload[start:end)``. Serves in
    ``_CHUNK`` slices and, once, raises ``exc`` after ``fail_after`` bytes —
    a socket timeout or a cut connection mid-download. ``fail_after=None``
    reads cleanly to EOF."""

    def __init__(self, payload, start, end, fail_after, exc):
        self._payload = payload
        self._pos = start
        self._end = len(payload) if end is None else end
        self._served = 0
        self._fail_after = fail_after
        self._exc = exc
        self._raised = False

    def read(self, n=-1):
        if (self._fail_after is not None and not self._raised
                and self._served >= self._fail_after):
            self._raised = True
            raise self._exc
        if self._pos >= self._end:
            return b""
        take = min(_CHUNK, self._end - self._pos)
        if self._fail_after is not None and not self._raised:
            take = min(take, self._fail_after - self._served)
        chunk = self._payload[self._pos:self._pos + take]
        self._pos += take
        self._served += take
        return chunk


class _FakeRaw:
    def __init__(self, reader, status=206, will_close=False):
        self.status = status
        self.will_close = will_close
        self._reader = reader

    def read(self, n=-1):
        return self._reader.read(n)


class _FakeServer:
    """Holds the full object and hands out connection bodies, recording the
    absolute byte offset each reconnect asked to resume from."""

    def __init__(self, payload, cuts, *,
                 exc_factory=lambda: socket.timeout("timed out"), status=206):
        self._payload = payload
        self._cuts = list(cuts)
        self._exc_factory = exc_factory
        self._status = status
        self._i = 0
        self.range_starts: list[int] = []

    def _reader(self, start, end=None):
        cut = self._cuts[self._i] if self._i < len(self._cuts) else None
        self._i += 1
        return _FlakyReader(self._payload, start, end, cut, self._exc_factory())

    def source(self):
        """The original (pre-:class:`HTTPStream`) body — consumes ``cuts[0]``."""
        return self._reader(0)


class _FakeConn:
    def __init__(self, server):
        self._server = server
        self.sock = None
        self._raw = None

    def connect(self):
        self.sock = object()

    def request(self, method, path, body=None, headers=None):
        rng = (headers or {}).get("Range", "")
        start, end = 0, None
        if rng.startswith("bytes="):
            lo, _, hi = rng[len("bytes="):].partition("-")
            start = int(lo) if lo.strip() else 0
            end = int(hi) + 1 if hi.strip() else None  # Range end is inclusive
        self._server.range_starts.append(start)
        self._raw = _FakeRaw(self._server._reader(start, end), status=self._server._status)

    def getresponse(self):
        return self._raw

    def close(self):
        self.sock = None


class _FakeSession:
    def __init__(self, conn):
        self._conn = conn

    def _get_connection(self, scheme, host, port, timeout):
        return self._conn


def _make_stream(payload, cuts, *, exc_factory=lambda: socket.timeout("timed out"),
                 max_retries=4, status=206):
    server = _FakeServer(payload, cuts, exc_factory=exc_factory, status=status)
    conn = _FakeConn(server)
    req = HTTPRequest.prepare(method="GET", url=_URL)
    stream = HTTPStream(server.source(), request=req, session=_FakeSession(conn),
                        max_retries=max_retries)
    return stream, server


# -- socket timeout / cut → resume at the same position ---------------------

def test_socket_timeout_resumes_at_received_offset_and_completes():
    payload = bytes(range(256)) * 4  # 1024 bytes, all distinct positions
    stream, server = _make_stream(payload, cuts=[400])  # source dies after 400 B
    body = bytes(stream.read_mv(-1))
    assert body == payload                 # complete, in order, no gap/dup
    assert server.range_starts == [400]    # resumed at the exact byte it stopped


def test_connection_reset_is_transient_and_resumes():
    payload = b"abcdefghij" * 100  # 1000 bytes
    stream, server = _make_stream(
        payload, cuts=[330], exc_factory=lambda: ConnectionResetError("reset"),
    )
    assert bytes(stream.read_mv(-1)) == payload
    assert server.range_starts == [330]


def test_ssl_unexpected_eof_is_transient_and_resumes():
    payload = b"X" * 700
    stream, server = _make_stream(
        payload, cuts=[256],
        exc_factory=lambda: ssl.SSLError("UNEXPECTED_EOF_WHILE_READING"),
    )
    assert bytes(stream.read_mv(-1)) == payload
    assert server.range_starts == [256]


def test_multiple_disconnects_resume_progressively():
    payload = bytes(i % 256 for i in range(2000))
    # Source dies at 500; the first reconnect (from 500) dies 300 B later at
    # absolute 800; the second finishes the object.
    stream, server = _make_stream(payload, cuts=[500, 300])
    assert bytes(stream.read_mv(-1)) == payload
    assert server.range_starts == [500, 800]


def test_bounded_range_resume_stays_within_original_slice():
    # Caller asked for slice bytes=100-199 (100 B). The body cuts 40 B in →
    # the resume must continue at absolute 140 AND keep the 199 upper bound,
    # delivering exactly payload[100:200] — not a re-read from the wrong place.
    payload = bytes(i % 251 for i in range(256))
    server = _FakeServer(payload, cuts=[40])
    conn = _FakeConn(server)
    req = HTTPRequest.prepare(method="GET", url=_URL, headers={"Range": "bytes=100-199"})
    stream = HTTPStream(server._reader(100, 200), request=req, session=_FakeSession(conn))
    assert bytes(stream.read_mv(-1)) == payload[100:200]
    assert server.range_starts == [140]


# -- exhaustion & non-resumable conditions ----------------------------------

def test_retries_exhausted_raises_the_transient_error():
    payload = b"Y" * 1000
    # Every connection dies after 50 B; with max_retries=2 the third cut wins.
    stream, server = _make_stream(payload, cuts=[50, 50, 50], max_retries=2)
    with pytest.raises(socket.timeout):
        stream.read_mv(-1)
    assert server.range_starts == [50, 100]  # original + 2 reconnects, then give up


def test_non_transient_error_is_not_retried():
    payload = b"Z" * 500
    stream, server = _make_stream(
        payload, cuts=[120], exc_factory=lambda: ValueError("decode boom"),
    )
    with pytest.raises(ValueError):
        stream.read_mv(-1)
    assert server.range_starts == []  # a logic error must not trigger a reconnect


def test_cannot_resume_without_request_or_session():
    payload = b"Q" * 300
    server = _FakeServer(payload, cuts=[80])
    # No request/session attached → a transient read error has nothing to
    # replay from and must propagate unchanged.
    stream = HTTPStream(server.source())
    with pytest.raises(socket.timeout):
        stream.read_mv(-1)
    assert server.range_starts == []
