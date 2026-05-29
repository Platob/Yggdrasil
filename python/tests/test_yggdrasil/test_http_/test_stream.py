"""Resume semantics for :class:`HTTPStream`.

The resumable body must continue from the correct *absolute* offset
after a mid-flight disconnect. For a plain whole-object read that's just
``bytes=<received>-``; for a request that already carried a bounded
``Range`` (a random-seek slice), the resume has to continue from
``base + received`` and keep the original upper bound — otherwise it
re-reads from the wrong part of the object and silently corrupts the
slice.
"""

from __future__ import annotations

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.stream import HTTPStream


class _FakeRaw:
    status = 206

    def read(self, *a):
        return b""


class _FakeConn:
    def __init__(self):
        self.requested = None

    def connect(self):
        pass

    def request(self, method, path, body=None, headers=None):
        self.requested = (method, path, dict(headers or {}))

    def getresponse(self):
        return _FakeRaw()

    def close(self):
        pass


class _FakeSession:
    def __init__(self, conn):
        self._conn = conn

    def _get_connection(self, scheme, host, port, timeout):
        return self._conn


def _resume(range_header, received):
    """Drive a resume and return the Range header the re-request carried."""
    headers = {"Range": range_header} if range_header else None
    req = HTTPRequest.prepare(
        method="GET",
        url="https://host.example/api/2.0/fs/files/Volumes/c/s/v/x",
        headers=headers,
    )
    conn = _FakeConn()
    stream = HTTPStream(request=req, session=_FakeSession(conn))
    stream._open_range_connection(received)
    return conn.requested[2].get("Range")


def test_whole_object_resume_from_received():
    # No original Range — resume from the absolute received count.
    assert _resume(None, 1024) == "bytes=1024-"


def test_whole_object_resume_at_zero_sends_no_range():
    # Nothing received yet and no original range → full re-request.
    assert _resume(None, 0) is None


def test_bounded_range_resume_keeps_base_and_upper_bound():
    # Original slice was bytes 100-199; 30 already consumed → continue at
    # absolute 130 and KEEP the 199 upper bound.
    assert _resume("bytes=100-199", 30) == "bytes=130-199"


def test_bounded_range_resume_at_zero_reissues_original_slice():
    # A failure before any byte landed re-requests the original slice.
    assert _resume("bytes=100-199", 0) == "bytes=100-199"


def test_open_ended_range_resume_advances_base():
    # Original ``bytes=500-`` (read-to-EOF from 500); 40 consumed →
    # continue at absolute 540, still open-ended.
    assert _resume("bytes=500-", 40) == "bytes=540-"
