"""Multipart (parallel) upload for :class:`VolumePath`, over a localhost server.

Exercises the real Databricks Files multipart protocol end to end against
a fake workspace + cloud-storage server: ``initiate-upload`` →
``create-upload-part-urls`` → parallel ``PUT`` to presigned part URLs
(ETags) → ``complete-upload``, with ``create-abort-upload-url`` + DELETE
on failure. Drives a real :class:`HTTPSession` through :class:`VolumePath`
so the orchestration, concurrency, and fallback all run for real.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.http_ import HTTPSession
from yggdrasil.url import URL


class _MultipartServerState:
    def __init__(self) -> None:
        self.initiate_status = 200          # flip to 404 to force fallback
        self.fail_part: int | None = None   # part_number to fail (→ abort)
        self.parts: dict[int, bytes] = {}   # uploaded part bodies
        self.completed: bytes | None = None # assembled object
        self.single_put: bytes | None = None
        self.aborted = False
        self.active_parts = 0
        self.max_concurrent_parts = 0
        self.initiate_calls = 0
        self._lock = threading.Lock()


def _make_handler(state: _MultipartServerState, host_holder: dict):
    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, *a):
            pass

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length", "0"))
            return self.rfile.read(n) if n else b""

        def _send(self, status, body=b"", headers=None):
            self.send_response(status)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj).encode(),
                       {"Content-Type": "application/json"})

        def do_POST(self):  # noqa: N802
            split = urlsplit(self.path)
            path, query = split.path, parse_qs(split.query)
            body = self._read_body()
            if path == "/api/2.0/fs/create-upload-part-urls":
                req = json.loads(body)
                host = host_holder["host"]
                start = req["start_part_number"]
                count = req["count"]
                token = req["session_token"]
                urls = [
                    {
                        "url": f"{host}/_part/{token}/{start + i}",
                        "part_number": start + i,
                        "headers": [],
                    }
                    for i in range(count)
                ]
                self._json(200, {"upload_part_urls": urls})
                return
            if path == "/api/2.0/fs/create-abort-upload-url":
                token = json.loads(body)["session_token"]
                self._json(200, {"abort_upload_url": {
                    "url": f"{host_holder['host']}/_abort/{token}", "headers": [],
                }})
                return
            if path.startswith("/api/2.0/fs/files"):
                action = query.get("action", [""])[0]
                if action == "initiate-upload":
                    state.initiate_calls += 1
                    if state.initiate_status != 200:
                        self._json(state.initiate_status, {"message": "no multipart"})
                        return
                    self._json(200, {"multipart_upload": {"session_token": "tok-1"}})
                    return
                if action == "complete-upload":
                    parts = json.loads(body)["parts"]
                    assembled = b"".join(
                        state.parts[p["part_number"]]
                        for p in sorted(parts, key=lambda p: p["part_number"])
                    )
                    state.completed = assembled
                    self._send(200)
                    return
            self._send(404)

        def do_PUT(self):  # noqa: N802
            split = urlsplit(self.path)
            path = split.path
            body = self._read_body()
            if path.startswith("/_part/"):
                part_number = int(path.rsplit("/", 1)[1])
                with state._lock:
                    state.active_parts += 1
                    state.max_concurrent_parts = max(
                        state.max_concurrent_parts, state.active_parts,
                    )
                try:
                    time.sleep(0.02)  # widen the overlap window for concurrency
                    if state.fail_part == part_number:
                        self._send(500)
                        return
                    state.parts[part_number] = body
                    self._send(200, headers={"ETag": f'"etag-{part_number}"'})
                finally:
                    with state._lock:
                        state.active_parts -= 1
                return
            if path.startswith("/api/2.0/fs/files"):
                state.single_put = body
                self._send(204)
                return
            self._send(404)

        def do_DELETE(self):  # noqa: N802
            if urlsplit(self.path).path.startswith("/_abort/"):
                state.aborted = True
                self._send(200)
                return
            self._send(404)

    return _H


class _Client:
    def __init__(self, host: str) -> None:
        self.base_url = URL.from_(host)
        self._session = HTTPSession(base_url=host, verify=False)

    def files_session(self):
        return self._session

    def files_authorization(self):
        return "Bearer test"


class _Service:
    def __init__(self, client):
        self.client = client


@pytest.fixture
def server():
    state = _MultipartServerState()
    host_holder: dict = {}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state, host_holder))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host_holder["host"] = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        yield state, host_holder["host"]
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture(autouse=True)
def _small_multipart(monkeypatch):
    # Trigger multipart on tiny payloads so the test is fast.
    monkeypatch.setattr(VolumePath, "MULTIPART_MIN_SIZE", 1024)
    monkeypatch.setattr(VolumePath, "MULTIPART_PART_SIZE", 256)


def _volume(host: str) -> VolumePath:
    return VolumePath("/Volumes/c/s/v/big.bin", service=_Service(_Client(host)))


class TestMultipartUpload:

    def test_large_upload_runs_parallel_parts_and_assembles(self, server):
        state, host = server
        payload = bytes(range(256)) * 8  # 2048 B → 8 parts of 256 B
        _volume(host).write_bytes(payload, overwrite=True)

        assert state.completed == payload          # parts assembled in order
        assert len(state.parts) == 8
        assert state.single_put is None            # took the multipart path
        assert state.max_concurrent_parts > 1      # parts ran concurrently

    def test_falls_back_to_single_put_when_unsupported(self, server):
        state, host = server
        state.initiate_status = 404                # workspace lacks multipart
        payload = bytes(range(256)) * 8
        _volume(host).write_bytes(payload, overwrite=True)

        assert state.initiate_calls == 1
        assert state.single_put == payload         # fell back to one PUT
        assert state.completed is None

    def test_small_upload_uses_single_put(self, server):
        state, host = server
        payload = b"tiny"                          # below MULTIPART_MIN_SIZE
        _volume(host).write_bytes(payload, overwrite=True)

        assert state.single_put == payload
        assert state.initiate_calls == 0

    def test_part_failure_aborts_and_raises(self, server):
        state, host = server
        state.fail_part = 3                        # one part 500s
        payload = bytes(range(256)) * 8
        with pytest.raises(OSError):
            _volume(host).write_bytes(payload, overwrite=True)
        assert state.aborted is True               # session cleaned up
        assert state.completed is None
