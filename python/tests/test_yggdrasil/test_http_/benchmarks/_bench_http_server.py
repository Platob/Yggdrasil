"""Shared localhost HTTP server fixture for HTTPSession benchmarks.

Used by :mod:`bench_http_live` and :mod:`bench_http_concurrent` so both
benches measure against the same wire shape:

* HTTP/1.1 with keep-alive (so the pool reuses one socket per host),
* ``TCP_NODELAY`` on the per-request socket (otherwise Nagle +
  delayed-ACK inflates small-payload loopback latency to ~40 ms),
* fixed-size JSON payloads keyed by request path
  (``/tiny`` / ``/kib1`` / ``/kib64`` / ``/mib2``),
* any query string is dropped before key resolution, so cache-MISS
  scenarios that append ``?probe=N`` still hit the right payload.

Reach for :func:`start_bench_server` to bring it up in a daemon thread;
the returned base URL (``http://127.0.0.1:<port>``) is what
:class:`HTTPSession` should be pointed at.
"""
from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


__all__ = ["PAYLOADS", "BenchHandler", "start_bench_server"]


def _build_payload(size_bytes: int) -> bytes:
    """Return a JSON object with a ``data`` string padded to *size_bytes*."""
    # 11 bytes of framing for ``{"data":""}`` plus the body. The exact
    # size doesn't matter to the bench — we just want a fixed,
    # reproducible payload per scenario.
    pad = max(0, size_bytes - 11)
    return ('{"data":"' + ("x" * pad) + '"}').encode("ascii")


#: Payload size → bytes. Same key set ``bench_http_live`` documents.
PAYLOADS: dict[str, bytes] = {
    "tiny": _build_payload(16),
    "kib1": _build_payload(1024),
    "kib64": _build_payload(64 * 1024),
    "mib2": _build_payload(2 * 1024 * 1024),
}


class BenchHandler(BaseHTTPRequestHandler):
    """Serve a fixed-size JSON payload based on the request path.

    Path conventions: ``/tiny`` / ``/kib1`` / ``/kib64`` / ``/mib2`` map
    to the matching :data:`PAYLOADS` entry. Anything else returns 404 so
    a typo in a bench scenario fails loudly instead of yielding silent
    misses.

    HTTP/1.1 with persistent connections — the stdlib default of
    HTTP/1.0 forces one TCP handshake per request, which under Linux's
    delayed-ACK on small payloads inflates loopback latency to ~40 ms /
    request. With keep-alive the pool reuses one socket across the whole
    scenario and the bench measures actual request-dispatch cost.
    """

    protocol_version = "HTTP/1.1"

    # Silence the noisy default access log — every bench call would
    # otherwise dump a line to stderr.
    def log_message(self, format, *args):  # noqa: A002, D401 - stdlib name
        return

    def setup(self) -> None:  # noqa: D401 - stdlib name
        super().setup()
        # Disable Nagle on the per-request socket. ``BaseHTTPRequestHandler``
        # writes the status line, headers, and body in separate ``send`` calls,
        # which under Nagle + Linux's client-side delayed-ACK on loopback
        # stalls every small response by ~40 ms. The fix has to be on the
        # accepted socket the handler talks to — setting it on the listener
        # alone doesn't help. With TCP_NODELAY the small-payload numbers
        # actually reflect the dispatch cost instead of a kernel quirk.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _serve(self, path: str) -> None:
        # Drop the query — cache-MISS scenarios append ``?probe=N`` to
        # force a fresh URL identity per call. Without the strip those
        # paths would 404 because the query slips into the key match.
        key = path.split("?", 1)[0].strip("/").split("/", 1)[0]
        payload = PAYLOADS.get(key)
        if payload is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - stdlib name
        self._serve(self.path)


def start_bench_server() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    """Bind a server to a random localhost port and run it in a daemon thread.

    Returns the server, the serving thread, and the resolved base URL
    (``http://127.0.0.1:<port>``). The thread is daemonic so the bench
    process exits cleanly even if a scenario short-circuits the teardown.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), BenchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"
