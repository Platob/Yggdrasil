"""Resumable HTTP response stream — :class:`HTTPStream`.

:class:`HTTPStream` wraps a live HTTP socket body with automatic
retry-on-disconnect. When the source read raises a transient error
(SSL EOF, connection reset, socket timeout), it reconnects to the
origin with a ``Range: bytes=<received>-`` header and continues
pulling from where it left off — transparent to the consumer.

Used by :meth:`HTTPResponse.from_wire` as the buffer for response
bodies so large downloads survive mid-flight connection drops.
"""

from __future__ import annotations

import logging
import ssl
import socket
from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.path.memory_stream import MemoryStream

if TYPE_CHECKING:
    from yggdrasil.http_.request import HTTPRequest

__all__ = ["HTTPStream"]

logger = logging.getLogger(__name__)

_TRANSIENT_SSL = ("EOF", "UNEXPECTED_EOF", "Connection reset")


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError, ConnectionResetError, BrokenPipeError)):
        return True
    if isinstance(exc, ssl.SSLError):
        msg = str(exc)
        return any(t in msg for t in _TRANSIENT_SSL)
    if isinstance(exc, OSError):
        return True
    return False


class HTTPStream(MemoryStream):
    """MemoryStream with HTTP range-request retry on transient failures."""

    __slots__ = (
        "_request", "_session_ref", "_max_retries", "_retry_count",
    )

    def __init__(
        self,
        source: Any = None,
        *,
        request: "HTTPRequest | None" = None,
        session: Any = None,
        max_retries: int = 4,
        content_encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(source, content_encoding=content_encoding, **kwargs)
        self._request = request
        self._session_ref = session
        self._max_retries = max_retries
        self._retry_count = 0

    def _pull_one(self, want: int) -> int:
        try:
            return super()._pull_one(want)
        except Exception as exc:
            if not _is_transient(exc):
                raise
            if self._request is None or self._session_ref is None:
                raise
            if self._retry_count >= self._max_retries:
                logger.error(
                    "HTTPStream exhausted %d retries for %s %s at offset %d",
                    self._max_retries, self._request.method,
                    self._request.url, self.size,
                )
                raise
            self._retry_count += 1
            logger.warning(
                "HTTPStream retry %d/%d for %s %s at offset %d: %s",
                self._retry_count, self._max_retries,
                self._request.method, self._request.url,
                self.size, exc,
            )
            self._reconnect()
            return super()._pull_one(want)

    def _reconnect(self) -> None:
        """Re-issue the request with a Range header from current offset."""
        import http.client
        session = self._session_ref
        request = self._request
        offset = self.size

        range_request = request.copy(
            headers={**dict(request.headers), "Range": f"bytes={offset}-"},
        )

        url = range_request.url
        scheme = url.scheme
        host = url.host
        port = url.port or (443 if scheme == "https" else 80)
        path = url.path or "/"
        if url.query:
            path = f"{path}?{url.query}"

        conn = session._get_connection(scheme, host, port, None)
        try:
            conn.connect()
            send_headers = dict(range_request.headers)
            send_headers.setdefault(
                "Host", f"{host}:{port}" if port not in (80, 443) else host,
            )
            conn.request(request.method, path, body=None, headers=send_headers)
            raw = conn.getresponse()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

        if raw.status not in (200, 206):
            try:
                conn.close()
            except Exception:
                pass
            raise OSError(
                f"Range retry returned {raw.status} (expected 206) "
                f"for {request.method} {request.url}"
            )

        self._eof = False
        self._read_chunk = raw.read

        logger.info(
            "HTTPStream reconnected at offset %d for %s %s (status=%d)",
            offset, request.method, request.url, raw.status,
        )
