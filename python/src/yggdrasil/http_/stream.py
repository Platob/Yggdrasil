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

from yggdrasil.io.io_stats import format_bytes
from yggdrasil.path.memory_stream import MemoryStream

if TYPE_CHECKING:
    from yggdrasil.http_.request import HTTPRequest

__all__ = ["HTTPStream"]

logger = logging.getLogger(__name__)

_TRANSIENT_SSL = ("EOF", "UNEXPECTED_EOF", "Connection reset")

# Transient connection EOF / reset mid-stream is usually a one-off (a dropped
# keep-alive socket); the range-reconnect picks right back up. Keep those retries
# at DEBUG and only escalate to WARNING once they persist to the 4th retry, so a
# single blip mid-download doesn't surface as a warning.
_EOF_LOG_AFTER = 4


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError, ConnectionResetError, BrokenPipeError)):
        return True
    if isinstance(exc, ssl.SSLError):
        msg = str(exc)
        return any(t in msg for t in _TRANSIENT_SSL)
    if isinstance(exc, OSError):
        return True
    return False


_COMPRESS_THRESHOLD = 4 * 1024 * 1024   # 4 MiB
_MAX_PICKLE_SIZE = 4 * 1024 * 1024      # 4 MiB


class HTTPStream(MemoryStream):
    """MemoryStream with HTTP range-request retry on transient failures."""

    __slots__ = (
        "_request", "_session_ref", "_max_retries", "_retry_count",
        "_live_conn", "_live_pool_key",
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
        self._live_conn: Any = None
        self._live_pool_key: Any = None

    def __getstate__(self) -> dict:
        self._pull_to_eof()
        total = self.size
        body = bytes(self.read_mv(total, 0))
        compressed = None
        if len(body) > _COMPRESS_THRESHOLD:
            import zlib
            compressed = zlib.compress(body, level=1)
            if len(compressed) >= len(body):
                compressed = None
        payload = compressed if compressed is not None else body
        if len(payload) <= _MAX_PICKLE_SIZE:
            return {
                "body": payload,
                "compressed": compressed is not None,
                "size": total,
            }
        if self._request is None or self._session_ref is None:
            raise OverflowError(
                f"HTTPStream body too large to pickle: "
                f"{len(payload)} bytes (limit={_MAX_PICKLE_SIZE}) "
                f"and no request/session to replay from."
            )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "HTTPStream pickling %s %s as replay (body=%s > %s limit)",
                self._request.method, self._request.url,
                format_bytes(total), format_bytes(_MAX_PICKLE_SIZE),
            )
        return {
            "replay": True,
            "request": self._request,
            "session": self._session_ref,
            "offset": total,
        }

    def __setstate__(self, state: dict) -> None:
        if state.get("replay"):
            request = state["request"]
            session = state["session"]
            offset = state["offset"]
            self.__init__(request=request, session=session)
            self._reconnect_at(offset)
        else:
            body = state["body"]
            if state.get("compressed"):
                import zlib
                body = zlib.decompress(body)
            self.__init__(source=body)
            self._pull_to_eof()

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
            logger.log(
                logging.WARNING if self._retry_count >= _EOF_LOG_AFTER else logging.DEBUG,
                "HTTPStream retry %d/%d for %s %s at offset %d: %s",
                self._retry_count, self._max_retries,
                self._request.method, self._request.url,
                self.size, exc,
            )
            self._reconnect()
            return super()._pull_one(want)

    def _reconnect(self) -> None:
        """Re-issue the request with a Range header from current offset."""
        self._open_range_connection(self.size)

    def _reconnect_at(self, offset: int) -> None:
        """Open a range connection and skip to *offset*.

        Used on deserialization: the body up to *offset* was already
        consumed in the original process; re-issue with
        ``Range: bytes=<offset>-`` so the server skips the head.
        When the server responds with 200 (doesn't support Range),
        drain the first *offset* bytes locally.
        """
        raw = self._open_range_connection(offset)
        if raw.status == 200 and offset > 0:
            remaining = offset
            while remaining > 0:
                chunk = raw.read(min(remaining, 1024 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "HTTPStream drained %s to reach offset %d",
                    format_bytes(offset - remaining), offset,
                )

    def _open_range_connection(self, offset: int) -> Any:
        """Issue a Range request and bind the response as the new source.

        *offset* is the number of body bytes already consumed. When the
        original request itself carried a ``Range: bytes=<base>-[<end>]``
        (a bounded / offset read), the resume must continue from
        ``base + offset`` and preserve the original upper bound — not
        from ``offset`` alone, which would be an absolute address into
        the wrong part of the object and silently corrupt the read.

        Stores the new connection in ``_live_conn`` / ``_live_pool_key``
        so :meth:`HTTPResponse.release_conn` can return it to the pool
        instead of the original (now-broken) connection.
        """
        # Close any previous reconnect connection — it's being replaced.
        old = self._live_conn
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
            self._live_conn = None
            self._live_pool_key = None

        session = self._session_ref
        request = self._request
        headers = {**dict(request.headers)}
        base, end = 0, ""
        original = headers.get("Range")
        if original and original.startswith("bytes="):
            spec = original[len("bytes="):].split("-", 1)
            if spec[0].strip().isdigit():
                base = int(spec[0].strip())
            if len(spec) > 1:
                end = spec[1].strip()
        start = base + offset
        if start > 0 or end:
            headers["Range"] = f"bytes={start}-{end}"

        url = request.url
        scheme = url.scheme
        host = url.host
        port = url.port or (443 if scheme == "https" else 80)
        path = url.path or "/"
        if url.query:
            path = f"{path}?{url.query}"

        conn = session._get_connection(scheme, host, port, None)
        try:
            if conn.sock is None:
                conn.connect()
            headers.setdefault(
                "Host", f"{host}:{port}" if port not in (80, 443) else host,
            )
            headers.setdefault("Connection", "keep-alive")
            conn.request(request.method, path, body=None, headers=headers)
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
                f"Range request returned {raw.status} (expected 200/206) "
                f"for {request.method} {request.url}"
            )

        pool_key = (scheme, host, port)
        self._live_conn = conn
        self._live_pool_key = None if getattr(raw, "will_close", False) else pool_key
        self._eof = False
        self._read_chunk = raw.read

        logger.info(
            "HTTPStream connected at offset %d for %s %s (status=%d)",
            offset, request.method, request.url, raw.status,
        )
        return raw
