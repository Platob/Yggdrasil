"""Stdlib-only HTTP connection pool — the implementation yggdrasil ships with.

Replaces the historical ``urllib3`` dependency: every HTTP call site in the
codebase routes through this module instead. Built on :mod:`http.client` +
:mod:`urllib.parse` (stdlib) so the package has no third-party HTTP
transport requirement.

Surface (urllib3-shaped, by design — keeps the call sites readable):

* :class:`PoolManager` with ``.request(method, url, body, headers, timeout,
  preload_content, decode_content, redirect)``
* :class:`HTTPResponse` exposing ``.status`` / ``.headers`` / ``.read`` /
  ``.stream`` / ``.release_conn`` / ``.drain_conn``
* :class:`Retry` (subclassable; ``history`` carries :class:`RequestHistory`
  entries so :class:`yggdrasil.http_.session._TieredRetry` works unchanged)
* :class:`Timeout` carrying ``connect`` / ``read`` / ``total``
* :class:`HTTPHeaderDict` case-insensitive multi-value header dict
* :func:`disable_warnings`
* :mod:`exceptions` namespace: ``HTTPError``, ``NewConnectionError``,
  ``TimeoutError``, ``ConnectTimeoutError``, ``ReadTimeoutError``,
  ``ProxyError``, ``DecodeError``, ``InvalidChunkLength``, ``IncompleteRead``,
  ``PoolError``, ``ClosedPoolError``, ``EmptyPoolError``, ``HostChangedError``,
  ``LocationValueError``, ``LocationParseError``, ``SSLError``,
  ``SecurityWarning``, ``InsecureRequestWarning``, ``MaxRetryError``

Covers yggdrasil's actual call shapes; it is not a general-purpose
reimplementation of urllib3.
"""
from __future__ import annotations

import collections
import collections.abc
import http.client
import logging
import socket
import ssl
import threading
import time
import types
import warnings
import zlib
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit


LOGGER = logging.getLogger(__name__)

__all__ = [
    "exceptions",
    "PoolManager",
    "Retry",
    "Timeout",
    "BaseHTTPResponse",
    "HTTPResponse",
    "HTTPHeaderDict",
    "disable_warnings",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HTTPError(Exception):
    """Root of the stdlib-shim urllib3 exception hierarchy."""


class PoolError(HTTPError):
    def __init__(self, pool: Any = None, message: str = "") -> None:
        super().__init__(message)
        self.pool = pool


class RequestError(PoolError):
    def __init__(self, pool: Any = None, url: str = "", message: str = "") -> None:
        super().__init__(pool, message)
        self.url = url


class SSLError(HTTPError):
    pass


class ProxyError(HTTPError):
    def __init__(self, message: str = "", error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.original_error = error


class DecodeError(HTTPError):
    pass


class ProtocolError(HTTPError):
    pass


class ConnectionError(HTTPError):
    pass


class NewConnectionError(ConnectionError, RequestError):
    def __init__(self, pool: Any = None, message: str = "") -> None:
        # urllib3 signature: (pool, message)
        super().__init__(pool, "", message)


class TimeoutError(HTTPError):  # noqa: A001 — mirrors urllib3 spelling
    pass


class ConnectTimeoutError(TimeoutError):
    pass


class ReadTimeoutError(TimeoutError, RequestError):
    def __init__(self, pool: Any = None, url: str = "", message: str = "") -> None:
        super().__init__(pool, url, message)


class MaxRetryError(RequestError):
    def __init__(self, pool: Any = None, url: str = "", reason: Optional[Exception] = None) -> None:
        message = f"Max retries exceeded with url: {url} (Caused by {reason!r})"
        super().__init__(pool, url, message)
        self.reason = reason


class ClosedPoolError(PoolError):
    pass


class EmptyPoolError(PoolError):
    pass


class HostChangedError(RequestError):
    def __init__(self, pool: Any = None, url: str = "", retries: int = 3) -> None:
        message = f"Tried to open a foreign host with url: {url}"
        super().__init__(pool, url, message)
        self.retries = retries


class LocationValueError(ValueError, HTTPError):
    pass


class LocationParseError(LocationValueError):
    def __init__(self, location: str = "") -> None:
        super().__init__(f"Failed to parse: {location}")
        self.location = location


class InvalidChunkLength(HTTPError, http.client.IncompleteRead):
    def __init__(self, response: Any = None, length: int = 0) -> None:
        http.client.IncompleteRead.__init__(self, partial=b"", expected=length)
        self.response = response
        self.length = length


class IncompleteRead(HTTPError, http.client.IncompleteRead):
    def __init__(self, partial: int = 0, expected: Optional[int] = None) -> None:
        http.client.IncompleteRead.__init__(self, partial=b"" if isinstance(partial, int) else partial, expected=expected)


class SecurityWarning(Warning):
    pass


class InsecureRequestWarning(SecurityWarning):
    pass


# Bundle the exception classes into a module-like namespace mirroring
# ``urllib3.exceptions`` so ``from yggdrasil._urllib3 import exceptions``
# followed by ``exceptions.HTTPError`` works the same on either branch.
exceptions = types.SimpleNamespace(
    HTTPError=HTTPError,
    PoolError=PoolError,
    RequestError=RequestError,
    SSLError=SSLError,
    ProxyError=ProxyError,
    DecodeError=DecodeError,
    ProtocolError=ProtocolError,
    ConnectionError=ConnectionError,
    NewConnectionError=NewConnectionError,
    TimeoutError=TimeoutError,
    ConnectTimeoutError=ConnectTimeoutError,
    ReadTimeoutError=ReadTimeoutError,
    MaxRetryError=MaxRetryError,
    ClosedPoolError=ClosedPoolError,
    EmptyPoolError=EmptyPoolError,
    HostChangedError=HostChangedError,
    LocationValueError=LocationValueError,
    LocationParseError=LocationParseError,
    InvalidChunkLength=InvalidChunkLength,
    IncompleteRead=IncompleteRead,
    SecurityWarning=SecurityWarning,
    InsecureRequestWarning=InsecureRequestWarning,
)


# ---------------------------------------------------------------------------
# disable_warnings
# ---------------------------------------------------------------------------

def disable_warnings(category: type = SecurityWarning) -> None:
    """Filter out warnings of ``category`` (default :class:`SecurityWarning`)."""
    warnings.filterwarnings("ignore", category=category)


# ---------------------------------------------------------------------------
# HTTPHeaderDict
# ---------------------------------------------------------------------------

class HTTPHeaderDict(collections.abc.MutableMapping):
    """Case-insensitive, multi-value header dict mirroring ``urllib3``'s.

    Stores values per lowercase key but preserves the first-seen original
    casing when iterating. Multi-value headers (Set-Cookie, …) are joined
    with ``, `` on read, matching urllib3's collapsing behavior.
    """

    def __init__(self, headers: Any = None, **kwargs: str) -> None:
        # _store maps lowercase key -> (original_case, [values])
        self._store: dict[str, Tuple[str, list[str]]] = {}
        if headers is not None:
            self.extend(headers)
        if kwargs:
            self.extend(kwargs)

    # MutableMapping protocol -------------------------------------------------
    def __setitem__(self, key: str, value: str) -> None:
        self._store[key.lower()] = (key, [value])

    def __getitem__(self, key: str) -> str:
        _, values = self._store[key.lower()]
        return ", ".join(values)

    def __delitem__(self, key: str) -> None:
        del self._store[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return (original for original, _ in self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self._store

    # Multi-value helpers -----------------------------------------------------
    def add(self, key: str, value: str) -> None:
        slot = self._store.get(key.lower())
        if slot is None:
            self._store[key.lower()] = (key, [value])
        else:
            slot[1].append(value)

    def extend(self, other: Any) -> None:
        if isinstance(other, HTTPHeaderDict):
            for original, values in other._store.values():
                for v in values:
                    self.add(original, v)
            return
        if hasattr(other, "items"):
            other = other.items()
        for k, v in other:
            self.add(k, v)

    def getlist(self, key: str) -> list[str]:
        slot = self._store.get(key.lower())
        return list(slot[1]) if slot is not None else []

    def __repr__(self) -> str:
        return f"HTTPHeaderDict({dict(self.items())!r})"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class Timeout:
    """Per-call timeout config — ``urllib3.Timeout`` shaped subset.

    ``connect`` / ``read`` / ``total`` map onto ``http.client`` connect and
    read deadlines; the shim does not implement the full DEFAULT-sentinel
    machinery (None means "no timeout", a number means "seconds").
    """

    DEFAULT_TIMEOUT = socket._GLOBAL_DEFAULT_TIMEOUT  # type: ignore[attr-defined]

    def __init__(
        self,
        total: Optional[float] = None,
        connect: Optional[float] = None,
        read: Optional[float] = None,
    ) -> None:
        self.total = total
        self.connect = connect
        self.read = read

    @property
    def connect_timeout(self) -> Optional[float]:
        if self.connect is not None:
            return self.connect
        return self.total

    @property
    def read_timeout(self) -> Optional[float]:
        if self.read is not None:
            return self.read
        return self.total

    def __repr__(self) -> str:
        return f"Timeout(total={self.total!r}, connect={self.connect!r}, read={self.read!r})"


def _resolve_timeout(timeout: Any) -> Tuple[Optional[float], Optional[float]]:
    """Normalize a ``timeout`` argument to ``(connect, read)`` seconds."""
    if timeout is None:
        return None, None
    if isinstance(timeout, Timeout):
        return timeout.connect_timeout, timeout.read_timeout
    if isinstance(timeout, (int, float)):
        return float(timeout), float(timeout)
    if isinstance(timeout, tuple) and len(timeout) == 2:
        return timeout[0], timeout[1]
    return None, None


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

class RequestHistory(NamedTuple):
    """Single retry-attempt record. Mirrors ``urllib3.util.retry.RequestHistory``."""
    method: Optional[str]
    url: Optional[str]
    error: Optional[Exception]
    status: Optional[int]
    redirect_location: Optional[str]


class Retry:
    """Minimal subclassable retry policy mirroring ``urllib3.Retry``.

    Implements the surface yggdrasil's :class:`_TieredRetry` reaches into:
    ``BACKOFF_MAX`` (class attr), ``history`` (tuple of
    :class:`RequestHistory`), ``get_backoff_time()``,
    ``get_retry_after(response)``, ``increment(...)``, ``new(**overrides)``,
    ``is_exhausted()``, ``sleep(response=None)``.

    The shim's :class:`PoolManager` drives this in its retry loop.
    """

    DEFAULT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "DELETE", "TRACE"})
    DEFAULT_BACKOFF_MAX = 120.0
    BACKOFF_MAX: float = DEFAULT_BACKOFF_MAX

    def __init__(
        self,
        total: Optional[int] = 10,
        connect: Optional[int] = None,
        read: Optional[int] = None,
        redirect: Optional[int] = None,
        status: Optional[int] = None,
        other: Optional[int] = None,
        allowed_methods: Optional[Iterable[str]] = DEFAULT_METHODS,
        status_forcelist: Optional[Iterable[int]] = None,
        backoff_factor: float = 0.0,
        backoff_max: Optional[float] = None,
        raise_on_redirect: bool = True,
        raise_on_status: bool = True,
        history: Tuple[RequestHistory, ...] = (),
        respect_retry_after_header: bool = True,
        remove_headers_on_redirect: Optional[Iterable[str]] = None,
        **_: Any,
    ) -> None:
        self.total = total
        self.connect = connect
        self.read = read
        self.redirect = redirect
        self.status = status
        self.other = other
        self.allowed_methods = (
            frozenset(m.upper() for m in allowed_methods) if allowed_methods else None
        )
        self.status_forcelist = frozenset(status_forcelist) if status_forcelist else frozenset()
        self.backoff_factor = backoff_factor
        if backoff_max is not None:
            self.BACKOFF_MAX = backoff_max  # type: ignore[misc]  # instance shadows class attr
        self.raise_on_redirect = raise_on_redirect
        self.raise_on_status = raise_on_status
        self.history = tuple(history)
        self.respect_retry_after_header = respect_retry_after_header
        self.remove_headers_on_redirect = (
            frozenset(h.lower() for h in remove_headers_on_redirect)
            if remove_headers_on_redirect
            else frozenset()
        )

    # --- urllib3 compat surface ---------------------------------------------
    def new(self, **kwargs: Any) -> "Retry":
        params = dict(
            total=self.total,
            connect=self.connect,
            read=self.read,
            redirect=self.redirect,
            status=self.status,
            other=self.other,
            allowed_methods=self.allowed_methods,
            status_forcelist=self.status_forcelist,
            backoff_factor=self.backoff_factor,
            backoff_max=self.BACKOFF_MAX,
            raise_on_redirect=self.raise_on_redirect,
            raise_on_status=self.raise_on_status,
            history=self.history,
            respect_retry_after_header=self.respect_retry_after_header,
            remove_headers_on_redirect=self.remove_headers_on_redirect,
        )
        params.update(kwargs)
        return type(self)(**params)

    def get_backoff_time(self) -> float:
        # urllib3 short-circuits to 0 before the second consecutive error.
        consecutive_errors = [h for h in reversed(self.history) if h.redirect_location is None]
        if len(consecutive_errors) <= 1:
            return 0.0
        backoff = self.backoff_factor * (2 ** (len(consecutive_errors) - 1))
        return float(min(self.BACKOFF_MAX, backoff))

    def get_retry_after(self, response: "BaseHTTPResponse") -> Optional[float]:
        if not self.respect_retry_after_header:
            return None
        value = response.headers.get("Retry-After") if response.headers else None
        if not value:
            return None
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            pass
        try:
            import datetime as _dt
            delta = parsedate_to_datetime(value) - _dt.datetime.now(_dt.timezone.utc)
            return max(0.0, delta.total_seconds())
        except Exception:
            return None

    def sleep(self, response: Optional["BaseHTTPResponse"] = None) -> None:
        wait = 0.0
        if response is not None:
            after = self.get_retry_after(response)
            if after is not None:
                wait = after
        if wait <= 0:
            wait = self.get_backoff_time()
        if wait > 0:
            time.sleep(wait)

    def is_exhausted(self) -> bool:
        retry_counts = [c for c in (self.total, self.connect, self.read, self.redirect, self.status, self.other) if c is not None]
        return any(c < 0 for c in retry_counts)

    def is_retry(self, method: str, status_code: int, has_retry_after: bool = False) -> bool:
        if self.allowed_methods is not None and method.upper() not in self.allowed_methods:
            return False
        if status_code in self.status_forcelist:
            return True
        return has_retry_after and self.respect_retry_after_header

    def increment(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
        response: Optional["BaseHTTPResponse"] = None,
        error: Optional[Exception] = None,
        _pool: Any = None,
        _stacktrace: Any = None,
    ) -> "Retry":
        total = self.total - 1 if self.total is not None else None
        connect = self.connect
        read = self.read
        status_count = self.status
        other = self.other
        redirect = self.redirect
        cause = "unknown"

        if error and isinstance(error, ConnectTimeoutError):
            if connect is not None:
                connect -= 1
            cause = "connection error"
        elif error and isinstance(error, (ReadTimeoutError, ProtocolError)):
            if read is not None:
                read -= 1
            cause = "read error"
        elif error is not None:
            if other is not None:
                other -= 1
            cause = "other error"
        elif response is not None and self.is_retry(method or "", response.status, False):
            if status_count is not None:
                status_count -= 1
            cause = f"too many {response.status} error responses"

        history = self.history + (
            RequestHistory(
                method=method,
                url=url,
                error=error,
                status=response.status if response is not None else None,
                redirect_location=None,
            ),
        )
        new = self.new(
            total=total,
            connect=connect,
            read=read,
            redirect=redirect,
            status=status_count,
            other=other,
            history=history,
        )
        if new.is_exhausted():
            raise MaxRetryError(_pool, url or "", error or Exception(cause))
        return new


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class BaseHTTPResponse:
    """ABC marker — every concrete shim response inherits from this."""

    status: int
    headers: HTTPHeaderDict


class _DecodingReader:
    """Wraps a chunked source iterator and decodes gzip/deflate on the fly."""

    def __init__(self, raw_read, content_encoding: Optional[str]) -> None:
        self._raw_read = raw_read
        self._encoding = (content_encoding or "").lower()
        self._decoder: Any = None
        if self._encoding in ("gzip", "x-gzip"):
            self._decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif self._encoding == "deflate":
            self._decoder = zlib.decompressobj()

    def read(self, amt: Optional[int] = None) -> bytes:
        chunk = self._raw_read(amt) if amt is not None else self._raw_read()
        if not chunk:
            if self._decoder is not None:
                tail = self._decoder.flush()
                self._decoder = None
                return tail
            return b""
        if self._decoder is not None:
            return self._decoder.decompress(chunk)
        return chunk


class HTTPResponse(BaseHTTPResponse):
    """Stdlib-backed urllib3-shaped HTTP response.

    Wraps :class:`http.client.HTTPResponse` and exposes:

    * ``.status`` — int status code
    * ``.headers`` — :class:`HTTPHeaderDict`
    * ``.read([amt])`` — body bytes (decodes Content-Encoding when
      ``decode_content=True``)
    * ``.stream(amt=...)`` — iterator yielding chunks
    * ``.release_conn()`` — returns the underlying connection to the pool
    * ``.drain_conn()`` — read remaining bytes so the connection is reusable
    """

    def __init__(
        self,
        body: Any = None,
        headers: Any = None,
        status: int = 0,
        preload_content: bool = True,
        decode_content: bool = True,
        *,
        request_url: Optional[str] = None,
        request_method: Optional[str] = None,
        pool: Optional["PoolManager"] = None,
        connection: Optional[http.client.HTTPConnection] = None,
        pool_key: Optional[Tuple[str, str, int]] = None,
        **_: Any,
    ) -> None:
        # Two construction shapes are supported, mirroring urllib3's own
        # ``HTTPResponse``:
        #   1. internal "just sent a wire request" — ``body`` is the live
        #      :class:`http.client.HTTPResponse` we own + need to drain;
        #   2. external "wrap pre-decoded bytes" — ``body`` is a
        #      :class:`io.BytesIO` / :class:`bytes` / file-like the caller
        #      already populated (the urllib3-shim path in
        #      :mod:`yggdrasil.exceptions.http.ResponseError.urllib3_response`).
        if isinstance(body, http.client.HTTPResponse):
            self._raw = body
            raw_status = body.status
            raw_headers = body.getheaders()
        else:
            self._raw = None
            raw_status = None
            raw_headers = None

        self.status = raw_status if raw_status is not None else status
        self.headers = headers if isinstance(headers, HTTPHeaderDict) else HTTPHeaderDict(headers or {})
        if raw_headers is not None and not headers:
            for k, v in raw_headers:
                self.headers.add(k, v)

        self._url = request_url
        self._method = request_method
        self._decode_content = decode_content
        self._pool = pool
        self._connection = connection
        self._pool_key = pool_key
        self._released = False

        # Pre-buffered body (BytesIO / bytes) bypasses the decoder entirely.
        self._body: Optional[bytes]
        if self._raw is not None:
            self._body = None
            self._decoder = _DecodingReader(
                self._raw.read,
                self.headers.get("Content-Encoding") if decode_content else None,
            )
        elif isinstance(body, (bytes, bytearray)):
            self._body = bytes(body)
            self._decoder = _DecodingReader(lambda *_: b"", None)
        elif body is None:
            self._body = b""
            self._decoder = _DecodingReader(lambda *_: b"", None)
        else:
            # file-like (BytesIO, etc.) — read lazily.
            self._body = None
            self._decoder = _DecodingReader(body.read, None)

        if preload_content and self._body is None:
            self._body = self._read_all()
            self.release_conn()

    # ---- core API ----------------------------------------------------------
    def read(self, amt: Optional[int] = None) -> bytes:
        if self._body is not None:
            if amt is None:
                data, self._body = self._body, b""
                return data
            data, self._body = self._body[:amt], self._body[amt:]
            return data
        return self._decoder.read(amt)

    def _read_all(self) -> bytes:
        chunks: list[bytes] = []
        while True:
            chunk = self._decoder.read(1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def stream(self, amt: int = 65536, decode_content: Optional[bool] = None) -> Iterator[bytes]:
        if self._body is not None:
            # Already buffered (preload_content=True path).
            data = self._body
            self._body = b""
            for i in range(0, len(data), amt):
                yield data[i:i + amt]
            return
        try:
            while True:
                chunk = self._decoder.read(amt)
                if not chunk:
                    return
                yield chunk
        finally:
            self.release_conn()

    def release_conn(self) -> None:
        if self._released:
            return
        self._released = True
        if self._pool is not None and self._connection is not None and self._pool_key is not None:
            self._pool._release_connection(self._pool_key, self._connection)
        elif self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass

    def drain_conn(self) -> None:
        if self._body is not None or self._released:
            return
        try:
            while self._decoder.read(1 << 16):
                pass
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._raw is not None:
                self._raw.close()
        finally:
            self.release_conn()

    @property
    def data(self) -> bytes:
        if self._body is None:
            self._body = self._read_all()
            self.release_conn()
        return self._body


# ---------------------------------------------------------------------------
# PoolManager
# ---------------------------------------------------------------------------

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class PoolManager:
    """Stdlib-backed urllib3 ``PoolManager`` substitute.

    Maintains a per-``(scheme, host, port)`` ``deque`` of idle
    :class:`http.client.HTTPConnection` / :class:`HTTPSConnection` instances
    so repeated requests to the same host reuse sockets — the same
    keep-alive guarantee urllib3 provides. Implements the
    ``.request(method, url, body, headers, timeout, preload_content,
    decode_content, redirect)`` shape yggdrasil's HTTP stack calls.

    The retry loop consults the :class:`Retry` object (when supplied)
    exactly like urllib3 does: status retries through ``status_forcelist``,
    network retries on socket/SSL/timeout, status-aware backoff via
    ``get_backoff_time`` / ``get_retry_after``.
    """

    def __init__(
        self,
        num_pools: int = 10,
        maxsize: int = 1,
        block: bool = False,
        retries: Any = None,
        cert_reqs: Optional[str] = None,
        ca_certs: Optional[str] = None,
        timeout: Any = None,
        assert_hostname: Any = None,
        **_: Any,
    ) -> None:
        self.num_pools = num_pools
        self.maxsize = maxsize
        self.block = block
        self.retries = retries
        self.cert_reqs = cert_reqs
        self.ca_certs = ca_certs
        self.timeout = timeout
        self.assert_hostname = assert_hostname
        self._pools: dict[Tuple[str, str, int], collections.deque] = {}
        self._lock = threading.Lock()
        self._closed = False

    # ---- pool plumbing -----------------------------------------------------
    def _get_connection(self, scheme: str, host: str, port: int, connect_timeout: Optional[float]) -> http.client.HTTPConnection:
        if self._closed:
            raise ClosedPoolError(self, "Pool is closed.")
        key = (scheme, host, port)
        with self._lock:
            pool = self._pools.get(key)
            if pool is not None and pool:
                return pool.popleft()
        if scheme == "https":
            ssl_ctx: Optional[ssl.SSLContext]
            if self.cert_reqs == "CERT_NONE":
                ssl_ctx = ssl._create_unverified_context()  # type: ignore[attr-defined]
            else:
                ssl_ctx = ssl.create_default_context(cafile=self.ca_certs)
            conn = http.client.HTTPSConnection(
                host, port=port, timeout=connect_timeout, context=ssl_ctx,
            )
            if self.assert_hostname is False:
                # match urllib3's assert_hostname=False — skip hostname check
                ssl_ctx.check_hostname = False
        else:
            conn = http.client.HTTPConnection(host, port=port, timeout=connect_timeout)
        return conn

    def _release_connection(self, key: Tuple[str, str, int], conn: http.client.HTTPConnection) -> None:
        with self._lock:
            pool = self._pools.setdefault(key, collections.deque())
            if len(pool) < self.maxsize:
                pool.append(conn)
                return
        try:
            conn.close()
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            pools, self._pools = self._pools, {}
        for pool in pools.values():
            while pool:
                try:
                    pool.popleft().close()
                except Exception:
                    pass

    def close(self) -> None:
        self._closed = True
        self.clear()

    # ---- request -----------------------------------------------------------
    def request(
        self,
        method: str,
        url: str,
        body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Any = None,
        preload_content: bool = True,
        decode_content: bool = True,
        redirect: bool = True,
        retries: Any = None,
        **_: Any,
    ) -> HTTPResponse:
        retries_obj = retries if retries is not None else self.retries
        if retries_obj is not None and not isinstance(retries_obj, Retry):
            # urllib3 accepts a bare int for retries — promote it.
            retries_obj = Retry(total=int(retries_obj))
        effective_timeout = timeout if timeout is not None else self.timeout

        # Cap redirects via the retry policy when present.
        max_redirects = 10
        visited_redirects = 0
        current_url = url
        current_method = method
        current_body = body
        current_headers = dict(headers or {})

        while True:
            try:
                response = self._send_once(
                    method=current_method,
                    url=current_url,
                    body=current_body,
                    headers=current_headers,
                    timeout=effective_timeout,
                    preload_content=preload_content,
                    decode_content=decode_content,
                )
            except (socket.timeout, TimeoutError) as exc:
                if retries_obj is None:
                    raise ReadTimeoutError(self, current_url, str(exc)) from exc
                wrapped = ReadTimeoutError(self, current_url, str(exc))
                retries_obj = retries_obj.increment(method=current_method, url=current_url, error=wrapped, _pool=self)
                retries_obj.sleep()
                continue
            except ssl.SSLError as exc:
                raise SSLError(str(exc)) from exc
            except (OSError, http.client.HTTPException) as exc:
                if retries_obj is None:
                    raise NewConnectionError(self, str(exc)) from exc
                wrapped_oserr = NewConnectionError(self, str(exc))
                retries_obj = retries_obj.increment(method=current_method, url=current_url, error=wrapped_oserr, _pool=self)
                retries_obj.sleep()
                continue

            # Redirect handling
            if redirect and response.status in _REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if location and visited_redirects < max_redirects:
                    response.drain_conn()
                    response.release_conn()
                    visited_redirects += 1
                    # Resolve relative locations.
                    if "://" not in location:
                        parts = urlsplit(current_url)
                        location = urlunsplit((parts.scheme, parts.netloc, location, "", "")) if location.startswith("/") else urlunsplit((parts.scheme, parts.netloc, parts.path.rsplit("/", 1)[0] + "/" + location, "", ""))
                    current_url = location
                    if response.status in (301, 302, 303) and current_method.upper() != "HEAD":
                        current_method = "GET"
                        current_body = None
                        current_headers.pop("Content-Length", None)
                        current_headers.pop("Content-Type", None)
                    continue

            # Retry on status_forcelist
            if retries_obj is not None and retries_obj.is_retry(current_method, response.status, response.headers.get("Retry-After") is not None):
                try:
                    next_retries = retries_obj.increment(method=current_method, url=current_url, response=response, _pool=self)
                except MaxRetryError:
                    if retries_obj.raise_on_status:
                        raise
                    return response
                response.drain_conn()
                response.release_conn()
                next_retries.sleep(response=response)
                retries_obj = next_retries
                continue

            return response

    # ---- low-level single send --------------------------------------------
    def _send_once(
        self,
        *,
        method: str,
        url: str,
        body: Any,
        headers: Mapping[str, str],
        timeout: Any,
        preload_content: bool,
        decode_content: bool,
    ) -> HTTPResponse:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            raise LocationParseError(url)
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https"):
            raise LocationValueError(f"Unsupported scheme: {scheme!r}")
        host = parts.hostname or ""
        port = parts.port or (443 if scheme == "https" else 80)
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"

        connect_timeout, read_timeout = _resolve_timeout(timeout)
        key = (scheme, host, port)
        conn = self._get_connection(scheme, host, port, connect_timeout)
        try:
            if read_timeout is not None:
                conn.timeout = read_timeout
            # Stringify headers; http.client expects str values.
            send_headers = {k: str(v) for k, v in headers.items()}
            send_headers.setdefault("Host", f"{host}:{port}" if port not in (80, 443) else host)
            if body is not None and "Content-Length" not in send_headers and not hasattr(body, "read"):
                if isinstance(body, (bytes, bytearray)):
                    send_headers["Content-Length"] = str(len(body))
            conn.request(method, path, body=body, headers=send_headers)
            raw = conn.getresponse()
        except (socket.timeout,) as exc:
            try:
                conn.close()
            except Exception:
                pass
            raise ReadTimeoutError(self, url, str(exc)) from exc
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

        return HTTPResponse(
            raw,
            request_url=url,
            request_method=method,
            decode_content=decode_content,
            preload_content=preload_content,
            pool=self,
            connection=conn,
            pool_key=key,
        )
