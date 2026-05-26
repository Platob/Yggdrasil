"""HTTP exception hierarchy — lives under :mod:`yggdrasil.exceptions`.

Every type defined here subclasses :class:`YGGException` through
:class:`HTTPError`, so a generic ``except YGGException`` catches
every error the library deliberately raises (HTTP-bound included,
alongside :class:`CastError` and any future global type). Keeping
the hierarchy at the package root (and not under ``io/``) means
non-IO modules can ``raise NotFoundError(...)`` / ``raise
ForbiddenError(...)`` / ``raise TooManyRequests(...)`` without
pulling :mod:`yggdrasil.io` first; see ``AGENTS.md`` →
"Centralise exceptions in :mod:`yggdrasil.exceptions`" for the rule.

The hierarchy also:

1. Inherits from the stdlib-backed transport exceptions in
   :mod:`yggdrasil.http_.exceptions` (urllib3-shaped, but stdlib-only)
   so transport-level ``except`` blocks still match across the
   transport/business boundary.
2. Carries a bound ``Response`` (and/or ``PreparedRequest``) on every
   exception so callers never need to thread response objects through
   manually.
3. Mirrors the public transport exceptions that have HTTP-level semantics,
   adding typed ``response`` / ``request`` attributes where applicable.

Hierarchy (transport exceptions sourced from
:mod:`yggdrasil.http_.exceptions`; additions marked with *):
─────────────────────────────────────────────────────
HTTPError                           ← _http_pool.HTTPError
  ├── RequestError *                ← request-bound base
  │     ├── ConnectionError        ← _http_pool.NewConnectionError
  │     ├── TimeoutError           ← _http_pool.TimeoutError
  │     │     ├── ConnectTimeoutError
  │     │     └── ReadTimeoutError
  │     └── ProxyError             ← _http_pool.ProxyError
  │
  ├── ResponseError *              ← response-bound base
  │     ├── HTTPStatusError  *     ← 4xx / 5xx (was HTTPError in response.py)
  │     │     ├── ClientError      ← 4xx
  │     │     │     ├── BadRequest            (400)
  │     │     │     ├── UnauthorizedError     (401)
  │     │     │     ├── ForbiddenError        (403)
  │     │     │     ├── NotFoundError         (404)
  │     │     │     ├── MethodNotAllowed      (405)
  │     │     │     ├── ConflictError         (409)
  │     │     │     ├── GoneError             (410)
  │     │     │     ├── UnprocessableEntity   (422)
  │     │     │     ├── TooManyRequests       (429)
  │     │     └── ServerError      ← 5xx
  │     │           ├── InternalServerError   (500)
  │     │           ├── BadGatewayError       (502)
  │     │           ├── ServiceUnavailable    (503)
  │     │           └── GatewayTimeout        (504)
  │     ├── DecodeError            ← _http_pool.DecodeError
  │     ├── InvalidChunkLength     ← _http_pool.InvalidChunkLength
  │     └── IncompleteRead         ← _http_pool.IncompleteRead
  │
  ├── PoolError *                  ← _http_pool.PoolError
  │     ├── ClosedPoolError        ← _http_pool.ClosedPoolError
  │     ├── EmptyPoolError         ← _http_pool.EmptyPoolError
  │     └── HostChangedError       ← _http_pool.HostChangedError
  │
  ├── LocationError *
  │     ├── LocationValueError     ← _http_pool.LocationValueError
  │     └── LocationParseError     ← _http_pool.LocationParseError
  │
  ├── SecurityWarning (Warning)    ← _http_pool.SecurityWarning
  ├── InsecureRequestWarning       ← _http_pool.InsecureRequestWarning
  ├── SSLError                     ← _http_pool.SSLError
  └── CacheError *                 ← cache backend failures
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.http_.exceptions import exceptions as _u3

from .base import YGGException

if TYPE_CHECKING:
    from yggdrasil.http_.request import PreparedRequest
    from yggdrasil.http_.response import Response
    from starlette.responses import JSONResponse as StarletteResponse
    from fastapi.responses import JSONResponse as FastAPIJSONResponse

__all__ = [
    # Base
    "HTTPError",
    # Request-bound
    "RequestError",
    "AuthRequiredError",
    "ConnectionError",
    "TimeoutError",
    "ConnectTimeoutError",
    "ReadTimeoutError",
    "ProxyError",
    # Response-bound
    "ResponseError",
    "HTTPStatusError",
    "ClientError",
    "BadRequest",
    "UnauthorizedError",
    "ForbiddenError",
    "NotFoundError",
    "MethodNotAllowed",
    "ConflictError",
    "GoneError",
    "UnprocessableEntity",
    "TooManyRequests",
    "ServerError",
    "InternalServerError",
    "BadGatewayError",
    "ServiceUnavailable",
    "GatewayTimeout",
    "DecodeError",
    "InvalidChunkLength",
    "IncompleteRead",
    # Pool
    "PoolError",
    "ClosedPoolError",
    "EmptyPoolError",
    "HostChangedError",
    # Location
    "LocationError",
    "LocationValueError",
    "LocationParseError",
    # Security / SSL
    "SecurityWarning",
    "InsecureRequestWarning",
    "SSLError",
    # Cache
    "CacheError",
    # Factory
    "make_for_status",
    "from_urllib3",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reason_phrase(status_code: int) -> str:
    try:
        from http import HTTPStatus
        return HTTPStatus(status_code).phrase
    except Exception:
        return ""


def _body_snippet(response: "Response", max_bytes: int = 2048) -> str:
    try:
        raw = response.buffer.to_bytes() if response.buffer else b""
        if not raw:
            return ""
        from yggdrasil.http_.response import _get_charset
        charset = _get_charset(response.headers)
        text = raw[:max_bytes].decode(charset, errors="replace").strip()
        suffix = "…" if len(raw) > max_bytes else ""
        return f"\nResponse body ({min(len(raw), max_bytes)} bytes){suffix}:\n{text}"
    except Exception:
        return ""


def _fmt(response: "Response") -> str:
    status = response.status_code
    reason = _reason_phrase(status)
    method = response.request.method
    url = response.request.url.to_string()
    base = f"{status}{(' ' + reason) if reason else ''} for {method} {url}"
    return base + _body_snippet(response)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class HTTPError(YGGException, _u3.HTTPError):
    """
    Root of the yggdrasil HTTP exception hierarchy.

    Inherits from :class:`urllib3.exceptions.HTTPError` so all
    urllib3-aware catch blocks match, AND from
    :class:`~yggdrasil.exceptions.YGGException` so the library-wide
    ``except YGGException`` catches every HTTP failure too.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else repr(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.args[0]!r})" if self.args else f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Request-bound errors (no response available yet)
# ---------------------------------------------------------------------------

class RequestError(HTTPError):
    """
    Base for errors that occur before/during sending a request,
    when no Response is available yet.
    """

    def __init__(self, message: str, *, request: "PreparedRequest"):
        super().__init__(message)
        self.request = request

    def __repr__(self) -> str:
        url = self.request.url.to_string() if self.request else "?"
        return f"{type(self).__name__}(url={url!r})"


class AuthRequiredError(RequestError):
    """
    Raised when an operation needs an :class:`Authorization` handler
    but none is bound — either on the request or on the session.

    Fired by :meth:`Session.refresh_auth` when ``force=True`` (the
    default) and the caller has not configured any auth source. The
    silent no-op branch is reserved for ``force=False``, which is what
    :meth:`Session.prepare_request_before_send` uses on steady-state
    sends (a request to a public endpoint shouldn't fail just because
    the session doesn't have a token to refresh).
    """


class ConnectionError(RequestError, _u3.NewConnectionError):  # type: ignore[misc]
    """Failed to open a connection to the host."""

    def __init__(self, message: str, *, request: "PreparedRequest", pool: Any = None):
        # urllib3.NewConnectionError expects (pool, message)
        _u3.NewConnectionError.__init__(self, pool or object(), message)
        RequestError.__init__(self, message, request=request)


class TimeoutError(RequestError, _u3.TimeoutError):  # type: ignore[misc]
    """Base for all timeout variants."""

    def __init__(
        self,
        message: str,
        *,
        request: "PreparedRequest",
        timeout: float | None = None,
    ):
        super().__init__(message, request=request)
        self.timeout = timeout


class ConnectTimeoutError(TimeoutError, _u3.ConnectTimeoutError):  # type: ignore[misc]
    """Timed out while establishing the TCP connection."""


class ReadTimeoutError(TimeoutError, _u3.ReadTimeoutError):  # type: ignore[misc]
    """Timed out while reading the response body."""

    def __init__(
        self,
        message: str,
        *,
        request: "PreparedRequest",
        timeout: float | None = None,
        pool: Any = None,
        url: str | None = None,
    ):
        _u3.ReadTimeoutError.__init__(self, pool or object(), url or "", message)
        TimeoutError.__init__(self, message, request=request, timeout=timeout)


class ProxyError(RequestError, _u3.ProxyError):  # type: ignore[misc]
    """Error communicating through a proxy."""

    def __init__(
        self,
        message: str,
        *,
        request: "PreparedRequest",
        original_error: Optional[Exception] = None,
    ):
        _u3.ProxyError.__init__(self, message, original_error)
        RequestError.__init__(self, message, request=request)
        self.original_error = original_error


# ---------------------------------------------------------------------------
# Response-bound errors (response is available)
# ---------------------------------------------------------------------------

class ResponseError(HTTPError):
    """
    Base for errors where a Response has been received.
    Always carries both ``response`` and ``request`` (via response.request).
    """

    def __init__(self, message: str, *, response: "Response"):
        super().__init__(message)
        self.response = response

    @property
    def request(self) -> "PreparedRequest":
        return self.response.request

    @property
    def status_code(self) -> int:
        return self.response.status_code

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"status_code={self.status_code}, "
            f"url={self.response.request.url.to_string()!r}"
            f")"
        )

    # ------------------------------------------------------------------
    # ASGI / Starlette / FastAPI
    # ------------------------------------------------------------------

    def to_starlette(self) -> "StarletteResponse":
        """
        Convert to a ``starlette.responses.JSONResponse`` suitable for
        returning directly from a Starlette or FastAPI exception handler.

        The JSON body follows the RFC 7807 Problem Details shape::

            {
                "status":  404,
                "error":   "Not Found",
                "detail":  "<exception message>",
                "path":    "/v1/contents/42/data",
                "method":  "GET"
            }

        ``retry_after`` is appended (and the ``Retry-After`` header set)
        when the exception carries that attribute (429 / 503).
        """
        from starlette.responses import JSONResponse as _JSONResponse

        status  = self.status_code
        reason  = _reason_phrase(status)
        method  = self.response.request.method
        path    = self.response.request.url.to_string()

        body: dict = {
            "status":  status,
            "error":   reason or type(self).__name__,
            "detail":  str(self),
            "path":    path,
            "method":  method,
        }

        headers: dict[str, str] = {}

        retry_after: float | None = getattr(self, "retry_after", None)
        if retry_after is not None:
            body["retry_after"] = retry_after
            headers["Retry-After"] = str(int(retry_after))

        return _JSONResponse(
            content=body,
            status_code=status,
            headers=headers or None,
        )

    def to_fastapi(self) -> "FastAPIJSONResponse":
        """
        Convert to a ``fastapi.responses.JSONResponse``.

        FastAPI's ``JSONResponse`` is a subclass of Starlette's, but
        returning the FastAPI type keeps FastAPI's exception handler
        machinery (background tasks, middleware hooks, OpenAPI error
        modelling) working correctly.

        Falls back to ``to_starlette()`` transparently when FastAPI is
        not installed.
        """
        try:
            from fastapi.responses import JSONResponse as _FastAPIJSONResponse
        except ImportError:
            return self.to_starlette()  # type: ignore[return-value]

        status  = self.status_code
        reason  = _reason_phrase(status)
        method  = self.response.request.method
        path    = self.response.request.url.to_string()

        body: dict = {
            "status":  status,
            "error":   reason or type(self).__name__,
            "detail":  str(self),
            "path":    path,
            "method":  method,
        }

        headers: dict[str, str] = {}

        retry_after: float | None = getattr(self, "retry_after", None)
        if retry_after is not None:
            body["retry_after"] = retry_after
            headers["Retry-After"] = str(int(retry_after))

        return _FastAPIJSONResponse(
            content=body,
            status_code=status,
            headers=headers or None,
        )

    # ------------------------------------------------------------------
    # urllib3 HTTPResponse shim — alias for the bound :class:`HTTPResponse`
    # ------------------------------------------------------------------

    @property
    def urllib3_response(self):
        """Return the bound :class:`~yggdrasil.http_.HTTPResponse`.

        After the pool / response merge, the high-level
        :class:`HTTPResponse` carries the full urllib3-shaped surface
        (``.status`` / ``.headers`` / ``.read`` / ``.stream`` /
        ``.release_conn`` / ``.drain_conn`` / ``.data``) directly, so
        the explicit shim is gone — the response IS the shim.
        """
        return self.response


# ---------------------------------------------------------------------------
# HTTP Status errors  (4xx / 5xx)
# ---------------------------------------------------------------------------

class HTTPStatusError(ResponseError):
    """
    Raised when the server returns a 4xx or 5xx status code.

    This is the direct replacement for the old ``HTTPError`` in response.py.
    ``raise_for_status()`` should raise this (or a more specific subclass).
    """


class ClientError(HTTPStatusError):
    """4xx — client-side errors."""


class ServerError(HTTPStatusError):
    """5xx — server-side errors."""


# ---- 4xx ----------------------------------------------------------------

class BadRequest(ClientError):
    """400 Bad Request."""


class UnauthorizedError(ClientError):
    """401 Unauthorized."""


class ForbiddenError(ClientError):
    """403 Forbidden."""


class NotFoundError(ClientError):
    """404 Not Found."""


class MethodNotAllowed(ClientError):
    """405 Method Not Allowed."""


class ConflictError(ClientError):
    """409 Conflict."""


class GoneError(ClientError):
    """410 Gone."""


class UnprocessableEntity(ClientError):
    """422 Unprocessable Entity."""


class TooManyRequests(ClientError):
    """
    429 Too Many Requests.

    Carries ``retry_after`` (seconds) when the upstream Retry-After header
    is present and parseable.
    """

    def __init__(self, message: str, *, response: "Response"):
        super().__init__(message, response=response)
        self.retry_after: float | None = _parse_retry_after(response)


# ---- 5xx ----------------------------------------------------------------

class InternalServerError(ServerError):
    """500 Internal Server Error."""


class BadGatewayError(ServerError):
    """502 Bad Gateway."""


class ServiceUnavailable(ServerError):
    """
    503 Service Unavailable.

    Carries ``retry_after`` when present.
    """

    def __init__(self, message: str, *, response: "Response"):
        super().__init__(message, response=response)
        self.retry_after: float | None = _parse_retry_after(response)


class GatewayTimeout(ServerError):
    """504 Gateway Timeout."""


# ---- Response decode/integrity errors -----------------------------------

class DecodeError(ResponseError, _u3.DecodeError):  # type: ignore[misc]
    """Failed to decode the response body (e.g. bad gzip/zstd stream)."""

    def __init__(self, message: str, *, response: "Response"):
        _u3.DecodeError.__init__(self, message)
        ResponseError.__init__(self, message, response=response)


class InvalidChunkLength(ResponseError, _u3.InvalidChunkLength):  # type: ignore[misc]
    """Received a chunked-encoding frame with an invalid length header."""

    def __init__(self, message: str, *, response: "Response", length: int = 0):
        # urllib3.InvalidChunkLength(response, length)
        _u3.InvalidChunkLength.__init__(self, response.urllib3_response if hasattr(response, 'urllib3_response') else None, length)  # type: ignore[arg-type]
        ResponseError.__init__(self, message, response=response)
        self.length = length


class IncompleteRead(ResponseError, _u3.IncompleteRead):  # type: ignore[misc]
    """
    The server closed the connection before sending the full body.
    ``partial`` is the bytes received so far; ``expected`` is the
    Content-Length (None if unknown).
    """

    def __init__(
        self,
        message: str,
        *,
        response: "Response",
        partial: int = 0,
        expected: int | None = None,
    ):
        _u3.IncompleteRead.__init__(self, partial, expected)
        ResponseError.__init__(self, message, response=response)
        self.partial = partial
        self.expected = expected


# ---------------------------------------------------------------------------
# Pool errors
# ---------------------------------------------------------------------------

class PoolError(HTTPError):
    """Base for connection-pool errors (no request/response yet)."""


class ClosedPoolError(PoolError, _u3.ClosedPoolError):  # type: ignore[misc]
    """Attempted to use a connection pool that has been closed."""

    def __init__(self, message: str, *, pool: Any = None):
        _u3.ClosedPoolError.__init__(self, pool or object(), message)
        PoolError.__init__(self, message)


class EmptyPoolError(PoolError, _u3.EmptyPoolError):  # type: ignore[misc]
    """No connections available in the pool and ``block=True``."""

    def __init__(self, message: str, *, pool: Any = None):
        _u3.EmptyPoolError.__init__(self, pool or object(), message)
        PoolError.__init__(self, message)


class HostChangedError(PoolError, _u3.HostChangedError):  # type: ignore[misc]
    """Request was made to a different host than the pool was created for."""

    def __init__(self, message: str, *, pool: Any = None, url: str = "", retries: int = 0):
        _u3.HostChangedError.__init__(self, pool or object(), url, retries)
        PoolError.__init__(self, message)


# ---------------------------------------------------------------------------
# Location errors
# ---------------------------------------------------------------------------

class LocationError(HTTPError):
    """Base for URL / location parse errors."""


class LocationValueError(LocationError, _u3.LocationValueError):  # type: ignore[misc]
    """A required location component (host, port …) is missing or invalid."""

    def __init__(self, message: str):
        _u3.LocationValueError.__init__(self, message)
        LocationError.__init__(self, message)


class LocationParseError(LocationError, _u3.LocationParseError):  # type: ignore[misc]
    """The URL string could not be parsed."""

    def __init__(self, message: str, *, location: str = ""):
        _u3.LocationParseError.__init__(self, location or message)
        LocationError.__init__(self, message)
        self.location = location


# ---------------------------------------------------------------------------
# Security / SSL
# ---------------------------------------------------------------------------

class SSLError(ResponseError, _u3.SSLError):  # type: ignore[misc]
    """TLS/SSL handshake or certificate validation failure."""

    def __init__(self, message: str, *, response: "Response"):
        _u3.SSLError.__init__(self, message)
        ResponseError.__init__(self, message, response=response)


try:
    class SecurityWarning(Warning, _u3.SecurityWarning):  # type: ignore[misc]
        """Issued for insecure connections or weak TLS configurations."""
except TypeError:
    class SecurityWarning(Warning):  # type: ignore[misc]
        """Issued for insecure connections or weak TLS configurations."""


class InsecureRequestWarning(SecurityWarning, _u3.InsecureRequestWarning):  # type: ignore[misc]
    """Issued when a request is made to an HTTPS URL without cert verification."""

# ---------------------------------------------------------------------------
# Cache errors (no urllib3 equivalent)
# ---------------------------------------------------------------------------

class CacheError(HTTPError):
    """
    Raised when a cache backend (e.g. Databricks Delta table) fails to
    read or write a cached response.
    """

    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause

    def __repr__(self) -> str:
        cause_str = f", cause={self.cause!r}" if self.cause else ""
        return f"{type(self).__name__}({self.args[0]!r}{cause_str})"


# ---------------------------------------------------------------------------
# Status-code → exception class dispatch table
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[int, type[HTTPStatusError]] = {
    400: BadRequest,
    401: UnauthorizedError,
    403: ForbiddenError,
    404: NotFoundError,
    405: MethodNotAllowed,
    409: ConflictError,
    410: GoneError,
    422: UnprocessableEntity,
    429: TooManyRequests,
    500: InternalServerError,
    502: BadGatewayError,
    503: ServiceUnavailable,
    504: GatewayTimeout,
}


def _status_cls(status_code: int) -> type[HTTPStatusError]:
    """Return the most specific exception class for a given status code."""
    if status_code in _STATUS_MAP:
        return _STATUS_MAP[status_code]
    if 400 <= status_code < 500:
        return ClientError
    if 500 <= status_code < 600:
        return ServerError
    return HTTPStatusError


def _parse_retry_after(response: "Response") -> Optional[float]:
    """
    Parse the Retry-After header from a response.
    Returns seconds as float, or None if absent / unparseable.
    """
    for k, v in (response.headers or {}).items():
        if k.lower() == "retry-after":
            v = v.strip()
            # Could be an integer number of seconds or an HTTP-date.
            try:
                return float(v)
            except ValueError:
                pass
            try:
                from email.utils import parsedate_to_datetime
                import datetime as _dt
                delta = parsedate_to_datetime(v) - _dt.datetime.now(_dt.timezone.utc)
                return max(0.0, delta.total_seconds())
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def make_for_status(response: "Response", *, max_body: int = 2048) -> Optional["HTTPError"]:
    """
    Raise the most specific ``HTTPStatusError`` subclass for 4xx/5xx responses.
    No-op for 1xx/2xx/3xx.

    Replaces ``Response.raise_for_status()`` — call this as a free function
    or keep the method as a thin wrapper.

    Example
    -------
    >>> error = make_for_status(response)
    """
    if 200 <= response.status_code < 400:
        return

    cls = _status_cls(response.status_code)
    msg = _fmt(response)
    return cls(msg, response=response)


def from_urllib3(
    exc: _u3.HTTPError,
    *,
    request: Optional["PreparedRequest"] = None,
    response: Optional["Response"] = None,
) -> "HTTPError":
    """
    Wrap an arbitrary ``urllib3.exceptions.HTTPError`` in the yggdrasil
    hierarchy, preserving the original as ``__cause__``.

    Useful in HTTP adapter layers that catch raw urllib3 errors and need to
    re-raise them with richer context.

    Example
    -------
    >>> try:
    ...     pool.urlopen(...)
    ... except urllib3.exceptions.TimeoutError as e:
    ...     raise from_urllib3(e, request=req) from e
    """
    message = str(exc)

    # Response-bound
    if response is not None:
        if isinstance(exc, _u3.DecodeError):
            return DecodeError(message, response=response)
        if isinstance(exc, _u3.IncompleteRead):
            return IncompleteRead(message, response=response)
        wrapped: ResponseError = ResponseError(message, response=response)
        wrapped.__cause__ = exc
        return wrapped

    # Request-bound
    if request is not None:
        if isinstance(exc, _u3.ConnectTimeoutError):
            return ConnectTimeoutError(message, request=request)
        if isinstance(exc, _u3.ReadTimeoutError):
            return ReadTimeoutError(message, request=request)
        if isinstance(exc, _u3.TimeoutError):
            return TimeoutError(message, request=request)
        if isinstance(exc, _u3.ProxyError):
            return ProxyError(message, request=request, original_error=exc)
        if isinstance(exc, _u3.NewConnectionError):
            return ConnectionError(message, request=request)
        req_wrapped: RequestError = RequestError(message, request=request)
        req_wrapped.__cause__ = exc
        return req_wrapped

    # Pool / location — no request or response
    if isinstance(exc, _u3.ClosedPoolError):
        return ClosedPoolError(message)
    if isinstance(exc, _u3.EmptyPoolError):
        return EmptyPoolError(message)
    if isinstance(exc, _u3.HostChangedError):
        return HostChangedError(message)
    if isinstance(exc, _u3.LocationValueError):
        return LocationValueError(message)
    if isinstance(exc, _u3.LocationParseError):
        return LocationParseError(message)
    if isinstance(exc, _u3.SSLError):
        # SSLError without a response — fall back to root
        fallback = HTTPError(message)
        fallback.__cause__ = exc
        return fallback

    # Unknown urllib3 error
    fallback = HTTPError(message)
    fallback.__cause__ = exc
    return fallback