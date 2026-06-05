"""Stdlib-only HTTP exception hierarchy + ``exceptions`` namespace.

Mirrors the ``urllib3.exceptions`` shape so the transport layer
:mod:`yggdrasil.http_.session` and the public :mod:`yggdrasil.exceptions.http`
hierarchy can reuse the same base classes without taking a urllib3
runtime dependency. :func:`disable_warnings` is bundled here because
its only argument is one of the :class:`SecurityWarning` subclasses
defined alongside.
"""
from __future__ import annotations

import http.client
import types
import warnings
from typing import Any, Optional


__all__ = [
    "HTTPError",
    "PoolError",
    "RequestError",
    "SSLError",
    "ProxyError",
    "DecodeError",
    "ProtocolError",
    "ConnectionError",
    "NewConnectionError",
    "TimeoutError",
    "ConnectTimeoutError",
    "ReadTimeoutError",
    "MaxRetryError",
    "ClosedPoolError",
    "EmptyPoolError",
    "HostChangedError",
    "LocationValueError",
    "LocationParseError",
    "InvalidChunkLength",
    "IncompleteRead",
    "SecurityWarning",
    "InsecureRequestWarning",
    "exceptions",
    "disable_warnings",
]


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
        http.client.IncompleteRead.__init__(
            self,
            partial=b"" if isinstance(partial, int) else partial,
            expected=expected,
        )


class SecurityWarning(Warning):
    pass


class InsecureRequestWarning(SecurityWarning):
    pass


# Bundle the exception classes into a module-like namespace mirroring
# ``urllib3.exceptions`` so ``from yggdrasil.http_.exceptions import exceptions``
# followed by ``exceptions.HTTPError`` keeps the urllib3-shaped surface
# every external caller already targets.
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


def disable_warnings(category: type = SecurityWarning) -> None:
    """Filter out warnings of ``category`` (default :class:`SecurityWarning`)."""
    warnings.filterwarnings("ignore", category=category)
