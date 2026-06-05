"""Subclassable retry policy — :class:`Retry` + :class:`RequestHistory`.

Mirrors the surface :class:`yggdrasil.http_.session._TieredRetry` reaches
into (``BACKOFF_MAX``, ``history``, ``get_backoff_time``,
``get_retry_after``, ``increment``, ``new``, ``is_exhausted``,
``sleep``). The :class:`yggdrasil.http_.session.HTTPSession` retry
loop drives this in its wire-send path.
"""
from __future__ import annotations

import time
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Iterable, NamedTuple, Optional, Tuple

from .exceptions import (
    ConnectTimeoutError,
    MaxRetryError,
    ProtocolError,
    ReadTimeoutError,
)

if TYPE_CHECKING:
    from .response import HTTPResponse


__all__ = ["Retry", "RequestHistory"]


class RequestHistory(NamedTuple):
    """Single retry-attempt record. Mirrors ``urllib3.util.retry.RequestHistory``."""

    method: Optional[str]
    url: Optional[str]
    error: Optional[Exception]
    status: Optional[int]
    redirect_location: Optional[str]


class Retry:
    """Minimal subclassable retry policy mirroring ``urllib3.Retry``."""

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
            self.BACKOFF_MAX = backoff_max  # type: ignore[misc]
        self.raise_on_redirect = raise_on_redirect
        self.raise_on_status = raise_on_status
        self.history = tuple(history)
        self.respect_retry_after_header = respect_retry_after_header
        self.remove_headers_on_redirect = (
            frozenset(h.lower() for h in remove_headers_on_redirect)
            if remove_headers_on_redirect
            else frozenset()
        )

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
        consecutive_errors = [h for h in reversed(self.history) if h.redirect_location is None]
        if len(consecutive_errors) <= 1:
            return 0.0
        backoff = self.backoff_factor * (2 ** (len(consecutive_errors) - 1))
        return float(min(self.BACKOFF_MAX, backoff))

    def get_retry_after(self, response: "HTTPResponse") -> Optional[float]:
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

    def sleep(self, response: "Optional[HTTPResponse]" = None) -> None:
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
        retry_counts = [
            c for c in (self.total, self.connect, self.read, self.redirect, self.status, self.other)
            if c is not None
        ]
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
        response: "Optional[HTTPResponse]" = None,
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
