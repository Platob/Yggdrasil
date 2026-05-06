"""Generic retry wrapper for remote-path SDK calls.

Two categories of transient failure are worth retrying:

* **Transient backend errors** (InternalError, BadRequest, 500/503,
  socket timeouts) — retried up to 4 times with incremental sleep
  (1 s, 2 s, 4 s, 8 s).
* **Permission errors** — retried once with a 1 s sleep, in case the
  SDK was racing a credential refresh / token mint. Anything beyond
  that is the caller's misconfiguration; we surface the error.

Anything else (NotFound, AlreadyExists, FileNotFoundError,
ValueError, …) is **not** retried — those are deterministic.

This module deliberately avoids importing the boto / databricks SDK
exception classes; it duck-types on the exception's class name and
the optional ``response`` attribute that boto-style errors carry.
That keeps the retry layer usable with mocks in tests.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar


__all__ = ["retry_sdk_call", "is_transient", "is_permission"]


LOGGER = logging.getLogger(__name__)


_T = TypeVar("_T")


#: Class names that signal transient backend errors. Keep in sync
#: with the SDKs that hit this path: boto3 surfaces these via
#: ``ClientError`` so we duck-type on the response code instead.
_TRANSIENT_NAMES = frozenset({
    "InternalError", "BadRequest", "ServerError", "ServiceUnavailable",
    "ThrottlingException", "RequestTimeout", "RequestTimeoutException",
    "ConnectionError", "EndpointConnectionError",
    "ReadTimeoutError", "ReadTimeout", "TooManyRequests", "RetryError",
})

#: Class names that signal permission/auth errors. One retry for
#: credential-refresh races; anything beyond that is a real config
#: problem.
_PERMISSION_NAMES = frozenset({
    "PermissionDenied", "AccessDenied", "AccessDeniedException",
    "Forbidden", "Unauthorized", "InvalidAccessKeyId",
    "ExpiredToken", "ExpiredTokenException", "TokenExpired",
    "AuthenticationError", "Unauthenticated",
})

#: HTTP status codes treated as transient (covers 5xx + 429).
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})

#: HTTP status codes treated as permission errors.
_PERMISSION_STATUSES = frozenset({401, 403})

#: AWS / SDK error codes treated as transient.
_TRANSIENT_CODES = frozenset({
    "InternalError", "InternalServerError", "BadRequest",
    "RequestTimeout", "ThrottlingException", "Throttling",
    "ServiceUnavailable", "SlowDown", "TooManyRequests",
})

#: AWS / SDK error codes treated as permission errors.
_PERMISSION_CODES = frozenset({
    "AccessDenied", "Forbidden", "InvalidAccessKeyId",
    "ExpiredToken", "Unauthorized",
})


def _http_status(exc: BaseException) -> "int | None":
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        meta = response.get("ResponseMetadata") or {}
        status = meta.get("HTTPStatusCode")
        if isinstance(status, int):
            return status
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if isinstance(code, str):
            return code
    return ""


def is_transient(exc: BaseException) -> bool:
    """True when *exc* should be retried as a transient backend error."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    if _http_status(exc) in _TRANSIENT_STATUSES:
        return True
    if _error_code(exc) in _TRANSIENT_CODES:
        return True
    return False


def is_permission(exc: BaseException) -> bool:
    """True when *exc* looks like an auth / permission failure."""
    if type(exc).__name__ in _PERMISSION_NAMES:
        return True
    if _http_status(exc) in _PERMISSION_STATUSES:
        return True
    if _error_code(exc) in _PERMISSION_CODES:
        return True
    return False


def retry_sdk_call(
    func: Callable[..., _T],
    *args: Any,
    max_transient_retries: int = 4,
    max_permission_retries: int = 1,
    base_sleep: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> _T:
    """Call *func(*args, **kwargs)* with the Databricks/AWS retry policy.

    Sleep schedule for transient errors: ``base_sleep * 2**attempt``
    seconds (1, 2, 4, 8 by default). Permission errors get exactly
    one *base_sleep* nap before retrying once.

    The *sleep* callable is injected so tests can pass a no-op or a
    spy. The default is :func:`time.sleep`.

    Non-transient, non-permission errors propagate immediately —
    that includes :class:`FileNotFoundError`, :class:`ValueError`,
    :class:`KeyError`, and any custom error type the caller wants
    to surface deterministically.
    """
    transient_attempt = 0
    permission_attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except BaseException as exc:
            if is_transient(exc):
                if transient_attempt >= max_transient_retries:
                    raise
                wait = base_sleep * (2 ** transient_attempt)
                LOGGER.info(
                    "Transient error from %s (attempt %d/%d); sleeping %.1fs: %s",
                    getattr(func, "__name__", func),
                    transient_attempt + 1, max_transient_retries, wait, exc,
                )
                sleep(wait)
                transient_attempt += 1
                continue
            if is_permission(exc):
                if permission_attempt >= max_permission_retries:
                    raise
                LOGGER.info(
                    "Permission error from %s (attempt %d/%d); sleeping %.1fs: %s",
                    getattr(func, "__name__", func),
                    permission_attempt + 1, max_permission_retries, base_sleep, exc,
                )
                sleep(base_sleep)
                permission_attempt += 1
                continue
            raise
