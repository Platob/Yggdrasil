"""Shared helpers across the Databricks path/IO subclasses.

Pulled out of the per-class files so the SDK-error tuples and the
mtime coercion helper aren't duplicated four times.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

from databricks.sdk.errors import (
    DeadlineExceeded,
    InternalError,
    OperationTimeout,
    RequestLimitExceeded,
    TemporarilyUnavailable,
    TooManyRequests,
)
from databricks.sdk.errors.platform import (
    AlreadyExists,
    BadRequest,
    NotFound,
    ResourceAlreadyExists,
    ResourceDoesNotExist,
)


__all__ = [
    "NOT_FOUND_ERRORS",
    "ALREADY_EXISTS_ERRORS",
    "SDK_ERRORS",
    "TRANSIENT_ERRORS",
    "retry_sdk_call",
    "coerce_mtime",
]


LOGGER = logging.getLogger(__name__)


# ``requests`` and ``urllib3`` ship with the Databricks SDK; import
# softly so a stripped-down install (or a test that replaces the SDK
# transport) doesn't blow up the path module on import.
_NETWORK_ERROR_TYPES: list[Type[BaseException]] = []
try:
    from requests.exceptions import (
        ChunkedEncodingError as _ReqChunkedEncodingError,
        ConnectionError as _ReqConnectionError,
        ReadTimeout as _ReqReadTimeout,
        Timeout as _ReqTimeout,
    )
    _NETWORK_ERROR_TYPES.extend([
        _ReqReadTimeout, _ReqTimeout,
        _ReqConnectionError, _ReqChunkedEncodingError,
    ])
except ImportError:
    pass

try:
    from urllib3.exceptions import (
        ProtocolError as _U3ProtocolError,
        ReadTimeoutError as _U3ReadTimeoutError,
    )
    _NETWORK_ERROR_TYPES.extend([_U3ReadTimeoutError, _U3ProtocolError])
except ImportError:
    pass


# Errors that mean "the remote has no such object yet."
NOT_FOUND_ERRORS = (NotFound, ResourceDoesNotExist)

# Errors that mean "the thing already exists." Different SDK
# versions raise different shapes — fold the variants together so
# subclasses can ``except ALREADY_EXISTS_ERRORS:`` once and not have
# to track which API surface uses which class.
ALREADY_EXISTS_ERRORS = (AlreadyExists, ResourceAlreadyExists)

# Catch-all for transient + missing errors that subclasses treat as
# "fall through to the directory-listing probe" or "swallow under
# allow_not_found=True". Includes ``BadRequest`` because the
# Workspace/DBFS APIs return 400 (not 404) for many missing-resource
# cases. ``InternalError`` joins because some SDK paths surface
# transient platform issues that are also worth retrying as a
# missing-fallback.
SDK_ERRORS = NOT_FOUND_ERRORS + (BadRequest, InternalError)


# Errors that justify a retry of an SDK call. Three families:
#   - Python builtins for transport: ``TimeoutError``,
#     ``ConnectionError``, ``OSError`` (covers BrokenPipeError and
#     similar low-level socket failures).
#   - ``requests`` / ``urllib3`` exceptions raised inside the SDK's
#     HTTP client — most importantly ``requests.ReadTimeout``, which
#     is a subclass of ``OSError`` but is *not* a Python
#     ``TimeoutError`` and won't be caught by code that only handles
#     the builtin.
#   - Databricks SDK platform errors that signal a server-side
#     transient: 5xx (``InternalError``), 429 (``TooManyRequests`` /
#     ``RequestLimitExceeded``), 503 (``TemporarilyUnavailable``),
#     504 (``DeadlineExceeded``), and the SDK's own
#     ``OperationTimeout``.
TRANSIENT_ERRORS: Tuple[Type[BaseException], ...] = tuple(
    dict.fromkeys(  # de-dup while preserving order
        [
            TimeoutError,
            ConnectionError,
            *_NETWORK_ERROR_TYPES,
            DeadlineExceeded,
            InternalError,
            OperationTimeout,
            RequestLimitExceeded,
            TemporarilyUnavailable,
            TooManyRequests,
        ]
    )
)


T = TypeVar("T")


def retry_sdk_call(
    fn: Callable[..., T],
    *args: Any,
    tries: int = 5,
    delay: float = 0.5,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    transient: Tuple[Type[BaseException], ...] = TRANSIENT_ERRORS,
    logger: Optional[logging.Logger] = None,
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    **kwargs: Any,
) -> T:
    """Call ``fn(*args, **kwargs)`` with retry on transient errors.

    Catches network/transport flakes (``requests.ReadTimeout``,
    ``urllib3.ReadTimeoutError``, builtin ``TimeoutError`` /
    ``ConnectionError``) plus Databricks SDK transient platform
    errors (5xx, 429, 503, 504, ``OperationTimeout``). Anything not
    in ``transient`` propagates immediately so that semantic errors
    — ``NotFound``, ``BadRequest``, ``PermissionDenied``,
    ``AlreadyExists`` — are not retried.

    The defaults give five attempts with exponential backoff capped
    at 30s (0.5, 1.0, 2.0, 4.0s between attempts), tuned for the
    60s SDK read timeout.

    ``on_retry``, if provided, is invoked just before each retry
    with ``(attempt, last_exc)``. Upload paths use this hook to
    seek a streaming :class:`BytesIO` payload back to the original
    position so the next attempt replays the same bytes.
    """
    if tries < 1:
        raise ValueError("tries must be >= 1")
    log = logger or LOGGER
    sleep_for = delay
    last_exc: BaseException | None = None
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except transient as exc:
            last_exc = exc
            if attempt >= tries:
                log.warning(
                    "SDK call %s failed after %d attempts: %r",
                    getattr(fn, "__qualname__", fn),
                    attempt,
                    exc,
                )
                raise
            log.info(
                "SDK call %s transient error on attempt %d/%d (%r); "
                "retrying in %.2fs",
                getattr(fn, "__qualname__", fn),
                attempt,
                tries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
            if on_retry is not None:
                on_retry(attempt, exc)
            sleep_for = min(sleep_for * backoff, max_delay)
    # Unreachable — the loop either returns or raises.
    raise last_exc  # type: ignore[misc]


def coerce_mtime(value: Any) -> Optional[float]:
    """Normalize an SDK-returned mtime into seconds-since-epoch.

    SDK responses vary across services — DBFS returns ms-int,
    Workspace returns ms-int, Files API returns RFC-2822 strings or
    datetimes depending on field. Funnel everything through here so
    the per-class :meth:`_refresh_stat` impls don't each grow their
    own parser.
    """
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # Heuristic: anything past ~year 2286 in seconds is almost
    # certainly a millisecond value. Most Databricks APIs return ms.
    if v > 10_000_000_000:
        v /= 1000.0
    return v