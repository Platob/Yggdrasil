"""Shared helpers across the Databricks path/IO subclasses.

Pulled out of the per-class files so the SDK-error tuples and the
mtime coercion helper aren't duplicated four times.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from databricks.sdk.errors import InternalError
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
    "coerce_mtime",
]


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