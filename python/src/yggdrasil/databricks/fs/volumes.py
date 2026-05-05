"""Volume-level helpers: status probes and UC metadata reads."""
from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
from typing import Tuple, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    NotFound,
    ResourceDoesNotExist,
)

from ._errors import TRANSIENT_ERRORS, retry_sdk_call

__all__ = [
    "get_volume_status",
    "get_volume_metadata",
]

_NOT_FOUND = (NotFound, ResourceDoesNotExist)
# Transient retry set: builtin TimeoutError/ConnectionError plus
# requests.ReadTimeout, urllib3 ReadTimeoutError, SDK 5xx/429/503/504.
# BadRequest and PermissionDenied always raise — never caught.
_RETRY = TRANSIENT_ERRORS


def _call(fn, *args, **kwargs):
    """Retry transient errors with exponential backoff; fatal errors propagate."""
    return retry_sdk_call(
        fn, *args,
        tries=5, delay=0.25, backoff=2.0, max_delay=8.0,
        transient=_RETRY,
        **kwargs,
    )


def get_volume_status(
    sdk: WorkspaceClient,
    full_path: str,
    check_file_first: bool = True,
    raise_error: bool = True,
) -> Tuple[Optional[bool], Optional[bool], Optional[int], Optional[dt.datetime]]:
    client = sdk.files
    probes = [
        (client.get_metadata, True),
        (client.get_directory_metadata, False),
    ]
    if not check_file_first:
        probes.reverse()

    last_exc: Optional[BaseException] = None
    for probe, is_file in probes:
        try:
            info = _call(probe, full_path)
        except _NOT_FOUND as e:
            last_exc = e
            continue
        size = getattr(info, "content_length", None) if is_file else None
        return is_file, not is_file, size, _parse_mtime(info)

    if raise_error and last_exc is not None:
        raise last_exc
    return None, None, None, None


def get_volume_metadata(
    sdk: WorkspaceClient,
    full_name: str,
    include_browse: bool = False,
    raise_error: bool = True,
):
    try:
        return _call(sdk.volumes.read, name=full_name, include_browse=include_browse)
    except _NOT_FOUND:
        if raise_error:
            raise
        return None
    # BadRequest / PermissionDenied propagate unconditionally.


def _parse_mtime(info) -> dt.datetime:
    if not info:
        return dt.datetime.now(tz=dt.timezone.utc)

    for key in ("last_modified", "modified_at", "modification_time", "mtime"):
        value = getattr(info, key, None)
        if value is not None:
            return _coerce_mtime(value)
    return dt.datetime.now(tz=dt.timezone.utc)


def _coerce_mtime(value) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return (
            value.replace(tzinfo=dt.timezone.utc)
            if value.tzinfo is None
            else value.astimezone(dt.timezone.utc)
        )

    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("empty mtime string")

        # ISO-8601 (Z handled via replace for pre-3.11 compatibility)
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None

        # RFC 2822 / HTTP-date fallback
        if parsed is None:
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError, IndexError):
                parsed = None

        if parsed is None:
            raise ValueError(f"Unsupported mtime string format: {value!r}")

        return (
            parsed.replace(tzinfo=dt.timezone.utc)
            if parsed.tzinfo is None
            else parsed.astimezone(dt.timezone.utc)
        )

    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 10_000_000_000:  # ms → s
            ts /= 1000.0
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)

    raise TypeError(f"Unsupported mtime type: {type(value).__name__}")