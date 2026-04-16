"""Volume-level helpers: status probes and UC metadata reads."""
from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
from typing import Tuple, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    NotFound,
    ResourceDoesNotExist,
    BadRequest,
    PermissionDenied,
)

__all__ = [
    "get_volume_status",
    "get_volume_metadata",
]


def get_volume_status(
    sdk: WorkspaceClient,
    full_path: str,
    check_file_first: bool = True,
    raise_error: bool = True,
) -> Tuple[Optional[bool], Optional[bool], int | None, Optional[dt.datetime]]:
    client = sdk.files

    if check_file_first:
        try:
            info = client.get_metadata(full_path)
            return True, False, info.content_length, _parse_mtime(info)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass

        try:
            info = client.get_directory_metadata(full_path)
            return False, True, 0, _parse_mtime(info)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied) as e:
            last_exception = e
    else:
        try:
            info = client.get_directory_metadata(full_path)
            return False, True, 0, _parse_mtime(info)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass

        try:
            info = client.get_metadata(full_path)
            return True, False, info.content_length, _parse_mtime(info)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied) as e:
            last_exception = e

    if raise_error and last_exception is not None:
        raise last_exception

    return None, None, None, None


def get_volume_metadata(
    sdk: WorkspaceClient,
    full_name: str,
    include_browse: bool = False,
    raise_error: bool = True,
):
    client = sdk.volumes

    try:
        return client.read(
            name=full_name,
            include_browse=include_browse,
        )
    except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
        if raise_error:
            raise

    return None


def _parse_mtime(info) -> dt.datetime:
    if not info:
        return dt.datetime.now(tz=dt.timezone.utc)

    for key in ("last_modified", "modified_at", "modification_time", "mtime"):
        value = getattr(info, key, None)
        if value is None:
            continue

        if isinstance(value, dt.datetime):
            return (
                value.replace(tzinfo=dt.timezone.utc)
                if value.tzinfo is None
                else value.astimezone(dt.timezone.utc)
            )

        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue

            # 1. ISO-8601 / near-ISO
            try:
                parsed = dt.datetime.fromisoformat(value)
                return (
                    parsed.replace(tzinfo=dt.timezone.utc)
                    if parsed.tzinfo is None
                    else parsed.astimezone(dt.timezone.utc)
                )
            except ValueError:
                pass

            # 2. Common "Z" suffix variant
            try:
                parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
                return (
                    parsed.replace(tzinfo=dt.timezone.utc)
                    if parsed.tzinfo is None
                    else parsed.astimezone(dt.timezone.utc)
                )
            except ValueError:
                pass

            # 3. RFC 2822 / HTTP-date, e.g. "Sat, 11 Apr 2026 16:21:34 GMT"
            try:
                parsed = parsedate_to_datetime(value)
                return (
                    parsed.replace(tzinfo=dt.timezone.utc)
                    if parsed.tzinfo is None
                    else parsed.astimezone(dt.timezone.utc)
                )
            except (TypeError, ValueError, IndexError):
                pass

            raise ValueError(f"Unsupported mtime string format: {value!r}")

        # numeric timestamp
        ts = float(value)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)

    return dt.datetime.now(tz=dt.timezone.utc)

