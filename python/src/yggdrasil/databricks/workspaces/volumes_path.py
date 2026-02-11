import datetime as dt
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
    "get_volume_metadata"
]


def get_volume_status(
    sdk: WorkspaceClient,
    full_path: str,
    check_file_first: bool = True,
    raise_error: bool = True,
) -> Tuple[Optional[bool], Optional[bool], Optional[int], Optional[dt.datetime]]:
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
            include_browse=include_browse
        )
    except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
        if raise_error:
            raise

    return None



def _parse_mtime(info):
    if not info:
        return dt.datetime.now(tz=dt.timezone.utc)

    return dt.datetime.strptime(info.last_modified, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=dt.timezone.utc)