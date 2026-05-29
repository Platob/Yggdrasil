from __future__ import annotations

from .common import StrictModel


class FsEntry(StrictModel):
    path: str
    name: str
    is_dir: bool
    size: int = 0
    modified_at: str = ""


class FsListResponse(StrictModel):
    node_id: str
    path: str
    entries: list[FsEntry]
    # Paging: total entries in the directory, and the window this page covers.
    total: int = 0
    offset: int = 0


class FsReadResponse(StrictModel):
    path: str
    content: str
    encoding: str = "utf-8"
    size: int = 0
    # Byte offset this window starts at (for ranged reads).
    offset: int = 0
    # True when there are more bytes after this window (offset+len < size).
    truncated: bool = False


class FsWriteRequest(StrictModel):
    path: str
    content: str
    encoding: str = "utf-8"
    mkdir: bool = True


class FsMoveRequest(StrictModel):
    source: str
    destination: str
