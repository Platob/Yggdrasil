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


class FsReadResponse(StrictModel):
    path: str
    content: str
    encoding: str = "utf-8"
    size: int = 0


class FsWriteRequest(StrictModel):
    path: str
    content: str
    encoding: str = "utf-8"
    mkdir: bool = True


class FsMoveRequest(StrictModel):
    source: str
    destination: str
