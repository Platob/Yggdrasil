from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class FileInfo(StrictModel):
    path: str
    name: str
    is_dir: bool
    size: int = 0
    modified_at: str = ""
    created_at: str = ""


class DirectoryListing(StrictModel):
    node_id: str
    path: str
    entries: list[FileInfo]


class FileContent(StrictModel):
    path: str
    content: str  # base64 for binary, utf-8 for text
    encoding: str = "utf-8"  # utf-8 or base64
    size: int = 0


class FileWriteRequest(StrictModel):
    path: str
    content: str
    encoding: str = "utf-8"
    mkdir: bool = True  # create parent dirs


class FileMoveRequest(StrictModel):
    source: str
    destination: str
