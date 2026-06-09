"""Filesystem browser contracts: ls, read, write, delete."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FsEntry(BaseModel):
    name: str
    is_dir: bool
    size: int = 0
    modified: str = ""


class FsLsResult(BaseModel):
    path: str
    entries: list[FsEntry]
    total: int = 0


class FsReadResult(BaseModel):
    path: str
    content: str
    truncated: bool = False
    size: int = 0


class FsWriteRequest(BaseModel):
    path: str
    content: str


class FsNodeRoot(BaseModel):
    node_id: str
    home: str
    entries: list[FsEntry] = []
    mounts: list[dict[str, Any]] = []
