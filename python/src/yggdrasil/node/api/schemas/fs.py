"""Filesystem browsing schemas."""
from __future__ import annotations

from pydantic import BaseModel


class FsEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int
    modified: str  # ISO datetime


class FsListResult(BaseModel):
    entries: list[FsEntry]
    total: int


class FsReadResult(BaseModel):
    content: str
    truncated: bool
    size: int
