"""Filesystem service — node-rooted listing/read with scandir + paged model builds.

``ls`` uses :func:`os.scandir` (one stat per child, dirent type already cached)
instead of ``iterdir`` + repeated stats, and builds ``FsEntry`` models only for
the requested page so a 50k-entry directory pages cheaply. ``read`` is bounded
at ``max_read_bytes`` so previewing a giant file never pulls it all into memory.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from pydantic import BaseModel

from yggdrasil.exceptions.api import NotFoundError


class FsEntry(BaseModel):
    name: str
    size: int
    is_dir: bool
    mtime: str


class LsResult(BaseModel):
    entries: list[FsEntry]
    total: int


class ReadResult(BaseModel):
    content: bytes
    truncated: bool


class FsService:
    """Filesystem operations rooted at ``settings.node_home``."""

    def __init__(self, settings: object) -> None:
        self._root = Path(settings.node_home)
        self._max_read = getattr(settings, "max_read_bytes", 4 * 1024 * 1024)

    def _resolve(self, path: str) -> Path:
        full = (self._root / path).resolve()
        root = self._root.resolve()
        if full != root and root not in full.parents:
            raise NotFoundError(f"Path {path!r} escapes the node home.")
        return full

    async def ls(self, path: str, *, offset: int = 0, limit: int | None = None) -> LsResult:
        full = self._resolve(path)
        if not full.exists():
            raise NotFoundError(f"Path {path!r} not found.")

        # scandir once, capture (name, is_dir, dirent) without per-child re-stats.
        dirents = list(os.scandir(full))
        dirents.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
        total = len(dirents)

        window = dirents[offset:offset + limit] if limit is not None else dirents[offset:]
        entries: list[FsEntry] = []
        for e in window:
            st = e.stat()
            is_dir = e.is_dir()
            entries.append(
                FsEntry(
                    name=e.name,
                    size=0 if is_dir else st.st_size,
                    is_dir=is_dir,
                    mtime=dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat(),
                )
            )
        return LsResult(entries=entries, total=total)

    async def read(self, path: str) -> ReadResult:
        full = self._resolve(path)
        if not full.is_file():
            raise NotFoundError(f"File {path!r} not found.")
        size = full.stat().st_size
        with open(full, "rb") as fh:
            content = fh.read(self._max_read)
        return ReadResult(content=content, truncated=size > self._max_read)

    async def write(self, path: str, content: bytes) -> None:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(content)

    async def info(self, path: str) -> FsEntry:
        full = self._resolve(path)
        if not full.exists():
            raise NotFoundError(f"Path {path!r} not found.")
        st = full.stat()
        is_dir = full.is_dir()
        return FsEntry(
            name=full.name,
            size=0 if is_dir else st.st_size,
            is_dir=is_dir,
            mtime=dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat(),
        )
