"""Confined node filesystem service — the backend behind the file tree.

Rooted at ``settings.node_home``. The hot path is :meth:`ls`, which uses
``os.scandir``: the dirent caches its entry type and a single ``stat`` per
child, so a directory of N entries pays N stats instead of the 3-4 the old
``iterdir`` + per-child ``stat`` path cost. Every relative path is resolved
under the root and rejected if it escapes via ``..`` or an absolute path.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import AsyncIterator

from pydantic import BaseModel

from yggdrasil.exceptions.api import ForbiddenError, NotFoundError
from yggdrasil.node.config import Settings


class FsEntry(BaseModel):
    name: str
    is_dir: bool
    size: int
    mtime: str


class LsResult(BaseModel):
    path: str
    entries: list[FsEntry]


class FsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = Path(settings.node_home).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel: str) -> Path:
        candidate = (self.root / (rel or "")).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ForbiddenError(f"Path {rel!r} resolves outside the node root.")
        return candidate

    async def ls(self, rel: str = "") -> LsResult:
        target = self._resolve(rel)
        if not target.exists():
            raise NotFoundError(f"No such path: {rel!r}.")
        entries: list[FsEntry] = []
        with os.scandir(target) as it:
            for entry in it:
                # entry.is_dir() / entry.stat() are served from the cached dirent —
                # one syscall per child, not the old stat-thrice-per-entry cost.
                is_dir = entry.is_dir()
                st = entry.stat()
                entries.append(FsEntry(
                    name=entry.name,
                    is_dir=is_dir,
                    size=0 if is_dir else st.st_size,
                    mtime=dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat(),
                ))
        entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
        return LsResult(path=rel, entries=entries)

    async def read(self, rel: str, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
        target = self._resolve(rel)
        if not target.is_file():
            raise NotFoundError(f"No such file: {rel!r}.")
        with target.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def write(self, rel: str, data: bytes) -> None:
        target = self._resolve(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def mkdir(self, rel: str) -> None:
        self._resolve(rel).mkdir(parents=True, exist_ok=True)

    async def delete(self, rel: str) -> None:
        target = self._resolve(rel)
        if target.is_dir():
            import shutil

            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        else:
            raise NotFoundError(f"No such path: {rel!r}.")
