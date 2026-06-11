"""Filesystem service rooted at ``node_home``.

``ls`` is the hot path behind the front's lazy file tree. It ``scandir``s the
directory once (caching the dirent type so the sort key doesn't re-stat), sorts
directories-first then by name, and then builds pydantic ``FsEntry`` models for
*only the requested page* — a 50k-entry directory pages without constructing
50k models or a giant JSON payload per request.

``read`` is bounded: it pulls at most ``max_read_bytes`` no matter how large the
file is, so previewing a multi-GB log never loads the whole thing into RAM.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from pydantic import BaseModel


class FsEntry(BaseModel):
    name: str
    is_dir: bool
    size: int
    modified_at: str


class FsListResult(BaseModel):
    entries: list[FsEntry]
    total: int


class FsReadResult(BaseModel):
    content: str
    truncated: bool


class FsService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._root = Path(settings.node_home)
        self._max_read = getattr(settings, "max_read_bytes", 4 * 1024 * 1024)

    def _resolve(self, relative: str) -> Path:
        target = (self._root / relative).resolve() if relative else self._root.resolve()
        root = self._root.resolve()
        # Refuse to escape node_home via .. or symlinks.
        if target != root and root not in target.parents:
            raise ValueError(
                f"Path {relative!r} escapes node_home ({root}). "
                f"Pass a path under the node root."
            )
        return target

    async def ls(self, relative: str, offset: int = 0, limit: int | None = None) -> FsListResult:
        target = self._resolve(relative)
        if not target.exists():
            raise FileNotFoundError(f"No such path {relative!r} under node_home.")
        if not target.is_dir():
            raise NotADirectoryError(f"{relative!r} is a file, not a directory.")

        # One scandir pass: collect (name, is_dir, DirEntry) without building
        # any models yet. is_dir() here is served from the cached dirent type.
        scanned: list[tuple[str, bool, os.DirEntry]] = []
        with os.scandir(target) as it:
            for de in it:
                scanned.append((de.name, de.is_dir(), de))
        scanned.sort(key=lambda t: (not t[1], t[0].lower()))
        total = len(scanned)

        end = total if limit is None else min(total, offset + limit)
        window = scanned[offset:end]
        entries = []
        for name, is_dir, de in window:
            st = de.stat()
            entries.append(FsEntry(
                name=name,
                is_dir=is_dir,
                size=0 if is_dir else st.st_size,
                modified_at=dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat(),
            ))
        return FsListResult(entries=entries, total=total)

    async def read(self, relative: str) -> FsReadResult:
        target = self._resolve(relative)
        if not target.is_file():
            raise FileNotFoundError(f"No such file {relative!r} under node_home.")
        size = target.stat().st_size
        with open(target, "rb") as fh:
            raw = fh.read(self._max_read)
        truncated = size > self._max_read
        # Lossy decode for preview — node serves text previews, not exact bytes.
        return FsReadResult(content=raw.decode("utf-8", errors="replace"), truncated=truncated)
