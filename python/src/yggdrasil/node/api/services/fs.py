"""FsService — directory listing + bounded file preview.

``ls`` scandirs once (caching the dirent type, statting each child once) and
builds pydantic ``FsEntry`` models for ONLY the requested page, so a 50k-entry
directory pages cheaply. ``read`` caps at ``settings.max_read_bytes`` so
previewing a 1 GB log never pulls 1 GB into memory.

Paths are relative to ``settings.node_home`` and confined to it.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class FsEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int
    modified: str  # ISO-8601 UTC


class LsResult(BaseModel):
    entries: list[FsEntry]
    total: int
    offset: int = 0


class ReadResult(BaseModel):
    content: str
    truncated: bool
    size: int


class FsService:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.root = Path(settings.node_home)

    def _resolve(self, rel: str) -> Path:
        target = (self.root / rel.lstrip("/")).resolve()
        root = self.root.resolve()
        if target != root and root not in target.parents:
            raise ValueError(
                f"path {rel!r} escapes the node home {root}; stay within node_home."
            )
        return target

    async def ls(self, path: str, offset: int = 0, limit: int | None = None) -> LsResult:
        base = self._resolve(path)
        # One scandir pass: collect (name, is_dir) tuples cheaply, sort, then
        # stat ONLY the entries that fall in the requested page.
        names: list[tuple[bool, str]] = []
        with os.scandir(base) as it:
            for de in it:
                names.append((de.is_dir(), de.name))
        # dirs first, then case-insensitive name.
        names.sort(key=lambda t: (not t[0], t[1].lower()))
        total = len(names)

        end = total if limit is None else min(total, offset + limit)
        window = names[offset:end]

        entries: list[FsEntry] = []
        rel_prefix = path.strip("/")
        for is_dir, name in window:
            full = base / name
            st = full.stat()
            entries.append(FsEntry(
                name=name,
                path=f"{rel_prefix}/{name}" if rel_prefix else name,
                is_dir=is_dir,
                size=0 if is_dir else st.st_size,
                modified=dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat(),
            ))
        return LsResult(entries=entries, total=total, offset=offset)

    async def read(self, path: str) -> ReadResult:
        target = self._resolve(path)
        cap = self.settings.max_read_bytes
        size = target.stat().st_size
        with open(target, "rb") as fh:
            raw = fh.read(cap)
        truncated = size > cap
        return ReadResult(
            content=raw.decode("utf-8", errors="replace"),
            truncated=truncated,
            size=size,
        )
