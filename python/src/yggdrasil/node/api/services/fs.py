"""Filesystem browsing service.

Listing is the hot path behind the lazy file tree. It uses ``os.scandir`` so
each entry is stat'd once (the dirent caches the type), and it builds pydantic
FsEntry models for ONLY the requested page when a limit is given — a 50k-entry
directory pages cheaply instead of constructing 50k models per request.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

from ...config import Settings
from ..schemas.fs import FsEntry, FsListResult, FsReadResult


class FsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _resolve(self, path: str) -> Path:
        root = self.settings.node_home.resolve()
        rel = str(path or "").strip("/")
        target = (root / rel).resolve() if rel else root
        if target != root and root not in target.parents:
            raise ValueError(f"path {path!r} escapes the node home {root}")
        return target

    async def ls(self, path: str, *, offset: int = 0, limit: int = 0) -> FsListResult:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"no such path: {path!r}")
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path!r}")

        rel = str(path or "").strip("/")

        # One scandir pass collects (name, is_dir, size, mtime) tuples without
        # building any model. Sorting on the cheap tuple keeps dirs first.
        raw: list[tuple[str, bool, int, float]] = []
        with os.scandir(target) as it:
            for de in it:
                try:
                    st = de.stat(follow_symlinks=False)
                    is_dir = de.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                raw.append((de.name, is_dir, 0 if is_dir else st.st_size, st.st_mtime))
        raw.sort(key=lambda r: (not r[1], r[0].lower()))

        total = len(raw)
        window = raw[offset:offset + limit] if limit > 0 else raw[offset:]

        entries = [
            FsEntry(
                name=name,
                path=f"{rel}/{name}" if rel else name,
                is_dir=is_dir,
                size=size,
                modified=dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc).isoformat(),
            )
            for (name, is_dir, size, mtime) in window
        ]
        return FsListResult(entries=entries, total=total)

    async def read(self, path: str) -> FsReadResult:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"no such file: {path!r}")
        size = target.stat().st_size
        cap = self.settings.max_read_bytes
        # Pull at most `cap` bytes regardless of file size — previewing a 1 GB
        # log must not pull 1 GB into memory.
        with open(target, "rb") as fh:
            data = fh.read(cap)
        truncated = size > cap
        return FsReadResult(
            content=data.decode("utf-8", errors="replace"),
            truncated=truncated,
            size=size,
        )

    async def inspect(self, path: str) -> dict:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"no such file: {path!r}")
        st = target.stat()
        return {
            "path": str(path),
            "name": target.name,
            "size": st.st_size,
            "is_dir": target.is_dir(),
            "modified": dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat(),
        }
