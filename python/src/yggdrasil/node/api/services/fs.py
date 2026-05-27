from __future__ import annotations

import base64
import datetime as dt
import logging
import shutil
from pathlib import Path
from typing import AsyncIterator

from ...config import Settings
from ...exceptions import ForbiddenError, NotFoundError
from ..schemas.fs import (
    FsEntry,
    FsListResponse,
    FsMoveRequest,
    FsReadResponse,
    FsWriteRequest,
)

LOGGER = logging.getLogger(__name__)

_CHUNK_SIZE = 64 * 1024


class FsService:
    """Filesystem operations rooted at node_home.

    All paths are resolved relative to node_home with traversal
    protection. Mirrors the v1 FilesystemService but lives in the
    v2 API surface with its own schema types.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._root = settings.node_home
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        cleaned = path.lstrip("/")
        if not cleaned:
            return self._root
        resolved = (self._root / cleaned).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ForbiddenError(
                "Path traversal not allowed. Path must stay within the node's file root."
            )
        return resolved

    def _entry(self, resolved: Path) -> FsEntry:
        stat = resolved.stat()
        try:
            rel = str(resolved.relative_to(self._root))
        except ValueError:
            rel = resolved.name
        return FsEntry(
            path=rel,
            name=resolved.name,
            is_dir=resolved.is_dir(),
            size=stat.st_size if not resolved.is_dir() else 0,
            modified_at=dt.datetime.fromtimestamp(
                stat.st_mtime, tz=dt.timezone.utc
            ).isoformat(),
        )

    async def stat(self, path: str) -> FsEntry:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"Path not found: {path!r}")
        return self._entry(resolved)

    async def ls(self, path: str = "") -> FsListResponse:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"Directory not found: {path!r}")
        if not resolved.is_dir():
            raise ForbiddenError(f"Path is not a directory: {path!r}")

        entries = [
            self._entry(child)
            for child in sorted(
                resolved.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        ]

        try:
            display = str(resolved.relative_to(self._root))
        except ValueError:
            display = ""
        if display == ".":
            display = ""

        return FsListResponse(
            node_id=self.settings.node_id,
            path=display,
            entries=entries,
        )

    async def read(self, path: str) -> FsReadResponse:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"File not found: {path!r}")
        if resolved.is_dir():
            raise ForbiddenError(f"Cannot read a directory as a file: {path!r}")

        try:
            content = resolved.read_text(encoding="utf-8")
            encoding = "utf-8"
        except (UnicodeDecodeError, ValueError):
            content = base64.b64encode(resolved.read_bytes()).decode("ascii")
            encoding = "base64"

        rel = str(resolved.relative_to(self._root))
        return FsReadResponse(
            path=rel,
            content=content,
            encoding=encoding,
            size=resolved.stat().st_size,
        )

    async def write(self, req: FsWriteRequest) -> FsEntry:
        resolved = self._resolve(req.path)

        if req.mkdir:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        elif not resolved.parent.exists():
            raise NotFoundError(
                f"Parent directory does not exist: "
                f"{str(resolved.parent.relative_to(self._root))!r}. "
                f"Set mkdir=true to create it automatically."
            )

        if req.encoding == "base64":
            resolved.write_bytes(base64.b64decode(req.content))
        else:
            resolved.write_text(req.content, encoding="utf-8")

        LOGGER.info("Wrote file %r (%d bytes)", req.path, resolved.stat().st_size)
        return self._entry(resolved)

    async def delete(self, path: str) -> None:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"Path not found: {path!r}")
        if resolved == self._root.resolve():
            raise ForbiddenError("Cannot delete the file root directory.")

        if resolved.is_dir():
            shutil.rmtree(resolved)
            LOGGER.info("Deleted directory %r", path)
        else:
            resolved.unlink()
            LOGGER.info("Deleted file %r", path)

    async def move(self, req: FsMoveRequest) -> FsEntry:
        src = self._resolve(req.source)
        dst = self._resolve(req.destination)
        if not src.exists():
            raise NotFoundError(f"Source not found: {req.source!r}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        LOGGER.info("Moved %r -> %r", req.source, req.destination)
        return self._entry(dst)

    async def mkdir(self, path: str) -> FsEntry:
        resolved = self._resolve(path)
        resolved.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Created directory %r", path)
        return self._entry(resolved)

    async def stream_read(self, path: str) -> AsyncIterator[bytes]:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"File not found: {path!r}")
        if resolved.is_dir():
            raise ForbiddenError(f"Cannot stream a directory: {path!r}")

        with open(resolved, "rb") as f:
            while True:
                chunk = f.read(_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk

    async def stream_write(self, path: str, chunks: AsyncIterator[bytes]) -> FsEntry:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved, "wb") as f:
            async for chunk in chunks:
                f.write(chunk)

        LOGGER.info("Stream-wrote file %r (%d bytes)", path, resolved.stat().st_size)
        return self._entry(resolved)
