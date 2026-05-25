from __future__ import annotations

import base64
import datetime as dt
import logging
import shutil
from pathlib import Path
from typing import AsyncIterator

from ..config import Settings
from ..exceptions import ForbiddenError, NotFoundError
from ..schemas.filesystem import (
    DirectoryListing,
    FileContent,
    FileInfo,
    FileMoveRequest,
    FileWriteRequest,
)

LOGGER = logging.getLogger(__name__)


class FilesystemService:
    """Manage files within the node's home directory.

    The filesystem root is ``node_home`` so users can browse all node data.
    Path traversal protection prevents access above this root.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._root = settings.node_home
        self._root.mkdir(parents=True, exist_ok=True)
        # Create default Linux-style hierarchy under node_home
        for d in (
            "data/files", "data/files/tmp", "data/files/downloads",
            "data/files/documents", "data/files/data", "mirrors", "logs", "cache",
        ):
            (self._root / d).mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        """Resolve path relative to root, prevent directory traversal."""
        # Strip leading slash so it's always treated as relative
        cleaned = path.lstrip("/")
        if not cleaned:
            return self._root
        resolved = (self._root / cleaned).resolve()
        # Prevent traversal above root
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ForbiddenError(
                f"Path traversal not allowed. Path must stay within the node's file root."
            )
        return resolved

    def _stat_path(self, resolved: Path, *, relative_to: Path | None = None) -> FileInfo:
        """Build a FileInfo from a resolved Path."""
        rel_root = relative_to or self._root
        try:
            rel_path = str(resolved.relative_to(rel_root))
        except ValueError:
            rel_path = str(resolved.relative_to(self._root))

        stat = resolved.stat()
        return FileInfo(
            path=rel_path,
            name=resolved.name,
            is_dir=resolved.is_dir(),
            size=stat.st_size if not resolved.is_dir() else 0,
            modified_at=dt.datetime.fromtimestamp(
                stat.st_mtime, tz=dt.timezone.utc
            ).isoformat(),
            created_at=dt.datetime.fromtimestamp(
                stat.st_ctime, tz=dt.timezone.utc
            ).isoformat(),
        )

    async def list_dir(self, path: str = "") -> DirectoryListing:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"Directory not found: {path!r}")
        if not resolved.is_dir():
            raise ForbiddenError(f"Path is not a directory: {path!r}")

        entries: list[FileInfo] = []
        for child in sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            entries.append(self._stat_path(child, relative_to=self._root))

        # Relative display path
        try:
            display_path = str(resolved.relative_to(self._root))
        except ValueError:
            display_path = ""
        if display_path == ".":
            display_path = ""

        return DirectoryListing(
            node_id=self.settings.node_id,
            path=display_path,
            entries=entries,
        )

    async def read_file(self, path: str) -> FileContent:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"File not found: {path!r}")
        if resolved.is_dir():
            raise ForbiddenError(f"Cannot read a directory as a file: {path!r}")

        # Try text first, fall back to base64 for binary
        try:
            content = resolved.read_text(encoding="utf-8")
            encoding = "utf-8"
        except (UnicodeDecodeError, ValueError):
            content = base64.b64encode(resolved.read_bytes()).decode("ascii")
            encoding = "base64"

        rel_path = str(resolved.relative_to(self._root))
        return FileContent(
            path=rel_path,
            content=content,
            encoding=encoding,
            size=resolved.stat().st_size,
        )

    async def write_file(self, req: FileWriteRequest) -> FileInfo:
        resolved = self._resolve(req.path)

        if req.mkdir:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        elif not resolved.parent.exists():
            raise NotFoundError(
                f"Parent directory does not exist: {str(resolved.parent.relative_to(self._root))!r}. "
                f"Set mkdir=true to create it automatically."
            )

        if req.encoding == "base64":
            data = base64.b64decode(req.content)
            resolved.write_bytes(data)
        else:
            resolved.write_text(req.content, encoding="utf-8")

        LOGGER.info("Wrote file %r (%d bytes)", req.path, resolved.stat().st_size)
        return self._stat_path(resolved)

    async def delete(self, path: str) -> None:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"Path not found: {path!r}")

        # Prevent deleting the root itself
        if resolved == self._root.resolve():
            raise ForbiddenError("Cannot delete the file root directory.")

        if resolved.is_dir():
            shutil.rmtree(resolved)
            LOGGER.info("Deleted directory %r", path)
        else:
            resolved.unlink()
            LOGGER.info("Deleted file %r", path)

    async def move(self, req: FileMoveRequest) -> FileInfo:
        src = self._resolve(req.source)
        dst = self._resolve(req.destination)

        if not src.exists():
            raise NotFoundError(f"Source not found: {req.source!r}")

        # Ensure destination parent exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src), str(dst))
        LOGGER.info("Moved %r -> %r", req.source, req.destination)
        return self._stat_path(dst)

    async def mkdir(self, path: str) -> FileInfo:
        resolved = self._resolve(path)
        resolved.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Created directory %r", path)
        return self._stat_path(resolved)

    async def stat(self, path: str) -> FileInfo:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"Path not found: {path!r}")
        return self._stat_path(resolved)

    async def stream_read(self, path: str) -> AsyncIterator[bytes]:
        """Yield file contents in chunks for streaming download."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"File not found: {path!r}")
        if resolved.is_dir():
            raise ForbiddenError(f"Cannot stream a directory: {path!r}")

        chunk_size = 64 * 1024  # 64KB chunks
        with open(resolved, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def stream_write(self, path: str, chunks: AsyncIterator[bytes]) -> FileInfo:
        """Write streaming upload chunks to a file."""
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        with open(resolved, "wb") as f:
            async for chunk in chunks:
                f.write(chunk)

        LOGGER.info("Stream-wrote file %r (%d bytes)", path, resolved.stat().st_size)
        return self._stat_path(resolved)
