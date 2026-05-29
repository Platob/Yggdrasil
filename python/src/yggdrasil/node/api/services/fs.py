from __future__ import annotations

import base64
import datetime as dt
import logging
import os
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

    async def ls(self, path: str = "", *, offset: int = 0, limit: int | None = None) -> FsListResponse:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"Directory not found: {path!r}")
        if not resolved.is_dir():
            raise ForbiddenError(f"Path is not a directory: {path!r}")

        # os.scandir caches each entry's type (one readdir, no per-child stat
        # for is_dir) and hands back a stat with a single syscall. Collect raw
        # tuples, sort, then build FsEntry pydantic models for ONLY the
        # requested page — a 100k-entry directory pages without constructing
        # 100k models.
        root_str = str(self._root.resolve())
        raw: list[tuple[bool, str, str, int, float]] = []
        with os.scandir(resolved) as it:
            for de in it:
                try:
                    is_dir = de.is_dir()
                    st = de.stat()
                except OSError:
                    continue
                raw.append((
                    is_dir, de.name, os.path.relpath(de.path, root_str),
                    0 if is_dir else st.st_size, st.st_mtime,
                ))
        raw.sort(key=lambda r: (not r[0], r[1].lower()))
        total = len(raw)
        window = raw[offset:offset + limit] if limit is not None else raw[offset:]
        entries = [
            FsEntry(
                path=rel, name=name, is_dir=is_dir, size=size,
                modified_at=dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc).isoformat(),
            )
            for (is_dir, name, rel, size, mtime) in window
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
            total=total,
            offset=offset,
        )

    async def read(self, path: str, *, max_bytes: int | None = None, offset: int = 0) -> FsReadResponse:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"File not found: {path!r}")
        if resolved.is_dir():
            raise ForbiddenError(f"Cannot read a directory as a file: {path!r}")

        # Bound the read: never pull more than the cap into memory just to
        # preview. A caller may ask for a smaller window but not a larger one,
        # and ``offset`` reads a byte range so a remote NodePath can page.
        cap = self.settings.max_read_bytes if max_bytes is None else max_bytes
        cap = max(1, min(cap, self.settings.max_read_bytes))
        offset = max(0, offset)
        full_size = resolved.stat().st_size
        # Read cap+1 so we can tell there's more without loading it all.
        with open(resolved, "rb") as fh:
            if offset:
                fh.seek(offset)
            raw = fh.read(cap + 1)
        if len(raw) > cap:
            raw = raw[:cap]
        truncated = offset + len(raw) < full_size

        try:
            content = raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            # A clean truncation can split a trailing multibyte char; drop up
            # to 3 trailing bytes and retry before deciding the file is binary.
            text: str | None = None
            if truncated:
                for back in (1, 2, 3):
                    try:
                        text = raw[:-back].decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        continue
            if text is not None:
                content, encoding = text, "utf-8"
            else:
                content = base64.b64encode(raw).decode("ascii")
                encoding = "base64"

        rel = str(resolved.relative_to(self._root))
        return FsReadResponse(
            path=rel,
            content=content,
            encoding=encoding,
            size=full_size,
            offset=offset,
            truncated=truncated,
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

    def build_dir_zip(self, path: str) -> tuple[Path, str]:
        """Zip a directory into a temp file (bounded by du_max_entries) and
        return ``(zip_path, archive_name)``. The caller streams then unlinks the
        temp file. Writing to disk keeps memory flat regardless of folder size.
        """
        import tempfile
        import zipfile

        resolved = self._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"Path not found: {path!r}")
        if not resolved.is_dir():
            raise ForbiddenError(f"Not a directory: {path!r}")

        cap = self.settings.du_max_entries
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        count = 0
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in resolved.rglob("*"):
                if count >= cap:
                    break
                if p.is_file():
                    try:
                        zf.write(p, p.relative_to(resolved))
                        count += 1
                    except OSError:
                        continue
        archive_name = f"{resolved.name or 'node'}.zip"
        return Path(tmp.name), archive_name

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

    def head_lines(self, path: str, n: int = 100) -> list[str]:
        """First N lines. Inline because we never need it elsewhere."""
        resolved = self._resolve(path)
        if not resolved.exists() or resolved.is_dir():
            raise NotFoundError(f"File not found: {path!r}")
        out: list[str] = []
        with open(resolved, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                out.append(line.rstrip("\n"))
        return out

    def tail_lines(self, path: str, n: int = 100) -> list[str]:
        """Last N lines via a tail-from-end byte scan (no whole-file load)."""
        resolved = self._resolve(path)
        if not resolved.exists() or resolved.is_dir():
            raise NotFoundError(f"File not found: {path!r}")
        # Read backwards in 8 KB chunks until we have n+1 newlines or hit BOF.
        block = 8192
        with open(resolved, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                read = min(block, size)
                size -= read
                f.seek(size)
                data = f.read(read) + data
        text = data.decode("utf-8", errors="replace")
        return text.splitlines()[-n:]

    async def watch_tail(self, path: str, poll_seconds: float = 0.5) -> AsyncIterator[str]:
        """SSE-style tail -f. Yields each new line as it appears."""
        import asyncio
        resolved = self._resolve(path)
        if not resolved.exists() or resolved.is_dir():
            raise NotFoundError(f"File not found: {path!r}")
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)  # start at EOF
            while True:
                line = f.readline()
                if line:
                    yield line.rstrip("\n")
                else:
                    await asyncio.sleep(poll_seconds)

    def grep(self, path: str, pattern: str, *, max_matches: int = 200,
             case_sensitive: bool = False, regex: bool = False) -> tuple[list[dict], bool]:
        """Recursive substring/regex search over text files inside ``path``.

        Returns ``(matches, truncated)`` where each match is a
        {path, line_number, line, match} dict. Skips files whose first 1 KB is
        mostly non-text (heuristic null-byte check). The walk stops once it has
        either filled ``max_matches`` or visited ``du_max_entries`` tree nodes,
        so a huge tree never gets materialized or fully scanned — ``truncated``
        flags when a cap cut the walk short.
        """
        import re
        from ...exceptions import NotFoundError
        root = self._resolve(path) if path else self._root
        if not root.exists():
            raise NotFoundError(f"Path not found: {path!r}")
        flags = 0 if case_sensitive else re.IGNORECASE
        prog = re.compile(pattern if regex else re.escape(pattern), flags)
        results: list[dict] = []
        scan_cap = self.settings.du_max_entries
        scanned = 0
        truncated = False
        # rglob is a generator — iterate it lazily so we never hold the whole
        # tree in memory; bail the moment a cap is hit.
        targets = iter([root]) if root.is_file() else root.rglob("*")
        for p in targets:
            if len(results) >= max_matches:
                truncated = True
                break
            scanned += 1
            if scanned > scan_cap:
                truncated = True
                break
            if not p.is_file():
                continue
            try:
                # Quick binary skip
                with open(p, "rb") as fh:
                    head = fh.read(1024)
                if b"\x00" in head:
                    continue
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if len(results) >= max_matches:
                            truncated = True
                            break
                        m = prog.search(line)
                        if m:
                            rel = str(p.relative_to(self._root))
                            results.append({
                                "path": rel,
                                "line_number": lineno,
                                "line": line.rstrip("\n")[:500],
                                "match": m.group(0),
                            })
            except (OSError, UnicodeDecodeError):
                continue
        return results, truncated
