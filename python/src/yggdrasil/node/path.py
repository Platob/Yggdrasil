"""Confined filesystem access for the node — every path stays under a root.

:class:`NodePath` wraps a root directory and resolves relative paths inside
it, raising :class:`ForbiddenError` on any attempt to escape via ``..`` or
an absolute path. All node file I/O goes through one of these so the server
can never be coaxed into reading outside its home.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Iterator

from yggdrasil.exceptions.api import ForbiddenError, NotFoundError


class NodePath:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, rel: str) -> Path:
        candidate = (self.root / (rel or "")).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ForbiddenError(
                f"Path {rel!r} resolves outside the node root {str(self.root)!r}."
            )
        return candidate

    def _local_path(self, rel: str) -> Path:
        return self.resolve(rel)

    def ls(self, rel: str) -> list[dict]:
        target = self.resolve(rel)
        if not target.exists():
            raise NotFoundError(f"No such path: {rel!r}")
        out: list[dict] = []
        with os.scandir(target) as it:
            for entry in it:
                st = entry.stat()
                is_dir = entry.is_dir()
                out.append({
                    "name": entry.name,
                    "is_dir": is_dir,
                    "size": 0 if is_dir else st.st_size,
                    "mtime": dt.datetime.fromtimestamp(
                        st.st_mtime, tz=dt.timezone.utc
                    ).isoformat(),
                })
        out.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
        return out

    def read_text(self, rel: str) -> str:
        return self.resolve(rel).read_text()

    def write_text(self, rel: str, content: str) -> None:
        self.resolve(rel).write_text(content)

    def read_bytes(self, rel: str) -> bytes:
        return self.resolve(rel).read_bytes()

    def write_bytes(self, rel: str, data: bytes) -> None:
        self.resolve(rel).write_bytes(data)

    def mkdir(self, rel: str, parents: bool = True) -> None:
        self.resolve(rel).mkdir(parents=parents, exist_ok=True)

    def stat(self, rel: str) -> dict:
        st = self.resolve(rel).stat()
        return {
            "size": st.st_size,
            "mtime": dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat(),
            "is_dir": os.path.isdir(self.resolve(rel)),
        }

    def iterdir(self, rel: str) -> list[str]:
        return [e.name for e in os.scandir(self.resolve(rel))]

    def stream_read(self, rel: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        with self.resolve(rel).open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
