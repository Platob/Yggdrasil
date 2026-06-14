"""NodePath — a path object rooted at the node home directory.

Every ``NodePath`` resolves relative to ``settings.node_home``, so callers
work in node-relative terms and never touch absolute host paths. It mirrors the
slice of :class:`pathlib.Path` the node needs (read/write/stat/iterdir) plus a
chunked :meth:`stream_read` for streaming large files back over HTTP.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from yggdrasil.node.config import Settings


class NodePath:
    """A path rooted at ``settings.node_home``."""

    def __init__(self, path: str | Path, *, settings: Settings | None = None) -> None:
        if settings is None:
            from yggdrasil.node.config import Settings as _Settings

            settings = _Settings()
        self._settings = settings
        self._rel = Path(path)

    def _local_path(self) -> Path:
        return Path(self._settings.node_home) / self._rel

    def __truediv__(self, other: str | Path) -> NodePath:
        return NodePath(self._rel / other, settings=self._settings)

    def __fspath__(self) -> str:
        return str(self._local_path())

    def __repr__(self) -> str:
        return f"NodePath({str(self._rel)!r})"

    def mkdir(self, **kwargs) -> None:
        kwargs.setdefault("parents", True)
        kwargs.setdefault("exist_ok", True)
        self._local_path().mkdir(**kwargs)

    def write_text(self, content: str) -> None:
        self._local_path().write_text(content)

    def write_bytes(self, content: bytes) -> None:
        self._local_path().write_bytes(content)

    def read_text(self) -> str:
        return self._local_path().read_text()

    def read_bytes(self) -> bytes:
        return self._local_path().read_bytes()

    def stat(self) -> os.stat_result:
        return self._local_path().stat()

    def iterdir(self) -> Iterator[NodePath]:
        for child in self._local_path().iterdir():
            yield NodePath(self._rel / child.name, settings=self._settings)

    def exists(self) -> bool:
        return self._local_path().exists()

    def stream_read(self, chunk_size: int = 65536) -> Iterator[bytes]:
        with open(self._local_path(), "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                yield chunk
