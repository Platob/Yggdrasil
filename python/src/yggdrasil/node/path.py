"""NodePath — a thin path rooted at the node home.

A NodePath is a string-relative path resolved under ``Settings.node_home``
(via the ``YGG_NODE_HOME`` env default). It exposes the small slice of the
``pathlib`` surface the node + benchmarks use, plus a chunked ``stream_read``.
All paths are confined to the node home; ``..`` that escapes the root raises.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from yggdrasil.node.config import Settings


class NodePath:
    __slots__ = ("_rel", "_settings")

    def __init__(self, rel: str | os.PathLike = "", *, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._rel = str(rel).lstrip("/")

    def _local_path(self) -> Path:
        root = self._settings.node_home.resolve()
        target = (root / self._rel).resolve()
        if root not in target.parents and target != root:
            raise ValueError(
                f"NodePath {self._rel!r} resolves outside the node home {root}. "
                f"Paths must stay within node_home."
            )
        return target

    def __truediv__(self, other: str) -> NodePath:
        rel = f"{self._rel}/{other}".strip("/") if self._rel else str(other).strip("/")
        return NodePath(rel, settings=self._settings)

    def mkdir(self, *, parents: bool = True, exist_ok: bool = True) -> None:
        self._local_path().mkdir(parents=parents, exist_ok=exist_ok)

    def write_text(self, text: str) -> int:
        return self._local_path().write_text(text)

    def read_text(self) -> str:
        return self._local_path().read_text()

    def write_bytes(self, data: bytes) -> int:
        return self._local_path().write_bytes(data)

    def read_bytes(self) -> bytes:
        return self._local_path().read_bytes()

    def stat(self) -> os.stat_result:
        return self._local_path().stat()

    def exists(self) -> bool:
        return self._local_path().exists()

    def iterdir(self) -> Iterator[NodePath]:
        base = self._local_path()
        for entry in os.scandir(base):
            yield self / entry.name

    def stream_read(self, chunk_size: int = 65536) -> Iterator[bytes]:
        with open(self._local_path(), "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    return
                yield chunk

    def __fspath__(self) -> str:
        return str(self._local_path())

    def __repr__(self) -> str:
        return f"NodePath({self._rel!r})"
