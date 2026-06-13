"""NodePath — a path rooted at the node's home directory.

All node filesystem access goes through NodePath so paths are confined to
``settings.node_home`` and never escape it via ``..`` traversal.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

from .config import Settings

_DEFAULT_SETTINGS: Optional[Settings] = None


def _default_settings() -> Settings:
    global _DEFAULT_SETTINGS
    if _DEFAULT_SETTINGS is None:
        _DEFAULT_SETTINGS = Settings()
    return _DEFAULT_SETTINGS


class NodePath:
    """A path relative to the node home, with local filesystem operations."""

    def __init__(self, rel: str, *, settings: Optional[Settings] = None) -> None:
        self.settings = settings or _default_settings()
        self.rel = str(rel).strip("/")

    def _local_path(self) -> Path:
        root = self.settings.node_home.resolve()
        target = (root / self.rel).resolve() if self.rel else root
        # Confine to node_home — reject traversal that climbs out of the root.
        if root not in target.parents and target != root:
            raise ValueError(f"path {self.rel!r} escapes the node home {root}")
        return target

    def mkdir(self, *, parents: bool = True, exist_ok: bool = True) -> None:
        self._local_path().mkdir(parents=parents, exist_ok=exist_ok)

    def write_text(self, text: str) -> None:
        self._local_path().write_text(text)

    def read_text(self) -> str:
        return self._local_path().read_text()

    def write_bytes(self, data: bytes) -> None:
        self._local_path().write_bytes(data)

    def read_bytes(self) -> bytes:
        return self._local_path().read_bytes()

    def stat(self) -> os.stat_result:
        return self._local_path().stat()

    def iterdir(self) -> Iterator["NodePath"]:
        base = self.rel
        for child in self._local_path().iterdir():
            child_rel = f"{base}/{child.name}" if base else child.name
            yield NodePath(child_rel, settings=self.settings)

    def stream_read(self, chunk_size: int = 65536) -> Iterator[bytes]:
        with open(self._local_path(), "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def exists(self) -> bool:
        return self._local_path().exists()

    def __truediv__(self, other: str) -> "NodePath":
        child_rel = f"{self.rel}/{other}" if self.rel else str(other)
        return NodePath(child_rel, settings=self.settings)

    def __repr__(self) -> str:
        return f"NodePath({self.rel!r})"
