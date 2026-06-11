"""Path rooted at the node home.

A thin convenience over :class:`pathlib.Path` so callers address node-local
files by a relative key without knowing where ``node_home`` lives. Every op
delegates to the resolved local path; ``stream_read`` chunks bytes off disk so
a large file never lands in RAM whole.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


class NodePath:
    def __init__(self, relative: str, node_home: Path | None = None) -> None:
        self.relative = str(relative)
        self.node_home = Path(node_home) if node_home is not None else Path.home() / ".ygg" / "node"

    def _local_path(self) -> Path:
        return self.node_home / self.relative

    def __truediv__(self, other: str) -> "NodePath":
        return NodePath(str(Path(self.relative) / other), self.node_home)

    def __repr__(self) -> str:
        return f"<NodePath {self.relative!r} @ {self.node_home}>"

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> "NodePath":
        self._local_path().mkdir(parents=parents, exist_ok=exist_ok)
        return self

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

    def iterdir(self) -> Iterator[Path]:
        return self._local_path().iterdir()

    def stream_read(self, chunk_size: int = 65536) -> Iterator[bytes]:
        with open(self._local_path(), "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                yield chunk
