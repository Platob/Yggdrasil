"""Virtual node paths.

:class:`NodePath` is a thin, chainable view over the local filesystem
rooted at ``node_home / <name>``. It mirrors the subset of
:class:`pathlib.Path` the node needs — ``mkdir``, ``read_text`` /
``write_text``, ``read_bytes`` / ``write_bytes``, ``stat``, ``iterdir`` —
plus a chunked :meth:`NodePath.stream_read` so large files never have to
be loaded whole.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .config import Settings


def _node_home() -> Path:
    """Root for all node paths — the default :class:`Settings` node home."""
    return Settings().node_home


class NodePath:
    """A virtual path under ``node_home``.

    The first segment is the logical entry name; further segments are
    joined with ``/``::

        p = NodePath("project")
        (p / "src" / "main.py").write_text("...")
    """

    __slots__ = ("name", "_parts")

    def __init__(self, name: str, *parts: str) -> None:
        self.name = name
        self._parts = tuple(parts)

    def _local_path(self) -> Path:
        return _node_home().joinpath(self.name, *self._parts)

    def __truediv__(self, other: str) -> "NodePath":
        return NodePath(self.name, *self._parts, other)

    def __repr__(self) -> str:
        return f"NodePath({self._local_path()!r})"

    def __fspath__(self) -> str:
        return str(self._local_path())

    @property
    def exists(self) -> bool:
        return self._local_path().exists()

    def mkdir(self, *, parents: bool = True, exist_ok: bool = True) -> None:
        self._local_path().mkdir(parents=parents, exist_ok=exist_ok)

    def write_text(self, text: str, *, encoding: str = "utf-8") -> int:
        return self._local_path().write_text(text, encoding=encoding)

    def read_text(self, *, encoding: str = "utf-8") -> str:
        return self._local_path().read_text(encoding=encoding)

    def write_bytes(self, data: bytes) -> int:
        return self._local_path().write_bytes(data)

    def read_bytes(self) -> bytes:
        return self._local_path().read_bytes()

    def stat(self):
        return self._local_path().stat()

    def iterdir(self) -> Iterator["NodePath"]:
        local = self._local_path()
        for child in local.iterdir():
            yield NodePath(self.name, *self._parts, child.name)

    def stream_read(self, *, chunk_size: int = 65536) -> Iterator[bytes]:
        """Yield the file's bytes in ``chunk_size`` chunks without loading
        the whole file into memory at once."""
        with open(self._local_path(), "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                yield chunk
