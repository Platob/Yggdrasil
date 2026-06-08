"""Mesh store — shared memory the fleet's process agents coordinate through.

When :class:`~yggdrasil.loki.fleet.Fleet` runs a swarm it points every agent at
a shared workspace and a shared :class:`MeshStore` (a JSON key-value file). An
agent **publishes** what it produced — a file path, an API it defined, a
decision — and its peers **read** the mesh before doing redundant work, so the
swarm reuses results instead of recomputing them. That's the agents talking to
each other: shared files + shared memory.

The agent reaches it through the ``mesh`` tool (:mod:`yggdrasil.loki.tools`),
which is added to the toolbox whenever the ``LOKI_MESH`` env var names the store.
Writes are atomic (temp-file replace) under a best-effort cross-process lock, so
concurrent agents don't clobber each other.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

__all__ = ["MeshStore"]


class MeshStore:
    """A small JSON key-value store shared across the fleet's agent processes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), "utf-8")
        tmp.replace(self.path)                       # atomic publish

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """Best-effort cross-process exclusive lock (``fcntl`` where available)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path.with_suffix(".lock"), "w")
        try:
            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX)
            except Exception:
                pass                                 # no fcntl (Windows) → unlocked best-effort
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)
            except Exception:
                pass
            handle.close()

    def all(self) -> dict[str, Any]:
        """Every published entry (the peers' shared state)."""
        return self._load()

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def put(self, key: str, value: Any) -> Any:
        """Publish *value* under *key* for peers to read (read-modify-write)."""
        with self._lock():
            data = self._load()
            data[key] = value
            self._write(data)
        return value

    def append(self, key: str, value: Any) -> list:
        """Append *value* to a list under *key* — a shared running log."""
        with self._lock():
            data = self._load()
            current = data.get(key)
            items = list(current) if isinstance(current, list) else ([current] if current is not None else [])
            items.append(value)
            data[key] = items
            self._write(data)
        return items


def from_env() -> "MeshStore | None":
    """The mesh store named by ``LOKI_MESH`` (set by the fleet on each agent), or
    ``None`` when this process isn't part of a mesh."""
    path = os.getenv("LOKI_MESH")
    return MeshStore(path) if path else None
