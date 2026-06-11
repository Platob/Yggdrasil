"""Append-only audit log of node mutations.

Two backings: a bounded in-memory ring (``recent`` reads the tail cheaply) and
an ``audit.jsonl`` file for durability. The file handle is opened once and held
open — re-opening per entry dominated the cost of any mutation burst — and
flushed per write so a crash loses at most the OS buffer. A lock guards both so
concurrent request handlers don't interleave.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path


class AuditLog:
    def __init__(self, settings, max_entries: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._logs_root = Path(settings.logs_root)
        self._logs_root.mkdir(parents=True, exist_ok=True)
        # Held open for the node's lifetime; line-buffered append.
        self._fh = open(self._logs_root / "audit.jsonl", "a", buffering=1)

    def log(self, action: str, resource: str, resource_id, *, detail: str = "") -> None:
        entry = {
            "ts": time.time(),
            "action": action,
            "resource": resource,
            "resource_id": str(resource_id),
            "detail": detail,
        }
        line = json.dumps(entry, separators=(",", ":"))
        with self._lock:
            self._entries.append(entry)
            self._fh.write(line + "\n")

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            n = len(self._entries)
            if limit >= n:
                return list(self._entries)
            # Tail window, most-recent-last.
            return list(self._entries)[n - limit:]

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()
