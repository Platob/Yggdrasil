"""In-memory audit log with periodic disk flush.

The old log opened ``audit.jsonl`` per entry, which dominated any mutation
burst. This keeps a bounded ``deque`` in memory (the read path) and appends
buffered lines to ``audit.jsonl`` in batches — the file handle is opened once
per flush, not once per entry.
"""
from __future__ import annotations

import json
import time
from collections import deque
from typing import Any

from pydantic import BaseModel


class AuditEntry(BaseModel):
    action: str
    resource_type: str
    resource_id: Any
    detail: str = ""
    ts: float


class AuditLog:
    def __init__(self, settings: Any, max_entries: int = 10_000, flush_every: int = 256) -> None:
        self.settings = settings
        self._entries: deque[AuditEntry] = deque(maxlen=max_entries)
        self._pending: list[str] = []
        self._flush_every = flush_every
        self._path = None  # resolved lazily on first flush

    def log(self, action: str, resource_type: str, resource_id: Any, detail: str = "") -> None:
        entry = AuditEntry(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            detail=detail,
            ts=time.time(),
        )
        self._entries.append(entry)
        self._pending.append(
            json.dumps({"action": action, "resource_type": resource_type,
                        "resource_id": resource_id, "detail": detail, "ts": entry.ts})
        )
        if len(self._pending) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        if self._path is None:
            root = self.settings.logs_root
            root.mkdir(parents=True, exist_ok=True)
            self._path = root / "audit.jsonl"
        with open(self._path, "a") as fh:
            fh.write("\n".join(self._pending))
            fh.write("\n")
        self._pending.clear()

    def get_entries(self, limit: int = 100) -> list[AuditEntry]:
        # Most-recent first, bounded by limit; slice the deque tail.
        n = len(self._entries)
        if limit >= n:
            return list(reversed(self._entries))
        return list(reversed(list(self._entries)[n - limit:]))
