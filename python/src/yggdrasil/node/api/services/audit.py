"""Audit log — append-mostly mutation trail with batched flush.

Every mutating op (create/update/delete a function, write a file, …) logs
one :class:`AuditEntry`. The old shape opened ``audit.jsonl`` per entry,
which dominated any mutation burst; this keeps the live tail in a
``deque(maxlen=max_entries)`` and flushes to ``logs_root/audit.jsonl`` only
every :data:`_FLUSH_EVERY` entries (and on :meth:`close`). :meth:`recent`
serves the tail straight from memory.
"""
from __future__ import annotations

import datetime as dt
from collections import deque

import orjson
from pydantic import BaseModel

_FLUSH_EVERY = 1000


class AuditEntry(BaseModel):
    ts: str
    action: str
    resource_type: str
    resource_id: int | str
    detail: str


class AuditLog:
    def __init__(self, settings, max_entries: int = 10_000) -> None:
        self.logs_root = settings.logs_root
        self._entries: deque[AuditEntry] = deque(maxlen=max_entries)
        self._pending: list[AuditEntry] = []

    def log(self, action: str, resource_type: str, resource_id: int | str, detail: str = "") -> None:
        entry = AuditEntry(
            ts=dt.datetime.now(dt.timezone.utc).isoformat(),
            action=action, resource_type=resource_type,
            resource_id=resource_id, detail=detail,
        )
        self._entries.append(entry)
        self._pending.append(entry)
        if len(self._pending) >= _FLUSH_EVERY:
            self.flush()

    def recent(self, limit: int = 100) -> list[AuditEntry]:
        n = len(self._entries)
        start = max(0, n - limit)
        return [self._entries[i] for i in range(start, n)]

    def flush(self) -> None:
        if not self._pending or self.logs_root is None:
            self._pending.clear()
            return
        self.logs_root.mkdir(parents=True, exist_ok=True)
        with (self.logs_root / "audit.jsonl").open("ab") as f:
            f.write(b"".join(orjson.dumps(e.model_dump()) + b"\n" for e in self._pending))
        self._pending.clear()

    def close(self) -> None:
        self.flush()
