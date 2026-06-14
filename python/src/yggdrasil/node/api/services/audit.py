"""Audit log — in-memory bounded ring; the old per-entry jsonl write was the bottleneck.

Mutations append an :class:`AuditEntry` to an in-memory ring capped at
``max_entries``; we deliberately do NOT touch disk per entry (opening
``audit.jsonl`` on every mutation dominated burst latency). Callers that want
durability flush the ring explicitly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AuditEntry:
    ts: float
    action: str
    resource_type: str
    resource_id: int
    detail: str = ""


class AuditLog:
    """Bounded in-memory audit ring."""

    def __init__(self, settings: object, max_entries: int = 10_000) -> None:
        self._path = Path(settings.logs_root) / "audit.jsonl"
        self._max_entries = max_entries
        self._entries: list[AuditEntry] = []
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, action: str, resource_type: str, resource_id: int, detail: str = "") -> None:
        self._entries.append(
            AuditEntry(
                ts=time.time(),
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                detail=detail,
            )
        )
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def get(self, limit: int = 100) -> list[dict]:
        return [vars(e) for e in self._entries[-limit:]]
