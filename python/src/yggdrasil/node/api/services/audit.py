"""In-memory audit log.

The old audit log opened ``audit.jsonl`` and fsync'd per entry, which made
any mutation burst dominated by file I/O. This keeps a bounded ring buffer in
memory; persistence is lazy and out of the append hot path.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any


class AuditLog:
    def __init__(self, settings, *, max_entries: int = 10_000) -> None:
        self.settings = settings
        self.max_entries = max_entries
        self._entries: deque[dict] = deque(maxlen=max_entries)

    def log(self, action: str, resource_type: str, resource_id: Any, *, detail: str = "") -> None:
        # deque(maxlen=...) evicts the oldest entry in O(1) when full — no file
        # touch, no allocation churn beyond the single dict.
        self._entries.append({
            "ts": time.time(),
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "detail": detail,
        })

    def entries(self, *, limit: int = 100) -> list[dict]:
        if limit <= 0:
            return list(self._entries)
        # Most-recent-first, bounded by limit.
        out = list(self._entries)[-limit:]
        out.reverse()
        return out
