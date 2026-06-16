"""Append-only audit log.

Every mutation is one JSON line appended to ``logs_root/audit.jsonl``. The
file handle is opened **once** in append mode and reused for the life of
the instance — the old per-entry ``open()`` dominated mutation bursts. The
most recent ``max_entries`` are also kept in memory to back the GET
endpoint without re-reading the file.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any


class AuditLog:
    """JSONL audit trail with an in-memory tail.

    Only requires ``settings.logs_root`` — any object exposing that
    attribute works, which keeps the benchmark's stub settings valid.
    """

    def __init__(self, settings: Any, *, max_entries: int = 10_000) -> None:
        self.settings = settings
        self.max_entries = max_entries
        self._entries: deque[dict] = deque(maxlen=max_entries)

        root = Path(settings.logs_root)
        root.mkdir(parents=True, exist_ok=True)
        self._path = root / "audit.jsonl"
        # Open once and keep it — line-buffered so each entry is flushed
        # without a fresh open()/close() per write.
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")

    def log(
        self,
        action: str,
        kind: str,
        ref: int | str | None = None,
        *,
        detail: str = "",
    ) -> dict:
        entry = {
            "ts": time.time(),
            "action": action,
            "kind": kind,
            "ref": ref,
            "detail": detail,
        }
        self._fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._entries.append(entry)
        return entry

    def recent(self, limit: int = 20) -> list[dict]:
        if limit <= 0:
            return []
        entries = list(self._entries)
        return entries[-limit:]

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __del__(self) -> None:  # best-effort flush on GC
        try:
            self.close()
        except Exception:
            pass
