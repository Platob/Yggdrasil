from __future__ import annotations

import datetime as dt
import logging
from collections import deque
from threading import Lock

LOGGER = logging.getLogger("yggdrasil.audit")


class AuditLog:
    def __init__(self, max_entries: int = 10000):
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._lock = Lock()

    def log(self, operation: str, asset_type: str, asset_id: int, user_hash: int = 0, detail: str = ""):
        entry = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "operation": operation,
            "asset_type": asset_type,
            "asset_id": asset_id,
            "user_hash": user_hash,
            "detail": detail,
        }
        with self._lock:
            self._entries.append(entry)
        LOGGER.info("%s %s %d %s", operation, asset_type, asset_id, detail)

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            items = list(self._entries)
        return items[-limit:]
