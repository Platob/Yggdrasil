from __future__ import annotations

import datetime as dt
import json
import logging
from collections import deque
from pathlib import Path
from threading import Lock

LOGGER = logging.getLogger("yggdrasil.audit")


class AuditLog:
    def __init__(self, settings=None, max_entries: int = 10000):
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._lock = Lock()
        self._log_file: Path | None = None
        if settings is not None:
            log_path = Path(settings.logs_root) / "audit.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = log_path
            # Load recent entries on startup
            if log_path.exists():
                try:
                    with open(log_path, "r") as f:
                        for line in f.readlines()[-max_entries:]:
                            line = line.strip()
                            if line:
                                self._entries.append(json.loads(line))
                except Exception:
                    pass

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
            if self._log_file is not None:
                try:
                    with open(self._log_file, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                except Exception:
                    pass
        LOGGER.info("%s %s %d %s", operation, asset_type, asset_id, detail)

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            items = list(self._entries)
        return items[-limit:]
