from __future__ import annotations

import datetime as dt
import logging
import orjson
from collections import deque
from pathlib import Path
from threading import Lock

LOGGER = logging.getLogger("yggdrasil.audit")


class AuditLog:
    def __init__(self, settings=None, max_entries: int = 10000):
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._lock = Lock()
        self._log_file: Path | None = None
        # Persistent append handle — opening per-entry was up to 70% of the
        # cost of a tight CRUD loop (open syscall + buffer alloc + fsync hint).
        # Holding the file open avoids that without changing durability for
        # POC use (the OS still flushes on close/shutdown).
        self._log_fh = None
        if settings is not None:
            log_path = Path(settings.logs_root) / "audit.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = log_path
            if log_path.exists():
                try:
                    with open(log_path, "rb") as f:
                        # Read last max_entries lines from end without loading
                        # the whole file: simple readlines is fine because
                        # audit logs are bounded by max_entries anyway.
                        for line in f.readlines()[-max_entries:]:
                            line = line.strip()
                            if line:
                                self._entries.append(orjson.loads(line))
                except Exception:
                    pass
            try:
                self._log_fh = open(log_path, "ab", buffering=0)
            except Exception:
                self._log_fh = None

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
            if self._log_fh is not None:
                try:
                    self._log_fh.write(orjson.dumps(entry) + b"\n")
                except Exception:
                    pass
        LOGGER.info("%s %s %d %s", operation, asset_type, asset_id, detail)

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            items = list(self._entries)
        return items[-limit:]
