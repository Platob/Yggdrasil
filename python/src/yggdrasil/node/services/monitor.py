"""System monitor — caches a snapshot for ``_ttl`` seconds to bound collection cost."""
from __future__ import annotations

import time

from pydantic import BaseModel


class MonitorSnapshot(BaseModel):
    ts: float
    cpu_percent: float
    mem_percent: float
    disk_percent: float


class MonitorService:
    """Collects CPU/mem/disk, re-sampling at most once per ``_ttl`` seconds."""

    def __init__(self, settings: object, history_size: int = 1000) -> None:
        self._settings = settings
        self._history: list[MonitorSnapshot] = []
        self._history_size = history_size
        self._last_collect: float = 0.0
        self._ttl: float = 1.0

    def snapshot(self) -> MonitorSnapshot:
        now = time.time()
        if now - self._last_collect < self._ttl and self._history:
            return self._history[-1]
        self._last_collect = now
        try:
            import psutil

            snap = MonitorSnapshot(
                ts=now,
                cpu_percent=psutil.cpu_percent(),
                mem_percent=psutil.virtual_memory().percent,
                disk_percent=psutil.disk_usage("/").percent,
            )
        except Exception:
            snap = MonitorSnapshot(ts=now, cpu_percent=0.0, mem_percent=0.0, disk_percent=0.0)
        self._history.append(snap)
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size:]
        return snap

    def history(self, limit: int = 100) -> list[MonitorSnapshot]:
        return self._history[-limit:]
