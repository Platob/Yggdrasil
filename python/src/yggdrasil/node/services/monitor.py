"""MonitorService — throttled system metrics snapshot.

``snapshot`` returns cpu/mem/disk percentages. ``psutil.cpu_percent`` is the
expensive call, so collection is throttled: within ``min_interval`` seconds of
the last collect the cached snapshot is returned, which is what keeps the
``GET /api/monitor`` hot path cheap. The bench forces re-collect by resetting
``_last_collect``.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any

import psutil
from pydantic import BaseModel


class MonitorSnapshot(BaseModel):
    cpu_pct: float
    mem_pct: float
    disk_pct: float
    ts: float


class MonitorService:
    def __init__(self, settings: Any, history_size: int = 256, min_interval: float = 0.5) -> None:
        self.settings = settings
        self._history: deque[MonitorSnapshot] = deque(maxlen=history_size)
        self._min_interval = min_interval
        self._last_collect = 0.0
        self._last: MonitorSnapshot | None = None

    def snapshot(self) -> MonitorSnapshot:
        now = time.time()
        if self._last is not None and (now - self._last_collect) < self._min_interval:
            return self._last
        snap = MonitorSnapshot(
            cpu_pct=psutil.cpu_percent(interval=None),
            mem_pct=psutil.virtual_memory().percent,
            disk_pct=psutil.disk_usage(str(self.settings.node_home.anchor or "/")).percent,
            ts=now,
        )
        self._last = snap
        self._last_collect = now
        self._history.append(snap)
        return snap

    def history(self) -> list[MonitorSnapshot]:
        return list(self._history)
