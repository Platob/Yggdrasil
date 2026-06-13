"""System monitor service.

Collects CPU/memory snapshots into a ring buffer. Collection is cached for one
second so a burst of /api/monitor hits doesn't hammer psutil. Falls back to
/proc and os.getloadavg when psutil is unavailable.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass

from ..config import Settings

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


@dataclass
class Snapshot:
    ts: float
    cpu_pct: float
    mem_pct: float
    mem_mb: float
    load_1m: float


class MonitorService:
    def __init__(self, settings: Settings, *, history_size: int = 3600) -> None:
        self.settings = settings
        self._history: deque[Snapshot] = deque(maxlen=history_size)
        self._last_collect: float = 0.0
        self._last_snapshot: Snapshot | None = None

    def snapshot(self) -> Snapshot:
        now = time.time()
        if self._last_snapshot is not None and (now - self._last_collect) < 1.0:
            return self._last_snapshot

        if _HAS_PSUTIL:
            vm = psutil.virtual_memory()
            snap = Snapshot(
                ts=now,
                cpu_pct=psutil.cpu_percent(interval=None),
                mem_pct=vm.percent,
                mem_mb=vm.used / (1024 * 1024),
                load_1m=os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0,
            )
        else:
            load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
            ncpu = os.cpu_count() or 1
            snap = Snapshot(
                ts=now,
                cpu_pct=min(100.0, load / ncpu * 100.0),
                mem_pct=0.0,
                mem_mb=0.0,
                load_1m=load,
            )

        self._last_collect = now
        self._last_snapshot = snap
        self._history.append(snap)
        return snap

    def history(self, n: int = 60) -> list[Snapshot]:
        if n <= 0:
            return list(self._history)
        return list(self._history)[-n:]
