"""System resource snapshots for the node dashboard.

Uses psutil when installed for accurate cpu/mem/disk; falls back to ``/proc``
and ``os`` primitives otherwise so the endpoint always answers (the front polls
it on a timer and a missing optional dep should never 500). Snapshots are
cached for a short interval and ring-buffered for a small history.
"""
from __future__ import annotations

import collections
import datetime as dt
import os
import shutil
import time

from ..config import Settings

try:
    import psutil as _psutil
except ImportError:  # psutil is optional; we degrade to /proc + os
    _psutil = None


class MonitorService:
    def __init__(self, settings: Settings, history_size: int = 100) -> None:
        self.settings = settings
        self._history: collections.deque[dict] = collections.deque(maxlen=history_size)
        self._last_collect: float = 0.0
        self._cached: dict = {}

    def snapshot(self) -> dict:
        now = time.time()
        # Recollect at most ~5x/sec; the bench resets _last_collect=0 to force it.
        if self._cached and (now - self._last_collect) < 0.2:
            return self._cached
        snap = self._collect()
        self._cached = snap
        self._last_collect = now
        self._history.append(snap)
        return snap

    def history(self) -> list[dict]:
        return list(self._history)

    def _collect(self) -> dict:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        if _psutil is not None:
            return {
                "cpu_percent": _psutil.cpu_percent(interval=None),
                "memory_percent": _psutil.virtual_memory().percent,
                "disk_percent": _psutil.disk_usage(str(self.settings.node_home if self.settings.node_home.exists() else "/")).percent,
                "timestamp": ts,
            }
        # psutil-free fallback.
        try:
            load1, _, _ = os.getloadavg()
            ncpu = os.cpu_count() or 1
            cpu_percent = min(100.0, load1 / ncpu * 100.0)
        except (OSError, AttributeError):
            cpu_percent = 0.0
        mem_percent = 0.0
        try:
            with open("/proc/meminfo") as fh:
                meminfo = {}
                for line in fh:
                    k, _, rest = line.partition(":")
                    meminfo[k] = int(rest.strip().split()[0])  # kB
            total = meminfo.get("MemTotal", 0)
            avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            if total:
                mem_percent = (total - avail) / total * 100.0
        except (OSError, ValueError, IndexError):
            pass
        usage = shutil.disk_usage(str(self.settings.node_home) if self.settings.node_home.exists() else "/")
        disk_percent = usage.used / usage.total * 100.0 if usage.total else 0.0
        return {
            "cpu_percent": round(cpu_percent, 1),
            "memory_percent": round(mem_percent, 1),
            "disk_percent": round(disk_percent, 1),
            "timestamp": ts,
        }
