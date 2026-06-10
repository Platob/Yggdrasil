"""Node resource monitor — rate-limited CPU/memory/uptime snapshots.

:meth:`snapshot` returns the last collected sample unless ``_collect_interval``
seconds have elapsed since ``_last_collect`` (monotonic), so a tight poll loop
doesn't hammer ``psutil`` once per call. History keeps the last
``history_size`` collected samples in a deque. ``psutil`` is optional — without
it the snapshot falls back to stdlib ``os``/``resource`` figures.
"""
from __future__ import annotations

import os
import time
from collections import deque

from yggdrasil.node.config import Settings

try:
    import psutil

    _PROC = psutil.Process()
except Exception:  # psutil not installed — stdlib fallback
    psutil = None
    _PROC = None


class MonitorService:
    def __init__(self, settings: Settings, history_size: int = 100) -> None:
        self.settings = settings
        self._history: deque[dict] = deque(maxlen=history_size)
        self._collect_interval = 0.5
        self._started = time.monotonic()
        self._last_collect = 0.0
        self._last: dict = {}

    def snapshot(self) -> dict:
        now = time.monotonic()
        if self._last and (now - self._last_collect) < self._collect_interval:
            return self._last

        if psutil is not None:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            sample = {
                "cpu_percent": cpu,
                "memory_used": int(mem.used),
                "memory_total": int(mem.total),
                "memory_percent": float(mem.percent),
                "uptime_s": now - self._started,
            }
        else:
            load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
            try:
                import resource

                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            except Exception:
                rss = 0
            sample = {
                "cpu_percent": load,
                "memory_used": int(rss),
                "memory_total": 0,
                "memory_percent": 0.0,
                "uptime_s": now - self._started,
            }

        self._last = sample
        self._last_collect = now
        self._history.append(sample)
        return sample

    def history(self) -> list[dict]:
        return list(self._history)
