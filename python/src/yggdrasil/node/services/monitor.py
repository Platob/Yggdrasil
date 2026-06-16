"""System monitor.

Collects a lightweight resource snapshot (CPU %, memory %, process info,
uptime) and caches it for a short TTL so a burst of ``/monitor`` calls
doesn't hammer the OS. Uses ``psutil`` when available and falls back to
stdlib (``os``, ``resource``, ``/proc``) when it isn't.

The benchmark forces a fresh collect by resetting ``_last_collect`` to 0
between calls.
"""
from __future__ import annotations

import os
import time
from collections import deque

from ..config import Settings

try:  # optional, richer metrics when present
    import psutil  # type: ignore

    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover - psutil not installed
    psutil = None  # type: ignore
    _HAVE_PSUTIL = False


_TTL_SECONDS = 1.0


class MonitorService:
    """Cached system resource snapshots with bounded history."""

    def __init__(self, settings: Settings, *, history_size: int = 1000) -> None:
        self.settings = settings
        self._history: deque[dict] = deque(maxlen=history_size)
        self._started = time.time()
        self._last_collect = 0.0
        self._cached: dict | None = None

    def snapshot(self) -> dict:
        now = time.time()
        if self._cached is not None and (now - self._last_collect) < _TTL_SECONDS:
            return self._cached
        snap = self._collect()
        self._last_collect = now
        self._cached = snap
        self._history.append(snap)
        return snap

    def history(self) -> list[dict]:
        return list(self._history)

    def _collect(self) -> dict:
        now = time.time()
        if _HAVE_PSUTIL:
            vm = psutil.virtual_memory()
            proc = psutil.Process()
            return {
                "node_id": self.settings.node_id,
                "timestamp": now,
                "uptime": now - self._started,
                "cpu_percent": psutil.cpu_percent(interval=None),
                "mem_percent": vm.percent,
                "mem_used_mb": vm.used / (1024 * 1024),
                "mem_total_mb": vm.total / (1024 * 1024),
                "process_rss_mb": proc.memory_info().rss / (1024 * 1024),
                "num_threads": proc.num_threads(),
            }

        # stdlib fallback — no psutil. Load average stands in for CPU
        # pressure; memory is read from /proc/meminfo on Linux.
        try:
            load1, _, _ = os.getloadavg()
            ncpu = os.cpu_count() or 1
            cpu_percent = min(100.0, load1 / ncpu * 100.0)
        except (OSError, AttributeError):
            cpu_percent = 0.0

        mem_total_mb = mem_used_mb = mem_percent = 0.0
        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    meminfo[key.strip()] = int(rest.strip().split()[0])  # kB
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            if total_kb:
                mem_total_mb = total_kb / 1024
                mem_used_mb = (total_kb - avail_kb) / 1024
                mem_percent = (total_kb - avail_kb) / total_kb * 100.0
        except (OSError, ValueError):
            pass

        return {
            "node_id": self.settings.node_id,
            "timestamp": now,
            "uptime": now - self._started,
            "cpu_percent": cpu_percent,
            "mem_percent": mem_percent,
            "mem_used_mb": mem_used_mb,
            "mem_total_mb": mem_total_mb,
            "process_rss_mb": 0.0,
            "num_threads": 0,
        }
