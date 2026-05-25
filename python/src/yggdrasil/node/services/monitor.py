"""Node resource and network IO monitoring."""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from ..config import Settings
from ..schemas.monitor import NetworkSnapshot, ResourceSnapshot


class MonitorService:
    def __init__(self, settings: Settings, *, history_size: int = 120) -> None:
        self.settings = settings
        self._history: deque[ResourceSnapshot] = deque(maxlen=history_size)
        self._lock = Lock()
        self._last_collect: float = 0

    def snapshot(self) -> ResourceSnapshot:
        now = time.monotonic()
        if now - self._last_collect < 0.5:
            with self._lock:
                if self._history:
                    return self._history[-1]

        snap = self._collect()
        with self._lock:
            self._history.append(snap)
            self._last_collect = now
        return snap

    def history(self, limit: int = 60) -> list[ResourceSnapshot]:
        with self._lock:
            items = list(self._history)
        return items[-limit:]

    @staticmethod
    def _collect() -> ResourceSnapshot:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            import psutil
        except ImportError:
            return ResourceSnapshot(timestamp=ts)

        cpu = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        net = psutil.net_io_counters()

        return ResourceSnapshot(
            cpu_percent=cpu,
            memory_percent=mem.percent,
            memory_used_mb=mem.used / (1024 * 1024),
            memory_total_mb=mem.total / (1024 * 1024),
            disk_percent=disk.percent,
            network=NetworkSnapshot(
                bytes_sent=net.bytes_sent,
                bytes_recv=net.bytes_recv,
                packets_sent=net.packets_sent,
                packets_recv=net.packets_recv,
                timestamp=ts,
            ),
            timestamp=ts,
        )
