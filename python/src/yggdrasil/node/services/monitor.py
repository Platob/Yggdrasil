"""Node resource and network IO monitoring."""
from __future__ import annotations

import logging
import os
import platform
import socket
import time
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from ..config import Settings
from ..schemas.monitor import NetworkSnapshot, ResourceSnapshot

LOGGER = logging.getLogger(__name__)


def _static_system_info() -> dict[str, object]:
    """Collect host-level facts that never change at runtime."""
    info: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
    }
    try:
        import psutil
        info["cpu_logical"] = psutil.cpu_count(logical=True)
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["memory_total_mb"] = round(psutil.virtual_memory().total / (1024 * 1024), 1)
        info["boot_time"] = psutil.boot_time()
    except ImportError:
        info["cpu_logical"] = info["cpu_count"]
        info["cpu_physical"] = info["cpu_count"]
    return info


class MonitorService:
    def __init__(self, settings: Settings, *, history_size: int = 120) -> None:
        self.settings = settings
        self._history: deque[ResourceSnapshot] = deque(maxlen=history_size)
        self._lock = Lock()
        self._last_collect: float = 0
        # Pre-compute and cache the static host info once at startup.
        self._system_info: dict[str, object] = _static_system_info()

    @property
    def system_info(self) -> dict[str, object]:
        """Cached static info (hostname, cpu count, memory total, etc.)."""
        return self._system_info

    def snapshot(self) -> ResourceSnapshot:
        now = time.monotonic()
        # Coalesce: bursty callers reuse the previous sample for up to 500ms.
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

        # interval=0 → instant non-blocking sample; psutil computes delta
        # against the previous internal call. First call returns 0.0 — fine
        # for a long-running service.
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
