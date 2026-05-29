from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from ...config import Settings
from ..schemas.backend import GpuInfo, NetworkIO, NodeBackend
from ..schemas.common import NodeRole

_GPU_QUERY = (
    "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,"
    "power.draw,power.limit"
)


def _gpu_num(value: str) -> float:
    """Parse an nvidia-smi numeric field, tolerating ``[N/A]`` /
    ``[Not Supported]`` (returned for power on some cards) as 0.0."""
    try:
        return float(value)
    except ValueError:
        return 0.0


class BackendService:
    """Collects node resource metrics: CPU, RAM, GPU, disk, network."""

    def __init__(
        self,
        settings: Settings,
        *,
        role: NodeRole = NodeRole.HYBRID,
        history_size: int = 120,
    ) -> None:
        self.settings = settings
        self.role = role
        self._start_time = time.monotonic()
        self._history: deque[NodeBackend] = deque(maxlen=history_size)
        self._lock = Lock()
        self._last_collect: float = 0
        self._active_runs_fn = lambda: 0
        self._total_runs_fn = lambda: 0

    def bind_run_counters(self, active_fn, total_fn) -> None:
        self._active_runs_fn = active_fn
        self._total_runs_fn = total_fn

    def set_role(self, role: NodeRole) -> None:
        self.role = role

    def snapshot(self) -> NodeBackend:
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

    def history(self, limit: int = 60) -> list[NodeBackend]:
        with self._lock:
            items = list(self._history)
        return items[-limit:]

    def _collect(self) -> NodeBackend:
        ts = datetime.now(timezone.utc).isoformat()
        uptime = time.monotonic() - self._start_time

        cpu_count = os.cpu_count() or 1
        cpu_percent = 0.0
        mem_used = 0.0
        mem_total = 0.0
        disk_used = 0.0
        disk_total = 0.0
        net = NetworkIO()

        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            mem_used = mem.used / (1024 * 1024)
            mem_total = mem.total / (1024 * 1024)
            disk = psutil.disk_usage("/")
            disk_used = disk.used / (1024 * 1024)
            disk_total = disk.total / (1024 * 1024)
            nio = psutil.net_io_counters()
            net = NetworkIO(
                bytes_sent=nio.bytes_sent,
                bytes_recv=nio.bytes_recv,
                packets_sent=nio.packets_sent,
                packets_recv=nio.packets_recv,
            )
        except ImportError:
            pass

        gpus = self._collect_gpus()

        return NodeBackend(
            node_id=self.settings.node_id,
            role=self.role,
            hostname=platform.node(),
            platform=f"{platform.system()} {platform.release()}",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            memory_used_mb=round(mem_used, 1),
            memory_total_mb=round(mem_total, 1),
            disk_used_mb=round(disk_used, 1),
            disk_total_mb=round(disk_total, 1),
            gpus=gpus,
            network=net,
            uptime_seconds=round(uptime, 1),
            active_runs=self._active_runs_fn(),
            total_runs=self._total_runs_fn(),
            timestamp=ts,
        )

    @staticmethod
    def _collect_gpus() -> list[GpuInfo]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={_GPU_QUERY}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        gpus: list[GpuInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                gpus.append(GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_used_mb=_gpu_num(parts[2]),
                    memory_total_mb=_gpu_num(parts[3]),
                    utilization_percent=_gpu_num(parts[4]),
                    temperature_c=_gpu_num(parts[5]),
                    # Power can be ``[N/A]`` on some cards — tolerate it
                    # rather than dropping the whole GPU row.
                    power_draw_w=_gpu_num(parts[6]) if len(parts) > 6 else 0.0,
                    power_limit_w=_gpu_num(parts[7]) if len(parts) > 7 else 0.0,
                ))
            except (ValueError, IndexError):
                continue
        return gpus
