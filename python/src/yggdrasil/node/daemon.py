"""Bot daemon lifecycle — directory setup, log rotation, auto-spawn.

Called by every ``ygg`` CLI invocation to ensure a local bot is
running.  The daemon:

1. Creates ``~/.bot/{user_key}/`` with ``data/``, ``cache/``,
   ``spill/``, ``logs/`` subdirectories.
2. Purges log files older than ``log_retention_days`` (default 7).
3. Scans for an open port if the configured one is busy.
4. Starts the bot server in a background process if none is running.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from yggdrasil.node.config import Settings, _find_open_port, get_settings

LOGGER = logging.getLogger(__name__)

_PID_FILE = "node.pid"
_PORT_FILE = "node.port"


def ensure_directories(settings: Settings) -> None:
    for d in (settings.node_home, settings.data_root, settings.cache_root,
              settings.spill_root, settings.logs_root):
        d.mkdir(parents=True, exist_ok=True)


def cleanup_old_logs(settings: Settings) -> int:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=settings.log_retention_days)
    cutoff_ts = cutoff.timestamp()
    removed = 0
    if not settings.logs_root.exists():
        return 0
    for f in settings.logs_root.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff_ts:
            f.unlink(missing_ok=True)
            removed += 1
    if removed:
        LOGGER.debug("Cleaned up %d old log files", removed)
    return removed


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect((host, port))
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _is_node_running(settings: Settings) -> tuple[bool, int | None, int | None]:
    pid_path = settings.node_home / _PID_FILE
    port_path = settings.node_home / _PORT_FILE

    if not pid_path.exists():
        return False, None, None

    try:
        pid = int(pid_path.read_text().strip())
        port = int(port_path.read_text().strip()) if port_path.exists() else settings.port
    except (ValueError, OSError):
        return False, None, None

    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        pid_path.unlink(missing_ok=True)
        port_path.unlink(missing_ok=True)
        return False, None, None

    if _is_port_open("127.0.0.1", port):
        return True, pid, port

    return False, None, None


def _write_pid(settings: Settings, pid: int, port: int) -> None:
    (settings.node_home / _PID_FILE).write_text(str(pid))
    (settings.node_home / _PORT_FILE).write_text(str(port))


def get_node_url(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    ensure_directories(settings)
    port_path = settings.node_home / _PORT_FILE
    if port_path.exists():
        try:
            port = int(port_path.read_text().strip())
            return f"http://127.0.0.1:{port}"
        except (ValueError, OSError):
            pass
    return f"http://127.0.0.1:{settings.port}"


def spawn_node(settings: Settings | None = None, *, allow_remote: bool = False) -> tuple[int, int]:
    """Ensure a bot is running. Returns (pid, port).

    If a bot is already running, returns its pid/port.
    Otherwise spawns a new background process.
    """
    settings = settings or get_settings()
    ensure_directories(settings)
    cleanup_old_logs(settings)

    running, pid, port = _is_node_running(settings)
    if running:
        return pid, port

    port = _find_open_port(settings.port, settings.port + 100)

    log_file = settings.logs_root / f"node-{dt.date.today().isoformat()}.log"

    env = os.environ.copy()
    env["YGG_NODE_PORT"] = str(port)
    env["YGG_NODE_HOME"] = str(settings.node_home)
    if allow_remote:
        env["YGG_NODE_ALLOW_REMOTE"] = "1"

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "yggdrasil.node.main"],
            env=env,
            stdout=lf,
            stderr=lf,
            start_new_session=True,
        )

    for _ in range(30):
        time.sleep(0.2)
        if _is_port_open("127.0.0.1", port):
            break

    _write_pid(settings, proc.pid, port)
    LOGGER.info("Spawned node (pid=%d, port=%d, log=%s)", proc.pid, port, log_file)
    return proc.pid, port


def stop_node(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    pid_path = settings.node_home / _PID_FILE

    if not pid_path.exists():
        return False

    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass

    pid_path.unlink(missing_ok=True)
    (settings.node_home / _PORT_FILE).unlink(missing_ok=True)
    return True
