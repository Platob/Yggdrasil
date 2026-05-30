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
import platform
import signal
import socket
import subprocess
import sys
import time
from typing import Callable

from yggdrasil.node.config import Settings, _find_open_port, get_settings

LOGGER = logging.getLogger(__name__)

_PID_FILE = "node.pid"
_PORT_FILE = "node.port"


def ensure_directories(settings: Settings) -> None:
    for d in (settings.node_home, settings.data_root, settings.cache_root,
              settings.spill_root, settings.logs_root, settings.tmp_root,
              settings.stg_root, settings.saga_root, settings.saga_data_root,
              settings.saga_log_root):
        d.mkdir(parents=True, exist_ok=True)


def cleanup_tmp(settings: Settings) -> int:
    """Reclaim expired scratch entries from ``tmp/``, ``stg/`` and ``spill/``.

    Entries are named ``{prefix}-{start_ms}-{end_ms}-{suffix}`` so expiry is
    read from the filename (no per-entry stat). ``tmp`` uses ``tmp_ttl`` as the
    fallback for any foreign files; ``stg`` is name-only (persistent staging);
    legacy ``spill`` files fall back to ``tmp_ttl`` by mtime. Returns the count
    removed.
    """
    from . import scratch

    now = scratch.now_ms()
    removed = scratch.sweep(settings.tmp_root, now=now, fallback_ttl_seconds=settings.tmp_ttl)
    removed += scratch.sweep(settings.stg_root, now=now)
    removed += scratch.sweep(settings.spill_root, now=now, fallback_ttl_seconds=settings.tmp_ttl)
    if removed:
        LOGGER.info("scratch janitor reclaimed %d entries", removed)
    return removed


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


def spawn_node(
    settings: Settings | None = None,
    *,
    host: str = "0.0.0.0",
    ready_timeout: float | None = None,
    on_progress: Callable[[float, bool], None] | None = None,
) -> tuple[int, int]:
    """Ensure a node is running. Returns (pid, port).

    Defaults to public binding (0.0.0.0) with remote access enabled.

    Cold-booting the server imports FastAPI, pyarrow and the whole app,
    which can take well over 6s on Windows — so we wait up to
    ``ready_timeout`` seconds (default ``YGG_NODE_BOOT_TIMEOUT`` env or 30s)
    for the port to accept connections, and bail early if the child
    process dies during startup. ``on_progress(elapsed, ready)`` is called
    each poll tick (and once at the end) so callers can render a live
    spinner instead of hanging silently.
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
    env["YGG_NODE_HOST"] = host
    env["YGG_NODE_HOME"] = str(settings.node_home)
    env["YGG_NODE_NODE_ID"] = settings.node_id
    env["YGG_NODE_ALLOW_REMOTE"] = "1"

    popen_kwargs: dict = {}
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "yggdrasil.node.main"],
            env=env,
            stdout=lf,
            stderr=lf,
            **popen_kwargs,
        )

    budget = ready_timeout if ready_timeout is not None else float(
        os.getenv("YGG_NODE_BOOT_TIMEOUT", "30")
    )
    start = time.monotonic()
    deadline = start + budget
    ready = False
    while time.monotonic() < deadline:
        if _is_port_open("127.0.0.1", port):
            ready = True
            break
        # Child crashed during import/startup — stop waiting the full
        # budget for a port that will never open.
        if proc.poll() is not None:
            LOGGER.error(
                "Node process exited during startup (code=%s); see %s",
                proc.returncode, log_file,
            )
            break
        if on_progress is not None:
            on_progress(time.monotonic() - start, False)
        time.sleep(0.25)

    elapsed = time.monotonic() - start
    if on_progress is not None:
        on_progress(elapsed, ready)

    _write_pid(settings, proc.pid, port)
    if ready:
        LOGGER.info(
            "Spawned node ready (pid=%d, port=%d) in %.1fs, log=%s",
            proc.pid, port, elapsed, log_file,
        )
    else:
        LOGGER.warning(
            "Spawned node (pid=%d, port=%d) not responding after %.1fs; check %s",
            proc.pid, port, elapsed, log_file,
        )
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
