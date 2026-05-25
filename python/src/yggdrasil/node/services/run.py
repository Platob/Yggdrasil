from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import platform
import resource as resource_mod
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator

from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..exceptions import NotFoundError
from ..ids import make_id
from ..schemas.function import FunctionEntry
from ..schemas.run import (
    RunCreate,
    RunEntry,
    RunListResponse,
    RunResponse,
)
from .function import FunctionService
from .environment import EnvironmentService

LOGGER = logging.getLogger(__name__)


class RunService:
    def __init__(
        self,
        settings: Settings,
        function_service: FunctionService,
        environment_service: EnvironmentService,
    ) -> None:
        self.settings = settings
        self._function_service = function_service
        self._environment_service = environment_service
        self._runs: OrderedDict[int, RunEntry] = OrderedDict()
        self._lock = Lock()

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: RunCreate) -> RunResponse:
        # Validate function exists
        function = await self._function_service.get(req.function_id)

        # Determine environment
        env_id = req.environment_id or function.environment_id
        env_python: str | None = None
        if env_id is not None:
            env_python = self._environment_service.get_python_path(env_id)

        run_id = make_id(f"{req.function_id}:{time.monotonic()}")
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        entry = RunEntry(
            id=run_id,
            function_id=req.function_id,
            environment_id=env_id,
            status="pending",
            started_at=now,
            node_id=self.settings.node_id,
            max_memory_mb=req.max_memory_mb,
            max_cpu_percent=req.max_cpu_percent,
            timeout=req.timeout,
        )

        with self._lock:
            self._runs[run_id] = entry
            self._evict()

        LOGGER.info("Triggering run %r for function %r", run_id, req.function_id)

        # Determine effective timeout: explicit > settings default
        effective_timeout = req.timeout if req.timeout is not None else self.settings.max_python_timeout

        # Execute in threadpool
        result = await self._run(
            self._execute, run_id, function, env_python, req.args,
            effective_timeout, req.max_memory_mb,
        )

        # Bump function run counter
        self._function_service.increment_run_count(req.function_id)

        return RunResponse(run=result)

    async def get(self, run_id: int) -> RunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None:
            raise NotFoundError(f"Run {run_id!r} not found")
        return entry

    async def list(self, *, function_id: int | None = None) -> RunListResponse:
        with self._lock:
            items = list(self._runs.values())
        if function_id is not None:
            items = [r for r in items if r.function_id == function_id]
        return RunListResponse(
            node_id=self.settings.node_id,
            runs=items,
        )

    async def delete(self, run_id: int) -> RunResponse:
        with self._lock:
            entry = self._runs.pop(run_id, None)
        if entry is None:
            raise NotFoundError(f"Run {run_id!r} not found")
        LOGGER.info("Deleted run %r", run_id)
        return RunResponse(run=entry)

    async def stream_logs(self, run_id: int) -> AsyncIterator[dict[str, Any]]:
        """Async generator yielding log events for an SSE stream.

        If the run is already completed, yields the stored stdout/stderr
        as a batch then the complete event. For in-progress runs, yields
        whatever stdout/stderr has been captured.
        """
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None:
            raise NotFoundError(f"Run {run_id!r} not found")

        # Poll until the run settles (completed/failed/cancelled)
        while entry.status in ("pending", "running"):
            await asyncio.sleep(0.3)
            with self._lock:
                entry = self._runs.get(run_id)
            if entry is None:
                return

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        if entry.stdout:
            for line in entry.stdout.splitlines():
                yield {"type": "stdout", "line": line, "timestamp": now}
        if entry.stderr:
            for line in entry.stderr.splitlines():
                yield {"type": "stderr", "line": line, "timestamp": now}

        yield {
            "type": "complete",
            "returncode": entry.returncode,
            "duration": entry.duration,
        }

    # -- internals ----------------------------------------------------------

    def _make_preexec_fn(self, max_memory_mb: int | None):
        """Build a preexec_fn that sets resource limits in the child process (Linux only)."""
        if max_memory_mb is None or platform.system() != "Linux":
            return None

        memory_bytes = max_memory_mb * 1024 * 1024

        def _set_limits():
            # RLIMIT_AS = virtual memory limit
            resource_mod.setrlimit(resource_mod.RLIMIT_AS, (memory_bytes, memory_bytes))

        return _set_limits

    def _execute(
        self,
        run_id: int,
        function: FunctionEntry,
        env_python: str | None,
        args: dict[str, Any],
        timeout: float,
        max_memory_mb: int | None,
    ) -> RunEntry:
        self._update_entry(run_id, status="running")

        python_bin = env_python or sys.executable
        t0 = time.monotonic()
        tmp = None

        try:
            # Write function code to a temp file
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
            )
            # Inject args into the script namespace
            preamble = f"import json\nargs = json.loads({json.dumps(json.dumps(args))!r})\n"
            tmp.write(preamble + function.code)
            tmp.flush()
            tmp.close()

            preexec_fn = self._make_preexec_fn(max_memory_mb)

            proc = subprocess.run(
                [python_bin, tmp.name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                preexec_fn=preexec_fn,
            )

            duration = round(time.monotonic() - t0, 3)
            status = "completed" if proc.returncode == 0 else "failed"

            entry = self._update_entry(
                run_id,
                status=status,
                completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration=duration,
                returncode=proc.returncode,
                stdout=proc.stdout or None,
                stderr=proc.stderr or None,
            )

            LOGGER.info("Run %r %s (rc=%s, %.2fs)", run_id, status, proc.returncode, duration)
            return entry

        except subprocess.TimeoutExpired:
            duration = round(time.monotonic() - t0, 3)
            entry = self._update_entry(
                run_id,
                status="failed",
                completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration=duration,
                stderr=f"Timed out after {timeout:.0f}s",
            )
            LOGGER.error("Run %r timed out after %.0fs", run_id, timeout)
            return entry

        except Exception as exc:
            duration = round(time.monotonic() - t0, 3)
            entry = self._update_entry(
                run_id,
                status="failed",
                completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration=duration,
                stderr=str(exc),
            )
            LOGGER.error("Run %r failed: %s", run_id, exc)
            return entry

        finally:
            if tmp is not None:
                Path(tmp.name).unlink(missing_ok=True)

    def _update_entry(self, run_id: int, **updates) -> RunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is not None:
                entry = entry.model_copy(update=updates)
                self._runs[run_id] = entry
                return entry
        # Should not happen in normal flow, but satisfy type checker
        raise NotFoundError(f"Run {run_id!r} not found")

    def _evict(self) -> None:
        while len(self._runs) > self.settings.max_runs_history:
            self._runs.popitem(last=False)
