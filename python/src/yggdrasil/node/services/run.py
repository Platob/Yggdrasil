from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import subprocess
import sys
import tempfile
import time
import uuid
from collections import OrderedDict
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator

from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..exceptions import NotFoundError
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
        self._runs: OrderedDict[str, RunEntry] = OrderedDict()
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

        run_id = uuid.uuid4().hex[:12]
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        entry = RunEntry(
            id=run_id,
            function_id=req.function_id,
            environment_id=env_id,
            status="pending",
            started_at=now,
            node_id=self.settings.node_id,
        )

        with self._lock:
            self._runs[run_id] = entry
            self._evict()

        LOGGER.info("Triggering run %r for function %r", run_id, req.function_id)

        # Execute in threadpool
        result = await self._run(
            self._execute, run_id, function, env_python, req.args,
        )

        # Bump function run counter
        self._function_service.increment_run_count(req.function_id)

        return RunResponse(run=result)

    async def get(self, run_id: str) -> RunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None:
            raise NotFoundError(f"Run {run_id!r} not found")
        return entry

    async def list(self, *, function_id: str | None = None) -> RunListResponse:
        with self._lock:
            items = list(self._runs.values())
        if function_id is not None:
            items = [r for r in items if r.function_id == function_id]
        return RunListResponse(
            node_id=self.settings.node_id,
            runs=items,
        )

    async def delete(self, run_id: str) -> RunResponse:
        with self._lock:
            entry = self._runs.pop(run_id, None)
        if entry is None:
            raise NotFoundError(f"Run {run_id!r} not found")
        LOGGER.info("Deleted run %r", run_id)
        return RunResponse(run=entry)

    async def stream_logs(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
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

    def _execute(
        self,
        run_id: str,
        function: FunctionEntry,
        env_python: str | None,
        args: dict[str, Any],
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

            proc = subprocess.run(
                [python_bin, tmp.name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.settings.max_python_timeout,
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
                stderr=f"Timed out after {self.settings.max_python_timeout:.0f}s",
            )
            LOGGER.error("Run %r timed out after %.0fs", run_id, self.settings.max_python_timeout)
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

    def _update_entry(self, run_id: str, **updates) -> RunEntry:
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
