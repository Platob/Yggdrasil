from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from collections import OrderedDict
from functools import partial
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
from .execution import PyEnvironment, PyFunction
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
        function = await self._function_service.get(req.function_id)

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

        effective_timeout = req.timeout if req.timeout is not None else self.settings.max_python_timeout

        result = await self._run(
            self._execute, run_id, function, env_python, req.args,
            effective_timeout, req.max_memory_mb,
        )

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
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None:
            raise NotFoundError(f"Run {run_id!r} not found")

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
        run_id: int,
        function: FunctionEntry,
        env_python: str | None,
        args: dict[str, Any],
        timeout: float,
        max_memory_mb: int | None,
    ) -> RunEntry:
        self._update_entry(run_id, status="running")

        py_env = PyEnvironment(
            python_bin=env_python,
            node_env={
                "YGG_RUNTIME_VERSION": self.settings.app_version,
                "YGG_NODE_ID": self.settings.node_id,
                "YGG_NODE_PORT": str(self.settings.port),
            },
        )

        exe = PyFunction(
            code=function.code,
            args=args,
            timeout=timeout,
            max_memory_mb=max_memory_mb,
        )

        execution = py_env.execute(exe)

        entry = self._update_entry(
            run_id,
            status=execution.status,
            completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            duration=execution.duration,
            returncode=execution.returncode,
            stdout=execution.stdout,
            stderr=execution.stderr,
            result=execution.result,
        )

        LOGGER.info(
            "Run %r %s (rc=%s, %.2fs)",
            run_id, execution.status, execution.returncode, execution.duration or 0,
        )
        return entry

    def _update_entry(self, run_id: int, **updates) -> RunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is not None:
                entry = entry.model_copy(update=updates)
                self._runs[run_id] = entry
                return entry
        raise NotFoundError(f"Run {run_id!r} not found")

    def _evict(self) -> None:
        while len(self._runs) > self.settings.max_runs_history:
            self._runs.popitem(last=False)
