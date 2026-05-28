from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import platform
import resource as resource_mod
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator

from fastapi.concurrency import run_in_threadpool

from yggdrasil.dataclasses.expiring import ExpiringDict

from ...config import Settings
from ...exceptions import NotFoundError
from ...ids import make_id
from ..schemas.pyfuncrun import (
    PyFuncRunCreate,
    PyFuncRunEntry,
    PyFuncRunListResponse,
    PyFuncRunResponse,
)
from .pyenv import PyEnvService
from .pyfunc import PyFuncService

LOGGER = logging.getLogger(__name__)


class PyFuncRunService:
    """Execution service: composes PyEnv + PyFunc + metadata + args/kwargs."""

    def __init__(
        self,
        settings: Settings,
        pyenv_service: PyEnvService,
        pyfunc_service: PyFuncService,
    ) -> None:
        self.settings = settings
        self._pyenv = pyenv_service
        self._pyfunc = pyfunc_service
        self._runs: ExpiringDict[int, PyFuncRunEntry] = ExpiringDict(default_ttl=None, max_size=settings.max_runs_history)
        self._lock = Lock()

    async def _pool(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: PyFuncRunCreate) -> PyFuncRunResponse:
        func = await self._pyfunc.get(req.func_id)

        env_id = req.env_id or func.env_id
        env_python: str | None = None
        if env_id is not None:
            env_python = self._pyenv.get_python_path(env_id)

        run_id = make_id(f"{req.func_id}:{time.monotonic()}")
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        entry = PyFuncRunEntry(
            id=run_id,
            func_id=req.func_id,
            env_id=env_id,
            status="pending",
            args=list(req.args),
            kwargs=dict(req.kwargs),
            started_at=now,
            node_id=self.settings.node_id,
        )
        with self._lock:
            self._runs.set(run_id, entry)

        effective_timeout = (
            req.timeout if req.timeout is not None
            else self.settings.max_python_timeout
        )

        result = await self._pool(
            self._execute, run_id, func.code, env_python,
            list(req.args), dict(req.kwargs),
            effective_timeout, req.max_memory_mb,
        )

        self._pyfunc.increment_run_count(req.func_id)
        self._pyfunc.record_run_completion(
            req.func_id,
            duration_ms=(result.duration or 0.0) * 1000,
            success=(result.status == "completed"),
        )
        return PyFuncRunResponse(run=result)

    async def get(self, run_id: int) -> PyFuncRunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")
        return entry

    async def list(self, *, func_id: int | None = None) -> PyFuncRunListResponse:
        with self._lock:
            items = list(self._runs.values())
        if func_id is not None:
            items = [r for r in items if r.func_id == func_id]
        return PyFuncRunListResponse(
            node_id=self.settings.node_id, runs=items,
        )

    async def cancel(self, run_id: int) -> PyFuncRunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")
        if entry.status not in ("pending", "running"):
            raise NotFoundError(
                f"PyFuncRun {run_id!r} is already {entry.status!r} — "
                f"only pending/running runs can be cancelled"
            )
        return self._update_entry(
            run_id,
            status="cancelled",
            completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            progress=1.0,
        )

    async def delete(self, run_id: int) -> PyFuncRunResponse:
        with self._lock:
            entry = self._runs.pop(run_id, None)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")
        return PyFuncRunResponse(run=entry)

    async def wait(self, run_id: int, *, timeout: float = 600.0, interval: float = 0.3) -> PyFuncRunEntry:
        """Block until run completes or timeout. Returns final entry."""
        t0 = time.monotonic()
        while True:
            entry = await self.get(run_id)
            if entry.status not in ("pending", "running"):
                return entry
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"Run {run_id} timed out after {timeout}s")
            await asyncio.sleep(interval)

    async def stream_logs(self, run_id: int) -> AsyncIterator[dict[str, Any]]:
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")

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
            "result_type": entry.result_type,
        }

    async def stream_state(self, run_id: int) -> AsyncIterator[dict]:
        """Yield the full run entry as a dict every 0.5s until the run is done."""
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")

        while True:
            with self._lock:
                entry = self._runs.get(run_id)
            if entry is None:
                return
            yield entry.model_dump()
            if entry.status not in ("pending", "running"):
                return
            await asyncio.sleep(0.5)

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(
                1 for r in self._runs.values()
                if r.status in ("pending", "running")
            )

    @property
    def total_count(self) -> int:
        with self._lock:
            return len(self._runs)

    # -- internals ----------------------------------------------------------

    def _execute(
        self,
        run_id: int,
        func_code: str,
        env_python: str | None,
        args: list[Any],
        kwargs: dict[str, Any],
        timeout: float,
        max_memory_mb: int | None,
    ) -> PyFuncRunEntry:
        self._update_entry(run_id, status="running")

        python_bin = env_python or sys.executable
        t0 = time.monotonic()
        tmp = None
        outputs_file = None

        try:
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
            preamble = (
                "import json as _json\n"
                f"_args = _json.loads({json.dumps(json.dumps(args))!r})\n"
                f"_kwargs = _json.loads({json.dumps(json.dumps(kwargs))!r})\n"
            )
            outputs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            outputs_file.close()

            tmp.write(preamble + func_code)
            tmp.flush()
            tmp.close()

            env = os.environ.copy()
            env["YGG_RUNTIME_VERSION"] = self.settings.app_version
            env["YGG_NODE_ID"] = self.settings.node_id
            env["YGG_NODE_PORT"] = str(self.settings.port)
            env["__ygg_inputs__"] = json.dumps({"args": args, "kwargs": kwargs})
            env["__ygg_outputs_file__"] = outputs_file.name

            preexec_fn = None
            if max_memory_mb and platform.system() == "Linux":
                mem_bytes = max_memory_mb * 1024 * 1024

                def _limit():
                    resource_mod.setrlimit(
                        resource_mod.RLIMIT_AS, (mem_bytes, mem_bytes)
                    )

                preexec_fn = _limit

            proc = subprocess.run(
                [python_bin, tmp.name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                preexec_fn=preexec_fn,
                env=env,
            )

            duration = round(time.monotonic() - t0, 3)
            status = "completed" if proc.returncode == 0 else "failed"

            result: Any = None
            result_type: str | None = None
            op = Path(outputs_file.name)
            if op.exists() and op.stat().st_size > 0:
                try:
                    with open(op) as f:
                        result = json.load(f)
                    result_type = "json"
                except (json.JSONDecodeError, OSError):
                    pass

            stdout_text = proc.stdout or None
            stderr_text = proc.stderr or None
            total_lines = 0
            if stdout_text:
                total_lines += len(stdout_text.splitlines())
            if stderr_text:
                total_lines += len(stderr_text.splitlines())

            return self._update_entry(
                run_id,
                status=status,
                completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration=duration,
                returncode=proc.returncode,
                stdout=stdout_text,
                stderr=stderr_text,
                result=result,
                result_type=result_type,
                progress=1.0,
                log_lines=total_lines,
            )

        except subprocess.TimeoutExpired:
            return self._update_entry(
                run_id,
                status="failed",
                completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration=round(time.monotonic() - t0, 3),
                stderr=f"Timed out after {timeout:.0f}s",
                progress=1.0,
                log_lines=1,
            )
        except Exception as exc:
            return self._update_entry(
                run_id,
                status="failed",
                completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                duration=round(time.monotonic() - t0, 3),
                stderr=str(exc),
                progress=1.0,
                log_lines=1,
            )
        finally:
            if tmp is not None:
                Path(tmp.name).unlink(missing_ok=True)
            if outputs_file is not None:
                Path(outputs_file.name).unlink(missing_ok=True)

    def _update_entry(self, run_id: int, **updates) -> PyFuncRunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is not None:
                entry = entry.model_copy(update=updates)
                self._runs.set(run_id, entry)
                return entry
        raise NotFoundError(f"PyFuncRun {run_id!r} not found")
