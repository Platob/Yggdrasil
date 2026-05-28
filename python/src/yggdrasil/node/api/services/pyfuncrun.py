from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import platform
import resource as resource_mod
import signal
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator

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

_TERMINAL_EVENT_TYPES = {"complete", "error"}


class _RunRuntime:
    """Per-run execution state: process handle, log buffers, subscriber bus."""

    __slots__ = (
        "process", "stdout_lines", "stderr_lines",
        "stdout_truncated", "stderr_truncated",
        "subscribers", "replay", "task", "completed",
    )

    def __init__(self, max_lines: int) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.stdout_lines: deque[str] = deque(maxlen=max_lines)
        self.stderr_lines: deque[str] = deque(maxlen=max_lines)
        self.stdout_truncated = False
        self.stderr_truncated = False
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self.replay: deque[dict[str, Any]] = deque(maxlen=1024)
        self.task: asyncio.Task | None = None
        self.completed = asyncio.Event()


class PyFuncRunService:
    """Execution service: composes PyEnv + PyFunc + metadata + args/kwargs.

    POST returns immediately with a pending entry. The subprocess runs in
    a fire-and-forget asyncio.Task; stdout/stderr are streamed line-by-line
    onto a per-run event bus. Subscribers (SSE log/state endpoints) attach
    to that bus.
    """

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
        self._runtimes: dict[int, _RunRuntime] = {}
        self._lock = Lock()

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
        runtime = _RunRuntime(max_lines=self.settings.max_log_lines_per_stream)
        with self._lock:
            self._runs.set(run_id, entry)
            self._runtimes[run_id] = runtime

        effective_timeout = (
            req.timeout if req.timeout is not None
            else self.settings.max_python_timeout
        )

        runtime.task = asyncio.create_task(
            self._supervise(
                run_id, func.code, env_python,
                list(req.args), dict(req.kwargs),
                effective_timeout, req.max_memory_mb,
            ),
            name=f"pyfuncrun-{run_id}",
        )
        return PyFuncRunResponse(run=entry)

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
            runtime = self._runtimes.get(run_id)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")
        if entry.status not in ("pending", "running"):
            raise NotFoundError(
                f"PyFuncRun {run_id!r} is already {entry.status!r} — "
                f"only pending/running runs can be cancelled"
            )

        entry = self._update_entry(run_id, cancellation_requested=True)

        if runtime is not None and runtime.process is not None:
            self._terminate_process(runtime.process)
            try:
                await asyncio.wait_for(
                    runtime.process.wait(),
                    timeout=self.settings.run_cancel_grace_seconds,
                )
            except asyncio.TimeoutError:
                self._kill_process(runtime.process)
                try:
                    await asyncio.wait_for(runtime.process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    LOGGER.warning("Process for run %r did not die after SIGKILL", run_id)
            # supervisor task will finalize status as 'cancelled' (see _supervise)
            try:
                await asyncio.wait_for(runtime.completed.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            with self._lock:
                entry = self._runs.get(run_id, entry)
            return entry

        # No live process — finalize directly. Happens if cancel races startup.
        entry = self._update_entry(
            run_id,
            status="cancelled",
            completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            progress=1.0,
        )
        if runtime is not None:
            self._publish(runtime, {
                "type": "complete",
                "run_id": run_id,
                "status": "cancelled",
                "returncode": None,
                "duration": entry.duration or 0.0,
            })
            self._close_runtime(run_id)
        return entry

    async def delete(self, run_id: int) -> PyFuncRunResponse:
        with self._lock:
            entry = self._runs.pop(run_id, None)
            runtime = self._runtimes.pop(run_id, None)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")
        if runtime is not None and runtime.task is not None and not runtime.task.done():
            runtime.task.cancel()
        return PyFuncRunResponse(run=entry)

    async def wait(self, run_id: int, *, timeout: float = 600.0, interval: float = 0.1) -> PyFuncRunEntry:
        """Block until run completes or timeout. Returns final entry."""
        with self._lock:
            runtime = self._runtimes.get(run_id)
            entry = self._runs.get(run_id)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")
        if entry.status not in ("pending", "running"):
            return entry
        if runtime is not None:
            try:
                await asyncio.wait_for(runtime.completed.wait(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(f"Run {run_id} timed out after {timeout}s") from exc
            return await self.get(run_id)

        # No runtime (e.g. already cleaned up): fall back to short poll
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
            runtime = self._runtimes.get(run_id)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")

        if runtime is None or entry.status not in ("pending", "running"):
            # Terminal: replay stored stdout/stderr once.
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            for line in (entry.stdout or "").splitlines():
                yield {"type": "stdout", "run_id": run_id, "line": line, "timestamp": now}
            for line in (entry.stderr or "").splitlines():
                yield {"type": "stderr", "run_id": run_id, "line": line, "timestamp": now}
            yield {
                "type": "complete",
                "run_id": run_id,
                "status": entry.status,
                "returncode": entry.returncode,
                "duration": entry.duration,
            }
            return

        async for event in self._subscribe(runtime):
            yield event

    async def stream_state(self, run_id: int) -> AsyncIterator[dict]:
        """Yield run-entry snapshots until the run is terminal."""
        with self._lock:
            entry = self._runs.get(run_id)
            runtime = self._runtimes.get(run_id)
        if entry is None:
            raise NotFoundError(f"PyFuncRun {run_id!r} not found")

        if runtime is None or entry.status not in ("pending", "running"):
            yield entry.model_dump()
            return

        # Snapshot upfront so SSE consumers see the initial state.
        yield entry.model_dump()
        async for event in self._subscribe(runtime):
            if event["type"] in ("state", "heartbeat", "complete", "error"):
                current = await self.get(run_id)
                yield current.model_dump()

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

    # -- internals: subscription bus ---------------------------------------

    async def _subscribe(self, runtime: _RunRuntime) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        # Send replay first so a late subscriber sees historical context.
        for event in list(runtime.replay):
            await queue.put(event)
        runtime.subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
                if event["type"] in _TERMINAL_EVENT_TYPES:
                    return
        finally:
            try:
                runtime.subscribers.remove(queue)
            except ValueError:
                pass

    def _publish(self, runtime: _RunRuntime, event: dict[str, Any]) -> None:
        runtime.replay.append(event)
        for q in list(runtime.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # -- internals: supervisor & subprocess --------------------------------

    async def _supervise(
        self,
        run_id: int,
        func_code: str,
        env_python: str | None,
        args: list[Any],
        kwargs: dict[str, Any],
        timeout: float,
        max_memory_mb: int | None,
    ) -> None:
        runtime = self._runtimes.get(run_id)
        if runtime is None:
            return

        python_bin = env_python or sys.executable
        t0 = time.monotonic()
        tmp_path: Path | None = None
        outputs_path: Path | None = None
        heartbeat_task: asyncio.Task | None = None
        stdout_task: asyncio.Task | None = None
        stderr_task: asyncio.Task | None = None
        proc: asyncio.subprocess.Process | None = None
        cancelled_externally = False

        try:
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
            preamble = (
                "import json as _json\n"
                f"_args = _json.loads({json.dumps(json.dumps(args))!r})\n"
                f"_kwargs = _json.loads({json.dumps(json.dumps(kwargs))!r})\n"
            )
            outputs = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            outputs.close()
            tmp.write(preamble + func_code)
            tmp.flush()
            tmp.close()
            tmp_path = Path(tmp.name)
            outputs_path = Path(outputs.name)

            env = os.environ.copy()
            env["YGG_RUNTIME_VERSION"] = self.settings.app_version
            env["YGG_NODE_ID"] = self.settings.node_id
            env["YGG_NODE_PORT"] = str(self.settings.port)
            env["__ygg_inputs__"] = json.dumps({"args": args, "kwargs": kwargs})
            env["__ygg_outputs_file__"] = str(outputs_path)
            # Force line-buffered stdout so we can stream output live.
            env["PYTHONUNBUFFERED"] = "1"

            preexec_fn = None
            if max_memory_mb and platform.system() == "Linux":
                mem_bytes = max_memory_mb * 1024 * 1024

                def _preexec() -> None:
                    # New session so cancel/timeout can signal the whole tree.
                    os.setsid()
                    resource_mod.setrlimit(
                        resource_mod.RLIMIT_AS, (mem_bytes, mem_bytes)
                    )

                preexec_fn = _preexec
            elif platform.system() != "Windows":
                preexec_fn = os.setsid

            creationflags = 0
            if platform.system() == "Windows":
                creationflags = getattr(__import__("subprocess"), "CREATE_NEW_PROCESS_GROUP", 0)

            proc = await asyncio.create_subprocess_exec(
                python_bin, str(tmp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                preexec_fn=preexec_fn,
                creationflags=creationflags,
            )
            runtime.process = proc
            self._update_entry(
                run_id, status="running", pid=proc.pid,
            )
            self._publish(runtime, {
                "type": "state",
                "run_id": run_id,
                "status": "running",
                "progress": 0.0,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            })

            stdout_task = asyncio.create_task(
                self._drain_stream(run_id, runtime, proc.stdout, "stdout"),
                name=f"pyfuncrun-{run_id}-stdout",
            )
            stderr_task = asyncio.create_task(
                self._drain_stream(run_id, runtime, proc.stderr, "stderr"),
                name=f"pyfuncrun-{run_id}-stderr",
            )
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(run_id, runtime),
                name=f"pyfuncrun-{run_id}-heartbeat",
            )

            try:
                returncode = await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self._terminate_process(proc)
                try:
                    returncode = await asyncio.wait_for(
                        proc.wait(), timeout=self.settings.run_cancel_grace_seconds,
                    )
                except asyncio.TimeoutError:
                    self._kill_process(proc)
                    returncode = await proc.wait()
                duration = round(time.monotonic() - t0, 3)
                # Drain readers so all output up to termination is captured.
                await self._join_readers(stdout_task, stderr_task)
                self._finalize(
                    run_id, runtime, status="failed",
                    returncode=returncode, duration=duration,
                    error=f"timed out after {timeout:.0f}s",
                    outputs_path=outputs_path,
                )
                return

            duration = round(time.monotonic() - t0, 3)
            await self._join_readers(stdout_task, stderr_task)

            # Determine final status: cancellation_requested wins over returncode.
            with self._lock:
                snapshot = self._runs.get(run_id)
            requested_cancel = bool(snapshot and snapshot.cancellation_requested)
            if requested_cancel:
                status = "cancelled"
                cancelled_externally = True
            else:
                status = "completed" if returncode == 0 else "failed"

            self._finalize(
                run_id, runtime, status=status,
                returncode=returncode, duration=duration,
                outputs_path=outputs_path,
            )

        except asyncio.CancelledError:
            if proc is not None and proc.returncode is None:
                self._kill_process(proc)
            raise
        except Exception as exc:
            LOGGER.exception("Supervisor failure for run %r", run_id)
            self._finalize(
                run_id, runtime, status="failed",
                returncode=None, duration=round(time.monotonic() - t0, 3),
                error=str(exc), outputs_path=outputs_path,
            )
        finally:
            if heartbeat_task is not None and not heartbeat_task.done():
                heartbeat_task.cancel()
            for t in (stdout_task, stderr_task, heartbeat_task):
                if t is not None:
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if outputs_path is not None:
                outputs_path.unlink(missing_ok=True)

            # Update PyFunc metrics after completion.
            with self._lock:
                final = self._runs.get(run_id)
            if final is not None:
                self._pyfunc.increment_run_count(final.func_id)
                self._pyfunc.record_run_completion(
                    final.func_id,
                    duration_ms=(final.duration or 0.0) * 1000,
                    success=(final.status == "completed"),
                )

            runtime.completed.set()
            # Unblock any subscriber still waiting on the bus.
            for q in list(runtime.subscribers):
                try:
                    q.put_nowait({"type": "complete", "run_id": run_id, "status": "closed"})
                except asyncio.QueueFull:
                    pass

    async def _drain_stream(
        self, run_id: int, runtime: _RunRuntime,
        stream: asyncio.StreamReader | None, kind: str,
    ) -> None:
        if stream is None:
            return
        buf = runtime.stdout_lines if kind == "stdout" else runtime.stderr_lines
        max_lines = self.settings.max_log_lines_per_stream
        while True:
            try:
                raw = await stream.readline()
            except (asyncio.LimitOverrunError, ValueError):
                # Line too long for default 64KB buffer — read what we can.
                raw = await stream.read(65536)
            if not raw:
                return
            line = raw.decode(errors="replace").rstrip("\r\n")
            if len(buf) >= max_lines:
                if kind == "stdout":
                    runtime.stdout_truncated = True
                else:
                    runtime.stderr_truncated = True
            buf.append(line)
            self._publish(runtime, {
                "type": kind,
                "run_id": run_id,
                "line": line,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            })

    async def _heartbeat_loop(self, run_id: int, runtime: _RunRuntime) -> None:
        interval = self.settings.run_heartbeat_interval
        while True:
            await asyncio.sleep(interval)
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            self._update_entry(run_id, heartbeat_at=now)
            self._publish(runtime, {
                "type": "heartbeat",
                "run_id": run_id,
                "timestamp": now,
            })

    async def _join_readers(self, *tasks: asyncio.Task | None) -> None:
        for t in tasks:
            if t is None:
                continue
            try:
                await asyncio.wait_for(t, timeout=2.0)
            except asyncio.TimeoutError:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            except (asyncio.CancelledError, Exception):
                pass

    def _finalize(
        self,
        run_id: int,
        runtime: _RunRuntime,
        *,
        status: str,
        returncode: int | None,
        duration: float,
        error: str | None = None,
        outputs_path: Path | None = None,
    ) -> None:
        stdout_text = "\n".join(runtime.stdout_lines) if runtime.stdout_lines else None
        stderr_text = "\n".join(runtime.stderr_lines) if runtime.stderr_lines else None
        result: Any = None
        result_type: str | None = None
        if outputs_path is not None and outputs_path.exists() and outputs_path.stat().st_size > 0:
            try:
                with open(outputs_path) as f:
                    result = json.load(f)
                result_type = "json"
            except (json.JSONDecodeError, OSError):
                pass

        total_lines = len(runtime.stdout_lines) + len(runtime.stderr_lines)
        entry = self._update_entry(
            run_id,
            status=status,
            completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            duration=duration,
            returncode=returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            result=result,
            result_type=result_type,
            error=error,
            progress=1.0,
            log_lines=total_lines,
            stdout_truncated=runtime.stdout_truncated,
            stderr_truncated=runtime.stderr_truncated,
        )
        self._publish(runtime, {
            "type": "complete",
            "run_id": run_id,
            "status": status,
            "returncode": returncode,
            "duration": duration,
        })
        if error is not None:
            self._publish(runtime, {
                "type": "error",
                "run_id": run_id,
                "error": error,
                "timestamp": entry.completed_at,
            })

    def _terminate_process(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            if platform.system() == "Windows":
                proc.terminate()
            else:
                # New process group from preexec setsid; signal whole group
                # so child threads/subprocesses also die.
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    def _kill_process(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            if platform.system() == "Windows":
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _close_runtime(self, run_id: int) -> None:
        with self._lock:
            runtime = self._runtimes.pop(run_id, None)
        if runtime is None:
            return
        runtime.completed.set()
        for q in list(runtime.subscribers):
            try:
                q.put_nowait({"type": "complete", "run_id": run_id, "status": "closed"})
            except asyncio.QueueFull:
                pass

    def _update_entry(self, run_id: int, **updates) -> PyFuncRunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is not None:
                entry = entry.model_copy(update=updates)
                self._runs.set(run_id, entry)
                return entry
        raise NotFoundError(f"PyFuncRun {run_id!r} not found")
