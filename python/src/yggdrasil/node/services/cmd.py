from __future__ import annotations

import datetime as dt
import logging
import uuid
from collections import OrderedDict
from functools import partial
from threading import Lock

from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..exceptions import NotFoundError, TimeoutError
from ..schemas.cmd import CmdEntry, CmdListResponse, CmdRequest, CmdResponse
from .environment.py import PyEnvironment
from .execution.shell import ShellCommand

LOGGER = logging.getLogger(__name__)


class CmdService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._history: OrderedDict[str, CmdEntry] = OrderedDict()
        self._results: dict[str, CmdResponse] = {}
        self._lock = Lock()
        self._env = PyEnvironment()

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    async def execute(self, req: CmdRequest) -> CmdResponse:
        cmd_id = uuid.uuid4().hex[:12]
        timeout = req.timeout
        if timeout is not None:
            timeout = min(timeout, self.settings.max_cmd_timeout)
        else:
            timeout = self.settings.max_cmd_timeout

        LOGGER.info("Executing command %r (id=%s, timeout=%.0fs)", req.command, cmd_id, timeout)

        entry = CmdEntry(
            id=cmd_id,
            command=req.command,
            status="running",
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self._store_entry(entry)

        try:
            response = await self._run(
                self._exec_sync, cmd_id, req, timeout
            )
        except Exception as exc:
            entry = entry.model_copy(update={"status": "failed"})
            self._store_entry(entry)
            raise

        entry = entry.model_copy(update={
            "status": response.status,
            "returncode": response.returncode,
            "duration": response.duration,
        })
        self._store_entry(entry)
        return response

    def _exec_sync(
        self, cmd_id: str, req: CmdRequest, timeout: float
    ) -> CmdResponse:
        exe = ShellCommand(
            command=req.command,
            cwd=req.cwd,
            env=dict(req.env) if req.env else {},
            stdin=req.stdin,
            timeout=timeout,
        )

        execution = self._env.execute_shell_command(exe)

        if execution.status == "failed" and execution.stderr and "Timed out" in execution.stderr:
            raise TimeoutError(
                f"Command timed out after {timeout:.0f}s: {req.command}"
            )

        LOGGER.info("Completed command %r (id=%s, rc=%s, %.2fs)", req.command, cmd_id, execution.returncode, execution.duration or 0)

        response = CmdResponse(
            id=cmd_id,
            node_id=self.settings.node_id,
            command=req.command,
            returncode=execution.returncode,
            stdout=execution.stdout,
            stderr=execution.stderr,
            duration=execution.duration,
            status=execution.status,
        )
        with self._lock:
            self._results[cmd_id] = response
        return response

    async def get(self, cmd_id: str) -> CmdResponse:
        with self._lock:
            result = self._results.get(cmd_id)
        if result is None:
            raise NotFoundError(f"Command execution {cmd_id!r} not found")
        return result

    async def list_history(self) -> CmdListResponse:
        with self._lock:
            items = list(self._history.values())
        return CmdListResponse(
            node_id=self.settings.node_id,
            items=items,
        )

    async def delete(self, cmd_id: str) -> CmdResponse:
        with self._lock:
            result = self._results.pop(cmd_id, None)
            self._history.pop(cmd_id, None)
        if result is None:
            raise NotFoundError(f"Command execution {cmd_id!r} not found")
        return result

    def _store_entry(self, entry: CmdEntry) -> None:
        with self._lock:
            self._history[entry.id] = entry
            while len(self._history) > self.settings.job_max_history:
                oldest_id, _ = self._history.popitem(last=False)
                self._results.pop(oldest_id, None)
