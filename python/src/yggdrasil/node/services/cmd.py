from __future__ import annotations

import datetime as dt
import logging
import subprocess
import time
import uuid
from collections import OrderedDict
from functools import partial
from threading import Lock

from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..exceptions import NotFoundError, TimeoutError
from ..schemas.cmd import CmdEntry, CmdListResponse, CmdRequest, CmdResponse

LOGGER = logging.getLogger(__name__)


class CmdService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._history: OrderedDict[str, CmdEntry] = OrderedDict()
        self._results: dict[str, CmdResponse] = {}
        self._lock = Lock()

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
        env = dict(req.env) if req.env else None
        t0 = time.monotonic()

        try:
            proc = subprocess.run(
                req.command,
                cwd=req.cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                input=req.stdin,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - t0
            raise TimeoutError(
                f"Command timed out after {timeout:.0f}s: {req.command}"
            )

        duration = time.monotonic() - t0
        status = "completed" if proc.returncode == 0 else "failed"

        LOGGER.info("Completed command %r (id=%s, rc=%d, %.2fs)", req.command, cmd_id, proc.returncode, duration)

        response = CmdResponse(
            id=cmd_id,
            node_id=self.settings.node_id,
            command=req.command,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration=round(duration, 3),
            status=status,
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
