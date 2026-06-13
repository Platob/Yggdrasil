"""Function management service.

Functions are stored in memory and upserted by name (CLAUDE.md: upsert by
default). IDs are assigned once and immutable. Execution runs the code in a
subprocess with a timeout so a runaway function can't wedge the node.
"""
from __future__ import annotations

import subprocess
import sys
import time

from ..config import Settings
from ..schemas.function import Function, FunctionCreate, FunctionResponse, RunResponse


class FunctionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._by_id: dict[int, Function] = {}
        self._id_by_name: dict[str, int] = {}
        self._runs: dict[int, RunResponse] = {}
        self._next_id = 1
        self._next_run_id = 1

    async def create(self, req: FunctionCreate) -> FunctionResponse:
        existing_id = self._id_by_name.get(req.name)
        if existing_id is not None:
            # Upsert: keep the immutable id, replace the body.
            old = self._by_id[existing_id]
            fn = Function(
                id=existing_id,
                name=req.name,
                code=req.code,
                language=req.language,
                created_at=old.created_at,
            )
        else:
            fn = Function(
                id=self._next_id,
                name=req.name,
                code=req.code,
                language=req.language,
                created_at=time.time(),
            )
            self._next_id += 1
            self._id_by_name[req.name] = fn.id
        self._by_id[fn.id] = fn
        return FunctionResponse(function=fn)

    async def get(self, id: int) -> Function:
        fn = self._by_id.get(id)
        if fn is None:
            raise KeyError(f"no function with id {id}")
        return fn

    async def delete(self, id: int) -> None:
        fn = self._by_id.pop(id, None)
        if fn is not None:
            self._id_by_name.pop(fn.name, None)

    async def list(self) -> list[Function]:
        return list(self._by_id.values())

    async def run(self, id: int, *, timeout: float = 30.0) -> RunResponse:
        fn = await self.get(id)
        run_id = self._next_run_id
        self._next_run_id += 1
        started = time.time()
        proc = subprocess.run(
            [sys.executable, "-c", fn.code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        run = RunResponse(
            id=run_id,
            function_id=id,
            status="ok" if proc.returncode == 0 else "error",
            stdout=proc.stdout,
            stderr=proc.stderr,
            started_at=started,
            finished_at=time.time(),
        )
        self._runs[run_id] = run
        return run

    async def get_run(self, run_id: int) -> RunResponse:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"no run with id {run_id}")
        return run
