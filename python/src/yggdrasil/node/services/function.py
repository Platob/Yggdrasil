"""In-memory python function registry.

Upsert by name: creating a function whose name already exists updates the
stored code in place and keeps the original ``id``. IDs are immutable once
assigned. Running a function records a :class:`RunData` entry (execution is
captured but not sandboxed here — that is the node runtime's job).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from yggdrasil.exceptions.node import NodeNotFoundError

from ..config import Settings
from ..schemas.function import (
    FunctionCreate,
    FunctionData,
    FunctionResponse,
    RunData,
    RunResponse,
)


def _make_id() -> str:
    return uuid4().hex


class FunctionService:
    """Function CRUD + run history, all in process memory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._by_id: dict[str, FunctionData] = {}
        self._id_by_name: dict[str, str] = {}
        self._runs: dict[str, RunData] = {}

    async def create(self, payload: FunctionCreate) -> FunctionResponse:
        async with self._lock:
            existing_id = self._id_by_name.get(payload.name)
            if existing_id is not None:
                func = self._by_id[existing_id]
                func.code = payload.code
                func.language = payload.language
                func.updated_at = datetime.now(timezone.utc)
                return FunctionResponse(function=func)
            func = FunctionData(
                id=_make_id(),
                name=payload.name,
                code=payload.code,
                language=payload.language,
            )
            self._by_id[func.id] = func
            self._id_by_name[func.name] = func.id
            return FunctionResponse(function=func)

    async def get(self, function_id: str) -> FunctionData:
        async with self._lock:
            func = self._by_id.get(function_id)
            if func is None:
                raise NodeNotFoundError(
                    f"No function with id {function_id!r}. "
                    f"{len(self._by_id)} function(s) registered."
                )
            return func

    async def list(self) -> list[FunctionData]:
        async with self._lock:
            return list(self._by_id.values())

    async def delete(self, function_id: str) -> None:
        async with self._lock:
            func = self._by_id.pop(function_id, None)
            if func is None:
                raise NodeNotFoundError(
                    f"No function with id {function_id!r} to delete."
                )
            self._id_by_name.pop(func.name, None)

    async def run(self, function_id: str) -> RunResponse:
        async with self._lock:
            func = self._by_id.get(function_id)
            if func is None:
                raise NodeNotFoundError(
                    f"No function with id {function_id!r} to run."
                )
            run = RunData(id=_make_id(), function_id=function_id)
            self._runs[run.id] = run
            return RunResponse(run=run)

    async def get_run(self, run_id: str) -> RunData:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise NodeNotFoundError(f"No run with id {run_id!r}.")
            return run
