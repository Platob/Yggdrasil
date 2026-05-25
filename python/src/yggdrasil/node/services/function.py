from __future__ import annotations

import datetime as dt
import logging
import uuid
from collections import OrderedDict
from threading import Lock

from ..config import Settings
from ..exceptions import NotFoundError
from ..schemas.function import (
    FunctionCreate,
    FunctionEntry,
    FunctionListResponse,
    FunctionResponse,
    FunctionUpdate,
)

LOGGER = logging.getLogger(__name__)


class FunctionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._functions: OrderedDict[str, FunctionEntry] = OrderedDict()
        self._lock = Lock()

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: FunctionCreate) -> FunctionResponse:
        func_id = uuid.uuid4().hex[:12]
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        entry = FunctionEntry(
            id=func_id,
            name=req.name,
            language=req.language,
            code=req.code,
            description=req.description,
            python_version=req.python_version,
            dependencies=list(req.dependencies),
            environment_id=req.environment_id,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._functions[func_id] = entry
            self._evict()

        LOGGER.info("Created function %r (name=%r)", func_id, req.name)
        return FunctionResponse(function=entry)

    async def get(self, func_id: str) -> FunctionEntry:
        with self._lock:
            entry = self._functions.get(func_id)
        if entry is None:
            raise NotFoundError(f"Function {func_id!r} not found")
        return entry

    async def list(self) -> FunctionListResponse:
        with self._lock:
            items = list(self._functions.values())
        return FunctionListResponse(
            node_id=self.settings.node_id,
            functions=items,
        )

    async def update(self, func_id: str, req: FunctionUpdate) -> FunctionResponse:
        with self._lock:
            entry = self._functions.get(func_id)
        if entry is None:
            raise NotFoundError(f"Function {func_id!r} not found")

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        updates: dict = {"updated_at": now}
        if req.name is not None:
            updates["name"] = req.name
        if req.code is not None:
            updates["code"] = req.code
        if req.description is not None:
            updates["description"] = req.description
        if req.python_version is not None:
            updates["python_version"] = req.python_version
        if req.dependencies is not None:
            updates["dependencies"] = list(req.dependencies)
        if req.environment_id is not None:
            updates["environment_id"] = req.environment_id

        updated = entry.model_copy(update=updates)
        with self._lock:
            self._functions[func_id] = updated

        LOGGER.info("Updated function %r", func_id)
        return FunctionResponse(function=updated)

    async def delete(self, func_id: str) -> FunctionResponse:
        with self._lock:
            entry = self._functions.pop(func_id, None)
        if entry is None:
            raise NotFoundError(f"Function {func_id!r} not found")
        LOGGER.info("Deleted function %r", func_id)
        return FunctionResponse(function=entry)

    def increment_run_count(self, func_id: str) -> None:
        """Bump the run counter for a function (called by RunService)."""
        with self._lock:
            entry = self._functions.get(func_id)
            if entry is not None:
                self._functions[func_id] = entry.model_copy(
                    update={"run_count": entry.run_count + 1}
                )

    # -- internals ----------------------------------------------------------

    def _evict(self) -> None:
        while len(self._functions) > self.settings.max_functions:
            self._functions.popitem(last=False)
