from __future__ import annotations

import datetime as dt
import logging
import time
from collections import OrderedDict
from threading import Lock

from ..config import Settings
from ..exceptions import NotFoundError
from ..ids import make_id
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
        self._functions: OrderedDict[int, FunctionEntry] = OrderedDict()
        self._by_name: dict[str, int] = {}  # name -> id secondary index for O(1) lookup
        self._lock = Lock()

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: FunctionCreate) -> FunctionResponse:
        """Create-or-update: if a function with the same name exists, update it."""
        return await self.create_or_update(req)

    async def create_or_update(self, req: FunctionCreate) -> FunctionResponse:
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        with self._lock:
            # O(1) name lookup via secondary index
            existing_id = self._by_name.get(req.name)
            existing = self._functions.get(existing_id) if existing_id is not None else None
            if existing:
                # Update in place
                updates: dict = {"updated_at": now}
                updates["code"] = req.code
                updates["language"] = req.language
                if req.description:
                    updates["description"] = req.description
                if req.python_version is not None:
                    updates["python_version"] = req.python_version
                if req.dependencies:
                    updates["dependencies"] = list(req.dependencies)
                if req.environment_id is not None:
                    updates["environment_id"] = req.environment_id

                updated = existing.model_copy(update=updates)
                self._functions[existing.id] = updated
                LOGGER.info("Upserted function %r (name=%r, mode=update)", existing.id, req.name)
                return FunctionResponse(function=updated)
            else:
                # Create new
                func_id = make_id(req.name)
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
                self._functions[func_id] = entry
                self._by_name[req.name] = func_id
                self._evict()
                LOGGER.info("Upserted function %r (name=%r, mode=create)", func_id, req.name)
                return FunctionResponse(function=entry)

    async def get(self, func_id: int) -> FunctionEntry:
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

    async def update(self, func_id: int, req: FunctionUpdate) -> FunctionResponse:
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
            # Keep the name index consistent when name changes
            if req.name is not None and req.name != entry.name:
                self._by_name.pop(entry.name, None)
                self._by_name[req.name] = func_id

        LOGGER.info("Updated function %r", func_id)
        return FunctionResponse(function=updated)

    async def delete(self, func_id: int) -> FunctionResponse:
        with self._lock:
            entry = self._functions.pop(func_id, None)
            if entry is not None:
                self._by_name.pop(entry.name, None)
        if entry is None:
            raise NotFoundError(f"Function {func_id!r} not found")
        LOGGER.info("Deleted function %r", func_id)
        return FunctionResponse(function=entry)

    async def clone(self, func_id: int, new_name: str | None = None) -> FunctionResponse:
        """Clone a function with a new ID and name."""
        with self._lock:
            entry = self._functions.get(func_id)
        if entry is None:
            raise NotFoundError(f"Function {func_id!r} not found")

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        clone_name = new_name or f"{entry.name}_clone"
        clone_id = make_id(clone_name)

        cloned = FunctionEntry(
            id=clone_id,
            name=clone_name,
            language=entry.language,
            code=entry.code,
            description=entry.description,
            python_version=entry.python_version,
            dependencies=list(entry.dependencies),
            environment_id=entry.environment_id,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._functions[clone_id] = cloned
            self._by_name[clone_name] = clone_id
            self._evict()

        LOGGER.info("Cloned function %r -> %r (name=%r)", func_id, clone_id, clone_name)
        return FunctionResponse(function=cloned)

    def increment_run_count(self, func_id: int) -> None:
        """Bump the run counter and touch last_used_at (called by RunService)."""
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            entry = self._functions.get(func_id)
            if entry is not None:
                self._functions[func_id] = entry.model_copy(
                    update={"run_count": entry.run_count + 1, "last_used_at": now}
                )

    # -- internals ----------------------------------------------------------

    def _evict(self) -> None:
        while len(self._functions) > self.settings.max_functions:
            self._functions.popitem(last=False)
