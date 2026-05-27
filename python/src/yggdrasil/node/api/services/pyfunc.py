from __future__ import annotations

import datetime as dt
import logging
from collections import OrderedDict
from threading import Lock

from ...config import Settings
from ...exceptions import NotFoundError
from ...ids import make_id
from ..schemas.pyfunc import (
    PyFuncCreate,
    PyFuncEntry,
    PyFuncListResponse,
    PyFuncResponse,
    PyFuncUpdate,
)

LOGGER = logging.getLogger(__name__)


class PyFuncService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._funcs: OrderedDict[int, PyFuncEntry] = OrderedDict()
        self._lock = Lock()

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: PyFuncCreate) -> PyFuncResponse:
        now = self._now()
        with self._lock:
            existing = next(
                (f for f in self._funcs.values() if f.name == req.name), None
            )
            if existing:
                updates: dict = {
                    "updated_at": now,
                    "code": req.code,
                }
                if req.description:
                    updates["description"] = req.description
                if req.python_version is not None:
                    updates["python_version"] = req.python_version
                if req.dependencies:
                    updates["dependencies"] = list(req.dependencies)
                if req.env_id is not None:
                    updates["env_id"] = req.env_id
                updated = existing.model_copy(update=updates)
                self._funcs[existing.id] = updated
                return PyFuncResponse(func=updated)

            func_id = make_id(req.name)
            entry = PyFuncEntry(
                id=func_id,
                name=req.name,
                code=req.code,
                description=req.description,
                python_version=req.python_version,
                dependencies=list(req.dependencies),
                env_id=req.env_id,
                created_at=now,
                updated_at=now,
            )
            self._funcs[func_id] = entry
            self._evict()
            return PyFuncResponse(func=entry)

    async def get(self, func_id: int) -> PyFuncEntry:
        with self._lock:
            entry = self._funcs.get(func_id)
        if entry is None:
            raise NotFoundError(f"PyFunc {func_id!r} not found")
        return entry

    async def list(self) -> PyFuncListResponse:
        with self._lock:
            items = list(self._funcs.values())
        return PyFuncListResponse(node_id=self.settings.node_id, funcs=items)

    async def update(self, func_id: int, req: PyFuncUpdate) -> PyFuncResponse:
        with self._lock:
            entry = self._funcs.get(func_id)
        if entry is None:
            raise NotFoundError(f"PyFunc {func_id!r} not found")

        now = self._now()
        updates: dict = {"updated_at": now}
        for field in ("name", "code", "description", "python_version", "dependencies", "env_id"):
            val = getattr(req, field)
            if val is not None:
                updates[field] = list(val) if field == "dependencies" else val

        updated = entry.model_copy(update=updates)
        with self._lock:
            self._funcs[func_id] = updated
        return PyFuncResponse(func=updated)

    async def delete(self, func_id: int) -> PyFuncResponse:
        with self._lock:
            entry = self._funcs.pop(func_id, None)
        if entry is None:
            raise NotFoundError(f"PyFunc {func_id!r} not found")
        return PyFuncResponse(func=entry)

    def increment_run_count(self, func_id: int) -> None:
        now = self._now()
        with self._lock:
            entry = self._funcs.get(func_id)
            if entry is not None:
                self._funcs[func_id] = entry.model_copy(
                    update={"run_count": entry.run_count + 1, "last_run_at": now}
                )

    # -- internals ----------------------------------------------------------

    def _evict(self) -> None:
        while len(self._funcs) > self.settings.max_functions:
            self._funcs.popitem(last=False)

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()
