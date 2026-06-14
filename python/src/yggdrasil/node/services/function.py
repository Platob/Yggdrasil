"""Function registry service — upsert by name, immutable int64 id once assigned."""
from __future__ import annotations

import time

import xxhash

from yggdrasil.exceptions.api import NotFoundError
from yggdrasil.node.schemas.function import FunctionCreate, FunctionInfo, FunctionResponse


class FunctionService:
    """In-memory function registry keyed by immutable int64 id, upsert by name."""

    def __init__(self, settings: object) -> None:
        self._settings = settings
        self._functions: dict[int, FunctionInfo] = {}
        self._names: dict[str, int] = {}

    async def create(self, req: FunctionCreate) -> FunctionResponse:
        now = time.time()
        existing = self._names.get(req.name)
        if existing is not None:
            prior = self._functions[existing]
            updated = FunctionInfo(
                id=existing,
                name=req.name,
                language=req.language,
                created_at=prior.created_at,
                updated_at=now,
            )
            self._functions[existing] = updated
            return FunctionResponse(function=updated)

        fid = (xxhash.xxh32(req.name).intdigest() << 32) | (int(now * 1000) & 0xFFFFFFFF)
        fn = FunctionInfo(
            id=fid, name=req.name, language=req.language, created_at=now, updated_at=now
        )
        self._functions[fid] = fn
        self._names[req.name] = fid
        return FunctionResponse(function=fn)

    async def get(self, fid: int) -> FunctionResponse:
        fn = self._functions.get(fid)
        if fn is None:
            raise NotFoundError(f"Function {fid} not found.")
        return FunctionResponse(function=fn)

    async def delete(self, fid: int) -> None:
        fn = self._functions.pop(fid, None)
        if fn is not None:
            self._names.pop(fn.name, None)

    async def list(self) -> list[FunctionInfo]:
        return list(self._functions.values())
