"""In-memory pyfunc registry — upsert-by-name CRUD over user functions.

A function record is keyed by an xxhash composite id derived from its
name (the semantic key) and the create timestamp. Upsert by default:
re-creating a function with the same name replaces the stored code but
keeps a fresh id stamped at write time.
"""
from __future__ import annotations

import datetime as dt
import time

import xxhash

from yggdrasil.exceptions.api import NotFoundError
from yggdrasil.node.config import Settings
from yggdrasil.node.schemas.function import (
    FunctionCreate,
    FunctionRecord,
    FunctionResponse,
)


class FunctionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._by_id: dict[int, FunctionRecord] = {}
        self._by_name: dict[str, int] = {}

    async def create(self, payload: FunctionCreate) -> FunctionResponse:
        ts_ms = time.time_ns() // 1_000_000
        fid = (xxhash.xxh32(payload.name.encode()).intdigest() << 32) | (ts_ms & 0xFFFFFFFF)
        # Upsert by name: drop the prior id so list/get don't return a stale twin.
        old = self._by_name.get(payload.name)
        if old is not None:
            self._by_id.pop(old, None)
        record = FunctionRecord(
            id=fid, name=payload.name, code=payload.code,
            language=payload.language, description=payload.description,
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self._by_id[fid] = record
        self._by_name[payload.name] = fid
        return FunctionResponse(function=record)

    async def get(self, id: int) -> FunctionRecord:
        record = self._by_id.get(id)
        if record is None:
            raise NotFoundError(f"No function with id {id}.")
        return record

    async def delete(self, id: int) -> None:
        record = self._by_id.pop(id, None)
        if record is None:
            raise NotFoundError(f"No function with id {id}.")
        if self._by_name.get(record.name) == id:
            del self._by_name[record.name]

    async def list(self) -> list[FunctionRecord]:
        return list(self._by_id.values())
