"""Store and run small Python functions.

Functions are upserted by name (create-if-new, replace-code-if-seen) so a
repeated deploy of the same named function is idempotent and keeps its id. A
run executes the stored source with ``exec`` in a throwaway namespace, capturing
stdout; failures are caught and surfaced as a failed run, never as a 500.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import itertools

import xxhash

from ..config import Settings
from ..schemas.function import (
    FunctionCreate,
    FunctionRecord,
    FunctionResponse,
    RunRecord,
    RunResponse,
)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class FunctionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._by_id: dict[str, FunctionRecord] = {}
        self._id_by_name: dict[str, str] = {}
        self._runs: dict[str, RunRecord] = {}
        self._seq = itertools.count(1)

    async def create(self, data: FunctionCreate) -> FunctionResponse:
        # Upsert by name: an existing name keeps its id, only the body changes.
        existing_id = self._id_by_name.get(data.name)
        if existing_id is not None:
            prev = self._by_id[existing_id]
            rec = prev.model_copy(update={"code": data.code, "language": data.language})
            self._by_id[existing_id] = rec
            return FunctionResponse(function=rec)
        seq = next(self._seq)
        fid = xxhash.xxh32(data.name.encode()).intdigest() << 32 | seq
        rec = FunctionRecord(
            id=str(fid),
            name=data.name,
            code=data.code,
            language=data.language,
            created_at=_now_iso(),
        )
        self._by_id[rec.id] = rec
        self._id_by_name[data.name] = rec.id
        return FunctionResponse(function=rec)

    async def get(self, func_id: str) -> FunctionResponse:
        rec = self._by_id.get(func_id)
        if rec is None:
            raise KeyError(
                f"No function {func_id!r}. Known ids: {sorted(self._by_id)[:10]}"
                f"{' …' if len(self._by_id) > 10 else ''}."
            )
        return FunctionResponse(function=rec)

    async def delete(self, func_id: str) -> None:
        rec = self._by_id.pop(func_id, None)
        if rec is not None:
            self._id_by_name.pop(rec.name, None)

    async def list(self) -> list[FunctionRecord]:
        return list(self._by_id.values())

    async def run(self, func_id: str) -> RunResponse:
        rec = self._by_id.get(func_id)
        if rec is None:
            raise KeyError(f"No function {func_id!r} to run.")
        run_id = str(xxhash.xxh32(f"run:{func_id}:{next(self._seq)}".encode()).intdigest())
        started = _now_iso()
        buf = io.StringIO()
        status, error = "success", None
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(rec.code, f"<function {rec.name}>", "exec"), {"__name__": "__main__"})
        except Exception as exc:  # a user function failing is a failed run, not a node error
            status, error = "error", f"{type(exc).__name__}: {exc}"
        run = RunRecord(
            id=run_id,
            function_id=func_id,
            status=status,
            output=buf.getvalue() or None,
            error=error,
            started_at=started,
            finished_at=_now_iso(),
        )
        self._runs[run_id] = run
        return RunResponse(run=run)

    async def get_run(self, run_id: str) -> RunResponse:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"No run {run_id!r}.")
        return RunResponse(run=run)
