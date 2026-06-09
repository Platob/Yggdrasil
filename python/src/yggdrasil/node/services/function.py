"""FunctionService — stored Python functions, upsert by name + async run.

Functions live in an in-memory dict keyed by their int id. ``create`` upserts
on name: a known name keeps its id and replaces the body; a new name mints a
fresh id (xxhash of the name composed with a millisecond timestamp, per the
ygg int64-id convention). ``run`` executes the stored code in a sandboxed
namespace and captures stdout/stderr into a ``RunResult``.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import time
from typing import Any

import xxhash

from yggdrasil.node.schemas.function import (
    Function,
    FunctionCreate,
    FunctionResponse,
    RunResult,
)


class FunctionService:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._by_id: dict[int, Function] = {}
        self._id_by_name: dict[str, int] = {}
        self._runs: dict[int, RunResult] = {}
        self._run_ids = itertools.count(1)

    async def create(self, req: FunctionCreate) -> FunctionResponse:
        existing = self._id_by_name.get(req.name)
        if existing is not None:
            fn = Function(id=existing, name=req.name, code=req.code, language=req.language)
        else:
            fid = (xxhash.xxh32(req.name.encode()).intdigest() << 32) | (int(time.time() * 1000) & 0xFFFFFFFF)
            fn = Function(id=fid, name=req.name, code=req.code, language=req.language)
            self._id_by_name[req.name] = fid
        self._by_id[fn.id] = fn
        return FunctionResponse(function=fn)

    async def get(self, id: int) -> FunctionResponse:
        fn = self._by_id.get(id)
        if fn is None:
            raise KeyError(f"no function with id {id}; known ids: {sorted(self._by_id)}.")
        return FunctionResponse(function=fn)

    async def list(self) -> list[Function]:
        return list(self._by_id.values())

    async def delete(self, id: int) -> None:
        fn = self._by_id.pop(id, None)
        if fn is not None:
            self._id_by_name.pop(fn.name, None)

    async def run(self, id: int) -> RunResult:
        fn = self._by_id.get(id)
        if fn is None:
            raise KeyError(f"no function with id {id} to run.")
        run = RunResult(id=next(self._run_ids), function_id=id, status="running")
        self._runs[run.id] = run
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                ns: dict[str, Any] = {}
                exec(compile(fn.code, f"<function:{fn.name}>", "exec"), ns, ns)
            run.status = "succeeded"
            run.result = ns.get("result")
        except Exception as exc:  # capture, don't propagate — it's a user script
            run.status = "failed"
            err.write(f"{type(exc).__name__}: {exc}")
        run.stdout = out.getvalue()
        run.stderr = err.getvalue()
        return run

    async def get_run(self, run_id: int) -> RunResult:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"no run with id {run_id}.")
        return run
