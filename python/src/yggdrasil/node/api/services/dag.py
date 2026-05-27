from __future__ import annotations

import datetime as dt
import hashlib
import json as json_mod
import logging
import time
from collections import deque
from threading import Lock
from typing import Any

import httpx

from yggdrasil.dataclasses.expiring import ExpiringDict

from ...config import Settings
from ...exceptions import NotFoundError
from ...ids import make_id
from ..schemas.dag import (
    DAGCreate,
    DAGEntry,
    DAGListResponse,
    DAGResponse,
    DAGRunEntry,
    DAGRunListResponse,
    DAGRunResponse,
    DAGStep,
)
from ..schemas.pyfuncrun import PyFuncRunCreate
from .pyfuncrun import PyFuncRunService

LOGGER = logging.getLogger(__name__)

_MAX_DAGS = 128
_MAX_DAG_RUNS = 256


class DAGService:
    def __init__(
        self,
        settings: Settings,
        pyfuncrun_service: PyFuncRunService,
    ) -> None:
        self.settings = settings
        self._pyfuncrun = pyfuncrun_service
        self._dags: ExpiringDict[int, DAGEntry] = ExpiringDict(default_ttl=None, max_size=_MAX_DAGS)
        self._runs: ExpiringDict[int, DAGRunEntry] = ExpiringDict(default_ttl=None, max_size=_MAX_DAG_RUNS)
        self._lock = Lock()

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: DAGCreate) -> DAGResponse:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            existing = next(
                (d for d in self._dags.values() if d.name == req.name), None
            )
            if existing:
                content_hash = hashlib.sha256(
                    (existing.name
                     + json_mod.dumps([s.model_dump() for s in req.steps], sort_keys=True)
                     + json_mod.dumps([e.model_dump() for e in req.edges], sort_keys=True)
                     ).encode()
                ).hexdigest()
                updated = existing.model_copy(update={
                    "description": req.description,
                    "steps": req.steps,
                    "edges": req.edges,
                    "updated_at": now,
                    "content_hash": content_hash,
                })
                self._dags[existing.id] = updated
                return DAGResponse(dag=updated)

            dag_id = make_id(req.name)
            content_hash = hashlib.sha256(
                (req.name
                 + json_mod.dumps([s.model_dump() for s in req.steps], sort_keys=True)
                 + json_mod.dumps([e.model_dump() for e in req.edges], sort_keys=True)
                 ).encode()
            ).hexdigest()
            entry = DAGEntry(
                id=dag_id,
                name=req.name,
                description=req.description,
                steps=req.steps,
                edges=req.edges,
                created_at=now,
                updated_at=now,
                content_hash=content_hash,
            )
            self._dags.set(dag_id, entry)
            return DAGResponse(dag=entry)

    async def get(self, dag_id: int) -> DAGEntry:
        with self._lock:
            entry = self._dags.get(dag_id)
        if entry is None:
            raise NotFoundError(f"DAG {dag_id!r} not found")
        return entry

    async def list(self) -> DAGListResponse:
        with self._lock:
            items = list(self._dags.values())
        return DAGListResponse(node_id=self.settings.node_id, dags=items)

    async def delete(self, dag_id: int) -> DAGResponse:
        with self._lock:
            entry = self._dags.pop(dag_id, None)
        if entry is None:
            raise NotFoundError(f"DAG {dag_id!r} not found")
        return DAGResponse(dag=entry)

    # -- execution ----------------------------------------------------------

    async def execute(self, dag_id: int) -> DAGRunResponse:
        dag = await self.get(dag_id)

        run_id = make_id(f"dagrun:{dag_id}:{time.monotonic()}")
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        t0 = time.monotonic()

        dag_run = DAGRunEntry(
            id=run_id,
            dag_id=dag_id,
            status="running",
            started_at=now,
            node_id=self.settings.node_id,
        )
        with self._lock:
            self._runs.set(run_id, dag_run)

        step_results: dict[str, Any] = {}
        status = "completed"

        try:
            ordered = self._topo_sort(dag.steps)
            for step in ordered:
                args = dict(step.ref.args)
                for edge in dag.edges:
                    if edge.to_step == step.id and edge.from_step in step_results:
                        prior = step_results[edge.from_step]
                        value = (
                            prior.get(edge.output_key, prior)
                            if isinstance(prior, dict) else prior
                        )
                        args[edge.input_key] = value

                if step.ref.node_url:
                    result = await self._execute_remote(step, args)
                else:
                    result = await self._execute_local(step, args)
                step_results[step.id] = result

        except Exception as exc:
            status = "failed"
            LOGGER.error("DAG run %r failed: %s", run_id, exc)

        duration = round(time.monotonic() - t0, 3)
        dag_run = dag_run.model_copy(update={
            "status": status,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "duration": duration,
            "step_results": step_results,
        })
        with self._lock:
            self._runs[run_id] = dag_run
            dag_entry = self._dags.get(dag_id)
            if dag_entry is not None:
                self._dags[dag_id] = dag_entry.model_copy(
                    update={"run_count": dag_entry.run_count + 1}
                )
        return DAGRunResponse(run=dag_run)

    async def get_run(self, dag_id: int, run_id: int) -> DAGRunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None or entry.dag_id != dag_id:
            raise NotFoundError(f"DAG run {run_id!r} not found")
        return entry

    async def list_runs(self, dag_id: int) -> DAGRunListResponse:
        await self.get(dag_id)
        with self._lock:
            items = [r for r in self._runs.values() if r.dag_id == dag_id]
        return DAGRunListResponse(node_id=self.settings.node_id, runs=items)

    # -- internals ----------------------------------------------------------

    def _topo_sort(self, steps: list[DAGStep]) -> list[DAGStep]:
        step_map = {s.id: s for s in steps}
        in_degree: dict[str, int] = {s.id: 0 for s in steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in steps}
        for s in steps:
            for dep in s.depends_on:
                if dep in adjacency:
                    adjacency[dep].append(s.id)
                    in_degree[s.id] += 1

        queue: deque[str] = deque(
            sid for sid, deg in in_degree.items() if deg == 0
        )
        result: list[DAGStep] = []
        while queue:
            sid = queue.popleft()
            result.append(step_map[sid])
            for neighbor in adjacency[sid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(steps):
            raise ValueError("DAG contains a cycle")
        return result

    async def _execute_local(self, step: DAGStep, args: dict[str, Any]) -> Any:
        req = PyFuncRunCreate(
            func_id=step.ref.func_id,
            env_id=step.ref.env_id,
            kwargs=args,
        )
        response = await self._pyfuncrun.create(req)
        run = response.run
        if run.status == "failed":
            raise RuntimeError(
                f"Step {step.id!r} failed: {run.stderr or 'unknown error'}"
            )
        if run.result is not None:
            return run.result
        if run.stdout:
            import json
            try:
                return json.loads(run.stdout.strip())
            except (json.JSONDecodeError, ValueError):
                return run.stdout.strip()
        return None

    async def _execute_remote(self, step: DAGStep, args: dict[str, Any]) -> Any:
        url = f"{step.ref.node_url}/api/v2/pyfuncrun"
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(url, json={
                "func_id": step.ref.func_id,
                "env_id": step.ref.env_id,
                "kwargs": args,
            })
            resp.raise_for_status()
            data = resp.json()

        run_data = data.get("run", data)
        if run_data.get("status") == "failed":
            raise RuntimeError(
                f"Remote step {step.id!r} failed: "
                f"{run_data.get('stderr', 'unknown error')}"
            )
        if run_data.get("result") is not None:
            return run_data["result"]
        stdout = run_data.get("stdout")
        if stdout:
            import json
            try:
                return json.loads(stdout.strip())
            except (json.JSONDecodeError, ValueError):
                return stdout.strip()
        return None

