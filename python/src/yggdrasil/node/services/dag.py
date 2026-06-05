from __future__ import annotations

import datetime as dt
import logging
import time
from collections import OrderedDict, deque
from threading import Lock
from typing import Any

import httpx

from ..config import Settings
from ..exceptions import NotFoundError
from ..ids import make_id
from ..schemas.dag import (
    DagCreate,
    DagEntry,
    DagListResponse,
    DagResponse,
    DagRunEntry,
    DagRunListResponse,
    DagRunResponse,
    DagStep,
)
from ..schemas.run import RunCreate
from .environment import EnvironmentService
from .function import FunctionService
from .run import RunService

LOGGER = logging.getLogger(__name__)

_MAX_DAGS = 128
_MAX_DAG_RUNS = 256


class DagService:
    def __init__(
        self,
        settings: Settings,
        function_service: FunctionService,
        environment_service: EnvironmentService,
        run_service: RunService,
    ) -> None:
        self.settings = settings
        self._function_service = function_service
        self._environment_service = environment_service
        self._run_service = run_service
        self._dags: OrderedDict[int, DagEntry] = OrderedDict()
        self._runs: OrderedDict[int, DagRunEntry] = OrderedDict()
        self._lock = Lock()

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: DagCreate) -> DagResponse:
        """Create-or-update: if a DAG with the same name exists, update it."""
        return await self.create_or_update(req)

    async def create_or_update(self, req: DagCreate) -> DagResponse:
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        with self._lock:
            existing = next(
                (d for d in self._dags.values() if d.name == req.name), None
            )
            if existing:
                updated = existing.model_copy(update={
                    "description": req.description,
                    "steps": req.steps,
                    "edges": req.edges,
                    "updated_at": now,
                })
                self._dags[existing.id] = updated
                LOGGER.info("Upserted DAG %r (name=%r, mode=update)", existing.id, req.name)
                return DagResponse(dag=updated)
            else:
                dag_id = make_id(req.name)
                entry = DagEntry(
                    id=dag_id,
                    name=req.name,
                    description=req.description,
                    steps=req.steps,
                    edges=req.edges,
                    created_at=now,
                    updated_at=now,
                )
                self._dags[dag_id] = entry
                self._evict_dags()
                LOGGER.info("Upserted DAG %r (name=%r, mode=create)", dag_id, req.name)
                return DagResponse(dag=entry)

    async def get(self, dag_id: int) -> DagEntry:
        with self._lock:
            entry = self._dags.get(dag_id)
        if entry is None:
            raise NotFoundError(f"DAG {dag_id!r} not found")
        return entry

    async def list(self) -> DagListResponse:
        with self._lock:
            items = list(self._dags.values())
        return DagListResponse(
            node_id=self.settings.node_id,
            dags=items,
        )

    async def delete(self, dag_id: int) -> DagResponse:
        with self._lock:
            entry = self._dags.pop(dag_id, None)
        if entry is None:
            raise NotFoundError(f"DAG {dag_id!r} not found")
        LOGGER.info("Deleted DAG %r", dag_id)
        return DagResponse(dag=entry)

    # -- execution ----------------------------------------------------------

    async def execute(self, dag_id: int) -> DagRunResponse:
        dag = await self.get(dag_id)

        run_id = make_id(f"dagrun:{dag_id}:{time.monotonic()}")
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        t0 = time.monotonic()

        dag_run = DagRunEntry(
            id=run_id,
            dag_id=dag_id,
            status="running",
            started_at=now,
        )

        with self._lock:
            self._runs[run_id] = dag_run
            self._evict_runs()

        step_results: dict[str, Any] = {}
        status = "completed"

        try:
            # Topological sort
            ordered = self._topo_sort(dag.steps)

            for step in ordered:
                # Build args: start with step.ref.args, overlay edge inputs
                args = dict(step.ref.args)
                for edge in dag.edges:
                    if edge.to_step == step.id and edge.from_step in step_results:
                        prior_result = step_results[edge.from_step]
                        # If prior result is a dict, try to extract output_key
                        if isinstance(prior_result, dict):
                            value = prior_result.get(edge.output_key, prior_result)
                        else:
                            value = prior_result
                        args[edge.input_key] = value

                # Execute step
                if step.ref.node_url:
                    # Remote execution
                    result = await self._execute_remote(step, args)
                else:
                    # Local execution
                    result = await self._execute_local(step, args)

                step_results[step.id] = result

        except Exception as exc:
            status = "failed"
            LOGGER.error("DAG run %r failed at step: %s", run_id, exc)

        duration = round(time.monotonic() - t0, 3)
        completed_at = dt.datetime.now(dt.timezone.utc).isoformat()

        dag_run = dag_run.model_copy(update={
            "status": status,
            "completed_at": completed_at,
            "duration": duration,
            "step_results": step_results,
        })

        with self._lock:
            self._runs[run_id] = dag_run
            # Increment DAG run count
            dag_entry = self._dags.get(dag_id)
            if dag_entry is not None:
                self._dags[dag_id] = dag_entry.model_copy(
                    update={"run_count": dag_entry.run_count + 1}
                )

        LOGGER.info("DAG run %r %s (%.2fs)", run_id, status, duration)
        return DagRunResponse(run=dag_run)

    async def get_run(self, dag_id: int, run_id: int) -> DagRunEntry:
        with self._lock:
            entry = self._runs.get(run_id)
        if entry is None or entry.dag_id != dag_id:
            raise NotFoundError(f"DAG run {run_id!r} not found")
        return entry

    async def list_runs(self, dag_id: int) -> DagRunListResponse:
        # Validate DAG exists
        await self.get(dag_id)
        with self._lock:
            items = [r for r in self._runs.values() if r.dag_id == dag_id]
        return DagRunListResponse(
            node_id=self.settings.node_id,
            runs=items,
        )

    # -- internals ----------------------------------------------------------

    def _topo_sort(self, steps: list[DagStep]) -> list[DagStep]:
        """Topological sort of steps by depends_on."""
        step_map = {s.id: s for s in steps}
        in_degree: dict[str, int] = {s.id: 0 for s in steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in steps}

        for s in steps:
            for dep in s.depends_on:
                if dep in adjacency:
                    adjacency[dep].append(s.id)
                    in_degree[s.id] += 1

        queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
        result: list[DagStep] = []

        while queue:
            sid = queue.popleft()
            result.append(step_map[sid])
            for neighbor in adjacency[sid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(steps):
            raise ValueError("DAG contains a cycle — cannot execute")

        return result

    async def _execute_local(self, step: DagStep, args: dict[str, Any]) -> Any:
        """Execute a step locally via RunService."""
        req = RunCreate(
            function_id=step.ref.function_id,
            environment_id=step.ref.environment_id,
            args=args,
        )
        response = await self._run_service.create(req)
        run = response.run

        if run.status == "failed":
            raise RuntimeError(
                f"Step {step.id!r} failed: {run.stderr or 'unknown error'}"
            )

        # Return stdout as the result (parsed as JSON if possible)
        if run.stdout:
            import json
            try:
                return json.loads(run.stdout.strip())
            except (json.JSONDecodeError, ValueError):
                return run.stdout.strip()
        return None

    async def _execute_remote(self, step: DagStep, args: dict[str, Any]) -> Any:
        """Execute a step on a remote node via HTTP."""
        url = f"{step.ref.node_url}/api/function/{step.ref.function_id}/run"

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(url, json={"args": args})
            resp.raise_for_status()
            data = resp.json()

        run_data = data.get("run", data)
        if run_data.get("status") == "failed":
            raise RuntimeError(
                f"Remote step {step.id!r} failed: {run_data.get('stderr', 'unknown error')}"
            )

        stdout = run_data.get("stdout")
        if stdout:
            import json
            try:
                return json.loads(stdout.strip())
            except (json.JSONDecodeError, ValueError):
                return stdout.strip()
        return None

    def _evict_dags(self) -> None:
        while len(self._dags) > _MAX_DAGS:
            self._dags.popitem(last=False)

    def _evict_runs(self) -> None:
        while len(self._runs) > _MAX_DAG_RUNS:
            self._runs.popitem(last=False)
