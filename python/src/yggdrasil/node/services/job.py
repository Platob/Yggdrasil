from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
import time
import uuid
from collections import OrderedDict
from functools import partial
from threading import Lock
from typing import Any

from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..exceptions import NotFoundError
from ..schemas.job import (
    JobCreateRequest,
    JobEntry,
    JobListResponse,
    JobResponse,
    RunEntry,
    RunListResponse,
    RunResponse,
    TaskDefinition,
)

LOGGER = logging.getLogger(__name__)


class JobService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: OrderedDict[str, _StoredJob] = OrderedDict()
        self._lock = Lock()

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    # -- job CRUD ----------------------------------------------------------

    async def create_job(self, req: JobCreateRequest) -> JobResponse:
        job_id = uuid.uuid4().hex[:12]
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        stored = _StoredJob(
            job_id=job_id,
            name=req.name,
            tasks=dict(req.tasks),
            schedule=req.schedule,
            created_at=now,
        )
        with self._lock:
            self._jobs[job_id] = stored
            self._evict_old_jobs()

        LOGGER.info("Created job %r (name=%r, tasks=%d)", job_id, req.name, len(req.tasks))

        return JobResponse(
            node_id=self.settings.node_id,
            job=stored.to_entry(),
        )

    async def get_job(self, job_id: str) -> JobResponse:
        stored = self._get_stored(job_id)
        return JobResponse(
            node_id=self.settings.node_id,
            job=stored.to_entry(),
        )

    async def list_jobs(self) -> JobListResponse:
        with self._lock:
            items = [s.to_entry() for s in self._jobs.values()]
        return JobListResponse(
            node_id=self.settings.node_id,
            items=items,
        )

    async def update_job(self, job_id: str, req: JobCreateRequest) -> JobResponse:
        stored = self._get_stored(job_id)
        stored.name = req.name
        stored.tasks = dict(req.tasks)
        stored.schedule = req.schedule
        LOGGER.info("Updated job %r", job_id)
        return JobResponse(
            node_id=self.settings.node_id,
            job=stored.to_entry(),
        )

    async def delete_job(self, job_id: str) -> JobResponse:
        with self._lock:
            stored = self._jobs.pop(job_id, None)
        if stored is None:
            raise NotFoundError(f"Job {job_id!r} not found")
        LOGGER.info("Deleted job %r", job_id)
        return JobResponse(
            node_id=self.settings.node_id,
            job=stored.to_entry(),
        )

    # -- run CRUD ----------------------------------------------------------

    async def trigger_run(self, job_id: str) -> RunResponse:
        stored = self._get_stored(job_id)
        run_id = uuid.uuid4().hex[:12]

        LOGGER.info("Triggering run %r for job %r", run_id, job_id)

        run_entry = await self._run(self._execute_run, stored, run_id)
        return RunResponse(
            node_id=self.settings.node_id,
            run=run_entry,
        )

    async def get_run(self, job_id: str, run_id: str) -> RunResponse:
        stored = self._get_stored(job_id)
        with self._lock:
            run = stored.runs.get(run_id)
        if run is None:
            raise NotFoundError(f"Run {run_id!r} not found on job {job_id!r}")
        return RunResponse(
            node_id=self.settings.node_id,
            run=run,
        )

    async def list_runs(self, job_id: str) -> RunListResponse:
        stored = self._get_stored(job_id)
        with self._lock:
            items = list(stored.runs.values())
        return RunListResponse(
            node_id=self.settings.node_id,
            items=items,
        )

    async def delete_run(self, job_id: str, run_id: str) -> RunResponse:
        stored = self._get_stored(job_id)
        with self._lock:
            run = stored.runs.pop(run_id, None)
        if run is None:
            raise NotFoundError(f"Run {run_id!r} not found on job {job_id!r}")
        LOGGER.info("Deleted run %r from job %r", run_id, job_id)
        return RunResponse(
            node_id=self.settings.node_id,
            run=run,
        )

    async def update_run(self, job_id: str, run_id: str) -> RunResponse:
        """Re-trigger a run (replay)."""
        stored = self._get_stored(job_id)
        with self._lock:
            old_run = stored.runs.get(run_id)
        if old_run is None:
            raise NotFoundError(f"Run {run_id!r} not found on job {job_id!r}")

        LOGGER.info("Re-triggering run %r for job %r", run_id, job_id)
        run_entry = await self._run(self._execute_run, stored, run_id)
        return RunResponse(
            node_id=self.settings.node_id,
            run=run_entry,
        )

    # -- internals ---------------------------------------------------------

    def _get_stored(self, job_id: str) -> _StoredJob:
        with self._lock:
            stored = self._jobs.get(job_id)
        if stored is None:
            raise NotFoundError(f"Job {job_id!r} not found")
        return stored

    def _execute_run(self, stored: _StoredJob, run_id: str) -> RunEntry:
        now = dt.datetime.now(dt.timezone.utc)
        task_results: dict[str, Any] = {}
        completed_tasks: set[str] = set()
        overall_status = "completed"
        t0 = time.monotonic()

        ordered = self._topo_sort(stored.tasks)

        for task_key in ordered:
            task = stored.tasks[task_key]
            unmet = set(task.depends_on) - completed_tasks
            if unmet:
                task_results[task_key] = {
                    "status": "skipped",
                    "reason": f"unmet dependencies: {sorted(unmet)}",
                }
                overall_status = "failed"
                continue

            result = self._run_task(task_key, task)
            task_results[task_key] = result
            if result.get("status") == "completed":
                completed_tasks.add(task_key)
            else:
                overall_status = "failed"

        duration = time.monotonic() - t0
        run_entry = RunEntry(
            run_id=run_id,
            job_id=stored.job_id,
            status=overall_status,
            started_at=now.isoformat(),
            finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            duration=round(duration, 3),
            task_results=task_results,
        )
        with self._lock:
            stored.runs[run_id] = run_entry
        return run_entry

    def _run_task(self, task_key: str, task: TaskDefinition) -> dict[str, Any]:
        LOGGER.debug("Running task %r (type=%s)", task_key, task.type)
        timeout = task.timeout or self.settings.max_cmd_timeout

        if task.type == "cmd":
            if not task.command:
                return {"status": "failed", "error": "No command specified"}
            try:
                proc = subprocess.run(
                    task.command,
                    env=dict(task.env) if task.env else None,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
                return {
                    "status": "completed" if proc.returncode == 0 else "failed",
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            except subprocess.TimeoutExpired:
                return {"status": "failed", "error": f"Timed out after {timeout:.0f}s"}
            except Exception as exc:
                return {"status": "failed", "error": str(exc)}

        if task.type == "python":
            if not task.code:
                return {"status": "failed", "error": "No code specified"}
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", task.code],
                    env=dict(task.env) if task.env else None,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
                return {
                    "status": "completed" if proc.returncode == 0 else "failed",
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            except subprocess.TimeoutExpired:
                return {"status": "failed", "error": f"Timed out after {timeout:.0f}s"}
            except Exception as exc:
                return {"status": "failed", "error": str(exc)}

        return {"status": "failed", "error": f"Unknown task type {task.type!r}"}

    @staticmethod
    def _topo_sort(tasks: dict[str, TaskDefinition]) -> list[str]:
        visited: set[str] = set()
        order: list[str] = []

        def visit(key: str) -> None:
            if key in visited:
                return
            visited.add(key)
            task = tasks.get(key)
            if task:
                for dep in task.depends_on:
                    if dep in tasks:
                        visit(dep)
            order.append(key)

        for key in tasks:
            visit(key)
        return order

    def _evict_old_jobs(self) -> None:
        while len(self._jobs) > self.settings.job_max_history:
            self._jobs.popitem(last=False)


class _StoredJob:
    __slots__ = ("job_id", "name", "tasks", "schedule", "created_at", "runs")

    def __init__(
        self,
        job_id: str,
        name: str | None,
        tasks: dict[str, TaskDefinition],
        schedule: str | None,
        created_at: str,
    ) -> None:
        self.job_id = job_id
        self.name = name
        self.tasks = tasks
        self.schedule = schedule
        self.created_at = created_at
        self.runs: OrderedDict[str, RunEntry] = OrderedDict()

    def to_entry(self) -> JobEntry:
        return JobEntry(
            job_id=self.job_id,
            name=self.name,
            task_keys=list(self.tasks),
            schedule=self.schedule,
            created_at=self.created_at,
            run_count=len(self.runs),
        )
