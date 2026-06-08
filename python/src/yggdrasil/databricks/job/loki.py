"""Loki skills for the **jobs** service (``dbc.jobs`` / ``dbc.job_runs``).

A job is a scheduled/triggerable workflow of tasks; a job run is one execution
of it. ``dbc.jobs`` lists jobs and triggers a run (returning the new run's id +
URL); ``dbc.job_runs`` lists run history, optionally scoped to a job.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksJobsSkill", "DatabricksJobRunsSkill"]


@register
class DatabricksJobsSkill(DatabricksServiceSkill):
    """List jobs, or trigger a run of one by name/id (returns the new run)."""

    name = "databricks-jobs"
    description = "List Databricks jobs, or trigger a run of one by name/id."
    preprompt = (
        "You manage Databricks jobs via dbc.jobs: list them, or trigger a run "
        "with job.run(parameters=…) — triggering a run is a real, billable "
        "action, so confirm intent and pass parameters explicitly."
    )

    def run(
        self,
        agent: "Loki",
        *,
        run: Optional[str] = None,
        parameters: Optional[dict] = None,
        **_: Any,
    ) -> dict[str, Any]:
        client = self._client(agent)
        if run:
            job = client.jobs.get(run, default=None)
            if job is None:
                return {"job": run, "found": False}
            job_run = job.run(parameters=parameters or None)
            return {
                "job": job.name or job.job_id,
                "job_id": job.job_id,
                "run_id": getattr(job_run, "run_id", None),
                "url": str(getattr(job_run, "url", "")) or None,
            }
        return {"jobs": names(client.jobs.list())}


@register
class DatabricksJobRunsSkill(DatabricksServiceSkill):
    """List job-run history — across, or scoped to one job (active/completed)."""

    name = "databricks-job-runs"
    description = "List Databricks job-run history, optionally scoped to a job."
    preprompt = (
        "You inspect job-run history via dbc.job_runs.list(job=…, active_only/"
        "completed_only). Use it to check whether a job is running or how its "
        "last runs went; each run has a run_id, state, and URL."
    )

    def run(
        self,
        agent: "Loki",
        *,
        job: Optional[str] = None,
        active_only: bool = False,
        completed_only: bool = False,
        limit: int = 25,
        **_: Any,
    ) -> dict[str, Any]:
        runs = self._client(agent).job_runs.list(
            job, active_only=active_only, completed_only=completed_only, limit=limit,
        )
        return {
            "job": job,
            "runs": [
                {"run_id": r.run_id, "job_id": r.job_id,
                 "state": str(getattr(r, "_state", "")), "url": str(getattr(r, "url", "")) or None}
                for r in runs
            ],
        }
