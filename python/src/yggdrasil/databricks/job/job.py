"""Databricks Job resource — individual job lifecycle management.

:class:`Job` wraps the SDK :class:`databricks.sdk.service.jobs.Job`
with cached settings, run triggering, and permission management.
Singleton-cached by ``(service, job_id)`` so repeat lookups for the
same job under the same client collapse to one live handle.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Iterator, Optional, TYPE_CHECKING

from databricks.sdk.service.jobs import (
    Job as SDKJob,
    JobAccessControlRequest,
    JobSettings,
    Task,
)

from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.url import URL

from ..resource import DatabricksResource
from .service import Jobs, JobRuns, _check_permission, _check_task, _resolve_cluster_id, _set_cached_name

if TYPE_CHECKING:
    from ..cluster import Cluster
    from .run import JobRun

__all__ = ["Job"]

LOGGER = logging.getLogger(__name__)


class Job(Singleton, DatabricksResource):
    """High-level Databricks job handle.

    Parameters
    ----------
    service:
        Parent :class:`Jobs` service.
    job_id:
        Databricks job id.
    name:
        Job display name.
    details:
        Pre-fetched SDK :class:`~databricks.sdk.service.jobs.Job`.
    """

    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: Jobs | None = None,
        job_id: int | None = None,
        name: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        return (cls, service, job_id)

    def __init__(
        self,
        service: Jobs | None = None,
        job_id: int | None = None,
        name: str | None = None,
        *,
        details: SDKJob | None = None,
        singleton_ttl: Any = ...,
    ):
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        if service is None:
            service = Jobs.current()

        super().__init__(service=service)
        self.service: Jobs = service
        self.job_id: int | None = job_id
        self.name: str | None = name
        self._details: SDKJob | None = details

        if self._details is not None:
            self.job_id = self._details.job_id
            if self._details.settings and self._details.settings.name:
                self.name = self._details.settings.name

        if self.name and not self.job_id:
            found = self.service.find(name=self.name, raise_error=True)
            self.job_id = found.job_id
            self._details = found._details

        self._initialized = True

    def __str__(self) -> str:
        return f"Job({self.name or self.job_id})"

    def __hash__(self):
        return hash((type(self), self.job_id))

    def __eq__(self, other):
        return isinstance(other, Job) and self.job_id == other.job_id

    @property
    def explore_url(self) -> URL:
        return self.client.base_url.with_path(f"/jobs/{self.job_id or 'unknown'}")

    # ------------------------------------------------------------------ #
    # Details
    # ------------------------------------------------------------------ #

    @property
    def details(self) -> SDKJob:
        if self._details is None and self.job_id is not None:
            self.refresh()
        return self._details

    @property
    def settings(self) -> JobSettings | None:
        d = self.details
        return d.settings if d else None

    @property
    def tasks(self) -> list[Task]:
        s = self.settings
        return s.tasks or [] if s else []

    @property
    def tags(self) -> dict[str, str]:
        s = self.settings
        return s.tags or {} if s else {}

    def refresh(self) -> "Job":
        sdk = self.client.workspace_client().jobs
        raw = sdk.get(job_id=self.job_id)
        self._details = raw
        if raw.settings and raw.settings.name:
            self.name = raw.settings.name
            _set_cached_name(self.client, self.name, self.job_id)
        return self

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(
        self,
        *,
        parameters: dict[str, str] | None = None,
        notebook_params: dict[str, str] | None = None,
        python_params: list[str] | None = None,
        jar_params: list[str] | None = None,
        wait: WaitingConfigArg = False,
        raise_error: bool = True,
    ) -> "JobRun":
        """Trigger a new run of this job.

        Parameters
        ----------
        parameters:
            Job-level parameters (``job_parameters`` in the SDK).
        notebook_params:
            Notebook-specific parameter overrides.
        python_params / jar_params:
            Task-type-specific parameter overrides.
        wait:
            ``False`` (default) = fire-and-forget; ``True`` = block until
            terminal; a number = timeout in seconds.
        raise_error:
            Raise on terminal failure when waiting.
        """
        from .run import JobRun

        sdk = self.client.workspace_client().jobs

        LOGGER.debug("Triggering run for job %r (id=%s)", self.name, self.job_id)

        response = sdk.run_now(
            job_id=self.job_id,
            job_parameters=parameters,
            notebook_params=notebook_params,
            python_params=python_params,
            jar_params=jar_params,
        )

        runs_svc = JobRuns(client=self.client)
        job_run = JobRun(
            service=runs_svc,
            run_id=response.run_id,
            job_id=self.job_id,
        )

        LOGGER.info("Triggered run %s for job %r", response.run_id, self.name)

        if wait is not False:
            job_run.wait(wait=wait, raise_error=raise_error)

        return job_run

    def run_and_wait(
        self,
        *,
        parameters: dict[str, str] | None = None,
        notebook_params: dict[str, str] | None = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "JobRun":
        return self.run(
            parameters=parameters,
            notebook_params=notebook_params,
            wait=wait,
            raise_error=raise_error,
        )

    # ------------------------------------------------------------------ #
    # Runs listing
    # ------------------------------------------------------------------ #

    @property
    def runs(self) -> "JobRuns":
        return JobRuns(client=self.client)

    def list_runs(
        self,
        *,
        active_only: bool = False,
        completed_only: bool = False,
        limit: int | None = None,
    ) -> Iterator["JobRun"]:
        return self.runs.list(
            job_id=self.job_id,
            active_only=active_only,
            completed_only=completed_only,
            limit=limit,
        )

    def latest_run(self) -> Optional["JobRun"]:
        return next(self.list_runs(limit=1), None)

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(
        self,
        *,
        tasks: list[Task | dict] | None = None,
        cluster: "Cluster | str | None" = None,
        permissions: list[str | JobAccessControlRequest] | None = None,
        tags: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
        max_concurrent_runs: int | None = None,
        **settings_kwargs: Any,
    ) -> "Job":
        sdk = self.client.workspace_client().jobs

        current = self.settings
        new_settings: dict[str, Any] = {}

        if current:
            new_settings = {
                k: v for k, v in current.as_shallow_dict().items()
                if v is not None
            }

        if tasks is not None:
            new_settings["tasks"] = [_check_task(t, cluster=cluster) for t in tasks]
        elif cluster is not None and "tasks" in new_settings:
            cid = _resolve_cluster_id(cluster)
            updated = []
            for t in (new_settings["tasks"] or []):
                if isinstance(t, Task) and t.existing_cluster_id is None and t.new_cluster is None:
                    t.existing_cluster_id = cid
                updated.append(t)
            new_settings["tasks"] = updated

        if tags is not None:
            merged = self.service.default_tags(update=True)
            merged.update(tags)
            new_settings["tags"] = merged

        if timeout_seconds is not None:
            new_settings["timeout_seconds"] = timeout_seconds
        if max_concurrent_runs is not None:
            new_settings["max_concurrent_runs"] = max_concurrent_runs

        new_settings.update(settings_kwargs)
        new_settings.pop("name", None)

        LOGGER.debug("Updating job %r (id=%s)", self.name, self.job_id)
        sdk.reset(
            job_id=self.job_id,
            new_settings=JobSettings(name=self.name, **new_settings),
        )
        self._details = None
        LOGGER.info("Updated job %r", self.name)

        if permissions:
            self.update_permissions(permissions)

        return self

    def update_permissions(
        self,
        permissions: list[str | JobAccessControlRequest],
    ) -> "Job":
        sdk = self.client.workspace_client().jobs
        checked = [_check_permission(p) for p in permissions]
        sdk.update_permissions(
            job_id=str(self.job_id),
            access_control_list=checked,
        )
        return self

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #

    def delete(self) -> None:
        if not self.job_id:
            return
        LOGGER.debug("Deleting job %r (id=%s)", self.name, self.job_id)
        self.client.workspace_client().jobs.delete(job_id=self.job_id)
        self.invalidate_singleton()
        LOGGER.info("Deleted job %r", self.name)

    # ------------------------------------------------------------------ #
    # Cancel all runs
    # ------------------------------------------------------------------ #

    def cancel_all_runs(self) -> "Job":
        sdk = self.client.workspace_client().jobs
        sdk.cancel_all_runs(job_id=self.job_id)
        LOGGER.info("Cancelled all runs for job %r", self.name)
        return self
