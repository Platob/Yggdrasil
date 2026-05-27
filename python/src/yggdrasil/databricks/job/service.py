"""Databricks Jobs service — collection-level job and run management.

:class:`Jobs` wraps the Databricks SDK ``JobsAPI`` with sensible defaults,
cluster integration, and the yggdrasil tagging contract.  :class:`JobRuns`
provides run-level listing and retrieval across all jobs or scoped to one.

Individual job lifecycle (run, update, delete) lives on the
:class:`~yggdrasil.databricks.job.job.Job` resource.  Individual run
lifecycle (wait, cancel, repair) lives on
:class:`~yggdrasil.databricks.job.run.JobRun`.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, TYPE_CHECKING

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.jobs import (
    JobAccessControlRequest,
    JobPermissionLevel,
    JobSettings,
    Task,
)

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.waiting import WaitingConfigArg

from ..client import DatabricksClient, DatabricksService

if TYPE_CHECKING:
    from ..cluster import Cluster
    from .job import Job
    from .run import JobRun

__all__ = [
    "Jobs",
    "JobRuns",
]

LOGGER = logging.getLogger(__name__)

# host -> ExpiringDict(job_name -> job_id)
_NAME_CACHE: dict[str, ExpiringDict[str, int]] = {}


def _cache_key(client: DatabricksClient) -> str:
    return client.base_url.to_string() if client.base_url else "default"


def _set_cached_name(client: DatabricksClient, name: str, job_id: int) -> None:
    host = _cache_key(client)
    bucket = _NAME_CACHE.get(host)
    if bucket is None:
        bucket = _NAME_CACHE[host] = ExpiringDict(default_ttl=3600)
    bucket[name] = job_id


def _get_cached_id(client: DatabricksClient, name: str) -> int | None:
    host = _cache_key(client)
    bucket = _NAME_CACHE.get(host)
    return bucket.get(name) if bucket else None


def _check_task(
    task: Any,
    *,
    cluster: "Cluster | str | None" = None,
) -> Task:
    if isinstance(task, Task):
        if cluster is not None and task.existing_cluster_id is None and task.new_cluster is None:
            task.existing_cluster_id = _resolve_cluster_id(cluster)
        return task
    if isinstance(task, dict):
        return Task(**task)
    raise TypeError(f"Expected Task or dict, got {type(task).__name__}")


def _resolve_cluster_id(cluster: "Cluster | str | None") -> str | None:
    if cluster is None:
        return None
    if isinstance(cluster, str):
        return cluster
    return cluster.cluster_id


def _check_permission(
    permission: str | JobAccessControlRequest,
) -> JobAccessControlRequest:
    if isinstance(permission, JobAccessControlRequest):
        return permission
    if isinstance(permission, str):
        if permission == "users":
            return JobAccessControlRequest(
                group_name=permission,
                permission_level=JobPermissionLevel.CAN_VIEW,
            )
        if "@" in permission:
            return JobAccessControlRequest(
                user_name=permission,
                permission_level=JobPermissionLevel.CAN_MANAGE,
            )
        return JobAccessControlRequest(
            group_name=permission,
            permission_level=JobPermissionLevel.CAN_MANAGE,
        )
    raise TypeError(f"Expected str or JobAccessControlRequest, got {type(permission).__name__}")


# ---------------------------------------------------------------------------
# Jobs service
# ---------------------------------------------------------------------------


class Jobs(DatabricksService):
    """Collection-level Databricks job management.

    Listing, finding, creating, and updating jobs live here.
    Individual job lifecycle operations live on the :class:`Job` resource.
    """

    def __iter__(self) -> Iterator["Job"]:
        yield from self.list()

    def __getitem__(self, key: str | int) -> "Job":
        if isinstance(key, int):
            return self.get(key)
        return self.find(name=key)

    # ------------------------------------------------------------------ #
    # List / Find / Get
    # ------------------------------------------------------------------ #

    def list(
        self,
        *,
        name: str | None = None,
        limit: int | None = None,
        expand_tasks: bool = False,
    ) -> Iterator["Job"]:
        from .job import Job

        sdk = self.client.workspace_client().jobs
        cnt = 0
        cap = limit if limit else float("inf")

        for raw in sdk.list(name=name, expand_tasks=expand_tasks):
            job = Job(service=self, job_id=raw.job_id, name=raw.settings.name if raw.settings else None, details=raw)
            if raw.settings and raw.settings.name:
                _set_cached_name(self.client, raw.settings.name, raw.job_id)
            yield job
            cnt += 1
            if cnt >= cap:
                break

    def get(self, job_id: int) -> "Job":
        from .job import Job

        sdk = self.client.workspace_client().jobs
        raw = sdk.get(job_id=job_id)
        name = raw.settings.name if raw.settings else None
        if name:
            _set_cached_name(self.client, name, raw.job_id)
        return Job(service=self, job_id=raw.job_id, name=name, details=raw)

    def find(
        self,
        obj: "Job | int | str | None" = None,
        *,
        job_id: int | None = None,
        name: str | None = None,
        raise_error: bool = True,
    ) -> Optional["Job"]:
        from .job import Job

        if obj is not None:
            if isinstance(obj, Job):
                return obj
            if isinstance(obj, int):
                job_id = job_id or obj
            elif isinstance(obj, str):
                name = name or obj
            else:
                raise TypeError(f"obj must be Job | int | str, got {type(obj).__name__}")

        if job_id:
            try:
                return self.get(job_id)
            except ResourceDoesNotExist:
                if raise_error:
                    raise
                return None

        if name:
            cached_id = _get_cached_id(self.client, name)
            if cached_id is not None:
                try:
                    return self.get(cached_id)
                except ResourceDoesNotExist:
                    pass

            for job in self.list(name=name):
                if job.name == name:
                    return job

            if raise_error:
                raise ResourceDoesNotExist(f"Cannot find job {name!r}")
            return None

        if raise_error:
            raise ValueError("Either job_id or name must be provided")
        return None

    # ------------------------------------------------------------------ #
    # Create / Update
    # ------------------------------------------------------------------ #

    def create(
        self,
        name: str,
        *,
        tasks: list[Task | dict] | None = None,
        cluster: "Cluster | str | None" = None,
        permissions: list[str | JobAccessControlRequest] | None = None,
        tags: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
        max_concurrent_runs: int | None = None,
        **settings_kwargs: Any,
    ) -> "Job":
        from .job import Job

        sdk = self.client.workspace_client().jobs
        checked_tasks = [_check_task(t, cluster=cluster) for t in tasks] if tasks else None

        merged_tags = self.default_tags(update=False)
        if tags:
            merged_tags.update(tags)

        settings = JobSettings(
            name=name,
            tasks=checked_tasks,
            tags=merged_tags,
            timeout_seconds=timeout_seconds or 3600,
            max_concurrent_runs=max_concurrent_runs or 1,
            **settings_kwargs,
        )

        LOGGER.debug("Creating job %r", name)
        response = sdk.create(
            name=settings.name,
            tasks=settings.tasks,
            tags=settings.tags,
            timeout_seconds=settings.timeout_seconds,
            max_concurrent_runs=settings.max_concurrent_runs,
            schedule=settings.schedule,
            email_notifications=settings.email_notifications,
            notification_settings=settings.notification_settings,
            health=settings.health,
            parameters=settings.parameters,
            environments=settings.environments,
            job_clusters=settings.job_clusters,
            git_source=settings.git_source,
            queue=settings.queue,
            run_as=settings.run_as,
            budget_policy_id=settings.budget_policy_id,
            continuous=settings.continuous,
            deployment=settings.deployment,
            description=settings.description,
            edit_mode=settings.edit_mode,
            format=settings.format,
            trigger=settings.trigger,
        )

        job = Job(service=self, job_id=response.job_id, name=name)
        _set_cached_name(self.client, name, response.job_id)

        if permissions:
            job.update_permissions(permissions)

        LOGGER.info("Created job %r (id=%s)", name, response.job_id)
        return job

    def create_or_update(
        self,
        obj: "Job | int | str | None" = None,
        *,
        name: str | None = None,
        job_id: int | None = None,
        tasks: list[Task | dict] | None = None,
        cluster: "Cluster | str | None" = None,
        permissions: list[str | JobAccessControlRequest] | None = None,
        tags: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
        max_concurrent_runs: int | None = None,
        **settings_kwargs: Any,
    ) -> "Job":
        if obj is not None:
            from .job import Job
            if isinstance(obj, Job):
                job_id = obj.job_id
                name = name or obj.name
            elif isinstance(obj, int):
                job_id = obj
            elif isinstance(obj, str):
                name = name or obj

        existing = self.find(job_id=job_id, name=name, raise_error=False) if (job_id or name) else None

        if existing is not None:
            return existing.update(
                tasks=tasks,
                cluster=cluster,
                permissions=permissions,
                tags=tags,
                timeout_seconds=timeout_seconds,
                max_concurrent_runs=max_concurrent_runs,
                **settings_kwargs,
            )

        if not name:
            raise ValueError("name is required to create a new job")

        return self.create(
            name=name,
            tasks=tasks,
            cluster=cluster,
            permissions=permissions,
            tags=tags,
            timeout_seconds=timeout_seconds,
            max_concurrent_runs=max_concurrent_runs,
            **settings_kwargs,
        )


# ---------------------------------------------------------------------------
# JobRuns service
# ---------------------------------------------------------------------------


class JobRuns(DatabricksService):
    """Collection-level run management — listing and retrieving runs."""

    def list(
        self,
        *,
        job_id: int | None = None,
        active_only: bool = False,
        completed_only: bool = False,
        expand_tasks: bool = False,
        limit: int | None = None,
    ) -> Iterator["JobRun"]:
        from .run import JobRun

        sdk = self.client.workspace_client().jobs
        cnt = 0
        cap = limit if limit else float("inf")

        for raw in sdk.list_runs(
            job_id=job_id,
            active_only=active_only,
            completed_only=completed_only,
            expand_tasks=expand_tasks,
        ):
            run = JobRun(service=self, run_id=raw.run_id, job_id=raw.job_id, details=raw)
            yield run
            cnt += 1
            if cnt >= cap:
                break

    def get(self, run_id: int) -> "JobRun":
        from .run import JobRun

        sdk = self.client.workspace_client().jobs
        raw = sdk.get_run(run_id=run_id)
        return JobRun(service=self, run_id=raw.run_id, job_id=raw.job_id, details=raw)
