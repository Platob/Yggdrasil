"""
Databricks Jobs service — collection-level management.

Exposes :class:`Jobs`, a thin layer around the Databricks SDK
``JobsAPI`` that handles:

- listing / finding jobs by id or name
- creating, updating and deleting jobs with sensible defaults
- short-lived in-process caching to avoid redundant API calls

Individual job lifecycle (run, refresh, update, …) lives on the
:class:`~yggdrasil.databricks.jobs.job.Job` resource returned by this
service.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Iterator, List, Optional, TYPE_CHECKING, Union

from databricks.sdk import JobsAPI
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.jobs import (
    JobAccessControlRequest,
    JobPermissionLevel,
    SubmitTask,
    Task,
)

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.waiting import WaitingConfigArg

from ..client import DatabricksClient, DatabricksService

if TYPE_CHECKING:
    from .job import Job
    from .run import JobRun


__all__ = ["Jobs"]

LOGGER = logging.getLogger(__name__)

_CREATE_ARG_NAMES = set(inspect.signature(JobsAPI.create).parameters.keys())
_SUBMIT_ARG_NAMES = set(inspect.signature(JobsAPI.submit).parameters.keys())

# host -> ExpiringDict(job_name -> job_id)
_NAME_ID_CACHE: dict[str, ExpiringDict[str, int]] = {}
_NAMED_JOBS: ExpiringDict[tuple[str, str], "Job"] = ExpiringDict(default_ttl=7200.0)


def _set_cached_job_id(client: DatabricksClient, name: str, job_id: int) -> None:
    host = client.base_url.to_string()
    existing = _NAME_ID_CACHE.get(host)
    if not existing:
        existing = _NAME_ID_CACHE[host] = ExpiringDict(default_ttl=60.0)
    existing[name] = job_id


def _get_cached_job_id(client: DatabricksClient, name: str) -> Optional[int]:
    host = client.base_url.to_string()
    existing = _NAME_ID_CACHE.get(host)
    return existing.get(name) if existing else None


class Jobs(DatabricksService):
    """Collection-level Databricks Jobs management service.

    Mirrors the shape of :class:`~yggdrasil.databricks.cluster.Clusters`
    and :class:`~yggdrasil.databricks.compute.instance_pool.InstancePools` so
    callers can switch between resources with the same vocabulary.
    """

    def __iter__(self) -> Iterator["Job"]:
        yield from self.list()

    # ------------------------------------------------------------------ #
    # SDK boundary
    # ------------------------------------------------------------------ #
    def _jobs_api(self) -> JobsAPI:
        return self.client.workspace_client().jobs

    # ------------------------------------------------------------------ #
    # Listing / finding
    # ------------------------------------------------------------------ #
    def list(
        self,
        *,
        name: str | None = None,
        limit: int | None = None,
        expand_tasks: bool = False,
    ) -> Iterator["Job"]:
        """Iterate over workspace jobs, optionally filtered by name."""
        from .job import Job

        cnt, limit = 0, limit or float("inf")
        # ``JobsAPI.list`` supports ``name`` server-side; pass it through.
        for entry in self._jobs_api().list(name=name, expand_tasks=expand_tasks):
            settings = entry.settings
            entry_name = settings.name if settings is not None else None

            if entry.job_id and entry_name:
                _set_cached_job_id(self.client, entry_name, entry.job_id)

            yield Job(
                service=self,
                job_id=entry.job_id,
                job_name=entry_name,
                details=entry,
            )

            cnt += 1
            if cnt >= limit:
                break

    def find(
        self,
        job_id: int | None = None,
        *,
        name: str | None = None,
        raise_error: bool | None = None,
    ) -> Optional["Job"]:
        """Look up a job by id or name. Returns ``None`` if absent.

        Identity precedence: ``job_id`` wins. When only ``name`` is
        provided, the per-host name cache is consulted first; on miss
        we fall back to the server-side ``list(name=…)`` filter.
        """
        if job_id is None and not name:
            raise ValueError("Either job_id or name must be provided")

        from .job import Job

        if job_id is None and name:
            job_id = _get_cached_job_id(self.client, name)

        if job_id is not None:
            try:
                details = self._jobs_api().get(job_id=job_id)
            except ResourceDoesNotExist:
                if raise_error:
                    raise ValueError(f"Cannot find databricks job {job_id!r}")
                return None

            settings = details.settings
            resolved_name = settings.name if settings is not None else None
            if resolved_name:
                _set_cached_job_id(self.client, resolved_name, details.job_id)

            return Job(
                service=self,
                job_id=details.job_id,
                job_name=resolved_name,
                details=details,
            )

        # last resort: server-side name filter (cheapest list scan path)
        for job in self.list(name=name, limit=1):
            return job

        if raise_error:
            raise ValueError(f"Cannot find databricks job {name!r}")
        return None

    def get(
        self,
        job_id: int | None = None,
        *,
        name: str | None = None,
    ) -> "Job":
        """Like :meth:`find` but raises if the job does not exist."""
        return self.find(job_id=job_id, name=name, raise_error=True)

    # ------------------------------------------------------------------ #
    # Create / update
    # ------------------------------------------------------------------ #
    def _normalize_create_kwargs(
        self,
        *,
        name: str | None,
        tasks: Optional[List[Task]],
        tags: Optional[dict[str, str]],
        **settings: Any,
    ) -> dict[str, Any]:
        """Apply default tags and drop unset keys.

        Defaults injected from :meth:`DatabricksService.default_tags`
        merge under any caller-supplied tags.
        """
        spec: dict[str, Any] = {}
        if name is not None:
            spec["name"] = name
        if tasks is not None:
            spec["tasks"] = list(tasks)

        for key, value in settings.items():
            if value is not None:
                spec[key] = value

        default_tags = self.default_tags(update=False)
        if tags:
            merged: dict[str, str] = dict(default_tags) if default_tags else {}
            merged.update({str(k): str(v) for k, v in tags.items()})
            spec["tags"] = merged
        elif default_tags:
            spec["tags"] = dict(default_tags)

        return spec

    def create(
        self,
        *,
        name: str,
        tasks: Optional[List[Task]] = None,
        permissions: Optional[list[Union[str, JobAccessControlRequest]]] = None,
        tags: Optional[dict[str, str]] = None,
        **settings: Any,
    ) -> "Job":
        """Create a new Databricks job and return its :class:`Job` wrapper."""
        from .job import Job

        spec = self._normalize_create_kwargs(
            name=name, tasks=tasks, tags=tags, **settings,
        )
        create_kwargs = {k: v for k, v in spec.items() if k in _CREATE_ARG_NAMES}

        if permissions:
            create_kwargs["access_control_list"] = [
                self.check_permission(p) for p in permissions
            ]

        LOGGER.debug("Creating job %r with spec %s", name, create_kwargs)
        response = self._jobs_api().create(**create_kwargs)

        job_id = response.job_id
        _set_cached_job_id(self.client, name, job_id)

        instance = Job(
            service=self,
            job_id=job_id,
            job_name=name,
        ).refresh()

        LOGGER.info("Created job %r", instance)
        return instance

    def create_or_update(
        self,
        *,
        job_id: int | None = None,
        name: str | None = None,
        tasks: Optional[List[Task]] = None,
        permissions: Optional[list[Union[str, JobAccessControlRequest]]] = None,
        tags: Optional[dict[str, str]] = None,
        **settings: Any,
    ) -> "Job":
        """Update an existing job by id/name, or create one if missing."""
        found = self.find(job_id=job_id, name=name)

        if found is not None:
            return found.update(
                name=name,
                tasks=tasks,
                permissions=permissions,
                tags=tags,
                **settings,
            )

        if not name:
            raise ValueError(
                "Cannot create a new job without name; pass name=... or an "
                f"existing job_id (received job_id={job_id!r})."
            )

        return self.create(
            name=name,
            tasks=tasks,
            permissions=permissions,
            tags=tags,
            **settings,
        )

    def delete(
        self,
        job_id: int | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Delete a job by id or name (no-op if it does not exist)."""
        found = self.find(job_id=job_id, name=name)
        if found is not None:
            found.delete()

    # ------------------------------------------------------------------ #
    # Submit one-off run (no persisted Job)
    # ------------------------------------------------------------------ #
    def submit(
        self,
        *,
        run_name: str | None = None,
        tasks: Optional[List[SubmitTask]] = None,
        wait: WaitingConfigArg = False,
        **submit_kwargs: Any,
    ) -> "JobRun":
        """Submit a one-off run without persisting a job definition.

        Wraps :meth:`JobsAPI.submit`. Returns a :class:`JobRun` bound to
        the resulting run id. When ``wait`` is truthy, blocks until the
        run terminates before returning.
        """
        from .run import JobRun

        kwargs = {
            k: v for k, v in submit_kwargs.items() if k in _SUBMIT_ARG_NAMES
        }
        if run_name is not None:
            kwargs["run_name"] = run_name
        if tasks is not None:
            kwargs["tasks"] = list(tasks)

        LOGGER.debug("Submitting one-off job run %r", run_name or "<unnamed>")
        waiter = self._jobs_api().submit(**kwargs)
        run_id = waiter.run_id

        instance = JobRun(service=self, run_id=run_id)
        LOGGER.info("Submitted job run %r", instance)

        if wait:
            instance.wait_for_status(wait=wait)
        return instance

    # ------------------------------------------------------------------ #
    # Run listing (across the workspace, not scoped to a single job)
    # ------------------------------------------------------------------ #
    def list_runs(
        self,
        *,
        job_id: int | None = None,
        active_only: bool | None = None,
        completed_only: bool | None = None,
        limit: int | None = None,
        expand_tasks: bool = False,
    ) -> Iterator["JobRun"]:
        """Iterate over runs, optionally scoped to a single job."""
        from .run import JobRun

        cnt, limit = 0, limit or float("inf")
        for base in self._jobs_api().list_runs(
            job_id=job_id,
            active_only=active_only,
            completed_only=completed_only,
            expand_tasks=expand_tasks,
        ):
            yield JobRun(
                service=self,
                run_id=base.run_id,
                details=base,
            )
            cnt += 1
            if cnt >= limit:
                break

    # ------------------------------------------------------------------ #
    # Permissions
    # ------------------------------------------------------------------ #
    @staticmethod
    def check_permission(
        permission: Union[str, JobAccessControlRequest],
    ) -> JobAccessControlRequest:
        """Normalize a permission spec into a :class:`JobAccessControlRequest`.

        Strings shaped as emails become user permissions; other strings
        become group permissions. Already-built request objects pass
        through unchanged.
        """
        if isinstance(permission, JobAccessControlRequest):
            return permission

        if isinstance(permission, str):
            if "@" in permission:
                group_name, user_name = None, permission
            else:
                group_name, user_name = permission, None
            return JobAccessControlRequest(
                group_name=group_name,
                user_name=user_name,
                permission_level=JobPermissionLevel.CAN_MANAGE,
            )

        raise ValueError(
            f"Invalid job permission spec {permission!r}; expected str or "
            f"JobAccessControlRequest, got {type(permission).__name__}."
        )
