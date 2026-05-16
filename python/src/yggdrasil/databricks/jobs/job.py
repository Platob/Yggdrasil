"""
Databricks Job resource — individual job lifecycle management.

Wraps a single workspace Job. Caches :class:`JobInfo`/:class:`Job`
details and exposes:

- :meth:`refresh` / :meth:`update` / :meth:`reset` / :meth:`delete`
- :meth:`run` and :meth:`runs` for triggering and inspecting runs
- :meth:`update_permissions` for managing the job ACL
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Iterator, List, Optional, TYPE_CHECKING, Union

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.jobs import (
    Job as JobInfo,
    JobAccessControlRequest,
    JobSettings,
    Task,
)

from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io.url import URL

from ..client import DatabricksResource

if TYPE_CHECKING:
    from .run import JobRun
    from .service import Jobs


__all__ = ["Job"]

LOGGER = logging.getLogger(__name__)


class Job(Singleton, DatabricksResource):
    """High-level wrapper around a single Databricks Job.

    Parameters
    ----------
    service
        Parent :class:`~yggdrasil.databricks.jobs.service.Jobs` service.
    job_id
        Databricks job id.
    job_name
        Job display name. When provided without ``job_id`` the job is
        resolved by name during construction.

    Notes
    -----
    Inherits :class:`Singleton` (``_SINGLETON_TTL = None``) so two
    callers asking for the same job under the same service collapse to
    one instance — same cached :class:`JobInfo`, same permission state.
    """

    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: "Jobs | None" = None,
        job_id: int | None = None,
        job_name: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        return (cls, service, job_id, job_name)

    def __init__(
        self,
        service: "Jobs | None" = None,
        job_id: int | None = None,
        job_name: str | None = None,
        *,
        details: Optional[JobInfo] = None,
        singleton_ttl: Any = ...,
    ):
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        if service is None:
            from .service import Jobs
            service = Jobs.current()

        super().__init__(service=service)
        self.service = service
        self.job_id = job_id
        self.job_name = job_name
        self._details = details

        if self.job_name and self.job_id is None:
            found = self.service.find(name=self.job_name, raise_error=True)
            self.job_id = found.job_id
            self._details = found._details

        self._initialized = True

    # ------------------------------------------------------------------ #
    # Identity / display
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.url().to_string()!r})"

    def __str__(self) -> str:
        return self.url().to_string()

    def url(self) -> URL:
        """Return the workspace UI URL for this job."""
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}/jobs/{self.job_id or 'unknown'}"
        )

    @property
    def explore_url(self) -> URL:
        """Alias for :meth:`url` — symmetry with other Databricks resources."""
        return self.url()

    # ------------------------------------------------------------------ #
    # Details
    # ------------------------------------------------------------------ #
    @property
    def details(self) -> Optional[JobInfo]:
        """Return cached job details, fetching lazily when not yet loaded."""
        if self._details is None and self.job_id is not None:
            self._details = self.service._jobs_api().get(job_id=self.job_id)
            settings = self._details.settings if self._details is not None else None
            if settings is not None and settings.name:
                self.job_name = settings.name
        return self._details

    def refresh(self) -> "Job":
        """Force-refresh cached job details and return self."""
        if self.job_id is None:
            return self
        self._details = self.service._jobs_api().get(job_id=self.job_id)
        settings = self._details.settings if self._details is not None else None
        if settings is not None and settings.name:
            self.job_name = settings.name
        return self

    @property
    def settings(self) -> Optional[JobSettings]:
        """Latest :class:`JobSettings` for this job (from cached details)."""
        details = self.details
        return details.settings if details is not None else None

    @property
    def name(self) -> Optional[str]:
        settings = self.settings
        return settings.name if settings is not None else self.job_name

    @property
    def creator_user_name(self) -> Optional[str]:
        details = self.details
        return details.creator_user_name if details is not None else None

    # ------------------------------------------------------------------ #
    # Update / delete
    # ------------------------------------------------------------------ #
    def reset(self, new_settings: JobSettings) -> "Job":
        """Replace the job settings outright via :meth:`JobsAPI.reset`.

        Use this when you want to drop fields the caller did not specify
        (true overwrite). For partial updates that preserve unspecified
        fields, use :meth:`update`.
        """
        if self.job_id is None:
            raise ValueError(f"Cannot reset {self}: job_id is not set")

        LOGGER.debug("Resetting %r settings", self)
        self.service._jobs_api().reset(
            job_id=self.job_id,
            new_settings=new_settings,
        )
        self.refresh()
        LOGGER.info("Reset %r", self)
        return self

    def update(
        self,
        *,
        name: str | None = None,
        tasks: Optional[List[Task]] = None,
        permissions: Optional[List[Union[str, JobAccessControlRequest]]] = None,
        tags: Optional[dict[str, str]] = None,
        fields_to_remove: Optional[List[str]] = None,
        **settings: Any,
    ) -> "Job":
        """Apply partial settings updates via :meth:`JobsAPI.update`.

        Only fields explicitly passed are sent to the API; existing
        fields are left untouched on the Databricks side. Pass
        ``fields_to_remove`` to drop specific top-level settings keys.
        """
        if self.job_id is None:
            raise ValueError(f"Cannot update {self}: job_id is not set")

        new_settings_fields: dict[str, Any] = {}
        if name is not None:
            new_settings_fields["name"] = name
        if tasks is not None:
            new_settings_fields["tasks"] = list(tasks)
        if tags is not None:
            new_settings_fields["tags"] = dict(tags)
        for key, value in settings.items():
            if value is not None:
                new_settings_fields[key] = value

        if new_settings_fields or fields_to_remove:
            new_settings = (
                JobSettings(**new_settings_fields) if new_settings_fields else None
            )

            LOGGER.debug(
                "Updating %r with settings=%r fields_to_remove=%r",
                self, new_settings_fields, fields_to_remove,
            )
            self.service._jobs_api().update(
                job_id=self.job_id,
                new_settings=new_settings,
                fields_to_remove=fields_to_remove,
            )
            self.refresh()

        if permissions:
            self.update_permissions(permissions=permissions)

        LOGGER.info("Updated %r", self)
        return self

    def delete(self) -> None:
        """Delete the job if it exists. Also drops the named-job cache entry."""
        if self.job_id is None:
            return

        from .service import _NAME_ID_CACHE

        LOGGER.debug("Deleting %r", self)
        try:
            self.service._jobs_api().delete(job_id=self.job_id)
        except ResourceDoesNotExist:
            LOGGER.debug("Job %r already deleted", self)

        host = self.client.base_url.to_string()
        host_cache = _NAME_ID_CACHE.get(host)
        if host_cache is not None and self.job_name:
            host_cache.pop(self.job_name, None)

        LOGGER.info("Deleted %r", self)

    # ------------------------------------------------------------------ #
    # Permissions
    # ------------------------------------------------------------------ #
    def update_permissions(
        self,
        permissions: Optional[List[Union[str, JobAccessControlRequest]]] = None,
    ) -> "Job":
        """Apply ACL entries to this job."""
        if not permissions or self.job_id is None:
            return self

        normalized = [self.service.check_permission(p) for p in permissions]
        self.service._jobs_api().update_permissions(
            job_id=str(self.job_id),
            access_control_list=normalized,
        )
        return self

    # ------------------------------------------------------------------ #
    # Runs
    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        job_parameters: Optional[dict[str, str]] = None,
        notebook_params: Optional[dict[str, str]] = None,
        python_params: Optional[List[str]] = None,
        python_named_params: Optional[dict[str, str]] = None,
        jar_params: Optional[List[str]] = None,
        spark_submit_params: Optional[List[str]] = None,
        sql_params: Optional[dict[str, str]] = None,
        dbt_commands: Optional[List[str]] = None,
        idempotency_token: str | None = None,
        only: Optional[List[str]] = None,
        wait: WaitingConfigArg = False,
        **run_kwargs: Any,
    ) -> "JobRun":
        """Trigger a run of this job via :meth:`JobsAPI.run_now`.

        Returns a :class:`JobRun` bound to the resulting run id. When
        ``wait`` is truthy, blocks until the run terminates before
        returning.
        """
        from .run import JobRun

        if self.job_id is None:
            raise ValueError(f"Cannot run {self}: job_id is not set")

        kwargs = {
            k: v for k, v in {
                "job_parameters": job_parameters,
                "notebook_params": notebook_params,
                "python_params": python_params,
                "python_named_params": python_named_params,
                "jar_params": jar_params,
                "spark_submit_params": spark_submit_params,
                "sql_params": sql_params,
                "dbt_commands": dbt_commands,
                "idempotency_token": idempotency_token,
                "only": only,
                **run_kwargs,
            }.items()
            if v is not None
        }

        LOGGER.debug("Triggering run of %r with %s", self, kwargs)
        waiter = self.service._jobs_api().run_now(job_id=self.job_id, **kwargs)
        run = JobRun(service=self.service, run_id=waiter.run_id)
        LOGGER.info("Triggered %r", run)

        if wait:
            run.wait_for_status(wait=wait)
        return run

    def runs(
        self,
        *,
        active_only: bool | None = None,
        completed_only: bool | None = None,
        limit: int | None = None,
        expand_tasks: bool = False,
    ) -> Iterator["JobRun"]:
        """Iterate over this job's runs."""
        return self.service.list_runs(
            job_id=self.job_id,
            active_only=active_only,
            completed_only=completed_only,
            limit=limit,
            expand_tasks=expand_tasks,
        )

    def cancel_all_runs(self, *, all_queued_runs: bool = False) -> "Job":
        """Cancel every in-flight run of this job."""
        if self.job_id is None:
            return self
        self.service._jobs_api().cancel_all_runs(
            job_id=self.job_id,
            all_queued_runs=all_queued_runs or None,
        )
        return self
