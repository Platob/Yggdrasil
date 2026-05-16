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
from typing import (
    Any, Callable, ClassVar, Dict, Iterator, List, Mapping, Optional,
    TYPE_CHECKING, Union,
)

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.jobs import (
    CronSchedule,
    Job as JobInfo,
    JobAccessControlRequest,
    JobParameterDefinition,
    JobSettings,
    PauseStatus,
    Task,
    TriggerSettings,
)

from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io.url import URL

from ..client import DatabricksClient, DatabricksResource

if TYPE_CHECKING:
    from .run import JobRun
    from .service import Jobs
    from .task import JobTask


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
    def __str__(self) -> str:
        return self.explore_url.to_string()

    @property
    def explore_url(self) -> URL:
        """Workspace UI URL pointing at this job's run history."""
        return self.client.base_url.with_path(f"/jobs/{self.job_id or 'unknown'}")

    def url(self) -> URL:
        """Deprecated alias for :attr:`explore_url` (method form)."""
        return self.explore_url

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

        LOGGER.debug("Resetting settings on job %r", self)
        self.service._jobs_api().reset(
            job_id=self.job_id,
            new_settings=new_settings,
        )
        self.refresh()
        LOGGER.info("Reset job %r", self)
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
                "Updating job %r (settings=%r, fields_to_remove=%r)",
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

        LOGGER.info("Updated job %r", self)
        return self

    def delete(self) -> None:
        """Delete the job if it exists. Also drops the named-job cache entry."""
        if self.job_id is None:
            return

        from .service import _NAME_ID_CACHE

        LOGGER.debug("Deleting job %r", self)
        try:
            self.service._jobs_api().delete(job_id=self.job_id)
        except ResourceDoesNotExist:
            LOGGER.debug("Job %r already deleted — skipping delete", self)

        host = self.client.base_url.to_string()
        host_cache = _NAME_ID_CACHE.get(host)
        if host_cache is not None and self.job_name:
            host_cache.pop(self.job_name, None)

        LOGGER.info("Deleted job %r", self)

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

        LOGGER.debug("Triggering job %r with %s", self, kwargs)
        waiter = self.service._jobs_api().run_now(job_id=self.job_id, **kwargs)
        run = JobRun(service=self.service, run_id=waiter.run_id)
        LOGGER.info("Triggered job run %r", run)

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

    # ------------------------------------------------------------------ #
    # Task factory — Prefect-style registration of Python callables
    # ------------------------------------------------------------------ #
    def task(
        self,
        task_key: str,
        /,
        *,
        order: Optional[int] = None,
        **task_fields: Any,
    ) -> "JobTask":
        """Construct a :class:`JobTask` handle bound to this job.

        The returned handle is not yet persisted on the job — call
        :meth:`JobTask.create` (idempotent: re-creates with the same
        ``task_key`` replace in place), or use :meth:`JobTask.decorate`
        to stage a Python callable's source onto it and push it
        through in one step::

            job = client.jobs.get_or_create(job_id=123, name="my-job")

            @job.task("do").decorate
            def do(a: str, i: int):
                print(a, i)

            @job.task("do2", order=0, existing_cluster_id="c-123").decorate
            def do2(x: int): ...

        Extra *task_fields* are forwarded to
        :class:`databricks.sdk.service.jobs.Task` so you can attach
        compute (``new_cluster=`` / ``existing_cluster_id=`` /
        ``job_cluster_key=``), dependencies, retries, description, etc.
        at construction time. *order* (when set) pins the task's
        position in the job's task list on the next
        :meth:`~JobTask.create` — slice semantics, so ``0``
        lands first and ``-1`` lands second-to-last.
        """
        from .task import JobTask

        details = Task(task_key=task_key, **task_fields)
        return JobTask(job=self, task_key=task_key, details=details, order=order)

    def pytask(
        self,
        func: Optional[Callable[..., Any]] = None,
        /,
        *,
        task_key: Optional[str] = None,
        order: Optional[int] = None,
        **task_fields: Any,
    ) -> Any:
        """Fastpath: stage a Python callable as a task in one decorator.

        Composes :meth:`Job.task` + :meth:`JobTask.decorate` into a
        single Prefect-style decorator. Equivalent to chaining
        ``@job.task(key, order=…, **fields).decorate``, but shorter
        for the common case where you just want the function staged.

        Usable bare or parametrized::

            @job.pytask
            def step(): ...

            @job.pytask(task_key="custom", order=0, environment_key="env-1")
            def step(): ...

        Bare form defaults ``task_key`` to ``func.__name__``. Any extra
        *task_fields* flow into :meth:`Job.task` so the same
        caller-wins / decorate-back-fills semantics apply: explicit
        fields beat the docstring-derived defaults. *order* (when set)
        pins the resulting task's position in the job's task list.
        """
        def _decorate(f: Callable[..., Any]) -> Callable[..., Any]:
            key = task_key or f.__name__
            return self.task(key, order=order, **task_fields).decorate(f)

        if func is None:
            return _decorate
        return _decorate(func)

    # ====================================================================== #
    # Skeleton — class-level template for subclasses
    # ====================================================================== #
    #
    # ``Job`` itself doubles as the base skeleton: subclasses declare
    # how each piece of a :class:`JobSettings` is derived by overriding
    # the ``default_*`` classmethods below, and call
    # :meth:`deploy` / :meth:`find_for` / :meth:`get_for` /
    # :meth:`delete_for` / :meth:`create_for` to drive the lifecycle
    # against the workspace. Every hook receives the caller's
    # ``**context`` kwargs (e.g. ``table=`` on :class:`AsyncInsertJob`)
    # so the subclass can look up structural inputs without re-deriving
    # them at every call site.
    #
    # The hooks return ``None`` / ``[]`` by default — a bare ``Job``
    # carries no skeleton; subclasses opt in by overriding only what
    # they need. Caller-supplied kwargs on :meth:`deploy` always win
    # over the hook defaults, so a one-off deviation never requires a
    # subclass override.
    # ------------------------------------------------------------------ #

    # Each ``default_*`` hook is also the *resolver* — it owns the
    # transform from caller kwargs to the resolved :class:`JobSettings`
    # field. The default implementations are passthroughs (``name=``
    # → name, ``tasks=`` → tasks, …) plus a few light coercions (cron
    # string → :class:`CronSchedule`); subclasses override to derive
    # the same fields from their own context (a :class:`Table`, a
    # :class:`Volume`, …) while still honoring the caller's overrides.
    # Hooks receive every :meth:`deploy` keyword via ``**kwargs`` and
    # return ``None`` / ``[]`` when there's nothing to set.

    @classmethod
    def default_name(cls, *, name: Optional[str] = None, **_: Any) -> Optional[str]:
        """Skeleton: resolve the job name. Override on subclasses."""
        return name

    @classmethod
    def default_tasks(
        cls,
        *,
        tasks: Optional[List[Task]] = None,
        **_: Any,
    ) -> List[Task]:
        """Skeleton: resolve the task list (empty unless overridden)."""
        return list(tasks) if tasks else []

    @classmethod
    def default_schedule(
        cls,
        *,
        schedule: Any = None,
        schedule_timezone: str = "UTC",
        schedule_pause_status: Any = None,
        **_: Any,
    ) -> Optional[CronSchedule]:
        """Skeleton: resolve the cron schedule.

        Accepts a pre-built :class:`CronSchedule`, a Quartz cron
        string (coerced with the matching ``schedule_timezone`` /
        ``schedule_pause_status`` kwargs), or ``None``.
        """
        if schedule is None:
            return None
        if isinstance(schedule, CronSchedule):
            return schedule
        if isinstance(schedule, str):
            resolved_pause: Any = schedule_pause_status
            if isinstance(resolved_pause, str):
                resolved_pause = PauseStatus(resolved_pause.upper())
            return CronSchedule(
                quartz_cron_expression=schedule,
                timezone_id=schedule_timezone,
                pause_status=resolved_pause,
            )
        raise TypeError(
            f"{cls.__name__}: ``schedule`` must be a CronSchedule, a "
            f"Quartz cron string, or None — got {type(schedule).__name__}."
        )

    @classmethod
    def default_trigger(
        cls,
        *,
        trigger: Optional[TriggerSettings] = None,
        **_: Any,
    ) -> Optional[TriggerSettings]:
        """Skeleton: resolve :class:`TriggerSettings` (file-arrival / …)."""
        return trigger

    @classmethod
    def default_parameters(
        cls,
        *,
        parameters: Optional[List[JobParameterDefinition]] = None,
        **_: Any,
    ) -> List[JobParameterDefinition]:
        """Skeleton: resolve job-level parameter definitions."""
        return list(parameters) if parameters else []

    @classmethod
    def default_description(
        cls,
        *,
        description: Optional[str] = None,
        **_: Any,
    ) -> Optional[str]:
        """Skeleton: resolve the job description."""
        return description

    @classmethod
    def default_tags(
        cls,
        *,
        tags: Optional[Mapping[str, str]] = None,
        **_: Any,
    ) -> Optional[Dict[str, str]]:
        """Skeleton: resolve the tag map."""
        return dict(tags) if tags else None

    @classmethod
    def default_settings(cls, **_: Any) -> Dict[str, Any]:
        """Skeleton: extra :class:`JobSettings` kwargs (``max_concurrent_runs``, …)."""
        return {}

    @classmethod
    def resolve_jobs(
        cls,
        *,
        service: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        **_context: Any,
    ) -> "Jobs":
        """Resolve the :class:`Jobs` service to operate against.

        Override to pull the service from a context-bound resource
        (e.g. ``table.client.jobs`` on :class:`AsyncInsertJob`); the
        default falls back to the explicit ``service`` / ``client``
        argument, then :meth:`DatabricksClient.current`.
        """
        if service is not None:
            return service
        if client is None:
            client = DatabricksClient.current()
        return client.jobs

    @classmethod
    def _build_skeleton_kwargs(cls, **context: Any) -> Dict[str, Any]:
        """Resolve every :class:`JobSettings` field via the ``default_*`` hooks.

        ``context`` carries the caller's full :meth:`deploy` kwarg set;
        each hook reads what it needs and returns the resolved field.
        Subclasses override hooks to add context-derived defaults
        without changing this assembly path.
        """
        resolved_name = cls.default_name(**context)
        if not resolved_name:
            raise ValueError(
                f"{cls.__name__}: cannot resolve job name; pass ``name=`` "
                f"or override ``default_name(cls, **context)``."
            )

        resolved_parameters = cls.default_parameters(**context)
        resolved_trigger = cls.default_trigger(**context)
        kwargs: Dict[str, Any] = {
            "name": resolved_name,
            "tasks": cls.default_tasks(**context),
            "schedule": cls.default_schedule(**context),
            "parameters": resolved_parameters if resolved_parameters else None,
            "description": cls.default_description(**context),
            "tags": cls.default_tags(**context),
            **cls.default_settings(**context),
        }
        # Trigger lands conditionally: ``Jobs.create_or_update`` filters
        # explicit ``None`` values at its SDK boundary, but the keyword
        # being absent is what most call sites assert on (no trigger
        # requested → no trigger field in the API payload).
        if resolved_trigger is not None:
            kwargs["trigger"] = resolved_trigger
        return kwargs

    @classmethod
    def _wrap(
        cls,
        underlying: "Job",
        *,
        service: "Jobs | None" = None,
        **context: Any,
    ) -> "Job":
        """Build a *cls* instance from an existing :class:`Job`.

        Subclasses override to thread context-bound arguments through
        their own ``__init__`` (e.g. ``AsyncInsertJob(table=...)``).
        The default constructs by id / name / details so plain
        ``Job.deploy(...)`` returns a singleton-cached :class:`Job`.
        """
        return cls(
            service=service,
            job_id=underlying.job_id,
            job_name=underlying.job_name,
            details=getattr(underlying, "_details", None),
        )

    # ------------------------------------------------------------------ #
    # Skeleton CRUD
    # ------------------------------------------------------------------ #
    @classmethod
    def deploy(
        cls,
        *,
        service: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        permissions: Optional[List[Union[str, JobAccessControlRequest]]] = None,
        **kwargs: Any,
    ) -> "Job":
        """Idempotent create-or-update using the class skeleton.

        Every :class:`JobSettings` field is resolved by the matching
        ``default_*`` classmethod, which receives the full *kwargs*
        bundle so subclasses can derive fields from structural inputs
        (a :class:`Table`, a :class:`Volume`, …) while still honoring
        caller overrides (``name=`` / ``tasks=`` / ``schedule=`` /
        ``trigger=`` / ``parameters=`` / ``description=`` / ``tags=``
        and any extra :class:`JobSettings` knobs through
        :meth:`default_settings`).
        """
        jobs = cls.resolve_jobs(service=service, client=client, **kwargs)
        api_kwargs = cls._build_skeleton_kwargs(**kwargs)
        underlying = jobs.create_or_update(permissions=permissions, **api_kwargs)
        LOGGER.info(
            "Deployed %s %r (job_id=%s)",
            cls.__name__, api_kwargs["name"], underlying.job_id,
        )
        return cls._wrap(underlying, service=jobs, **kwargs)

    @classmethod
    def create_for(
        cls,
        *,
        service: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        permissions: Optional[List[Union[str, JobAccessControlRequest]]] = None,
        **kwargs: Any,
    ) -> "Job":
        """Explicit create (errors when the named job already exists)."""
        jobs = cls.resolve_jobs(service=service, client=client, **kwargs)
        api_kwargs = cls._build_skeleton_kwargs(**kwargs)
        underlying = jobs.create(permissions=permissions, **api_kwargs)
        return cls._wrap(underlying, service=jobs, **kwargs)

    @classmethod
    def find_for(
        cls,
        *,
        service: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        **kwargs: Any,
    ) -> "Job | None":
        """Find the job for *kwargs* (via :meth:`default_name`), or ``None``."""
        jobs = cls.resolve_jobs(service=service, client=client, **kwargs)
        resolved_name = cls.default_name(**kwargs)
        if not resolved_name:
            raise ValueError(
                f"{cls.__name__}.find_for: cannot resolve job name; pass "
                f"``name=`` or override ``default_name(cls, **context)``."
            )
        found = jobs.find(name=resolved_name)
        if found is None:
            return None
        return cls._wrap(found, service=jobs, **kwargs)

    @classmethod
    def get_for(cls, **kwargs: Any) -> "Job":
        """Like :meth:`find_for` but raises when the job is absent."""
        found = cls.find_for(**kwargs)
        if found is None:
            raise ValueError(
                f"No {cls.__name__} found for context {kwargs!r}."
            )
        return found

    @classmethod
    def get_or_create(cls, **kwargs: Any) -> "Job":
        """Return the existing skeleton-named job, otherwise :meth:`create_for`."""
        found = cls.find_for(**kwargs)
        if found is not None:
            return found
        return cls.create_for(**kwargs)

    @classmethod
    def delete_for(
        cls,
        *,
        service: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        **kwargs: Any,
    ) -> None:
        """Delete the job for *kwargs* (no-op when it doesn't exist)."""
        jobs = cls.resolve_jobs(service=service, client=client, **kwargs)
        resolved_name = cls.default_name(**kwargs)
        if not resolved_name:
            raise ValueError(
                f"{cls.__name__}.delete_for: cannot resolve job name; pass "
                f"``name=`` or override ``default_name(cls, **context)``."
            )
        jobs.delete(name=resolved_name)
