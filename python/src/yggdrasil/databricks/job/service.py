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
    SubmitTask,
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


def _is_numeric(value: str) -> bool:
    return value.isdigit()


def _resolve_job_obj(
    obj: Any,
    *,
    job_id: int | None,
    name: str | None,
) -> tuple[int | None, str | None]:
    """Resolve a positional ``obj`` into ``(job_id, name)``.

    Accepts ``Job``, ``int`` (always id), or ``str`` (numeric → id,
    otherwise → name).
    """
    if obj is None:
        return job_id, name

    from .job import Job

    if isinstance(obj, Job):
        return obj.job_id or job_id, obj.name or name
    if isinstance(obj, int):
        return obj, name
    if isinstance(obj, str):
        if _is_numeric(obj):
            return int(obj), name
        return job_id, obj
    raise TypeError(f"obj must be Job | int | str, got {type(obj).__name__}")


def _resolve_run_obj(
    obj: Any,
    *,
    run_id: int | None,
    job_id: int | None,
) -> tuple[int | None, int | None]:
    """Resolve a positional ``obj`` into ``(run_id, job_id)``."""
    if obj is None:
        return run_id, job_id

    from .run import JobRun

    if isinstance(obj, JobRun):
        return obj.run_id or run_id, obj.job_id or job_id
    if isinstance(obj, int):
        return obj, job_id
    if isinstance(obj, str):
        if _is_numeric(obj):
            return int(obj), job_id
        raise ValueError(f"Run lookup by name is not supported; got {obj!r}")
    raise TypeError(f"obj must be JobRun | int | str, got {type(obj).__name__}")


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


def _resolve_submit_environment(client: DatabricksClient, environment: Any) -> Any:
    """Resolve a one-time-run *environment* argument into a ``JobEnvironment``.

    Discovery is owned by :class:`~yggdrasil.databricks.environments.service.Environments`
    (``dbc.environments``); this only maps its result onto the run's
    ``JobEnvironment``. Accepts:

    - a :class:`JobEnvironment` — returned as-is;
    - a ``str`` — a seeded serverless **base-environment**: a version-tagged stem
      name (``ygg-0.8.58-py311``), a **project name** (``ygg`` / ``yggdrasil`` /
      ``meteologica`` → that project's deployed env for the current Python), or a
      direct workspace path to its ``.yml`` spec; reused via
      ``Environment.base_environment`` so the run shares the cached image;
    - ``None`` — **auto**: the project :meth:`~yggdrasil.databricks.environments.service.Environments.default`
      (local pyproject / client project, else the seeded **ygg** base environment
      for the current Python), else ``None`` (the run falls back to the
      workspace's default serverless compute).

    A named environment that can't be found raises :class:`FileNotFoundError`
    — the miss is loud because the caller asked for a specific image.
    """
    from databricks.sdk.service.jobs import JobEnvironment

    if isinstance(environment, JobEnvironment):
        return environment

    envs = client.environments

    if isinstance(environment, str):
        env = envs.resolve(environment)
        if env is None or not env.serverless:
            from yggdrasil.databricks.environments import service as W
            raise FileNotFoundError(
                f"no serverless base environment {environment!r} found under "
                f"{W.WORKSPACE_ENV_DIR} (deploy one with `ygg databricks deploy`, "
                f"pass a workspace path to its .yml, or a JobEnvironment)."
            )
        return env.job_environment()

    # ``None`` → auto: client project, then seeded ygg, then workspace default.
    env = envs.resolve()
    if env is None or not env.serverless:
        LOGGER.debug(
            "no client-project or seeded ygg base environment found — "
            "submitting on the workspace default serverless compute",
        )
        return None
    LOGGER.info("defaulting to base environment %s", env.serverless)
    return env.job_environment()


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

    Listing, finding, creating, and updating jobs live here, as does
    :meth:`submit` for one-time runs that aren't backed by a persisted job.
    Individual job lifecycle operations live on the :class:`Job` resource.

    Getter methods accept a positional ``obj`` that can be a :class:`Job`,
    an ``int`` (job id), or a ``str``.  Strings that are purely numeric
    are treated as ids; everything else is treated as a job name::

        jobs.get(12345)            # by id
        jobs.get("12345")          # numeric string → by id
        jobs.get("my-etl-job")     # by name
        jobs["my-etl-job"]         # __getitem__ delegates to get
    """

    def __iter__(self) -> Iterator["Job"]:
        yield from self.list()

    def __getitem__(self, key: str | int) -> "Job":
        return self.get(key)

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
            job = Job(
                service=self,
                job_id=raw.job_id,
                name=raw.settings.name if raw.settings else None,
                details=raw,
            )
            if raw.settings and raw.settings.name:
                _set_cached_name(self.client, raw.settings.name, raw.job_id)
            yield job
            cnt += 1
            if cnt >= cap:
                break

    def get(
        self,
        obj: "Job | int | str | None" = None,
        *,
        job_id: int | None = None,
        name: str | None = None,
        default: Any = ...,
    ) -> Optional["Job"]:
        """Resolve a job by id or name.

        Parameters
        ----------
        obj:
            Positional shortcut — ``Job`` (returned as-is), ``int``
            (job id), or ``str`` (numeric → id, otherwise → name).
        job_id:
            Explicit job id.
        name:
            Explicit job name.
        default:
            Returned when the job is not found.  ``...`` (the default)
            raises :class:`ResourceDoesNotExist`.
        """
        from .job import Job

        job_id, name = _resolve_job_obj(obj, job_id=job_id, name=name)

        if job_id is not None:
            try:
                return self._get_by_id(job_id)
            except ResourceDoesNotExist:
                if default is not ...:
                    return default
                raise

        if name is not None:
            cached_id = _get_cached_id(self.client, name)
            if cached_id is not None:
                try:
                    return self._get_by_id(cached_id)
                except ResourceDoesNotExist:
                    pass

            for job in self.list(name=name):
                if job.name == name:
                    return job

            if default is not ...:
                return default
            raise ResourceDoesNotExist(f"Cannot find job {name!r}")

        if default is not ...:
            return default
        raise ValueError("Either obj, job_id, or name must be provided")

    def _get_by_id(self, job_id: int) -> "Job":
        from .job import Job

        sdk = self.client.workspace_client().jobs
        raw = sdk.get(job_id=job_id)
        job_name = raw.settings.name if raw.settings else None
        if job_name:
            _set_cached_name(self.client, job_name, raw.job_id)
        return Job(service=self, job_id=raw.job_id, name=job_name, details=raw)

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
        resolved_id, resolved_name = _resolve_job_obj(obj, job_id=job_id, name=name)

        existing = self.get(job_id=resolved_id, name=resolved_name, default=None)

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

        final_name = resolved_name or name
        if not final_name:
            raise ValueError("name is required to create a new job")

        return self.create(
            name=final_name,
            tasks=tasks,
            cluster=cluster,
            permissions=permissions,
            tags=tags,
            timeout_seconds=timeout_seconds,
            max_concurrent_runs=max_concurrent_runs,
            **settings_kwargs,
        )

    def delete(
        self,
        obj: "Job | int | str | None" = None,
        *,
        job_id: int | None = None,
        name: str | None = None,
    ) -> None:
        job = self.get(obj, job_id=job_id, name=name)
        job.delete()

    # ------------------------------------------------------------------ #
    # Submit (one-time run, no persisted job)
    # ------------------------------------------------------------------ #

    def submit(
        self,
        *,
        run_name: str | None = None,
        tasks: list["SubmitTask | dict"] | None = None,
        cluster: "Cluster | str | None" = None,
        environment: "Any | str | None" = None,
        timeout_seconds: int | None = None,
        wait: WaitingConfigArg = False,
        raise_error: bool = True,
        **submit_kwargs: Any,
    ) -> "JobRun":
        """Submit a one-time run without creating a persisted job.

        Mirrors the SDK ``jobs.submit`` one-shot API: the run executes
        immediately and is not backed by a saved job definition (so it
        has a ``run_id`` but no ``job_id``).  Returns an awaitable
        :class:`JobRun`, exactly like :meth:`Job.run`.

        Tasks are :class:`SubmitTask` (not :class:`Task`) — the SDK uses a
        distinct task type for one-time runs.  Dicts are coerced.  When a
        ``cluster`` is given, it backfills ``existing_cluster_id`` on any
        task that doesn't already pin a cluster.

        Parameters
        ----------
        run_name:
            Display name for the run.
        tasks:
            List of :class:`SubmitTask` or dicts.
        cluster:
            Default cluster for tasks that don't specify their own.
        environment:
            **Serverless** environment for tasks that don't pin a cluster.
            A :class:`JobEnvironment` is used directly; a ``str`` names a
            seeded serverless base environment (or a workspace path to its
            ``.yml`` spec) present in the shared environment path. When
            given, it's attached to the run's ``environments`` and its key
            is backfilled onto every cluster-less, key-less task — so the
            verbose ``environments=[…]`` + per-task ``environment_key``
            boilerplate collapses to one argument. ``None`` (default)
            leaves submit behaviour unchanged.
        timeout_seconds:
            Overall run timeout.
        wait:
            ``False`` (default) = fire-and-forget; ``True`` = block until
            terminal; a number = timeout in seconds.
        raise_error:
            Raise on terminal failure when waiting.
        """
        from .run import JobRun

        sdk = self.client.workspace_client().jobs

        checked_tasks: list[SubmitTask] | None = None
        if tasks:
            cid = _resolve_cluster_id(cluster)
            checked_tasks = []
            for t in tasks:
                if isinstance(t, dict):
                    t = SubmitTask(**t)
                elif not isinstance(t, SubmitTask):
                    raise TypeError(
                        f"Expected SubmitTask or dict, got {type(t).__name__}"
                    )
                if cid is not None and t.existing_cluster_id is None and t.new_cluster is None:
                    t.existing_cluster_id = cid
                checked_tasks.append(t)

        # Serverless environment defaulting: resolve an explicit ``environment``
        # (a JobEnvironment, or a seeded base-environment name/path) and attach
        # it to the run, backfilling its key onto every cluster-less, key-less
        # task. Caller-supplied ``environments=[…]`` still wins.
        environments = submit_kwargs.pop("environments", None)
        if environment is not None and environments is None:
            resolved = _resolve_submit_environment(self.client, environment)
            if resolved is not None:
                environments = [resolved]
        if environments:
            env_key = environments[0].environment_key
            for t in checked_tasks or []:
                if (
                    t.existing_cluster_id is None
                    and t.new_cluster is None
                    and getattr(t, "environment_key", None) is None
                ):
                    t.environment_key = env_key

        LOGGER.debug("Submitting one-time run %r", run_name)
        response = sdk.submit(
            run_name=run_name,
            tasks=checked_tasks,
            timeout_seconds=timeout_seconds,
            environments=environments,
            **submit_kwargs,
        )

        # Fetch the run so the handle carries job_id + run_page_url; this
        # makes repr/explore_url resolve to the canonical run page rather
        # than the vanity-host jobs list.
        raw = sdk.get_run(run_id=response.run_id)
        job_run = JobRun(
            service=JobRuns(client=self.client),
            run_id=raw.run_id,
            job_id=raw.job_id,
            details=raw,
        )

        LOGGER.info("Submitted one-time run %s (%r) — %r", raw.run_id, run_name, job_run)

        if wait is not False:
            job_run.wait(wait=wait, raise_error=raise_error)

        return job_run


# ---------------------------------------------------------------------------
# JobRuns service
# ---------------------------------------------------------------------------


class JobRuns(DatabricksService):
    """Collection-level run management — listing and retrieving runs.

    Getter methods accept a positional ``obj`` that can be a
    :class:`JobRun`, an ``int`` (run id), or a numeric ``str``::

        job_runs.get(98765)        # by run id
        job_runs.get("98765")      # numeric string → by run id
    """

    def __getitem__(self, key: int | str) -> "JobRun":
        return self.get(key)

    def list(
        self,
        obj: "Job | int | str | None" = None,
        *,
        job_id: int | None = None,
        name: str | None = None,
        active_only: bool = False,
        completed_only: bool = False,
        expand_tasks: bool = False,
        limit: int | None = None,
    ) -> Iterator["JobRun"]:
        """List runs, optionally scoped to a job.

        Parameters
        ----------
        obj:
            Positional shortcut for the owning job — ``Job``, ``int``
            (job id), or ``str`` (numeric → job id, otherwise → job name
            resolved via :meth:`Jobs.get`).
        """
        from .run import JobRun

        resolved_job_id, resolved_name = _resolve_job_obj(obj, job_id=job_id, name=name)

        if resolved_name and not resolved_job_id:
            from .job import Job
            found = Jobs(client=self.client).get(name=resolved_name, default=None)
            if found is not None:
                resolved_job_id = found.job_id

        sdk = self.client.workspace_client().jobs
        cnt = 0
        cap = limit if limit else float("inf")

        for raw in sdk.list_runs(
            job_id=resolved_job_id,
            active_only=active_only,
            completed_only=completed_only,
            expand_tasks=expand_tasks,
        ):
            run = JobRun(service=self, run_id=raw.run_id, job_id=raw.job_id, details=raw)
            yield run
            cnt += 1
            if cnt >= cap:
                break

    def get(
        self,
        obj: "JobRun | int | str | None" = None,
        *,
        run_id: int | None = None,
        default: Any = ...,
    ) -> Optional["JobRun"]:
        """Retrieve a single run by id.

        Parameters
        ----------
        obj:
            Positional shortcut — ``JobRun`` (returned as-is), ``int``
            (run id), or numeric ``str``.
        run_id:
            Explicit run id.
        default:
            Returned when the run is not found.  ``...`` raises.
        """
        from .run import JobRun

        resolved_run_id, _ = _resolve_run_obj(obj, run_id=run_id, job_id=None)

        if resolved_run_id is None:
            if default is not ...:
                return default
            raise ValueError("Either obj or run_id must be provided")

        try:
            sdk = self.client.workspace_client().jobs
            raw = sdk.get_run(run_id=resolved_run_id)
            return JobRun(service=self, run_id=raw.run_id, job_id=raw.job_id, details=raw)
        except ResourceDoesNotExist:
            if default is not ...:
                return default
            raise
