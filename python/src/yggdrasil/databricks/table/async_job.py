"""Per-table applier job for staged async inserts.

One Databricks Job per :class:`Table`. The job drains the table's
``stg_<table>/.sql/async/insert`` staging folder (via
:meth:`AsyncInsert.apply_schema` or a user-supplied task) and is
fired automatically by a *file-arrival trigger* watching the
``data/`` sub-folder — every newly staged Parquet payload kicks the
applier without needing a cron schedule.

The class is a thin pairing of (:class:`Table`, :class:`Job`):

- :attr:`table` is the source of identity. The job name, the trigger
  URL, and the notebook base-parameters are derived from it.
- :attr:`job` is the live :class:`Job` resource for run / refresh /
  delete.

CRUD classmethods all take a :class:`Table` and route through the
workspace :class:`Jobs` service — :meth:`find`, :meth:`get`,
:meth:`create`, :meth:`get_or_create`, :meth:`create_or_update`, and
:meth:`delete`. Instance helpers :meth:`run`, :meth:`update`, and
:meth:`refresh` forward to the underlying :class:`Job`.
"""
from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from databricks.sdk.service.jobs import (
    CronSchedule,
    FileArrivalTriggerConfiguration,
    JobAccessControlRequest,
    JobParameterDefinition,
    NotebookTask,
    PauseStatus,
    Task,
    TriggerSettings,
)

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.fs import VolumePath
    from yggdrasil.databricks.jobs.job import Job
    from yggdrasil.databricks.jobs.run import JobRun
    from yggdrasil.databricks.jobs.service import Jobs

    from .table import Table


__all__ = ["AsyncInsertJob"]

LOGGER = logging.getLogger(__name__)

# Defaults for the file-arrival trigger debounce window. Without
# these a busy staging folder retriggers the applier as fast as
# Databricks can detect new files; 60s on either side gives the
# applier a chance to drain a small batch in one run.
DEFAULT_MIN_TIME_BETWEEN_TRIGGERS_SECONDS: int = 60
DEFAULT_WAIT_AFTER_LAST_CHANGE_SECONDS: int = 60


class AsyncInsertJob:
    """Applier :class:`Job` bound to a single :class:`Table`.

    Identity is keyed off ``(catalog_name, schema_name, table_name)``
    on the bound table — one job per target table, watching the
    table's own ``stg_<table>/.sql/async/insert/data/`` folder via a
    file-arrival trigger.
    """

    JOB_NAME_PREFIX: ClassVar[str] = "ygg-async-insert"

    def __init__(self, table: "Table", job: "Job"):
        self.table = table
        self.job = job

    def __repr__(self) -> str:
        return (
            f"AsyncInsertJob(table={self.table.full_name(safe=False)!r}, "
            f"job_id={self.job.job_id!r})"
        )

    # ------------------------------------------------------------------ #
    # Identity helpers
    # ------------------------------------------------------------------ #
    @property
    def job_id(self) -> Optional[int]:
        return self.job.job_id

    @property
    def name(self) -> str:
        return type(self).job_name(self.table)

    @classmethod
    def job_name(cls, table: "Table") -> str:
        """Canonical applier-job name for *table*."""
        cat, sch, tbl = cls._identity(table)
        return f"{cls.JOB_NAME_PREFIX}-{cat}-{sch}-{tbl}"

    @classmethod
    def trigger_folder(cls, table: "Table") -> "VolumePath":
        """Staging ``data/`` folder watched by the file-arrival trigger."""
        from .async_write import ASYNC_INSERT_DATA_SUBDIR

        return table.staging_folder(temporary=False, async_write=True).joinpath(
            ASYNC_INSERT_DATA_SUBDIR,
        )

    @classmethod
    def trigger_url(cls, table: "Table") -> str:
        """File-arrival trigger URL — ``dbfs:/Volumes/<cat>/<sch>/<vol>/...data/``.

        Databricks UC volume URLs in file-arrival triggers must be the
        ``dbfs:/Volumes/...`` form with a trailing slash so the prefix
        match catches every new file under it.
        """
        path = cls.trigger_folder(table).full_path()
        if not path.endswith("/"):
            path = path + "/"
        return f"dbfs:{path}"

    # ------------------------------------------------------------------ #
    # CRUD — classmethods keyed off Table
    # ------------------------------------------------------------------ #
    @classmethod
    def find(
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
    ) -> "AsyncInsertJob | None":
        """Return the applier job bound to *table*, or ``None`` if absent."""
        jobs = cls._resolve_jobs(table, jobs=jobs, client=client)
        found = jobs.find(name=cls.job_name(table))
        return cls(table, found) if found is not None else None

    @classmethod
    def get(
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
    ) -> "AsyncInsertJob":
        """Like :meth:`find` but raises when the job doesn't exist."""
        jobs = cls._resolve_jobs(table, jobs=jobs, client=client)
        return cls(table, jobs.get(name=cls.job_name(table)))

    @classmethod
    def create(
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        task: Any = None,
        notebook_path: str | None = None,
        notebook_warehouse_id: str | None = None,
        notebook_base_parameters: Optional[Mapping[str, str]] = None,
        schedule: Any = None,
        schedule_timezone: str = "UTC",
        schedule_pause_status: Any = None,
        file_arrival_trigger: bool = True,
        min_time_between_triggers_seconds: int = (
            DEFAULT_MIN_TIME_BETWEEN_TRIGGERS_SECONDS
        ),
        wait_after_last_change_seconds: int = (
            DEFAULT_WAIT_AFTER_LAST_CHANGE_SECONDS
        ),
        trigger_pause_status: Any = None,
        parameters: Optional[Mapping[str, str]] = None,
        description: str | None = None,
        permissions: Optional[List[Union[str, JobAccessControlRequest]]] = None,
        tags: Optional[Mapping[str, str]] = None,
        **settings: Any,
    ) -> "AsyncInsertJob":
        """Create the applier job for *table* (errors if it already exists)."""
        jobs = cls._resolve_jobs(table, jobs=jobs, client=client)
        return cls(
            table,
            jobs.create(
                **cls._build_create_kwargs(
                    table=table,
                    task=task,
                    notebook_path=notebook_path,
                    notebook_warehouse_id=notebook_warehouse_id,
                    notebook_base_parameters=notebook_base_parameters,
                    schedule=schedule,
                    schedule_timezone=schedule_timezone,
                    schedule_pause_status=schedule_pause_status,
                    file_arrival_trigger=file_arrival_trigger,
                    min_time_between_triggers_seconds=min_time_between_triggers_seconds,
                    wait_after_last_change_seconds=wait_after_last_change_seconds,
                    trigger_pause_status=trigger_pause_status,
                    parameters=parameters,
                    description=description,
                    permissions=permissions,
                    tags=tags,
                    settings=settings,
                ),
            ),
        )

    @classmethod
    def get_or_create(
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        **create_kwargs: Any,
    ) -> "AsyncInsertJob":
        """Return the existing applier job for *table* or create one."""
        found = cls.find(table, jobs=jobs, client=client)
        if found is not None:
            return found
        return cls.create(table, jobs=jobs, client=client, **create_kwargs)

    @classmethod
    def create_or_update(
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        task: Any = None,
        notebook_path: str | None = None,
        notebook_warehouse_id: str | None = None,
        notebook_base_parameters: Optional[Mapping[str, str]] = None,
        schedule: Any = None,
        schedule_timezone: str = "UTC",
        schedule_pause_status: Any = None,
        file_arrival_trigger: bool = True,
        min_time_between_triggers_seconds: int = (
            DEFAULT_MIN_TIME_BETWEEN_TRIGGERS_SECONDS
        ),
        wait_after_last_change_seconds: int = (
            DEFAULT_WAIT_AFTER_LAST_CHANGE_SECONDS
        ),
        trigger_pause_status: Any = None,
        parameters: Optional[Mapping[str, str]] = None,
        description: str | None = None,
        permissions: Optional[List[Union[str, JobAccessControlRequest]]] = None,
        tags: Optional[Mapping[str, str]] = None,
        **settings: Any,
    ) -> "AsyncInsertJob":
        """Idempotent upsert — existing settings are replaced with the new spec."""
        jobs = cls._resolve_jobs(table, jobs=jobs, client=client)
        kwargs = cls._build_create_kwargs(
            table=table,
            task=task,
            notebook_path=notebook_path,
            notebook_warehouse_id=notebook_warehouse_id,
            notebook_base_parameters=notebook_base_parameters,
            schedule=schedule,
            schedule_timezone=schedule_timezone,
            schedule_pause_status=schedule_pause_status,
            file_arrival_trigger=file_arrival_trigger,
            min_time_between_triggers_seconds=min_time_between_triggers_seconds,
            wait_after_last_change_seconds=wait_after_last_change_seconds,
            trigger_pause_status=trigger_pause_status,
            parameters=parameters,
            description=description,
            permissions=permissions,
            tags=tags,
            settings=settings,
        )
        LOGGER.info(
            "Creating-or-updating async-insert job %r for table %s "
            "(file_arrival=%s schedule=%r tasks=%d)",
            kwargs["name"],
            table.full_name(safe=False),
            file_arrival_trigger,
            kwargs.get("schedule").quartz_cron_expression
            if kwargs.get("schedule") is not None else None,
            len(kwargs.get("tasks") or []),
        )
        return cls(table, jobs.create_or_update(**kwargs))

    @classmethod
    def delete(
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
    ) -> None:
        """Delete the applier job for *table* (no-op if it does not exist)."""
        jobs = cls._resolve_jobs(table, jobs=jobs, client=client)
        jobs.delete(name=cls.job_name(table))

    # ------------------------------------------------------------------ #
    # Instance forwarders
    # ------------------------------------------------------------------ #
    def refresh(self) -> "AsyncInsertJob":
        self.job.refresh()
        return self

    def update(self, **kwargs: Any) -> "AsyncInsertJob":
        self.job.update(**kwargs)
        return self

    def run(self, **kwargs: Any) -> "JobRun":
        return self.job.run(**kwargs)

    def delete_job(self) -> None:
        self.job.delete()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _identity(table: "Table") -> Tuple[str, str, str]:
        """Return the (catalog, schema, table) triple, validating presence."""
        cat = table.catalog_name
        sch = table.schema_name
        tbl = table.table_name
        if not cat or not sch or not tbl:
            raise ValueError(
                f"AsyncInsertJob needs a fully-qualified table "
                f"(catalog.schema.table) — got {table!r}."
            )
        return cat, sch, tbl

    @staticmethod
    def _resolve_jobs(
        table: "Table",
        *,
        jobs: "Jobs | None",
        client: "DatabricksClient | None",
    ) -> "Jobs":
        if jobs is not None:
            return jobs
        if client is None:
            client = getattr(table, "client", None)
            if client is None:
                from yggdrasil.databricks.client import DatabricksClient
                client = DatabricksClient.current()
        return client.jobs

    @classmethod
    def _build_create_kwargs(
        cls,
        *,
        table: "Table",
        task: Any,
        notebook_path: str | None,
        notebook_warehouse_id: str | None,
        notebook_base_parameters: Optional[Mapping[str, str]],
        schedule: Any,
        schedule_timezone: str,
        schedule_pause_status: Any,
        file_arrival_trigger: bool,
        min_time_between_triggers_seconds: int,
        wait_after_last_change_seconds: int,
        trigger_pause_status: Any,
        parameters: Optional[Mapping[str, str]],
        description: str | None,
        permissions: Optional[List[Union[str, JobAccessControlRequest]]],
        tags: Optional[Mapping[str, str]],
        settings: Mapping[str, Any],
    ) -> dict:
        cat, sch, tbl = cls._identity(table)
        name = cls.job_name(table)

        resolved_tasks = cls._resolve_tasks(
            task=task,
            notebook_path=notebook_path,
            notebook_warehouse_id=notebook_warehouse_id,
            notebook_base_parameters=notebook_base_parameters,
            table=table,
        )

        cron_schedule = cls._resolve_schedule(
            schedule=schedule,
            timezone_id=schedule_timezone,
            pause_status=schedule_pause_status,
        )

        trigger_settings = (
            cls._build_trigger(
                table=table,
                min_time_between_triggers_seconds=min_time_between_triggers_seconds,
                wait_after_last_change_seconds=wait_after_last_change_seconds,
                pause_status=trigger_pause_status,
            )
            if file_arrival_trigger else None
        )

        job_params: list[JobParameterDefinition] = [
            JobParameterDefinition(name="catalog_name", default=cat),
            JobParameterDefinition(name="schema_name", default=sch),
            JobParameterDefinition(name="table_name", default=tbl),
        ]
        if parameters:
            existing = {p.name: p for p in job_params}
            for k, v in parameters.items():
                if k in existing:
                    existing[k].default = str(v)
                else:
                    job_params.append(
                        JobParameterDefinition(name=str(k), default=str(v))
                    )

        if description is None:
            description = (
                f"Apply staged async inserts for {cat}.{sch}.{tbl}"
            )

        kwargs: dict[str, Any] = {
            "name": name,
            "tasks": resolved_tasks,
            "schedule": cron_schedule,
            "parameters": job_params,
            "description": description,
            "permissions": permissions,
            "tags": dict(tags) if tags else None,
            **settings,
        }
        if trigger_settings is not None:
            kwargs["trigger"] = trigger_settings
        return kwargs

    @staticmethod
    def _resolve_tasks(
        *,
        task: Any,
        notebook_path: str | None,
        notebook_warehouse_id: str | None,
        notebook_base_parameters: Optional[Mapping[str, str]],
        table: "Table",
    ) -> List[Task]:
        """Normalize the caller's task spec into a list of :class:`Task`."""
        if task is not None:
            if isinstance(task, Task):
                return [task]
            return list(task)

        if notebook_path:
            cat, sch, tbl = AsyncInsertJob._identity(table)
            base_params: dict[str, str] = {
                "catalog_name": cat,
                "schema_name": sch,
                "table_name": tbl,
            }
            if notebook_base_parameters:
                base_params.update(
                    {str(k): str(v) for k, v in notebook_base_parameters.items()}
                )
            return [
                Task(
                    task_key="apply",
                    notebook_task=NotebookTask(
                        notebook_path=notebook_path,
                        warehouse_id=notebook_warehouse_id,
                        base_parameters=base_params,
                    ),
                )
            ]

        LOGGER.warning(
            "AsyncInsertJob.create called without ``task`` or "
            "``notebook_path`` — the resulting job for %s will have no "
            "tasks. Attach tasks later via ``Jobs.create_or_update(...)``.",
            table.full_name(safe=False),
        )
        return []

    @staticmethod
    def _resolve_schedule(
        *,
        schedule: Any,
        timezone_id: str,
        pause_status: Any,
    ) -> "CronSchedule | None":
        """Coerce *schedule* into a :class:`CronSchedule` (or ``None``)."""
        if schedule is None:
            return None

        if isinstance(schedule, CronSchedule):
            return schedule

        if isinstance(schedule, str):
            resolved_pause: Any = pause_status
            if isinstance(resolved_pause, str):
                resolved_pause = PauseStatus(resolved_pause.upper())
            return CronSchedule(
                quartz_cron_expression=schedule,
                timezone_id=timezone_id,
                pause_status=resolved_pause,
            )

        raise TypeError(
            f"AsyncInsertJob: ``schedule`` must be a CronSchedule, a "
            f"Quartz cron string, or None — got {type(schedule).__name__}."
        )

    @classmethod
    def _build_trigger(
        cls,
        *,
        table: "Table",
        min_time_between_triggers_seconds: int,
        wait_after_last_change_seconds: int,
        pause_status: Any,
    ) -> TriggerSettings:
        """Build the :class:`TriggerSettings` watching the staging folder."""
        resolved_pause: Any = pause_status
        if isinstance(resolved_pause, str):
            resolved_pause = PauseStatus(resolved_pause.upper())

        return TriggerSettings(
            file_arrival=FileArrivalTriggerConfiguration(
                url=cls.trigger_url(table),
                min_time_between_triggers_seconds=min_time_between_triggers_seconds,
                wait_after_last_change_seconds=wait_after_last_change_seconds,
            ),
            pause_status=resolved_pause,
        )
