"""Per-table applier :class:`Job` for staged async inserts.

One Databricks Job per :class:`Table`. The job drains the table's
``stg_<table>/.sql/async/insert`` staging folder (via
:meth:`AsyncInsert.apply_schema` or a user-supplied task) and is
fired automatically by a *file-arrival trigger* watching the
``data/`` sub-folder — every newly staged Parquet payload kicks the
applier without needing a cron schedule.

:class:`AsyncInsertJob` inherits from :class:`Job`, so every Job
method (:meth:`refresh`, :meth:`update`, :meth:`run`,
:meth:`delete`, :meth:`runs`, :meth:`cancel_all_runs`,
:meth:`update_permissions`, …) flows through unchanged. The
class-level skeleton hooks (:meth:`default_name`,
:meth:`default_tasks`, :meth:`default_trigger`,
:meth:`default_parameters`, :meth:`default_description`) declare
what the job looks like; :meth:`Job.deploy` / :meth:`find_for` /
:meth:`get_for` / :meth:`delete_for` are inherited and drive the
lifecycle. The convenience aliases :meth:`create_or_update` /
:meth:`find` / :meth:`get` / :meth:`get_or_create` / :meth:`create`
preserve the table-first call shape callers already use.
"""
from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

from databricks.sdk.service.jobs import (
    FileArrivalTriggerConfiguration,
    Job as JobInfo,
    JobParameterDefinition,
    NotebookTask,
    PauseStatus,
    Task,
    TriggerSettings,
)

from yggdrasil.databricks.jobs.job import Job

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.fs import VolumePath
    from yggdrasil.databricks.jobs.service import Jobs

    from .async_write import AsyncInsert
    from .table import Table


__all__ = ["AsyncInsertJob"]

LOGGER = logging.getLogger(__name__)

# Defaults for the file-arrival trigger debounce window. Without
# these a busy staging folder retriggers the applier as fast as
# Databricks can detect new files; 60s on either side gives the
# applier a chance to drain a small batch in one run.
DEFAULT_MIN_TIME_BETWEEN_TRIGGERS_SECONDS: int = 60
DEFAULT_WAIT_AFTER_LAST_CHANGE_SECONDS: int = 60


class AsyncInsertJob(Job):
    """Applier :class:`Job` bound to a single :class:`Table`.

    Identity is keyed off ``(catalog_name, schema_name, table_name)``
    on the bound table — one job per target table, watching the
    table's own ``stg_<table>/.sql/async/insert/data/`` folder via a
    file-arrival trigger. The skeleton hooks (:meth:`default_name`,
    :meth:`default_tasks`, :meth:`default_trigger`, …) wire each
    :class:`JobSettings` field off the bound :class:`Table`, so
    callers only ever hand in the table plus per-deployment options.
    """

    JOB_NAME_PREFIX: ClassVar[str] = "ygg-async-insert"

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def _singleton_key(
        cls,
        table: "Table | None" = None,
        *,
        service: "Jobs | None" = None,
        job_id: int | None = None,
        job_name: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        # Derive job_name from *table* so two callers with the same
        # table collapse to one singleton even when the underlying
        # job_id hasn't been resolved yet.
        if not job_name and table is not None:
            try:
                job_name = cls.job_name_for(table)
            except ValueError:
                pass
        if service is None and table is not None:
            try:
                service = cls.resolve_jobs(table=table)
            except Exception:  # noqa: BLE001 — best-effort
                service = None
        return (cls, service, job_id, job_name)

    def __init__(
        self,
        table: "Table | None" = None,
        *,
        service: "Jobs | None" = None,
        job_id: int | None = None,
        job_name: str | None = None,
        details: Optional[JobInfo] = None,
        singleton_ttl: Any = ...,
    ):
        already_initialized = getattr(self, "_initialized", False)

        # Rebind a freshly-supplied table onto a previously-cached
        # singleton (e.g. instance came from a job_id-only lookup).
        if already_initialized:
            if table is not None and getattr(self, "table", None) is None:
                self.table = table
            return

        if table is not None and not job_name:
            job_name = type(self).job_name_for(table)
        if service is None and table is not None:
            service = type(self).resolve_jobs(table=table)

        super().__init__(
            service=service,
            job_id=job_id,
            job_name=job_name,
            details=details,
            singleton_ttl=singleton_ttl,
        )
        self.table = table

    def __repr__(self) -> str:
        table = getattr(self, "table", None)
        target = (
            table.full_name(safe=False)
            if table is not None and table.table_name else None
        )
        return (
            f"AsyncInsertJob(table={target!r}, job_id={self.job_id!r})"
        )

    # ------------------------------------------------------------------ #
    # Identity helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def job_name_for(cls, table: "Table") -> str:
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
    # Skeleton — hook overrides
    # ------------------------------------------------------------------ #
    @classmethod
    def resolve_jobs(
        cls,
        *,
        service: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        table: "Table | None" = None,
        **_context: Any,
    ) -> "Jobs":
        """Resolve the :class:`Jobs` service off *table* when available."""
        if service is not None:
            return service
        if client is None and table is not None:
            client = getattr(table, "client", None)
        return super().resolve_jobs(service=service, client=client)

    @classmethod
    def default_name(
        cls,
        *,
        table: "Table | None" = None,
        **_context: Any,
    ) -> Optional[str]:
        if table is None:
            return None
        return cls.job_name_for(table)

    # The "apply" task is declared once via the deferred class-level
    # decorator — no API calls fire at class-definition time. At deploy
    # the factory runs with the resolved context: caller can pass an
    # explicit ``task=`` (verbatim Task / list of Tasks) or a
    # ``notebook_path=`` (wrapped in a NotebookTask with the bound
    # table's identity as base parameters). When neither is supplied
    # the factory returns ``None`` and the job lands tasks-less.
    @Job.task_def("apply")
    def _apply_task(
        cls,
        *,
        table: "Table | None" = None,
        task: Any = None,
        notebook_path: Optional[str] = None,
        notebook_warehouse_id: Optional[str] = None,
        notebook_base_parameters: Optional[Mapping[str, str]] = None,
        **_context: Any,
    ) -> Any:
        if task is not None:
            return task if isinstance(task, Task) else None

        if notebook_path and table is not None:
            cat, sch, tbl = cls._identity(table)
            base_params: Dict[str, str] = {
                "catalog_name": cat,
                "schema_name": sch,
                "table_name": tbl,
            }
            if notebook_base_parameters:
                base_params.update(
                    {str(k): str(v) for k, v in notebook_base_parameters.items()}
                )
            return Task(
                task_key="apply",
                notebook_task=NotebookTask(
                    notebook_path=notebook_path,
                    warehouse_id=notebook_warehouse_id,
                    base_parameters=base_params,
                ),
            )

        return None

    @classmethod
    def default_tasks(
        cls,
        *,
        task: Any = None,
        **context: Any,
    ) -> List[Task]:
        """Honor ``task=[...]`` lists; otherwise fall through to ``@task_def``."""
        if isinstance(task, list):
            return list(task)
        return super().default_tasks(task=task, **context)

    @classmethod
    def default_trigger(
        cls,
        *,
        table: "Table | None" = None,
        file_arrival_trigger: bool = True,
        min_time_between_triggers_seconds: int = (
            DEFAULT_MIN_TIME_BETWEEN_TRIGGERS_SECONDS
        ),
        wait_after_last_change_seconds: int = (
            DEFAULT_WAIT_AFTER_LAST_CHANGE_SECONDS
        ),
        trigger_pause_status: Any = None,
        **_context: Any,
    ) -> Optional[TriggerSettings]:
        if not file_arrival_trigger or table is None:
            return None

        resolved_pause: Any = trigger_pause_status
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

    @classmethod
    def default_parameters(
        cls,
        *,
        table: "Table | None" = None,
        parameters: Optional[Mapping[str, str]] = None,
        **_context: Any,
    ) -> List[JobParameterDefinition]:
        if table is None:
            return []
        cat, sch, tbl = cls._identity(table)
        job_params: List[JobParameterDefinition] = [
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
        return job_params

    @classmethod
    def default_description(
        cls,
        *,
        table: "Table | None" = None,
        **_context: Any,
    ) -> Optional[str]:
        if table is None:
            return None
        cat, sch, tbl = cls._identity(table)
        return f"Apply staged async inserts for {cat}.{sch}.{tbl}"

    @classmethod
    def _wrap(
        cls,
        underlying: "Job",
        *,
        service: "Jobs | None" = None,
        table: "Table | None" = None,
        **_context: Any,
    ) -> "AsyncInsertJob":
        """Wire *underlying* into an :class:`AsyncInsertJob` bound to *table*."""
        return cls(
            table=table,
            service=service,
            job_id=underlying.job_id,
            job_name=underlying.job_name,
            details=getattr(underlying, "_details", None),
        )

    # ------------------------------------------------------------------ #
    # Convenience aliases — table-first call shape over Job.deploy / …
    # ------------------------------------------------------------------ #
    @classmethod
    def create_or_update(
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        **kwargs: Any,
    ) -> "AsyncInsertJob":
        """Alias: :meth:`Job.deploy` with *table* threaded through context."""
        return cls.deploy(service=jobs, client=client, table=table, **kwargs)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        **kwargs: Any,
    ) -> "AsyncInsertJob":
        """Alias: :meth:`Job.create_for` with *table* threaded through context."""
        return cls.create_for(service=jobs, client=client, table=table, **kwargs)

    @classmethod
    def find(  # type: ignore[override]
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
    ) -> "AsyncInsertJob | None":
        """Alias: :meth:`Job.find_for` with *table* threaded through context."""
        return cls.find_for(service=jobs, client=client, table=table)

    @classmethod
    def get(  # type: ignore[override]
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
    ) -> "AsyncInsertJob":
        """Alias: :meth:`Job.get_for` with *table* threaded through context."""
        return cls.get_for(service=jobs, client=client, table=table)

    @classmethod
    def get_or_create(  # type: ignore[override]
        cls,
        table: "Table",
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        **kwargs: Any,
    ) -> "AsyncInsertJob":
        """Alias: :meth:`Job.get_or_create` with *table* threaded through context."""
        return super().get_or_create(
            service=jobs, client=client, table=table, **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Discover staged inserts
    # ------------------------------------------------------------------ #
    def load_from_path(
        self,
        path: "VolumePath | str | None" = None,
        *,
        merge: bool = True,
    ) -> List["AsyncInsert"]:
        """Read the staged :class:`AsyncInsert` records under *path*.

        *path* defaults to the bound table's own
        ``stg_<table>/.sql/async/insert`` staging folder, so an
        applier task can simply call ``self.load_from_path()`` to
        discover everything currently queued for the target table.

        With ``merge=True`` (the default) returns one merged record
        per target — every overlapping append folds into a single
        ``INSERT INTO`` and a trailing overwrite drops everything
        before it (see :meth:`AsyncInsert.merge`). Pass ``merge=False``
        to get the raw per-file records back instead, useful when the
        caller wants to inspect each operation independently.
        """
        from .async_write import AsyncInsert, _iter_records

        if path is None:
            if self.table is None:
                raise ValueError(
                    f"AsyncInsertJob {self!r} has no bound table; pass an "
                    "explicit ``path`` to load_from_path."
                )
            path = self.table.staging_folder(
                temporary=False, async_write=True,
            )

        client = self.client if self.table is not None else None
        if merge:
            return AsyncInsert.merge(path, client=client)
        return list(_iter_records(path, client=client))

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
