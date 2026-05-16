"""Per-table applier-job settings for staged async inserts.

A staged async insert lands a Parquet payload plus a JSON metadata
file under the target table's ``stg_<table>/.sql/async/insert/``
folder; a downstream Databricks Job drains them back into the
target. :meth:`AsyncInsertJob.settings` returns the full
:class:`JobSettings`-shaped kwargs dict for the per-table applier
job — splat into :meth:`Jobs.get_or_create` /
:meth:`Jobs.create_or_update` / :meth:`Job.deploy`. The matching
:meth:`Table.async_job` helper drives the lifecycle in one call.

Identity is keyed off ``(catalog_name, schema_name, table_name)`` —
one job per target table, watching the table's own staging
``data/`` folder via a file-arrival trigger so newly staged
payloads kick the applier without a cron schedule.
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
)

from databricks.sdk.service.jobs import (
    CronSchedule,
    FileArrivalTriggerConfiguration,
    JobParameterDefinition,
    NotebookTask,
    PauseStatus,
    Task,
    TriggerSettings,
)

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.fs import VolumePath

    from .async_write import AsyncInsert
    from .table import Table


__all__ = ["AsyncInsertJob"]

LOGGER = logging.getLogger(__name__)


class AsyncInsertJob:
    """Namespace for the per-table async-insert applier job spec.

    Not instantiated. :meth:`settings` returns the full kwargs dict
    for the workspace Job; :meth:`load` reads the staged
    :class:`AsyncInsert` records back from the table's staging
    folder.
    """

    JOB_NAME_PREFIX: ClassVar[str] = "ygg-async-insert"
    DATA_SUBDIR: ClassVar[str] = "data"
    DEFAULT_MIN_TIME_BETWEEN_TRIGGERS_SECONDS: ClassVar[int] = 60
    DEFAULT_WAIT_AFTER_LAST_CHANGE_SECONDS: ClassVar[int] = 60

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise TypeError(
            "AsyncInsertJob is a namespace, not a class — call "
            "AsyncInsertJob.settings(table, ...) or table.async_job() "
            "to get a Job."
        )

    # ------------------------------------------------------------------ #
    # Job spec
    # ------------------------------------------------------------------ #
    @staticmethod
    def job_name(table: "Table") -> str:
        """Canonical applier-job name for *table*."""
        cat, sch, tbl = AsyncInsertJob._identity(table)
        return f"{AsyncInsertJob.JOB_NAME_PREFIX}-{cat}-{sch}-{tbl}"

    @staticmethod
    def trigger_folder(table: "Table") -> "VolumePath":
        """Staging ``data/`` folder watched by the file-arrival trigger."""
        return table.staging_folder(temporary=False, async_write=True).joinpath(
            AsyncInsertJob.DATA_SUBDIR,
        )

    @staticmethod
    def trigger_url(table: "Table") -> str:
        """File-arrival URL — ``dbfs:/Volumes/<cat>/<sch>/<vol>/...data/``."""
        path = AsyncInsertJob.trigger_folder(table).full_path()
        if not path.endswith("/"):
            path = path + "/"
        return f"dbfs:{path}"

    @staticmethod
    def settings(
        table: "Table",
        *,
        task: Any = None,
        notebook_path: Optional[str] = None,
        notebook_warehouse_id: Optional[str] = None,
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
        tags: Optional[Mapping[str, str]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the full kwargs dict for the per-table applier Job.

        Splat into :meth:`Jobs.get_or_create` / :meth:`Jobs.create_or_update`
        / :meth:`Job.deploy`. Every settable :class:`JobSettings` field
        is resolved off *table* (name, parameters, description) or the
        caller's overrides. Cron strings are coerced to
        :class:`CronSchedule`; file-arrival trigger watches the table's
        own staging ``data/`` folder unless ``file_arrival_trigger=False``.
        """
        cat, sch, tbl = AsyncInsertJob._identity(table)

        out: Dict[str, Any] = {
            "name": AsyncInsertJob.job_name(table),
            "tasks": AsyncInsertJob._resolve_tasks(
                table=table,
                task=task,
                notebook_path=notebook_path,
                notebook_warehouse_id=notebook_warehouse_id,
                notebook_base_parameters=notebook_base_parameters,
                cat=cat, sch=sch, tbl=tbl,
            ),
            "schedule": AsyncInsertJob._resolve_schedule(
                schedule=schedule,
                timezone_id=schedule_timezone,
                pause_status=schedule_pause_status,
            ),
            "parameters": AsyncInsertJob._resolve_parameters(
                cat=cat, sch=sch, tbl=tbl, overrides=parameters,
            ),
            "description": (
                description
                or f"Apply staged async inserts for {cat}.{sch}.{tbl}"
            ),
            "tags": dict(tags) if tags else None,
        }
        if file_arrival_trigger:
            out["trigger"] = TriggerSettings(
                file_arrival=FileArrivalTriggerConfiguration(
                    url=AsyncInsertJob.trigger_url(table),
                    min_time_between_triggers_seconds=min_time_between_triggers_seconds,
                    wait_after_last_change_seconds=wait_after_last_change_seconds,
                ),
                pause_status=AsyncInsertJob._coerce_pause(trigger_pause_status),
            )
        return out

    # ------------------------------------------------------------------ #
    # Read staged records (called from the applier task body)
    # ------------------------------------------------------------------ #
    @staticmethod
    def load(
        table: "Table",
        *,
        path: "VolumePath | str | None" = None,
        merge: bool = True,
        client: "DatabricksClient | None" = None,
    ) -> List["AsyncInsert"]:
        """Read the staged :class:`AsyncInsert` records under *path*.

        *path* defaults to *table*'s own
        ``stg_<table>/.sql/async/insert`` folder. With ``merge=True``
        (default) returns one merged record per target — every
        overlapping append folds into a single ``INSERT INTO`` and a
        trailing overwrite drops everything before it (see
        :meth:`AsyncInsert.merge`). Pass ``merge=False`` to get the
        raw per-file records back.
        """
        from .async_write import AsyncInsert, _iter_records

        if path is None:
            path = table.staging_folder(temporary=False, async_write=True)
        if client is None:
            client = getattr(table, "client", None)
        if merge:
            return AsyncInsert.merge(path, client=client)
        return list(_iter_records(path, client=client))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _identity(table: "Table") -> tuple[str, str, str]:
        cat, sch, tbl = table.catalog_name, table.schema_name, table.table_name
        if not cat or not sch or not tbl:
            raise ValueError(
                f"AsyncInsertJob needs a fully-qualified table "
                f"(catalog.schema.table) — got {table!r}."
            )
        return cat, sch, tbl

    @staticmethod
    def _resolve_tasks(
        *,
        table: "Table",
        task: Any,
        notebook_path: Optional[str],
        notebook_warehouse_id: Optional[str],
        notebook_base_parameters: Optional[Mapping[str, str]],
        cat: str, sch: str, tbl: str,
    ) -> List[Task]:
        if task is not None:
            return [task] if isinstance(task, Task) else list(task)

        if notebook_path:
            base_params: Dict[str, str] = {
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
            "AsyncInsertJob.settings called without ``task`` or "
            "``notebook_path`` — the resulting job for %s will have no "
            "tasks. Attach tasks later via ``Jobs.create_or_update(...)``.",
            table.full_name(safe=False),
        )
        return []

    @staticmethod
    def _resolve_schedule(
        *, schedule: Any, timezone_id: str, pause_status: Any,
    ) -> Optional[CronSchedule]:
        if schedule is None or isinstance(schedule, CronSchedule):
            return schedule
        if isinstance(schedule, str):
            return CronSchedule(
                quartz_cron_expression=schedule,
                timezone_id=timezone_id,
                pause_status=AsyncInsertJob._coerce_pause(pause_status),
            )
        raise TypeError(
            f"AsyncInsertJob: ``schedule`` must be a CronSchedule, a "
            f"Quartz cron string, or None — got {type(schedule).__name__}."
        )

    @staticmethod
    def _resolve_parameters(
        *,
        cat: str, sch: str, tbl: str,
        overrides: Optional[Mapping[str, str]],
    ) -> List[JobParameterDefinition]:
        job_params: List[JobParameterDefinition] = [
            JobParameterDefinition(name="catalog_name", default=cat),
            JobParameterDefinition(name="schema_name", default=sch),
            JobParameterDefinition(name="table_name", default=tbl),
        ]
        if overrides:
            existing = {p.name: p for p in job_params}
            for k, v in overrides.items():
                if k in existing:
                    existing[k].default = str(v)
                else:
                    job_params.append(
                        JobParameterDefinition(name=str(k), default=str(v))
                    )
        return job_params

    @staticmethod
    def _coerce_pause(pause: Any) -> Any:
        if isinstance(pause, str):
            return PauseStatus(pause.upper())
        return pause
