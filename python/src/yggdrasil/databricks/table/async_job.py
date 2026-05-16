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

import contextlib
import datetime as _dt
import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
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
    LOCK_FILENAME: ClassVar[str] = ".lock"
    DEFAULT_MIN_TIME_BETWEEN_TRIGGERS_SECONDS: ClassVar[int] = 60
    DEFAULT_WAIT_AFTER_LAST_CHANGE_SECONDS: ClassVar[int] = 60
    DEFAULT_LOCK_TIMEOUT_SECONDS: ClassVar[float] = 600.0
    DEFAULT_LOCK_POLL_SECONDS: ClassVar[float] = 2.0

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

    # ------------------------------------------------------------------ #
    # Lock — coordinate concurrent applier runs against the same table
    # ------------------------------------------------------------------ #
    @staticmethod
    @contextlib.contextmanager
    def lock(
        table: "Table",
        *,
        path: "VolumePath | str | None" = None,
        timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        interval: float = DEFAULT_LOCK_POLL_SECONDS,
        client: "DatabricksClient | None" = None,
    ) -> Iterator["VolumePath"]:
        """Exclusive applier lock on *table*'s staging folder.

        Other processes scanning the same folder block on entry
        until any pre-existing :attr:`LOCK_FILENAME` (``.lock``) file
        is removed; then this process claims the lock by creating
        the file and yields its :class:`VolumePath`. On exit
        (success or failure) the lock is removed so the next
        process can proceed.

        ``timeout`` (seconds) caps the wait; ``0`` or negative means
        wait indefinitely. ``interval`` is the polling cadence.
        Stale locks (process crashed mid-apply) are cleared by the
        next caller after ``timeout`` elapses — the wait surfaces a
        :class:`TimeoutError` which the caller can catch and
        :meth:`force_unlock` past.
        """
        from yggdrasil.databricks.path import DatabricksPath

        if path is None:
            path = table.staging_folder(temporary=False, async_write=True)
        elif not hasattr(path, "joinpath"):
            path = DatabricksPath.from_(path, client=client)

        lock_path = path.joinpath(AsyncInsertJob.LOCK_FILENAME)

        # Wait for any existing lock to be released.
        deadline = (time.time() + timeout) if timeout and timeout > 0 else None
        first_wait = True
        while lock_path.exists():
            if first_wait:
                LOGGER.info(
                    "Waiting for applier lock %r to be released "
                    "(timeout=%.0fs interval=%.1fs)",
                    lock_path, timeout, interval,
                )
                first_wait = False
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timed out after {timeout:.0f}s waiting for applier "
                    f"lock {lock_path!r} to be released."
                )
            time.sleep(interval)

        # Claim the lock. The body marker (acquire timestamp) lands
        # in the file so a stale-lock diagnosis can read when the
        # holder last started without crawling logs.
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        LOGGER.info("Claiming applier lock %r (acquired_at=%s)", lock_path, now)
        lock_path.write_bytes(now.encode("utf-8"))
        try:
            yield lock_path
        finally:
            try:
                lock_path.remove(missing_ok=True, wait=False, recursive=False)
                LOGGER.info("Released applier lock %r", lock_path)
            except Exception:  # noqa: BLE001 — best-effort
                LOGGER.exception(
                    "Failed to release applier lock %r; manual cleanup "
                    "may be required.",
                    lock_path,
                )

    @staticmethod
    def force_unlock(
        table: "Table",
        *,
        path: "VolumePath | str | None" = None,
        client: "DatabricksClient | None" = None,
    ) -> None:
        """Drop a stale ``.lock`` file unconditionally."""
        from yggdrasil.databricks.path import DatabricksPath

        if path is None:
            path = table.staging_folder(temporary=False, async_write=True)
        elif not hasattr(path, "joinpath"):
            path = DatabricksPath.from_(path, client=client)
        lock_path = path.joinpath(AsyncInsertJob.LOCK_FILENAME)
        try:
            lock_path.remove(missing_ok=True, wait=False, recursive=False)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to force-unlock %r", lock_path)

    # ------------------------------------------------------------------ #
    # Default applier — staged onto the job by ``Table.async_job``
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_records(
        catalog_name: str,
        schema_name: str,
        table_name: str,
    ) -> None:
        """Default applier task body.

        Resolves the workspace client, looks up the target table,
        takes an exclusive :meth:`lock` on its staging folder, and
        applies every staged :class:`AsyncInsert` record against the
        target via :class:`AsyncWrite`. Concurrent applier runs
        block on the lock so the staging folder is drained by at
        most one process at a time.

        Used by :meth:`Table.async_job` as the staged Python task
        when the job doesn't exist yet — ``inspect.getsource`` is
        the runtime contract here, so this body must stay
        self-contained (no module-level closures, no decorators
        beyond ``@staticmethod``).
        """
        from yggdrasil.databricks.client import DatabricksClient
        from yggdrasil.databricks.table.async_job import AsyncInsertJob
        from yggdrasil.databricks.table.async_write import AsyncWrite

        client = DatabricksClient.current()
        engine = client.sql(
            catalog_name=catalog_name, schema_name=schema_name,
        )
        table = engine.table(table_name)

        with AsyncInsertJob.lock(table):
            records = AsyncInsertJob.load(table)
            if not records:
                return
            AsyncWrite.from_records(
                records,
                executor=engine.warehouse(),
                client=client,
                wait=True,
                raise_error=True,
            )
